import json
from pathlib import Path

import numpy as np
import pytest

from eeg_pipeline.contracts import EEGWindow, WindowCompleteness
from experiment_learning.assignment import (
    ModelSelectionSource,
    cli_model_assignment,
    load_model_assignment,
    shadow_condition,
)
from experiment_learning.contracts import Condition
from foundations.config import load_resolved_config
from foundations.contracts import GazeSample, SceneFrame
from foundations.events import JsonlEventLogger
from gaze_interaction.contracts import BoundingBox, Detection, TrackedObject, TrackedScene
from integration.live_input import LiveInputMerger
from integration.live_workflow import run_live_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Detector:
    def detect(self, frame: SceneFrame):
        del frame
        return (Detection(BoundingBox(0.2, 0.2, 0.8, 0.8), "laptop", 0.9),)


class _Tracker:
    def update(self, frame: SceneFrame, detections):
        detection = detections[0]
        return TrackedScene(
            frame.timestamp,
            (
                TrackedObject(
                    1,
                    detection.box,
                    detection.label,
                    detection.confidence,
                    frame.timestamp,
                ),
            ),
        )


class _StepClock:
    def __init__(self) -> None:
        self.value = -0.01

    def now(self) -> float:
        self.value += 0.01
        return self.value


class _FakeMindLink:
    instances: list["_FakeMindLink"] = []

    def __init__(self, *, clock, frame_queue_size, on_disconnect) -> None:
        del frame_queue_size
        self.clock = clock
        self.on_disconnect = on_disconnect
        self.calls: list[str] = []
        self.__class__.instances.append(self)

    def connect(self, **kwargs) -> None:
        del kwargs
        self.calls.append("connect")

    def calibrate(self, **kwargs):
        del kwargs
        self.calls.append("calibrate")
        return ("ok",)

    def start_capture(self, **callbacks) -> None:
        self.calls.append("start_capture")
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        callbacks["on_scene_frame"](SceneFrame(0.0, image))
        for timestamp in (0.01, 0.03, 0.05):
            callbacks["on_gaze_sample"](
                GazeSample(timestamp, 0.5, 0.5, True, 1.0)
            )

    def stop_capture(self) -> None:
        self.calls.append("stop_capture")

    def close(self) -> None:
        self.calls.append("close")


class _FakeGuardian:
    instances: list["_FakeGuardian"] = []

    def __init__(self, *, clock, **kwargs) -> None:
        del kwargs
        self.clock = clock
        self.calls: list[str] = []
        self.recording_done = False
        self.recording_id = "fake-live-recording"
        self.lost_sample_count = 0
        self.lost_block_count = 0
        self.__class__.instances.append(self)

    def connect(self) -> None:
        self.calls.append("connect")

    def check_battery(self) -> float:
        self.calls.append("battery")
        return 95.0

    def start_impedance(self, *, mains_frequency_hz: int) -> None:
        assert mains_frequency_hz == 60
        self.calls.append("impedance_start")

    def latest_impedance(self) -> float:
        return 12_000.0

    def stop_impedance(self) -> None:
        self.calls.append("impedance_stop")

    def start(self, *, recording_seconds: int) -> None:
        assert recording_seconds > 0
        assert self.clock() >= 0.0
        self.calls.append("start")

    def check_health(self) -> None:
        self.calls.append("health")

    def finalize_before(self, timestamp: float):
        del timestamp
        return ()

    def window(self, start: float, end: float) -> EEGWindow:
        return EEGWindow(
            start,
            end,
            (),
            None,
            None,
            WindowCompleteness.EMPTY,
        )

    def stop(self) -> None:
        self.calls.append("stop")

    def close(self) -> None:
        self.calls.append("close")


def _config(tmp_path: Path, subject: str, active: str = "G") -> dict:
    config = load_resolved_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    config["output_root"] = str(tmp_path / "runs")
    config["runtime"]["subject_id"] = subject
    config["runtime"]["active_model"] = active
    config["session"]["maximum_duration_seconds"] = 0.08
    config["processing"]["reorder_hold_seconds"] = 0.0
    config["processing"]["idle_sleep_seconds"] = 0.001
    config["display"]["enabled"] = False
    config["diagnostics"]["write_mindlink_metadata"] = False
    return config


def _run(config: dict, *, start: bool = True, poll_key=lambda: -1) -> Path:
    return run_live_experiment(
        config,
        detector=_Detector(),
        tracker=_Tracker(),
        mindlink_factory=_FakeMindLink,
        guardian_factory=_FakeGuardian,
        clock_factory=_StepClock,
        start_gate=lambda: start,
        poll_key=poll_key,
        guardian_token_loader=lambda **_: "fake-token",
    )


def test_cli_assignment_derives_shadow_and_round_trips(tmp_path: Path) -> None:
    assignment = cli_model_assignment("P001", 2, Condition.E)
    assert assignment.source is ModelSelectionSource.CLI
    assert assignment.shadow_condition is Condition.G
    assert shadow_condition(Condition.G) is Condition.E
    path = tmp_path / "assignment.json"
    from experiment_learning.assignment import save_model_assignment

    save_model_assignment(path, assignment)
    assert load_model_assignment(path) == assignment


def test_live_input_merger_orders_streams_and_makes_gaze_overflow_hard(
    tmp_path: Path,
) -> None:
    failures: list[tuple[str, BaseException]] = []
    now = [0.0]
    merger = LiveInputMerger(
        scene_queue_size=2,
        gaze_queue_size=1,
        reorder_hold_seconds=1.0,
        event_logger=JsonlEventLogger(tmp_path / "events.jsonl"),
        failure_callback=lambda source, error: failures.append((source, error)),
        monotonic=lambda: now[0],
    )
    frame = SceneFrame(0.0, np.zeros((2, 2, 3), dtype=np.uint8))
    merger.on_gaze(GazeSample(0.1, 0.5, 0.5, True, 1.0))
    merger.on_scene(frame)
    assert merger.pop_ready() == ("scene", frame)
    merger.on_gaze(GazeSample(0.2, 0.5, 0.5, True, 1.0))
    merger.on_gaze(GazeSample(0.3, 0.5, 0.5, True, 1.0))
    assert failures[-1][0] == "mindlink_gaze_queue_overflow"


def test_pre_space_abort_creates_no_assignment_or_completed_session(tmp_path: Path) -> None:
    run = _run(_config(tmp_path, "P-abort"), start=False)
    summary = json.loads((run / "attempt_summary.json").read_text(encoding="utf-8"))
    lineage = tmp_path / "runs" / "subjects" / "P-abort" / "lineage"
    assert summary["setup_aborted"] is True
    assert summary["attempt_started"] is False
    assert not list(lineage.glob("assignment_session_*.json"))
    assert not list(lineage.glob("completed_session_*.json"))
    assert not list(lineage.glob("policy_session_*.json"))
    assert "start_capture" not in _FakeMindLink.instances[-1].calls
    assert "start" not in _FakeGuardian.instances[-1].calls


def test_successful_sequential_sessions_advance_training_and_metrics(tmp_path: Path) -> None:
    first = _run(_config(tmp_path, "P-sequential", "G"))
    second = _run(_config(tmp_path, "P-sequential", "E"))
    first_summary = json.loads(
        (first / "attempt_summary.json").read_text(encoding="utf-8")
    )
    second_summary = json.loads(
        (second / "attempt_summary.json").read_text(encoding="utf-8")
    )
    lineage = tmp_path / "runs" / "subjects" / "P-sequential" / "lineage"
    assert first_summary["successful"] is True
    assert second_summary["successful"] is True
    assert (first_summary["session_number"], second_summary["session_number"]) == (1, 2)
    assert load_model_assignment(
        lineage / "assignment_session_001.json"
    ).shadow_condition is Condition.E
    assert load_model_assignment(
        lineage / "assignment_session_002.json"
    ).shadow_condition is Condition.G
    assert (lineage / "completed_session_001.json").is_file()
    assert (lineage / "completed_session_002.json").is_file()
    assert (lineage / "policy_session_003.json").is_file()
    participant = json.loads(
        (lineage / "participant_analysis_summary.json").read_text(encoding="utf-8")
    )
    assert participant["completed_sessions"] == 2
    assert set(participant["by_active_condition"]) == {"E", "G"}


def test_live_action_records_trigger_and_later_presentation_timestamp(
    tmp_path: Path,
) -> None:
    learning_override = tmp_path / "fast_dwell.yaml"
    learning_override.write_text(
        "cold_start_policy:\n"
        "  base_threshold_s: 0.02\n"
        "  minimum_e_threshold_s: 0.01\n"
        "  base_search_min_s: 0.01\n"
        "trainer:\n"
        "  base_min_s: 0.01\n"
        "  minimum_e_threshold_s: 0.01\n",
        encoding="utf-8",
    )
    config = _config(tmp_path, "P-presentation", "G")
    config["subsystem_configs"]["experiment_learning"]["override"] = str(
        learning_override
    )
    run = _run(config)
    events = [
        json.loads(line)
        for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    trigger = next(event for event in events if event["name"] == "integration_dwell_trigger")
    presented = next(
        event for event in events if event["name"] == "integration_action_presented"
    )
    record = next(
        event
        for event in events
        if event["name"] == "experiment_episode_training_record"
    )
    assert trigger["payload"]["presentation_timestamp"] > trigger["timestamp"]
    assert presented["timestamp"] == trigger["payload"]["presentation_timestamp"]
    assert record["payload"]["action_timestamp"] == presented["timestamp"]


def test_terminated_session_does_not_advance_and_retry_assignment_is_fixed(
    tmp_path: Path,
) -> None:
    keys = iter((ord("q"), -1, -1))
    terminated = _run(
        _config(tmp_path, "P-retry", "G"), poll_key=lambda: next(keys, -1)
    )
    summary = json.loads(
        (terminated / "attempt_summary.json").read_text(encoding="utf-8")
    )
    lineage = tmp_path / "runs" / "subjects" / "P-retry" / "lineage"
    assert summary["successful"] is False
    assert summary["stop_reason"] == "operator_terminated"
    assert (lineage / "assignment_session_001.json").is_file()
    assert not (lineage / "completed_session_001.json").exists()
    with pytest.raises(ValueError, match="retry active model differs"):
        _run(_config(tmp_path, "P-retry", "E"))


def test_cleanup_failure_does_not_publish_completed_session(tmp_path: Path) -> None:
    class FailingStopGuardian(_FakeGuardian):
        def stop(self) -> None:
            self.calls.append("stop")
            raise RuntimeError("Guardian stop failed")

    config = _config(tmp_path, "P-cleanup-failure", "G")
    with pytest.raises(RuntimeError, match="Guardian stop failed"):
        run_live_experiment(
            config,
            detector=_Detector(),
            tracker=_Tracker(),
            mindlink_factory=_FakeMindLink,
            guardian_factory=FailingStopGuardian,
            clock_factory=_StepClock,
            start_gate=lambda: True,
            poll_key=lambda: -1,
            guardian_token_loader=lambda **_: "fake-token",
        )
    lineage = (
        tmp_path / "runs" / "subjects" / "P-cleanup-failure" / "lineage"
    )
    assert (lineage / "assignment_session_001.json").is_file()
    assert not (lineage / "completed_session_001.json").exists()
    calls = FailingStopGuardian.instances[-1].calls
    assert calls.index("stop") < calls.index("close")
    assert "health" in calls[calls.index("stop") + 1 :]


def test_csv_and_controlled_trials_remain_inactive_in_main_runner(
    tmp_path: Path,
) -> None:
    csv_config = _config(tmp_path, "P-csv")
    csv_config["model_selection"]["source"] = "csv"
    csv_config["model_selection"]["csv"]["enabled"] = True
    with pytest.raises(ValueError, match="requires CLI model selection"):
        _run(csv_config)

    controlled_config = _config(tmp_path, "P-controlled")
    controlled_config["controlled_intention"]["enabled"] = True
    with pytest.raises(ValueError, match="placeholders and are not implemented"):
        _run(controlled_config)
