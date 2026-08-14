import json
from pathlib import Path

import pytest

from eeg_pipeline.contracts import EEGSample, EEGWindow, WindowCompleteness
from eeg_pipeline.guardian import GuardianPreflight
from experiment_learning.policy import load_frozen_policy
from experiment_learning.sessions import load_completed_session
from foundations.config import load_resolved_config
from integration.analysis import generate_analysis
from integration.orchestrator import ScheduledFeedbackPress, TimedFeedbackDriver
from integration.workflow import run_integrated_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_CONFIG = PROJECT_ROOT / "configs" / "integration.yaml"


def _config(tmp_path: Path, *, participant_id: str = "test-P001") -> dict:
    config = load_resolved_config(INTEGRATION_CONFIG)
    config["output_root"] = str(tmp_path / "runs")
    config["participant"]["id"] = participant_id
    config["participant"]["artifact_directory"] = str(tmp_path / "participant")
    config["session"]["id_prefix"] = "test-session"
    config["session"]["maximum_duration_seconds"] = 3.0
    return config


def _events(run_directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def _named(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event["name"] == name]


def test_synthetic_end_to_end_uses_schedule_frozen_policy_and_n_plus_one(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first_run = run_integrated_experiment(config)
    second_run = run_integrated_experiment(config)

    first_events = _events(first_run)
    second_events = _events(second_run)
    first_start = _named(first_events, "integration_session_started")[0]["payload"]
    second_start = _named(second_events, "integration_session_started")[0]["payload"]
    assert (first_start["session_number"], first_start["active_condition"]) == (1, "G")
    assert (second_start["session_number"], second_start["active_condition"]) == (2, "E")

    decisions = _named(first_events, "experiment_policy_decision")
    records = _named(first_events, "experiment_episode_training_record")
    assert decisions and records
    for event in records:
        payload = event["payload"]
        if payload["eeg_window_end"] is not None:
            assert payload["eeg_window_end"] == payload["prediction_cutoff_timestamp"]

    gaze_events = _named(second_events, "integration_gaze_processed")
    frozen = next(
        event
        for event in gaze_events
        if event["payload"]["decision_newly_frozen"] is True
        and event["payload"]["applied_intent_score"] is None
    )
    episode_id = frozen["payload"]["episode_id"]
    assert any(
        event["timestamp"] > frozen["timestamp"]
        and event["payload"]["episode_id"] == episode_id
        and event["payload"]["applied_intent_score"] is not None
        for event in gaze_events
    )

    participant = Path(config["participant"]["artifact_directory"])
    completed_1, _ = load_completed_session(participant / "completed_session_001.json")
    completed_2, _ = load_completed_session(participant / "completed_session_002.json")
    assert completed_1.successful and completed_2.successful
    policy_3 = load_frozen_policy(
        participant / "policy_session_003.json",
        expected_participant_id="test-P001",
        expected_session=3,
    )
    assert [source.session_number for source in policy_3.source_attempts] == [1, 2]
    for run in (first_run, second_run):
        for name in (
            "raw_glasses.h5",
            "raw_eeg.h5",
            "completed_session.json",
            "analysis_summary.json",
            "learning_curve.csv",
            "resolved_integration_config.json",
            "resolved_gaze_interaction_config.json",
            "resolved_eeg_pipeline_config.json",
            "resolved_experiment_learning_config.json",
        ):
            assert (run / name).is_file()


def test_unusable_eeg_is_explicitly_excluded(tmp_path: Path) -> None:
    eeg_override = tmp_path / "bad_eeg.yaml"
    eeg_override.write_text(
        """source:\n  synthetic:\n    duration_seconds: 12.0\n    invalid_intervals:\n      - [0.0, 12.0]\n""",
        encoding="utf-8",
    )
    config = _config(tmp_path)
    config["subsystem_config_overrides"]["eeg_pipeline"] = str(eeg_override)
    run = run_integrated_experiment(config)
    summary = json.loads((run / "analysis_summary.json").read_text(encoding="utf-8"))

    assert summary["excluded_records"]["count"] > 0
    assert "eeg_rejected" in summary["excluded_records"]["reasons"]
    assert summary["training_eligible_records"] == 0


def test_invalid_gaze_does_not_fabricate_episode_decision_or_record(tmp_path: Path) -> None:
    gaze_override = tmp_path / "invalid_gaze.yaml"
    gaze_override.write_text(
        """source:\n  virtual:\n    duration_seconds: 2.0\n    scene_width: 40\n    scene_height: 30\n    gaze_invalid_probability: 1.0\n""",
        encoding="utf-8",
    )
    config = _config(tmp_path)
    config["subsystem_config_overrides"]["gaze_interaction"] = str(gaze_override)
    run = run_integrated_experiment(config)
    events = _events(run)

    gaze_events = _named(events, "integration_gaze_processed")
    assert gaze_events and all(event["payload"]["valid"] is False for event in gaze_events)
    assert not _named(events, "integration_episode_started")
    assert not _named(events, "experiment_policy_decision")
    assert not _named(events, "experiment_episode_training_record")


class _MismatchedPendingController:
    pending_feedback_episode_id = 1

    def accept_feedback(self, timestamp):  # pragma: no cover - mismatch fails first
        raise AssertionError(timestamp)

    def advance_time(self, timestamp):  # pragma: no cover - mismatch fails first
        raise AssertionError(timestamp)


def test_feedback_replay_rejects_wrong_episode_identity(tmp_path: Path) -> None:
    from foundations.events import JsonlEventLogger

    driver = TimedFeedbackDriver(
        [ScheduledFeedbackPress(1.0, 2)],
        event_logger=JsonlEventLogger(tmp_path / "events.jsonl"),
        session_id="S001",
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        driver.before_time(1.0, _MismatchedPendingController())


def test_saved_inputs_feedback_and_analysis_replay_deterministically(tmp_path: Path) -> None:
    original_config = _config(tmp_path / "original")
    original = run_integrated_experiment(original_config)
    gaze_override = tmp_path / "replay_gaze.yaml"
    gaze_override.write_text(
        f"source:\n  recording_path: {original / 'raw_glasses.h5'}\n  replay_paced: false\n",
        encoding="utf-8",
    )
    eeg_override = tmp_path / "replay_eeg.yaml"
    eeg_override.write_text(
        f"source:\n  mode: replay\n  replay_path: {original / 'raw_eeg.h5'}\n  replay_paced: false\n",
        encoding="utf-8",
    )
    replay_config = _config(tmp_path / "replay")
    replay_config["input"]["mode"] = "replay"
    replay_config["feedback"]["mode"] = "replay"
    replay_config["feedback"]["replay"]["events_path"] = str(original / "events.jsonl")
    replay_config["subsystem_config_overrides"]["gaze_interaction"] = str(gaze_override)
    replay_config["subsystem_config_overrides"]["eeg_pipeline"] = str(eeg_override)
    replay = run_integrated_experiment(replay_config)

    original_summary = json.loads((original / "analysis_summary.json").read_text(encoding="utf-8"))
    replay_summary = json.loads((replay / "analysis_summary.json").read_text(encoding="utf-8"))
    assert replay_summary == original_summary
    regenerated_dir = tmp_path / "regenerated"
    assert generate_analysis(replay / "events.jsonl", regenerated_dir) == replay_summary
    assert (regenerated_dir / "learning_curve.csv").read_text(encoding="utf-8") == (
        replay / "learning_curve.csv"
    ).read_text(encoding="utf-8")


def test_failed_attempt_does_not_advance_session_or_train(tmp_path: Path) -> None:
    config = _config(tmp_path)
    broken_recording = tmp_path / "broken.h5"
    broken_recording.write_text("not an HDF5 file", encoding="utf-8")
    broken_gaze = tmp_path / "broken_gaze.yaml"
    broken_gaze.write_text(
        f"source:\n  recording_path: {broken_recording}\n  replay_paced: false\n",
        encoding="utf-8",
    )
    config["input"]["mode"] = "replay"
    config["subsystem_config_overrides"]["gaze_interaction"] = str(broken_gaze)
    with pytest.raises(OSError):
        run_integrated_experiment(config)

    participant = Path(config["participant"]["artifact_directory"])
    assert not list(participant.glob("completed_session_*.json"))
    assert not (participant / "policy_session_002.json").exists()
    config["input"]["mode"] = "synthetic"
    config["subsystem_config_overrides"]["gaze_interaction"] = "configs/integration_gaze.yaml"
    resumed = run_integrated_experiment(config)
    started = _named(_events(resumed), "integration_session_started")[0]["payload"]
    assert (started["session_number"], started["active_condition"]) == (1, "G")


def test_live_guardian_preflight_gate_clock_start_and_independent_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class FakeClock:
        def __init__(self) -> None:
            calls.append("clock_start")

        def now(self) -> float:
            return 0.0

    class FakeGuardian:
        def __init__(self, *, clock, **kwargs) -> None:
            del kwargs
            self.clock = clock
            self.samples: list[EEGSample] = []
            self.recording_done = False
            self.recording_id = "fake-recording"

        def prepare(self, **kwargs) -> GuardianPreflight:
            del kwargs
            calls.append("prepare")
            return GuardianPreflight(90.0, 10_000.0)

        def start(self, *, recording_seconds: int) -> None:
            assert recording_seconds == 1
            assert self.clock() == 0.0
            calls.append("guardian_start")
            self.samples = [EEGSample(index / 250.0, 5.0) for index in range(126)]

        def check_health(self) -> None:
            pass

        def finalize_before(self, timestamp: float):
            ready = tuple(
                sample for sample in self.samples if sample.timestamp < timestamp
            )
            self.samples = [sample for sample in self.samples if sample not in ready]
            return ready

        def window(self, start: float, end: float) -> EEGWindow:
            selected = tuple(
                sample for sample in self.samples if start <= sample.timestamp <= end
            )
            return EEGWindow(
                start,
                end,
                selected,
                selected[0].timestamp if selected else None,
                selected[-1].timestamp if selected else None,
                (
                    WindowCompleteness.PARTIAL
                    if selected
                    else WindowCompleteness.EMPTY
                ),
            )

        def stop(self) -> None:
            calls.append("stop")

        def close(self) -> None:
            calls.append("close")

    def gate() -> bool:
        calls.append("gate")
        return True

    eeg_override = tmp_path / "live_eeg.yaml"
    eeg_override.write_text(
        "source:\n  mode: live\n  guardian:\n    api_token_env: TEST_UNUSED_TOKEN\n"
        "    api_token_file: null\n    recording_seconds: 1\n"
        "    impedance:\n      enabled: false\n",
        encoding="utf-8",
    )
    config = _config(tmp_path)
    monkeypatch.setenv("TEST_UNUSED_TOKEN", "fake-token")
    config["session"]["maximum_duration_seconds"] = 0.5
    config["subsystem_config_overrides"]["eeg_pipeline"] = str(eeg_override)
    run = run_integrated_experiment(
        config,
        guardian_factory=FakeGuardian,
        clock_factory=FakeClock,
        start_gate=gate,
    )

    assert calls[:4] == ["prepare", "gate", "clock_start", "guardian_start"]
    assert calls[-2:] == ["stop", "close"]
    events = _events(run)
    assert _named(events, "integration_guardian_preflight_completed")
    completion = _named(events, "integration_session_completed")[0]["payload"]
    assert completion["guardian_recording_id"] == "fake-recording"
    assert (run / "raw_eeg.h5").is_file()


def test_live_guardian_cleanup_continues_when_stop_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class FakeClock:
        def now(self) -> float:
            return 0.0

    class FailingStopGuardian:
        recording_done = False
        recording_id = None

        def __init__(self, **kwargs) -> None:
            del kwargs
            self.stopped = False

        def prepare(self, **kwargs) -> GuardianPreflight:
            del kwargs
            return GuardianPreflight(90.0, None)

        def start(self, *, recording_seconds: int) -> None:
            assert recording_seconds == 1

        def check_health(self) -> None:
            if self.stopped:
                calls.append("drain_after_stop")

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
            calls.append("stop")
            self.stopped = True
            raise RuntimeError("stop failed")

        def close(self) -> None:
            calls.append("close")

    eeg_override = tmp_path / "live_eeg.yaml"
    eeg_override.write_text(
        "source:\n  mode: live\n  guardian:\n    api_token_env: TEST_UNUSED_TOKEN\n"
        "    api_token_file: null\n    recording_seconds: 1\n"
        "    impedance:\n      enabled: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_UNUSED_TOKEN", "fake-token")
    config = _config(tmp_path)
    config["session"]["maximum_duration_seconds"] = 0.1
    config["subsystem_config_overrides"]["eeg_pipeline"] = str(eeg_override)

    with pytest.raises(RuntimeError, match="stop failed"):
        run_integrated_experiment(
            config,
            guardian_factory=FailingStopGuardian,
            clock_factory=FakeClock,
            start_gate=lambda: True,
        )

    assert calls[-3:] == ["stop", "drain_after_stop", "close"]
    participant = Path(config["participant"]["artifact_directory"])
    assert not list(participant.glob("completed_session_*.json"))
    run = next((Path(config["output_root"]) / "integration").iterdir())
    assert _named(_events(run), "integration_session_incomplete")
