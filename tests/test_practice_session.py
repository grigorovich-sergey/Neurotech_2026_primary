import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eeg_pipeline.contracts import EEGSample
from foundations.config import load_resolved_config
from foundations.contracts import GazeSample, SceneFrame
from foundations.operator_gate import format_impedance
from foundations.timebase import MonotonicClock
from gaze_interaction.contracts import BoundingBox, Detection, TrackedObject, TrackedScene
from gaze_interaction.dwell import DwellState, DwellTrigger
from gaze_interaction.episodes import CandidateEpisode, EpisodeEndReason
from gaze_interaction.pipeline import InteractionUpdate
from mindlink import FrameMetadata, GazeMetadata
from practice_session.runner import (
    _InteractionDecisionReporter,
    _TerminalReporter,
    _ThreadSafeEvents,
    PROJECT_ROOT,
    _overlay_gaze_indicator,
    _wait_for_start_signal,
    run_practice_session,
)
from scripts.run_practice_session import _apply_cli_overrides


PRACTICE_CONFIG = PROJECT_ROOT / "configs" / "practice_session.yaml"


class FakeDetector:
    def detect(self, frame: SceneFrame) -> tuple[Detection, ...]:
        return (Detection(BoundingBox(0.2, 0.2, 0.8, 0.8), "cup", 0.95),)


class FakeTracker:
    def update(
        self, frame: SceneFrame, detections: tuple[Detection, ...]
    ) -> TrackedScene:
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


class FakeMindLink:
    def __init__(self, *, clock, frame_queue_size: int, on_disconnect=None) -> None:
        self.clock = clock
        self.on_disconnect = on_disconnect
        self.log: list[str] = []
        self.dropped_frame_count = 0
        self.dropped_frame_timestamp_count = 0
        self.dropped_gaze_timestamp_count = 0

    def connect(self, **_: object) -> None:
        self.log.append("connect")

    def calibrate(self, **_: object) -> tuple[str]:
        self.log.append("calibrate")
        return ("ok",)

    def start_capture(
        self,
        *,
        on_scene_frame,
        on_gaze_sample,
        on_frame_metadata,
        on_gaze_metadata,
    ) -> None:
        self.log.append("start_capture")
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        frame = SceneFrame(0.0, image)
        on_scene_frame(frame)
        on_frame_metadata(FrameMetadata(0.0, 0.0, None, None, 0))
        for timestamp in (0.0, 0.05, 0.10, 0.15):
            gaze = GazeSample(timestamp, 0.5, 0.5, True, None)
            on_gaze_sample(gaze)
            on_gaze_metadata(
                GazeMetadata(timestamp, timestamp, 100.0 + timestamp, (30.0, 20.0))
            )

    def close(self) -> None:
        self.log.append("close")


def _config(tmp_path: Path, *, eeg_enabled: bool = False) -> dict:
    gaze_override = tmp_path / "practice_gaze_override.yaml"
    gaze_override.write_text(
        """tracker:
  frame_rate: 30
dwell:
  baseline_seconds: 0.1
  minimum_seconds: 0.1
  maximum_seconds: 0.1
  maximum_reduction_fraction: 0.0
""",
        encoding="utf-8",
    )
    config = load_resolved_config(PRACTICE_CONFIG)
    config["output_root"] = str(tmp_path / "runs")
    config["maximum_duration_seconds"] = 0.05
    config["eeg"]["enabled"] = eeg_enabled
    config["terminal"]["verbose_decisions"] = True
    config["display"]["enabled"] = False
    config["subsystem_config_overrides"]["gaze_interaction"] = str(gaze_override)
    return config


def _events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_practice_runs_live_gaze_path_without_experimental_artifacts(
    tmp_path: Path, capsys
) -> None:
    instances: list[FakeMindLink] = []

    def mindlink_factory(**kwargs) -> FakeMindLink:
        instance = FakeMindLink(**kwargs)
        instances.append(instance)
        return instance

    run = run_practice_session(
        _config(tmp_path),
        detector=FakeDetector(),
        tracker=FakeTracker(),
        mindlink_factory=mindlink_factory,
        start_gate=lambda: True,
    )

    assert instances[0].log == ["connect", "calibrate", "start_capture", "close"]
    event_names = [event["name"] for event in _events(run)]
    assert "practice_selection" in event_names
    summary = json.loads((run / "practice_summary.json").read_text(encoding="utf-8"))
    assert summary["run_type"] == "practice"
    assert summary["experimental_session"] is False
    assert summary["stop_reason"] == "duration_reached"
    assert summary["attempt_started_timestamp"] is not None
    assert summary["attempt_duration_seconds"] is not None
    assert summary["eeg_started"] is False
    assert summary["eeg_enabled"] is False
    assert summary["diagnostics"]["scene_received"] == 1
    assert summary["diagnostics"]["scene_processed"] == 1
    assert summary["diagnostics"]["gaze_received"] == 4
    assert summary["stream_rates_hz"]["scene_received"] > 0.0
    assert summary["stream_rates_hz"]["scene_processed"] > 0.0
    assert summary["tracker_configuration"] == {
        "frame_rate_hz": 30,
        "status": "provisional_pending_processed_rate_pilot",
    }
    environment = json.loads(
        (run / "environment_manifest.json").read_text(encoding="utf-8")
    )
    assert environment["schema"] == "neurotech.practice_environment.v1"
    assert set(environment["packages"]) == {
        "adhawk",
        "guardian",
        "opencv",
        "supervision",
        "ultralytics",
    }
    assert (run / "mindlink_frame_metadata.jsonl").is_file()
    assert (run / "mindlink_gaze_metadata.jsonl").is_file()
    assert not any("completed_session" in path.name for path in run.iterdir())
    assert not any("policy" in path.name or "training" in path.name for path in run.iterdir())
    terminal = capsys.readouterr().out
    assert "SELECTION triggered: cup (track 1)" in terminal
    assert "CANDIDATE start: episode=1 cup (track 1)" in terminal
    assert "TRIGGER: episode=1 cup (track 1)" in terminal
    assert "stopped: duration_reached | successful=True" in terminal


def test_concise_terminal_hides_decision_lines_but_keeps_events_and_selection(
    tmp_path: Path, capsys
) -> None:
    config = _config(tmp_path)
    config["terminal"]["verbose_decisions"] = False

    run = run_practice_session(
        config,
        detector=FakeDetector(),
        tracker=FakeTracker(),
        mindlink_factory=FakeMindLink,
        start_gate=lambda: True,
    )

    terminal = capsys.readouterr().out
    assert "attempt started" in terminal
    assert "SELECTION triggered: cup (track 1)" in terminal
    assert "stopped: duration_reached | successful=True" in terminal
    assert "CANDIDATE start:" not in terminal
    assert "DWELL 25%:" not in terminal
    assert "TRIGGER:" not in terminal
    assert "EPISODE end:" not in terminal
    event_names = [event["name"] for event in _events(run)]
    assert "practice_candidate_started" in event_names
    assert "practice_dwell_progress" in event_names
    assert "practice_dwell_trigger" in event_names
    assert "practice_episode_ended" in event_names
    assert "practice_selection" in event_names
    resolved = json.loads(
        (run / "resolved_practice_config.json").read_text(encoding="utf-8")
    )
    assert resolved["terminal"]["verbose_decisions"] is False


def test_terminal_style_cli_overrides_config_without_changing_eeg() -> None:
    config = load_resolved_config(PRACTICE_CONFIG)
    assert config["terminal"]["verbose_decisions"] is False
    assert config["eeg"]["enabled"] is False

    _apply_cli_overrides(
        config,
        SimpleNamespace(
            with_eeg=False,
            without_eeg=False,
            verbose_decisions=False,
            concise_decisions=True,
        ),
    )
    assert config["terminal"]["verbose_decisions"] is False
    assert config["eeg"]["enabled"] is False

    _apply_cli_overrides(
        config,
        SimpleNamespace(
            with_eeg=False,
            without_eeg=False,
            verbose_decisions=True,
            concise_decisions=False,
        ),
    )
    assert config["terminal"]["verbose_decisions"] is True


def test_practice_rejects_non_boolean_terminal_style(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["terminal"]["verbose_decisions"] = "yes"

    with pytest.raises(ValueError, match="terminal.verbose_decisions must be a bool"):
        run_practice_session(config)


def test_space_gate_precedes_attempt_clock_capture_and_display(
    tmp_path: Path, monkeypatch
) -> None:
    instances: list[FakeMindLink] = []
    clock_constructions: list[str] = []
    ui_calls: list[str] = []
    config = _config(tmp_path)
    config["display"]["enabled"] = True

    monkeypatch.setattr(
        "practice_session.runner.cv2.namedWindow",
        lambda *_: ui_calls.append("namedWindow"),
    )
    monkeypatch.setattr("practice_session.runner.cv2.imshow", lambda *_: None)
    monkeypatch.setattr("practice_session.runner.cv2.waitKey", lambda *_: -1)
    monkeypatch.setattr(
        "practice_session.runner.cv2.getWindowProperty", lambda *_: 1.0
    )
    monkeypatch.setattr(
        "practice_session.runner.cv2.destroyAllWindows",
        lambda: ui_calls.append("destroyAllWindows"),
    )

    def mindlink_factory(**kwargs) -> FakeMindLink:
        instance = FakeMindLink(**kwargs)
        instances.append(instance)
        return instance

    def clock_factory() -> MonotonicClock:
        clock_constructions.append("clock")
        return MonotonicClock()

    def start_gate() -> bool:
        assert instances[0].log == ["connect", "calibrate"]
        assert clock_constructions == []
        assert ui_calls == []
        return True

    run = run_practice_session(
        config,
        detector=FakeDetector(),
        tracker=FakeTracker(),
        mindlink_factory=mindlink_factory,
        clock_factory=clock_factory,
        start_gate=start_gate,
    )

    assert clock_constructions == ["clock"]
    assert ui_calls == ["namedWindow", "destroyAllWindows"]
    assert instances[0].log == ["connect", "calibrate", "start_capture", "close"]
    event_names = [event["name"] for event in _events(run)]
    assert event_names.index("practice_mindlink_calibrated") < event_names.index(
        "practice_run_started"
    )
    assert event_names.index("practice_run_started") < event_names.index(
        "practice_capture_started"
    )


def test_prestart_abort_never_starts_clock_capture_display_or_eeg(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances: list[FakeMindLink] = []
    clock_constructions: list[str] = []
    guardian_constructions: list[str] = []
    guardian_lifecycle: list[str] = []
    config = _config(tmp_path, eeg_enabled=True)
    config["recording"]["glasses_enabled"] = True

    def mindlink_factory(**kwargs) -> FakeMindLink:
        instance = FakeMindLink(**kwargs)
        instances.append(instance)
        return instance

    def clock_factory() -> MonotonicClock:
        clock_constructions.append("clock")
        return MonotonicClock()

    class FakeGuardianFitting:
        def connect(self) -> None:
            guardian_lifecycle.append("connect")

        def check_battery(self) -> float:
            guardian_lifecycle.append("battery")
            return 90.0

        def start_impedance(self, *, mains_frequency_hz: int) -> None:
            assert mains_frequency_hz == 60
            guardian_lifecycle.append("impedance_start")

        def latest_impedance(self) -> float:
            return 10_000.0

        def stop_impedance(self) -> None:
            guardian_lifecycle.append("impedance_stop")

        def close(self) -> None:
            guardian_lifecycle.append("close")

    def guardian_factory(**_: object) -> object:
        assert instances[0].log == ["connect", "calibrate"]
        guardian_constructions.append("guardian")
        return FakeGuardianFitting()

    monkeypatch.setenv("IDUN_API_TOKEN", "test-token")
    run = run_practice_session(
        config,
        detector=FakeDetector(),
        tracker=FakeTracker(),
        mindlink_factory=mindlink_factory,
        guardian_factory=guardian_factory,
        clock_factory=clock_factory,
        start_gate=lambda: False,
    )

    assert instances[0].log == ["connect", "calibrate", "close"]
    assert clock_constructions == []
    assert guardian_constructions == ["guardian"]
    assert guardian_lifecycle == [
        "connect",
        "battery",
        "impedance_start",
        "impedance_stop",
        "close",
    ]
    assert not (run / "practice_glasses.h5").exists()
    assert not (run / "practice_eeg.h5").exists()
    assert not (run / "mindlink_frame_metadata.jsonl").exists()
    assert not (run / "mindlink_gaze_metadata.jsonl").exists()
    summary = json.loads((run / "practice_summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "operator_abort_before_start"
    assert summary["attempt_started_timestamp"] is None
    assert summary["capture_started_timestamp"] is None
    assert summary["attempt_duration_seconds"] is None
    assert summary["capture_duration_seconds"] is None
    assert summary["eeg_prepared"] is True
    assert summary["eeg_started"] is False
    event_names = [event["name"] for event in _events(run)]
    assert "practice_setup_aborted" in event_names
    assert "practice_run_started" not in event_names
    assert "practice_capture_started" not in event_names
    terminal = capsys.readouterr().out
    assert (
        "[practice setup] calibration complete; MindLink acquisition is OFF" in terminal
    )
    assert "[practice setup] Guardian fitting; raw EEG is OFF" in terminal
    assert "[practice setup] aborted before acquisition start" in terminal


def test_start_signal_ignores_other_keys_and_accepts_only_start_or_abort() -> None:
    keys = iter(("x", "\n", " "))
    assert _wait_for_start_signal(lambda: next(keys)) is True
    assert _wait_for_start_signal(lambda: "q") is False
    assert _wait_for_start_signal(lambda: "\x1b") is False


def test_fitting_gate_refreshes_impedance_status_and_formats_units() -> None:
    keys = iter(("x", " "))
    readings = iter((None, 12_000.0))
    statuses: list[str] = []

    assert _wait_for_start_signal(
        lambda: next(keys),
        status=lambda: f"impedance {format_impedance(next(readings))}",
        emit=statuses.append,
    ) is True
    assert statuses == [
        "impedance waiting for first reading",
        "impedance 12.0 kOhm (12000 ohm)",
    ]


def test_decision_reporter_describes_instance_2_transitions(
    tmp_path: Path, capsys
) -> None:
    event_path = tmp_path / "decisions.jsonl"
    reporter = _InteractionDecisionReporter(
        _TerminalReporter(), _ThreadSafeEvents(event_path)
    )
    box = BoundingBox(0.2, 0.2, 0.8, 0.8)
    cup = TrackedObject(7, box, "cup", 0.95, 1.0)
    laptop = TrackedObject(3, box, "laptop", 0.94, 2.0)
    episode_1 = CandidateEpisode(1, 7, "cup", 1.0, 1.0, None, None)

    reporter.report(
        InteractionUpdate(
            GazeSample(1.0, 0.5, 0.5, True, None),
            1.0,
            cup,
            episode_1,
            None,
            DwellState(1, 0.0, 1.0, False),
            None,
        )
    )
    reporter.report(
        InteractionUpdate(
            GazeSample(1.3, 0.5, 0.5, True, None),
            1.0,
            cup,
            CandidateEpisode(1, 7, "cup", 1.0, 1.3, None, None),
            None,
            DwellState(1, 0.3, 1.0, False),
            None,
        )
    )
    reporter.report(
        InteractionUpdate(
            GazeSample(1.4, None, None, False, None),
            None,
            None,
            CandidateEpisode(1, 7, "cup", 1.0, 1.3, None, None),
            None,
            DwellState(1, 0.3, 1.0, False),
            None,
        )
    )
    reporter.report(
        InteractionUpdate(
            GazeSample(1.5, 0.5, 0.5, True, None),
            1.0,
            cup,
            CandidateEpisode(1, 7, "cup", 1.0, 1.5, None, None),
            None,
            DwellState(1, 0.4, 1.0, False),
            None,
        )
    )
    episode_2 = CandidateEpisode(2, 3, "laptop", 2.0, 2.0, None, None)
    reporter.report(
        InteractionUpdate(
            GazeSample(2.0, 0.5, 0.5, True, None),
            2.0,
            laptop,
            episode_2,
            CandidateEpisode(
                1,
                7,
                "cup",
                1.0,
                1.5,
                2.0,
                EpisodeEndReason.CANDIDATE_CHANGE,
            ),
            DwellState(2, 0.0, 1.0, False),
            None,
        )
    )
    reporter.report(
        InteractionUpdate(
            GazeSample(3.0, 0.5, 0.5, True, None),
            2.0,
            laptop,
            CandidateEpisode(2, 3, "laptop", 2.0, 3.0, None, None),
            None,
            DwellState(2, 1.0, 1.0, True),
            DwellTrigger(2, 3, 3.0, 1.0),
        )
    )
    reporter.finish(
        CandidateEpisode(
            2,
            3,
            "laptop",
            2.0,
            3.0,
            3.1,
            EpisodeEndReason.SOURCE_END,
        )
    )

    terminal = capsys.readouterr().out
    assert "CANDIDATE start: episode=1 cup (track 7)" in terminal
    assert "DWELL 25%: episode=1 cup (track 7)" in terminal
    assert "CANDIDATE pause: episode=1 cup (track 7) reason=invalid_gaze" in terminal
    assert "CANDIDATE resume: episode=1 cup (track 7)" in terminal
    assert "CANDIDATE switch: episode=1 cup (track 7) -> episode=2 laptop (track 3)" in terminal
    assert "TRIGGER: episode=2 laptop (track 3) dwell=1.000/1.000s" in terminal
    assert "EPISODE end: episode=2 laptop (track 3) reason=source_end" in terminal
    event_names = [
        json.loads(line)["name"]
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert event_names == [
        "practice_candidate_started",
        "practice_dwell_progress",
        "practice_candidate_paused",
        "practice_candidate_resumed",
        "practice_candidate_switched",
        "practice_dwell_progress",
        "practice_dwell_progress",
        "practice_dwell_progress",
        "practice_dwell_trigger",
        "practice_episode_ended",
    ]


def test_practice_gaze_indicator_is_visible_and_does_not_mutate_input() -> None:
    source = np.zeros((80, 120, 3), dtype=np.uint8)
    valid = GazeSample(0.0, 0.5, 0.5, True, None)

    rendered = _overlay_gaze_indicator(source, valid)

    assert np.array_equal(source, np.zeros_like(source))
    assert not np.array_equal(rendered, source)
    center = rendered[40, 60]
    assert center.max() > 0


def test_practice_gaze_indicator_omits_invalid_gaze() -> None:
    source = np.zeros((80, 120, 3), dtype=np.uint8)
    invalid = GazeSample(0.0, None, None, False, None)

    assert np.array_equal(_overlay_gaze_indicator(source, invalid), source)


def test_optional_eeg_uses_shared_clock_and_stops_with_practice(
    tmp_path: Path, monkeypatch
) -> None:
    mindlink_clocks = []
    guardian_clocks = []
    guardian_lifecycle: list[str] = []

    def mindlink_factory(**kwargs) -> FakeMindLink:
        mindlink_clocks.append(kwargs["clock"])
        return FakeMindLink(**kwargs)

    class FakeGuardian:
        def __init__(self, *, clock, **_: object) -> None:
            guardian_lifecycle.append("construct")
            self.clock = clock
            guardian_clocks.append(clock)
            self.samples: list[EEGSample] = []
            self.recording_done = False
            self.recording_id = None
            self.queue_overflowed = False

        def connect(self) -> None:
            guardian_lifecycle.append("connect")

        def check_battery(self) -> float:
            guardian_lifecycle.append("battery")
            return 88.0

        def start_impedance(self, *, mains_frequency_hz: int) -> None:
            assert mains_frequency_hz == 60
            guardian_lifecycle.append("impedance_start")

        def latest_impedance(self) -> float:
            return 12_000.0

        def stop_impedance(self) -> None:
            guardian_lifecycle.append("impedance_stop")

        def start(self, *, recording_seconds: int) -> None:
            guardian_lifecycle.append("start")
            assert recording_seconds > 0
            start = self.clock()
            for index in range(20):
                timestamp = start + index * 0.004
                self.samples.append(
                    EEGSample(
                        timestamp,
                        float(index),
                        host_receipt_timestamp=timestamp + 0.01,
                    )
                )

        def drain(self, *, cutoff_timestamp=None) -> tuple[EEGSample, ...]:
            ready = []
            while self.samples and (
                cutoff_timestamp is None
                or self.samples[0].timestamp <= cutoff_timestamp
            ):
                ready.append(self.samples.pop(0))
            return tuple(ready)

        def stop(self) -> None:
            guardian_lifecycle.append("stop")
            self.recording_done = True
            self.recording_id = "practice-recording"

        def close(self) -> None:
            guardian_lifecycle.append("close")

    def start_gate() -> bool:
        assert guardian_lifecycle == [
            "construct",
            "connect",
            "battery",
            "impedance_start",
        ]
        return True

    monkeypatch.setenv("IDUN_API_TOKEN", "test-token")
    run = run_practice_session(
        _config(tmp_path, eeg_enabled=True),
        detector=FakeDetector(),
        tracker=FakeTracker(),
        mindlink_factory=mindlink_factory,
        guardian_factory=FakeGuardian,
        start_gate=start_gate,
    )

    assert mindlink_clocks[0].__self__ is guardian_clocks[0].__self__
    assert guardian_lifecycle == [
        "construct",
        "connect",
        "battery",
        "impedance_start",
        "impedance_stop",
        "start",
        "stop",
        "close",
    ]
    summary = json.loads((run / "practice_summary.json").read_text(encoding="utf-8"))
    assert summary["successful"] is True
    assert summary["eeg_prepared"] is True
    assert summary["eeg_started"] is True
    assert summary["eeg"]["sample_count"] == 20
    assert summary["eeg"]["battery_percent"] == 88.0
    assert summary["eeg"]["impedance_ohms"] == 12_000.0
    assert summary["eeg"]["recording_id"] == "practice-recording"
    assert summary["eeg"]["gap_count"] == 0
    assert summary["eeg"]["mean_receipt_lag_seconds"] == pytest.approx(0.01)
    assert (run / "practice_eeg.h5").is_file()
    for artifact in run.iterdir():
        if artifact.suffix in {".json", ".jsonl"}:
            assert "test-token" not in artifact.read_text(encoding="utf-8")
