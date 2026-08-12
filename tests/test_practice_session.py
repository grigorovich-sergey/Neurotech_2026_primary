import json
from pathlib import Path
import time

import numpy as np

from eeg_pipeline.contracts import EEGSample
from foundations.config import load_resolved_config
from foundations.contracts import GazeSample, SceneFrame
from gaze_interaction.contracts import BoundingBox, Detection, TrackedObject, TrackedScene
from mindlink import FrameMetadata, GazeMetadata
from practice_session.runner import PROJECT_ROOT, run_practice_session


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
    config["display"]["enabled"] = False
    config["subsystem_config_overrides"]["gaze_interaction"] = str(gaze_override)
    return config


def _events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_practice_runs_live_gaze_path_without_experimental_artifacts(tmp_path: Path) -> None:
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
    )

    assert instances[0].log == ["connect", "calibrate", "start_capture", "close"]
    event_names = [event["name"] for event in _events(run)]
    assert "practice_selection" in event_names
    summary = json.loads((run / "practice_summary.json").read_text(encoding="utf-8"))
    assert summary["run_type"] == "practice"
    assert summary["experimental_session"] is False
    assert summary["stop_reason"] == "duration_reached"
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


def test_optional_eeg_uses_shared_clock_and_stops_with_practice(
    tmp_path: Path, monkeypatch
) -> None:
    mindlink_clocks = []
    guardian_clocks = []

    def mindlink_factory(**kwargs) -> FakeMindLink:
        mindlink_clocks.append(kwargs["clock"])
        return FakeMindLink(**kwargs)

    class FakeGuardian:
        def __init__(self, *, clock, **_: object) -> None:
            self.clock = clock
            guardian_clocks.append(clock)

        def run(self, *, on_sample, stop_requested, **_: object) -> float:
            start = self.clock()
            for index in range(20):
                on_sample(EEGSample(start + index * 0.004, float(index)))
            while not stop_requested():
                time.sleep(0.001)
            return 12_000.0

    monkeypatch.setenv("IDUN_API_TOKEN", "test-token")
    run = run_practice_session(
        _config(tmp_path, eeg_enabled=True),
        detector=FakeDetector(),
        tracker=FakeTracker(),
        mindlink_factory=mindlink_factory,
        guardian_factory=FakeGuardian,
    )

    assert mindlink_clocks[0].__self__ is guardian_clocks[0].__self__
    summary = json.loads((run / "practice_summary.json").read_text(encoding="utf-8"))
    assert summary["successful"] is True
    assert summary["eeg"]["sample_count"] == 20
    assert summary["eeg"]["impedance_ohms"] == 12_000.0
    assert (run / "practice_eeg.h5").is_file()
