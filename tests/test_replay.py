import json
from pathlib import Path

import numpy as np

from foundations.contracts import GazeSample, SceneFrame
from foundations.recording import HDF5Recorder, HDF5Replay
from foundations.workflow import run_virtual_glasses


def test_record_replay_preserves_values_timestamps_and_interleaving(tmp_path: Path) -> None:
    original = [
        SceneFrame(0.0, np.arange(18, dtype=np.uint8).reshape(2, 3, 3)),
        GazeSample(0.0, 0.2, 0.8, True, 0.9),
        GazeSample(0.1, None, None, False, None),
        SceneFrame(0.2, np.full((2, 3, 3), 5, dtype=np.uint8)),
    ]
    path = tmp_path / "recording.h5"
    with HDF5Recorder(path) as recorder:
        for sample in original:
            recorder.record(sample)

    replayed = list(HDF5Replay(path).samples())

    assert [type(item) for item in replayed] == [type(item) for item in original]
    for expected, actual in zip(original, replayed, strict=True):
        assert expected.timestamp == actual.timestamp
        if isinstance(expected, SceneFrame):
            assert isinstance(actual, SceneFrame)
            np.testing.assert_array_equal(expected.image, actual.image)
        else:
            assert expected == actual


def test_workflow_saves_resolved_config_events_and_recording(tmp_path: Path) -> None:
    config = {
        "mode": "record",
        "output_root": str(tmp_path),
        "seed": 3,
        "duration_seconds": 0.2,
        "scene": {
            "width": 4,
            "height": 3,
            "rate_hz": 5.0,
            "dropout_probability": 0.0,
        },
        "gaze": {
            "rate_hz": 10.0,
            "invalid_probability": 0.0,
            "dropout_probability": 0.0,
        },
        "recording_path": None,
        "replay_paced": False,
    }

    run_directory = run_virtual_glasses(config)

    assert json.loads((run_directory / "resolved_config.json").read_text()) == config
    assert (run_directory / "recording.h5").is_file()
    events = [
        json.loads(line)
        for line in (run_directory / "events.jsonl").read_text().splitlines()
    ]
    assert [event["name"] for event in events] == [
        "recording_started",
        "recording_finished",
    ]
