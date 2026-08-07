"""Orchestration for virtual-glasses recording and replay runs."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from foundations.contracts import SceneFrame
from foundations.events import Event, JsonlEventLogger
from foundations.recording import HDF5Recorder, HDF5Replay
from foundations.virtual_glasses import VirtualGlasses


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_positive(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")


def _require_probability(name: str, value: Any) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a number within [0, 1]")


def validate_virtual_glasses_config(config: dict[str, Any]) -> None:
    mode = config.get("mode")
    if mode not in {"record", "replay"}:
        raise ValueError("mode must be 'record' or 'replay'")
    if not isinstance(config.get("output_root"), str) or not config["output_root"]:
        raise ValueError("output_root must be a non-empty path string")
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    _require_positive("duration_seconds", config.get("duration_seconds"))

    scene = _require_mapping(config, "scene")
    for key in ("width", "height"):
        value = scene.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"scene.{key} must be a positive integer")
    _require_positive("scene.rate_hz", scene.get("rate_hz"))
    _require_probability("scene.dropout_probability", scene.get("dropout_probability"))

    gaze = _require_mapping(config, "gaze")
    _require_positive("gaze.rate_hz", gaze.get("rate_hz"))
    _require_probability("gaze.invalid_probability", gaze.get("invalid_probability"))
    _require_probability("gaze.dropout_probability", gaze.get("dropout_probability"))

    if not isinstance(config.get("replay_paced"), bool):
        raise ValueError("replay_paced must be a bool")
    recording_path = config.get("recording_path")
    if recording_path is not None and not isinstance(recording_path, str):
        raise ValueError("recording_path must be a path string or null")
    if mode == "replay" and not recording_path:
        raise ValueError("recording_path is required in replay mode")


def _new_run_directory(output_root: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = Path(output_root) / "virtual_glasses" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _save_resolved_config(config: dict[str, Any], run_directory: Path) -> None:
    path = run_directory / "resolved_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run_virtual_glasses(config: dict[str, Any]) -> Path:
    """Execute one configured virtual-glasses recording or replay run."""

    validate_virtual_glasses_config(config)
    if config["mode"] == "replay" and not Path(config["recording_path"]).is_file():
        raise FileNotFoundError(config["recording_path"])

    run_directory = _new_run_directory(config["output_root"])
    _save_resolved_config(config, run_directory)
    events = JsonlEventLogger(run_directory / "events.jsonl")

    if config["mode"] == "record":
        _record(config, run_directory, events)
    else:
        _replay(config, events)
    return run_directory


def _record(
    config: dict[str, Any], run_directory: Path, events: JsonlEventLogger
) -> None:
    scene = config["scene"]
    gaze = config["gaze"]
    source = VirtualGlasses(
        seed=config["seed"],
        duration_seconds=config["duration_seconds"],
        scene_width=scene["width"],
        scene_height=scene["height"],
        scene_rate_hz=scene["rate_hz"],
        gaze_rate_hz=gaze["rate_hz"],
        scene_dropout_probability=scene["dropout_probability"],
        gaze_dropout_probability=gaze["dropout_probability"],
        gaze_invalid_probability=gaze["invalid_probability"],
    )

    counts = {"scene": 0, "gaze": 0}
    events.log(Event(0.0, "recording_started"))

    def log_dropout(stream: str, timestamp: float) -> None:
        events.log(Event(timestamp, "sensor_dropout", {"stream": stream}))

    with HDF5Recorder(run_directory / "recording.h5") as recorder:
        for sample in source.samples(on_dropout=log_dropout):
            recorder.record(sample)
            if isinstance(sample, SceneFrame):
                counts["scene"] += 1
            else:
                counts["gaze"] += 1

    events.log(Event(config["duration_seconds"], "recording_finished", counts))


def _replay(config: dict[str, Any], events: JsonlEventLogger) -> None:
    events.log(Event(0.0, "replay_started", {"recording_path": config["recording_path"]}))
    count = 0
    last_timestamp = 0.0
    replay = HDF5Replay(config["recording_path"])
    for sample in replay.replay(paced=config["replay_paced"]):
        count += 1
        last_timestamp = max(last_timestamp, float(sample.timestamp))
    events.log(Event(last_timestamp, "replay_finished", {"samples": count}))


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        smoke_config = {
            "mode": "record",
            "output_root": directory,
            "seed": 1,
            "duration_seconds": 0.1,
            "scene": {"width": 4, "height": 3, "rate_hz": 5.0, "dropout_probability": 0.0},
            "gaze": {"rate_hz": 10.0, "invalid_probability": 0.0, "dropout_probability": 0.0},
            "recording_path": None,
            "replay_paced": False,
        }
        print(run_virtual_glasses(smoke_config))
