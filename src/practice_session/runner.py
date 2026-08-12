"""Live MindLink practice display with optional monitor-only Guardian EEG."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import queue
import sys
import threading
import time
from typing import Any

import cv2

from eeg_pipeline.buffer import EEGBuffer
from eeg_pipeline.contracts import EEGSample
from eeg_pipeline.guardian import GuardianAdapter
from eeg_pipeline.pipeline import EEGPipeline
from eeg_pipeline.processing import EEGFeatureExtractor, EEGPreprocessor, EEGQualityGate
from eeg_pipeline.recording import EEGHDF5Recorder
from foundations.config import load_resolved_config, save_resolved_config
from foundations.contracts import GazeSample, SceneFrame
from foundations.events import Event, JsonlEventLogger
from foundations.recording import HDF5Recorder
from foundations.timebase import MonotonicClock
from gaze_interaction.association import GazeAssociator
from gaze_interaction.detector import YOLOEDetector
from gaze_interaction.dwell import DwellController
from gaze_interaction.episodes import EpisodeTracker
from gaze_interaction.pipeline import GazeInteractionPipeline, InteractionUpdate, SceneUpdate
from gaze_interaction.tracker import ByteTrackAdapter
from gaze_interaction.visualization import render_diagnostic
from mindlink import FrameMetadata, GazeMetadata, MindLinkAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINDLINK_CONFIG = PROJECT_ROOT / "configs" / "mindlink.yaml"
DEFAULT_GAZE_CONFIG = PROJECT_ROOT / "configs" / "gaze_interaction.yaml"
DEFAULT_EEG_CONFIG = PROJECT_ROOT / "configs" / "eeg_pipeline.yaml"

_VERSION_TARGETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "opencv": ("cv2", ("opencv-python", "opencv-python-headless")),
    "ultralytics": ("ultralytics", ("ultralytics",)),
    "supervision": ("supervision", ("supervision",)),
    "adhawk": ("adhawkapi", ("adhawkapi", "adhawk-api")),
    "guardian": ("idun_guardian_sdk", ("idun-guardian-sdk",)),
}


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _positive_number(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _project_path(value: Any, *, name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string or null")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config.get("output_root"), str) or not config["output_root"]:
        raise ValueError("output_root must be a non-empty path string")
    _positive_number("maximum_duration_seconds", config.get("maximum_duration_seconds"))
    eeg = _mapping(config, "eeg")
    if not isinstance(eeg.get("enabled"), bool):
        raise ValueError("eeg.enabled must be a bool")
    recording = _mapping(config, "recording")
    for key in ("glasses_enabled", "eeg_enabled"):
        if not isinstance(recording.get(key), bool):
            raise ValueError(f"recording.{key} must be a bool")
    processing = _mapping(config, "processing")
    for key in ("scene_queue_size", "gaze_queue_size", "gaze_batch_size"):
        _positive_int(f"processing.{key}", processing.get(key))
    for key in (
        "idle_sleep_seconds",
        "eeg_status_window_seconds",
        "eeg_status_refresh_seconds",
    ):
        _positive_number(f"processing.{key}", processing.get(key))
    diagnostics = _mapping(config, "diagnostics")
    if not isinstance(diagnostics.get("write_mindlink_metadata"), bool):
        raise ValueError("diagnostics.write_mindlink_metadata must be a bool")
    display = _mapping(config, "display")
    if not isinstance(display.get("enabled"), bool):
        raise ValueError("display.enabled must be a bool")
    if not isinstance(display.get("window_name"), str) or not display["window_name"]:
        raise ValueError("display.window_name must be a non-empty string")
    for key in ("selection_banner_seconds", "no_frame_warning_seconds"):
        _positive_number(f"display.{key}", display.get(key))
    overrides = _mapping(config, "subsystem_config_overrides")
    for key in ("mindlink", "gaze_interaction", "eeg_pipeline"):
        _project_path(overrides.get(key), name=f"subsystem_config_overrides.{key}")


def _load_subsystem_configs(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overrides = config["subsystem_config_overrides"]
    mindlink = load_resolved_config(
        DEFAULT_MINDLINK_CONFIG,
        _project_path(overrides["mindlink"], name="mindlink override"),
    )
    gaze = load_resolved_config(
        DEFAULT_GAZE_CONFIG,
        _project_path(overrides["gaze_interaction"], name="gaze override"),
    )
    eeg = load_resolved_config(
        DEFAULT_EEG_CONFIG,
        _project_path(overrides["eeg_pipeline"], name="EEG override"),
    )
    return mindlink, gaze, eeg


def _new_run_directory(output_root: str) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / "practice" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _distribution_version(
    module_name: str, candidates: tuple[str, ...]
) -> dict[str, str | bool | None]:
    """Resolve a module's installed distribution without importing hardware SDKs."""

    distribution_names = list(candidates)
    for discovered in importlib_metadata.packages_distributions().get(module_name, ()):
        if discovered not in distribution_names:
            distribution_names.append(discovered)
    for distribution_name in distribution_names:
        try:
            version = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        return {
            "installed": True,
            "distribution": distribution_name,
            "version": version,
        }
    return {"installed": False, "distribution": None, "version": None}


def _write_environment_manifest(run_directory: Path) -> None:
    packages = {
        name: _distribution_version(module_name, candidates)
        for name, (module_name, candidates) in _VERSION_TARGETS.items()
    }
    manifest = {
        "schema": "neurotech.practice_environment.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
    }
    with (run_directory / "environment_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


class _ThreadSafeEvents:
    def __init__(self, path: Path) -> None:
        self._logger = JsonlEventLogger(path)
        self._lock = threading.Lock()

    def log(self, timestamp: float, name: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._logger.log(Event(float(timestamp), name, payload))


class _TerminalReporter:
    """Serialize concise operator-facing notices from the main and worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def report(self, timestamp: float, message: str) -> None:
        with self._lock:
            print(f"[practice {float(timestamp):8.3f}s] {message}", flush=True)


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        with self._lock:
            json.dump(payload, self._handle, allow_nan=False, separators=(",", ":"))
            self._handle.write("\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()


def _json_number(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


class _RunState:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self.reason: str | None = None
        self.failure: BaseException | None = None

    def request_stop(self, reason: str, failure: BaseException | None = None) -> None:
        with self._lock:
            if failure is not None and self.failure is None:
                self.reason = reason
                self.failure = failure
            elif self.reason is None:
                self.reason = reason
        self.stop.set()


class _Diagnostics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.scene_received = 0
        self.scene_processed = 0
        self.gaze_received = 0
        self.valid_gaze_received = 0
        self.scene_queue_drops = 0
        self.gaze_queue_drops = 0
        self.frame_adapter_drops = 0
        self.frame_timestamp_drops = 0
        self.gaze_timestamp_drops = 0
        self.eeg_samples = 0
        self.first_eeg_timestamp: float | None = None
        self.last_eeg_timestamp: float | None = None

    def scene(self) -> None:
        with self._lock:
            self.scene_received += 1

    def processed_scene(self) -> None:
        with self._lock:
            self.scene_processed += 1

    def gaze(self, valid: bool) -> None:
        with self._lock:
            self.gaze_received += 1
            self.valid_gaze_received += int(valid)

    def queue_drop(self, stream: str) -> int:
        with self._lock:
            if stream == "scene":
                self.scene_queue_drops += 1
                return self.scene_queue_drops
            self.gaze_queue_drops += 1
            return self.gaze_queue_drops

    def frame_metadata(self, metadata: FrameMetadata) -> None:
        with self._lock:
            self.frame_adapter_drops = metadata.dropped_frame_count

    def eeg(self, timestamp: float) -> None:
        with self._lock:
            self.eeg_samples += 1
            if self.first_eeg_timestamp is None:
                self.first_eeg_timestamp = timestamp
            self.last_eeg_timestamp = timestamp

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scene_received": self.scene_received,
                "scene_processed": self.scene_processed,
                "gaze_received": self.gaze_received,
                "valid_gaze_received": self.valid_gaze_received,
                "scene_queue_drops": self.scene_queue_drops,
                "gaze_queue_drops": self.gaze_queue_drops,
                "frame_adapter_drops": self.frame_adapter_drops,
                "frame_timestamp_drops": self.frame_timestamp_drops,
                "gaze_timestamp_drops": self.gaze_timestamp_drops,
                "eeg_samples": self.eeg_samples,
                "first_eeg_timestamp": self.first_eeg_timestamp,
                "last_eeg_timestamp": self.last_eeg_timestamp,
            }


class _EEGMonitor:
    def __init__(self, pipeline: EEGPipeline, recorder: EEGHDF5Recorder | None) -> None:
        self.pipeline = pipeline
        self.recorder = recorder
        self.lock = threading.Lock()
        self.sample_count = 0
        self.first_timestamp: float | None = None
        self.last_timestamp: float | None = None
        self.impedance_ohms: float | None = None
        self.status = "starting"
        self.quality_state = "not_available"
        self.quality_reasons: tuple[str, ...] = ()

    def ingest(self, sample: EEGSample) -> None:
        with self.lock:
            if self.recorder is not None:
                self.recorder.record(sample)
            self.pipeline.add_sample(sample)
            self.sample_count += 1
            if self.first_timestamp is None:
                self.first_timestamp = sample.timestamp
            self.last_timestamp = sample.timestamp
            self.status = "streaming"

    def refresh(self, window_seconds: float) -> None:
        with self.lock:
            if self.last_timestamp is None:
                return
            start = max(0.0, self.last_timestamp - window_seconds)
            window = self.pipeline.features(start, self.last_timestamp)
            self.quality_state = window.quality_state.value
            self.quality_reasons = window.quality_reasons

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            rate = None
            if (
                self.first_timestamp is not None
                and self.last_timestamp is not None
                and self.last_timestamp > self.first_timestamp
            ):
                rate = (self.sample_count - 1) / (
                    self.last_timestamp - self.first_timestamp
                )
            return {
                "status": self.status,
                "sample_count": self.sample_count,
                "sample_rate_hz": rate,
                "quality_state": self.quality_state,
                "quality_reasons": list(self.quality_reasons),
                "impedance_ohms": self.impedance_ohms,
            }

    def stopped(self, impedance_ohms: float | None) -> None:
        with self.lock:
            self.impedance_ohms = impedance_ohms
            self.status = "stopped"


def _build_gaze_pipeline(
    config: dict[str, Any], *, detector: Any | None, tracker: Any | None
) -> GazeInteractionPipeline:
    detector_config = _mapping(config, "detector")
    tracker_config = _mapping(config, "tracker")
    association = _mapping(config, "association")
    episode = _mapping(config, "episode")
    dwell = _mapping(config, "dwell")
    local_detector = detector or YOLOEDetector(
        detector_config["model"],
        confidence_threshold=detector_config["confidence_threshold"],
        image_size=detector_config["image_size"],
        device=detector_config["device"],
        category_filter=detector_config["category_filter"],
    )
    local_tracker = tracker or ByteTrackAdapter(
        activation_threshold=tracker_config["activation_threshold"],
        lost_track_buffer=tracker_config["lost_track_buffer"],
        matching_threshold=tracker_config["matching_threshold"],
        frame_rate=tracker_config["frame_rate"],
    )
    return GazeInteractionPipeline(
        detector=local_detector,
        tracker=local_tracker,
        associator=GazeAssociator(
            box_margin_normalized=association["box_margin_normalized"],
            max_scene_age_seconds=association["max_scene_age_seconds"],
        ),
        episode_tracker=EpisodeTracker(gap_grace_seconds=episode["gap_grace_seconds"]),
        dwell_controller=DwellController(
            baseline_seconds=dwell["baseline_seconds"],
            minimum_seconds=dwell["minimum_seconds"],
            maximum_seconds=dwell["maximum_seconds"],
            maximum_reduction_fraction=dwell["maximum_reduction_fraction"],
            max_sample_gap_seconds=dwell["max_sample_gap_seconds"],
        ),
    )


def _build_eeg_pipeline(config: dict[str, Any]) -> EEGPipeline:
    signal = _mapping(config, "signal")
    buffer_config = _mapping(config, "buffer")
    preprocessing = _mapping(config, "preprocessing")
    quality = _mapping(config, "quality")
    features = _mapping(config, "features")
    sample_rate = signal["sample_rate_hz"]
    return EEGPipeline(
        buffer=EEGBuffer(buffer_config["history_seconds"]),
        preprocessor=EEGPreprocessor(
            sample_rate_hz=sample_rate,
            low_hz=preprocessing["low_hz"],
            high_hz=preprocessing["high_hz"],
            order=preprocessing["order"],
        ),
        quality_gate=EEGQualityGate(
            sample_rate_hz=sample_rate,
            min_duration_seconds=quality["min_duration_seconds"],
            min_coverage=quality["min_coverage"],
            max_gap_seconds=quality["max_gap_seconds"],
            min_std_uv=quality["min_std_uv"],
            max_peak_to_peak_uv=quality["max_peak_to_peak_uv"],
        ),
        feature_extractor=EEGFeatureExtractor(
            sample_rate_hz=sample_rate,
            welch_nperseg_samples=features["welch_nperseg_samples"],
            welch_noverlap_samples=features["welch_noverlap_samples"],
        ),
    )


def _overlay_status(image_rgb: Any, lines: list[str], *, selection: str | None) -> Any:
    image = image_rgb.copy()
    y = 42
    for line in lines:
        cv2.putText(
            image,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20
    if selection is not None:
        cv2.putText(
            image,
            f"PRACTICE SELECTION: {selection}",
            (8, max(y + 10, image.shape[0] - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 215, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def _overlay_gaze_indicator(image_rgb: Any, gaze: GazeSample | None) -> Any:
    """Draw a high-contrast practice-only bullseye for the latest valid gaze."""

    image = image_rgb.copy()
    if (
        gaze is None
        or not gaze.valid
        or gaze.x_normalized is None
        or gaze.y_normalized is None
    ):
        return image
    height, width = image.shape[:2]
    point = (
        int(round(gaze.x_normalized * (width - 1))),
        int(round(gaze.y_normalized * (height - 1))),
    )
    cv2.circle(image, point, 18, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.circle(image, point, 18, (255, 60, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(
        image,
        point,
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    label_position = (min(point[0] + 22, max(0, width - 48)), max(14, point[1] - 20))
    cv2.putText(
        image,
        "GAZE",
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "GAZE",
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def run_practice_session(
    config: dict[str, Any],
    *,
    detector: Any | None = None,
    tracker: Any | None = None,
    mindlink_factory: Callable[..., Any] = MindLinkAdapter,
    guardian_factory: Callable[..., Any] = GuardianAdapter,
    clock_factory: Callable[[], MonotonicClock] = MonotonicClock,
) -> Path:
    """Run a live diagnostic; never create experimental or training artifacts."""

    _validate_config(config)
    resolved = deepcopy(config)
    mindlink_config, gaze_config, eeg_config = _load_subsystem_configs(resolved)
    gaze_pipeline = _build_gaze_pipeline(gaze_config, detector=detector, tracker=tracker)
    run_directory = _new_run_directory(resolved["output_root"])
    _write_environment_manifest(run_directory)
    save_resolved_config(resolved, run_directory / "resolved_practice_config.json")
    save_resolved_config(mindlink_config, run_directory / "resolved_mindlink_config.json")
    save_resolved_config(gaze_config, run_directory / "resolved_gaze_interaction_config.json")
    save_resolved_config(eeg_config, run_directory / "resolved_eeg_pipeline_config.json")

    events = _ThreadSafeEvents(run_directory / "events.jsonl")
    terminal = _TerminalReporter()
    state = _RunState()
    diagnostics = _Diagnostics()
    processing = resolved["processing"]
    display = resolved["display"]
    recording = resolved["recording"]
    scene_queue: queue.Queue[SceneFrame] = queue.Queue(
        maxsize=processing["scene_queue_size"]
    )
    gaze_queue: queue.Queue[GazeSample] = queue.Queue(
        maxsize=processing["gaze_queue_size"]
    )
    clock = clock_factory()
    event_start = clock.now()
    events.log(
        event_start,
        "practice_run_started",
        {"eeg_enabled": resolved["eeg"]["enabled"]},
    )

    glasses_recorder = (
        HDF5Recorder(run_directory / "practice_glasses.h5")
        if recording["glasses_enabled"]
        else None
    )
    glasses_lock = threading.Lock()
    metadata_enabled = resolved["diagnostics"]["write_mindlink_metadata"]
    frame_metadata_writer = (
        _JsonlWriter(run_directory / "mindlink_frame_metadata.jsonl")
        if metadata_enabled
        else None
    )
    gaze_metadata_writer = (
        _JsonlWriter(run_directory / "mindlink_gaze_metadata.jsonl")
        if metadata_enabled
        else None
    )
    latest_adapter_drop = [0]
    intentional_close = threading.Event()

    def fail(reason: str, exc: BaseException) -> None:
        timestamp = clock.now()
        events.log(
            timestamp,
            "practice_error",
            {"source": reason, "error": type(exc).__name__},
        )
        terminal.report(
            timestamp,
            f"ERROR {reason}: {type(exc).__name__}: {exc}",
        )
        state.request_stop(reason, exc)

    def record_glasses(sample: SceneFrame | GazeSample) -> None:
        if glasses_recorder is None:
            return
        try:
            with glasses_lock:
                glasses_recorder.record(sample)
        except BaseException as exc:
            fail("glasses_recording_error", exc)

    def enqueue_latest(target: queue.Queue[Any], sample: Any, stream: str) -> None:
        try:
            target.put_nowait(sample)
            return
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
        drop_count = diagnostics.queue_drop(stream)
        events.log(
            float(sample.timestamp),
            "practice_processing_queue_drop",
            {"stream": stream, "count": drop_count},
        )
        target.put_nowait(sample)

    def on_scene(frame: SceneFrame) -> None:
        diagnostics.scene()
        record_glasses(frame)
        enqueue_latest(scene_queue, frame, "scene")

    def on_gaze(gaze: GazeSample) -> None:
        diagnostics.gaze(gaze.valid)
        record_glasses(gaze)
        enqueue_latest(gaze_queue, gaze, "gaze")

    def on_frame_metadata(metadata: FrameMetadata) -> None:
        try:
            diagnostics.frame_metadata(metadata)
            if metadata.dropped_frame_count > latest_adapter_drop[0]:
                latest_adapter_drop[0] = metadata.dropped_frame_count
                events.log(
                    metadata.timestamp,
                    "mindlink_adapter_frame_drop",
                    {"count": metadata.dropped_frame_count},
                )
            if frame_metadata_writer is not None:
                frame_metadata_writer.write(
                    {
                        "timestamp": metadata.timestamp,
                        "host_receipt_timestamp": metadata.host_receipt_timestamp,
                        "vendor_frame_timestamp": (
                            None
                            if metadata.vendor_frame_timestamp is None
                            else metadata.vendor_frame_timestamp.isoformat()
                        ),
                        "tracker_timestamp": metadata.tracker_timestamp,
                        "dropped_frame_count": metadata.dropped_frame_count,
                    }
                )
        except BaseException as exc:
            fail("mindlink_frame_metadata_error", exc)

    def on_gaze_metadata(metadata: GazeMetadata) -> None:
        try:
            if gaze_metadata_writer is not None:
                gaze_metadata_writer.write(
                    {
                        "timestamp": metadata.timestamp,
                        "host_receipt_timestamp": metadata.host_receipt_timestamp,
                        "vendor_timestamp": metadata.vendor_timestamp,
                        "gaze_in_image": (
                            None
                            if metadata.gaze_in_image is None
                            else [_json_number(value) for value in metadata.gaze_in_image]
                        ),
                    }
                )
        except BaseException as exc:
            fail("mindlink_gaze_metadata_error", exc)

    def on_disconnect(error: Any) -> None:
        if intentional_close.is_set():
            return
        fail("mindlink_disconnect", RuntimeError(f"MindLink disconnected: {error}"))

    eeg_monitor: _EEGMonitor | None = None
    eeg_recorder: EEGHDF5Recorder | None = None
    guardian_thread: threading.Thread | None = None
    if resolved["eeg"]["enabled"]:
        eeg_pipeline = _build_eeg_pipeline(eeg_config)
        if recording["eeg_enabled"] and eeg_config["recording"]["enabled"]:
            eeg_recorder = EEGHDF5Recorder(
                run_directory / "practice_eeg.h5",
                sample_rate_hz=eeg_config["signal"]["sample_rate_hz"],
            )
        eeg_monitor = _EEGMonitor(eeg_pipeline, eeg_recorder)

    adapter: Any | None = None
    capture_started_at: float | None = None
    calibration_result: Any = None
    latest_frame: SceneFrame | None = None
    latest_scene_update: SceneUpdate | None = None
    latest_interaction: InteractionUpdate | None = None
    latest_gaze: GazeSample | None = None
    last_gaze_timestamp = 0.0
    selection_label: str | None = None
    selection_until = 0.0
    last_eeg_refresh = 0.0
    warned_no_frame = False

    try:
        capture = mindlink_config["capture"]
        adapter = mindlink_factory(
            clock=clock.now,
            frame_queue_size=capture["frame_queue_size"],
            on_disconnect=on_disconnect,
        )
        connection = mindlink_config["connection"]
        terminal.report(clock.now(), "connecting to MindLink")
        adapter.connect(
            connect_timeout_seconds=connection["connect_timeout_seconds"],
            tracker_ready_timeout_seconds=connection["tracker_ready_timeout_seconds"],
        )
        connected_at = clock.now()
        events.log(connected_at, "practice_mindlink_connected", {})
        terminal.report(connected_at, "MindLink connected")
        calibration = mindlink_config["calibration"]
        terminal.report(clock.now(), "starting MindLink calibration")
        calibration_result = adapter.calibrate(
            marker_size_mm=calibration["marker_size_mm"],
            returning_user=calibration["returning_user"],
            timeout_seconds=calibration["timeout_seconds"],
        )
        calibrated_at = clock.now()
        events.log(
            calibrated_at,
            "practice_mindlink_calibrated",
            {"result": repr(calibration_result)},
        )
        terminal.report(calibrated_at, "MindLink calibration complete")

        if eeg_monitor is not None:
            guardian = eeg_config["source"]["guardian"]
            token_name = guardian["api_token_env"]
            api_token = os.environ.get(token_name)
            if not api_token:
                raise RuntimeError(f"EEG practice requires environment variable {token_name}")
            guardian_adapter = guardian_factory(
                clock=clock.now,
                address=guardian["address"],
                api_token=api_token,
                debug=guardian["debug"],
            )

            def on_eeg_sample(sample: EEGSample) -> None:
                assert eeg_monitor is not None
                diagnostics.eeg(sample.timestamp)
                eeg_monitor.ingest(sample)

            def guardian_worker() -> None:
                assert eeg_monitor is not None
                try:
                    impedance = guardian["impedance"]
                    result = guardian_adapter.run(
                        recording_seconds=guardian["recording_seconds"],
                        on_sample=on_eeg_sample,
                        impedance_preflight_seconds=(
                            impedance["duration_seconds"] if impedance["enabled"] else None
                        ),
                        max_impedance_ohms=impedance["max_ohms"],
                        mains_frequency_hz=impedance["mains_frequency_hz"],
                        stop_requested=state.stop.is_set,
                    )
                    eeg_monitor.stopped(result)
                except BaseException as exc:
                    fail("guardian_error", exc)
                else:
                    if not state.stop.is_set():
                        state.request_stop("guardian_completed")

            guardian_thread = threading.Thread(
                target=guardian_worker,
                name="practice_guardian",
                daemon=True,
            )
            terminal.report(clock.now(), "starting Guardian EEG monitor")
            guardian_thread.start()

        if display["enabled"]:
            cv2.namedWindow(display["window_name"], cv2.WINDOW_NORMAL)
        capture_started_at = clock.now()
        adapter.start_capture(
            on_scene_frame=on_scene,
            on_gaze_sample=on_gaze,
            on_frame_metadata=on_frame_metadata,
            on_gaze_metadata=on_gaze_metadata,
        )
        events.log(capture_started_at, "practice_capture_started", {})
        terminal.report(capture_started_at, "capture started; press Q or Esc to stop")

        while not state.stop.is_set():
            now = clock.now()
            if now - capture_started_at >= resolved["maximum_duration_seconds"]:
                state.request_stop("duration_reached")
                break
            processed = False
            try:
                frame = scene_queue.get_nowait()
            except queue.Empty:
                frame = None
            if frame is not None:
                latest_scene_update = gaze_pipeline.process_scene(frame)
                diagnostics.processed_scene()
                latest_frame = frame
                processed = True

            for _ in range(processing["gaze_batch_size"]):
                try:
                    gaze = gaze_queue.get_nowait()
                except queue.Empty:
                    break
                latest_interaction = gaze_pipeline.process_gaze(gaze, intent_score=None)
                latest_gaze = gaze
                last_gaze_timestamp = float(gaze.timestamp)
                processed = True
                trigger = latest_interaction.dwell_trigger
                if trigger is not None:
                    label = None
                    if latest_scene_update is not None:
                        label = next(
                            (
                                item.label
                                for item in latest_scene_update.tracks
                                if item.track_id == trigger.track_id
                            ),
                            None,
                        )
                    selection_label = label or f"object #{trigger.track_id}"
                    selection_until = now + display["selection_banner_seconds"]
                    events.log(
                        trigger.timestamp,
                        "practice_selection",
                        {"track_id": trigger.track_id, "label": selection_label},
                    )
                    terminal.report(
                        trigger.timestamp,
                        (
                            f"SELECTION triggered: {selection_label} "
                            f"(track {trigger.track_id})"
                        ),
                    )

            if (
                eeg_monitor is not None
                and now - last_eeg_refresh
                >= processing["eeg_status_refresh_seconds"]
            ):
                eeg_monitor.refresh(processing["eeg_status_window_seconds"])
                last_eeg_refresh = now

            if (
                latest_frame is None
                and not warned_no_frame
                and now - capture_started_at
                >= display["no_frame_warning_seconds"]
            ):
                warned_no_frame = True
                events.log(now, "practice_no_frame_warning", {})
                terminal.report(now, "WARNING no scene frame received")

            if display["enabled"] and latest_frame is not None:
                snapshot = diagnostics.snapshot()
                elapsed = max(now - capture_started_at, 1e-9)
                gaze_validity = (
                    snapshot["valid_gaze_received"] / snapshot["gaze_received"]
                    if snapshot["gaze_received"]
                    else 0.0
                )
                lines = [
                    "PRACTICE - NOT AN EXPERIMENTAL SESSION",
                    "MindLink calibration: complete",
                    (
                        f"scene rx/proc {snapshot['scene_received'] / elapsed:.1f}/"
                        f"{snapshot['scene_processed'] / elapsed:.1f} Hz | "
                        f"gaze {snapshot['gaze_received'] / elapsed:.1f} Hz | "
                        f"valid {gaze_validity:.1%}"
                    ),
                    (
                        "drops adapter/frame-q/gaze-q: "
                        f"{snapshot['frame_adapter_drops']}/"
                        f"{snapshot['scene_queue_drops']}/"
                        f"{snapshot['gaze_queue_drops']}"
                    ),
                ]
                if latest_gaze is None:
                    lines.append("gaze indicator: waiting for sample")
                elif latest_gaze.valid:
                    lines.append(
                        "gaze indicator: VALID | "
                        f"x={latest_gaze.x_normalized:.3f} "
                        f"y={latest_gaze.y_normalized:.3f}"
                    )
                else:
                    lines.append("gaze indicator: INVALID / outside image")
                if eeg_monitor is None:
                    lines.append("EEG: off")
                else:
                    eeg_status = eeg_monitor.snapshot()
                    rate = eeg_status["sample_rate_hz"]
                    rate_text = "--" if rate is None else f"{rate:.1f} Hz"
                    lines.append(
                        f"EEG: {eeg_status['status']} | {rate_text} | "
                        f"quality {eeg_status['quality_state']}"
                    )
                image = render_diagnostic(
                    latest_frame,
                    tracks=(latest_scene_update.tracks if latest_scene_update else ()),
                    gaze=latest_gaze,
                    candidate=(latest_interaction.candidate if latest_interaction else None),
                    dwell_state=(latest_interaction.dwell_state if latest_interaction else None),
                    intent_score=None,
                )
                image = _overlay_gaze_indicator(image, latest_gaze)
                selection = selection_label if now <= selection_until else None
                image = _overlay_status(image, lines, selection=selection)
                cv2.imshow(display["window_name"], cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    state.request_stop("operator_stop")
                elif cv2.getWindowProperty(display["window_name"], cv2.WND_PROP_VISIBLE) < 1:
                    state.request_stop("window_closed")
            elif not processed:
                time.sleep(processing["idle_sleep_seconds"])
    except KeyboardInterrupt:
        state.request_stop("keyboard_interrupt")
    except BaseException as exc:
        fail("practice_runtime_error", exc)
    finally:
        state.stop.set()
        intentional_close.set()
        if adapter is not None:
            try:
                adapter.close()
            except BaseException as exc:
                if state.failure is None:
                    state.request_stop("mindlink_close_error", exc)
        if guardian_thread is not None:
            guardian_thread.join(timeout=15.0)
            if guardian_thread.is_alive() and state.failure is None:
                state.request_stop(
                    "guardian_stop_timeout",
                    RuntimeError("Guardian did not stop within 15 seconds"),
                )
        if last_gaze_timestamp > 0.0:
            try:
                gaze_pipeline.finish(last_gaze_timestamp)
            except ValueError:
                pass
        if eeg_recorder is not None:
            eeg_recorder.close()
        if glasses_recorder is not None:
            glasses_recorder.close()
        if frame_metadata_writer is not None:
            frame_metadata_writer.close()
        if gaze_metadata_writer is not None:
            gaze_metadata_writer.close()
        if display["enabled"]:
            cv2.destroyAllWindows()

    if adapter is not None:
        diagnostics.frame_timestamp_drops = int(
            getattr(adapter, "dropped_frame_timestamp_count", 0)
        )
        diagnostics.gaze_timestamp_drops = int(
            getattr(adapter, "dropped_gaze_timestamp_count", 0)
        )
    diagnostic_snapshot = diagnostics.snapshot()
    completed_at = clock.now()
    capture_duration = (
        None
        if capture_started_at is None
        else max(0.0, completed_at - capture_started_at)
    )
    rate_denominator = (
        capture_duration if capture_duration and capture_duration > 0 else None
    )
    stream_rates_hz = {
        "scene_received": (
            None
            if rate_denominator is None
            else diagnostic_snapshot["scene_received"] / rate_denominator
        ),
        "scene_processed": (
            None
            if rate_denominator is None
            else diagnostic_snapshot["scene_processed"] / rate_denominator
        ),
        "gaze_received": (
            None
            if rate_denominator is None
            else diagnostic_snapshot["gaze_received"] / rate_denominator
        ),
    }
    summary = {
        "schema": "neurotech.practice_summary.v1",
        "run_type": "practice",
        "experimental_session": False,
        "stop_reason": state.reason or "completed",
        "successful": state.failure is None,
        "started_timestamp": event_start,
        "capture_started_timestamp": capture_started_at,
        "completed_timestamp": completed_at,
        "capture_duration_seconds": capture_duration,
        "eeg_enabled": resolved["eeg"]["enabled"],
        "calibration_result": repr(calibration_result),
        "diagnostics": diagnostic_snapshot,
        "stream_rates_hz": stream_rates_hz,
        "tracker_configuration": {
            "frame_rate_hz": gaze_config["tracker"]["frame_rate"],
            "status": "provisional_pending_processed_rate_pilot",
        },
        "eeg": None if eeg_monitor is None else eeg_monitor.snapshot(),
        "error": None if state.failure is None else type(state.failure).__name__,
    }
    with (run_directory / "practice_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    events.log(
        completed_at,
        "practice_run_stopped",
        {
            "reason": summary["stop_reason"],
            "successful": summary["successful"],
        },
    )
    terminal.report(
        completed_at,
        (
            f"stopped: {summary['stop_reason']} | "
            f"successful={summary['successful']} | artifacts={run_directory}"
        ),
    )

    if state.failure is not None:
        raise RuntimeError(
            f"practice session failed ({state.reason}): {state.failure}"
        ) from state.failure
    return run_directory
