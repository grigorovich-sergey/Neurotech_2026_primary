"""Authoritative practice-style live experiment runner.

This is the hardware-tested live path. It keeps
the participant/session/learning contracts of the main experiment, but copies
the acquisition and display cadence that has proved responsive in
``run_practice_session``:

* latest-only scene and bounded latest-gaze queues;
* one scene inference followed by a gaze batch and one render per loop;
* persistent buffered JSONL output rather than opening the event file for every
  high-rate gaze/metadata event;
* raw glasses HDF5 recording disabled by default (opt in with
  ``--record-glasses``); and
* a detailed practice-like diagnostic overlay.

Run this file from the repository root. It intentionally favors live
responsiveness over the strict lossless/cross-stream reordering guarantees of
``integration.live_workflow``, which is retained as the strict reference path.
EEG, model prediction, scoring, feedback, session
completion, and training advancement still use the normal experiment modules.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
from typing import Any
from uuid import uuid4

import cv2
import h5py
import numpy as np

from eeg_pipeline.contracts import EEGSample
from eeg_pipeline.credentials import load_guardian_api_token
from eeg_pipeline.guardian import GuardianAdapter
from eeg_pipeline.recording import SCHEMA_VERSION, TIMEBASE, VALUE_UNITS
from experiment_learning.artifacts import (
    artifact_digest,
    immutable_write_json,
    load_json_object,
)
from experiment_learning.assignment import (
    ModelAssignment,
    cli_model_assignment,
    load_model_assignment,
    save_model_assignment,
)
from experiment_learning.contracts import Condition
from experiment_learning.guardian_source import GuardianEEGFeatureSource
from experiment_learning.policy import (
    FrozenSessionPolicy,
    create_cold_start_policy,
    save_frozen_policy,
)
from experiment_learning.sessions import save_completed_session
from experiment_learning.state_machine import ExperimentController
from experiment_learning.trainer import TrainerConfig, train_next_session_policy
from foundations.config import load_resolved_config, save_resolved_config
from foundations.contracts import GazeSample, SceneFrame
from foundations.events import Event
from foundations.operator_gate import format_impedance, wait_for_space_or_abort
from foundations.recording import HDF5Recorder
from foundations.timebase import MonotonicClock
from gaze_interaction.association import GazeAssociator
from gaze_interaction.detector import YOLOEDetector
from gaze_interaction.dwell import DwellController, DwellTrigger
from gaze_interaction.episodes import EpisodeTracker
from gaze_interaction.pipeline import GazeInteractionPipeline, InteractionUpdate, SceneUpdate
from gaze_interaction.tracker import ByteTrackAdapter
from gaze_interaction.visualization import render_diagnostic
from mindlink import FrameMetadata, GazeMetadata, MindLinkAdapter

from integration.analysis import generate_analysis, generate_participant_analysis
from integration.live_input import ContextualFeedbackDriver
from integration.orchestrator import IntegratedExperimentOrchestrator, IntegratedGazeUpdate
from integration.workflow import (
    PROJECT_ROOT,
    _build_eeg_pipeline,
    _completed_inputs,
    _ensure_policy,
    _mapping,
    _project_path,
)


_SUBJECT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class _BufferedEventLogger:
    """Thread-safe persistent JSONL writer for high-rate live events."""

    def __init__(self, path: Path, *, flush_seconds: float = 0.25) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._flush_seconds = float(flush_seconds)
        self._next_flush = time.monotonic() + self._flush_seconds

    def log(self, event: Event) -> None:
        line = json.dumps(
            {
                "timestamp": float(event.timestamp),
                "name": event.name,
                "payload": event.payload,
            },
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._lock:
            self._handle.write(line)
            self._handle.write("\n")
            now = time.monotonic()
            if now >= self._next_flush:
                self._handle.flush()
                self._next_flush = now + self._flush_seconds

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    def flush(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()


class _AsyncBatchEEGRecorder:
    """Persist finalized EEG off the UI thread with one resize per batch.

    Guardian finalization can release hundreds of 250 Hz samples at an episode
    decision cutoff.  The stock recorder resizes five HDF5 datasets once per
    sample; doing that synchronously is visible as a periodic UI freeze.
    """

    _STOP = object()

    def __init__(
        self,
        path: Path,
        *,
        sample_rate_hz: float,
        batch_size: int = 1000,
    ) -> None:
        self.path = path
        self.sample_rate_hz = float(sample_rate_hz)
        self.batch_size = int(batch_size)
        self._queue: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._failure_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="practice-style-eeg-writer",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("EEG writer did not initialize")
        self.check_health()

    def record(self, sample: EEGSample) -> None:
        if not isinstance(sample, EEGSample):
            raise TypeError("sample must be an EEGSample")
        if self._closed:
            raise RuntimeError("EEG recorder is closed")
        self.check_health()
        self._queue.put_nowait(sample)

    def check_health(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("background EEG recording failed") from failure

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._STOP)
        self._thread.join(timeout=60.0)
        if self._thread.is_alive():
            raise RuntimeError("background EEG writer did not stop within 60 seconds")
        self.check_health()

    def _set_failure(self, error: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = error

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(self.path, "w") as file:
                file.attrs["schema_version"] = SCHEMA_VERSION
                file.attrs["sample_rate_hz"] = self.sample_rate_hz
                file.attrs["timebase"] = TIMEBASE
                file.attrs["value_units"] = VALUE_UNITS
                for name, dtype in (
                    ("timestamp", np.float64),
                    ("value_uv", np.float64),
                    ("valid", np.bool_),
                    ("vendor_timestamp_unix", np.float64),
                    ("host_receipt_timestamp", np.float64),
                ):
                    file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
                self._ready.set()
                last_timestamp: float | None = None
                stopping = False
                while not stopping:
                    first = self._queue.get()
                    if first is self._STOP:
                        break
                    batch = [first]
                    while len(batch) < self.batch_size:
                        try:
                            # Briefly coalesce the burst produced by one
                            # Guardian finalization boundary.
                            item = self._queue.get(timeout=0.01)
                        except queue.Empty:
                            break
                        if item is self._STOP:
                            stopping = True
                            break
                        batch.append(item)
                    timestamps = np.asarray(
                        [float(sample.timestamp) for sample in batch], dtype=np.float64
                    )
                    if last_timestamp is not None and timestamps[0] < last_timestamp:
                        raise ValueError("EEG timestamps must be non-decreasing")
                    if len(timestamps) > 1 and np.any(timestamps[1:] < timestamps[:-1]):
                        raise ValueError("EEG timestamps must be non-decreasing")
                    values = {
                        "timestamp": timestamps,
                        "value_uv": np.asarray(
                            [sample.value_uv for sample in batch], dtype=np.float64
                        ),
                        "valid": np.asarray(
                            [sample.valid for sample in batch], dtype=np.bool_
                        ),
                        "vendor_timestamp_unix": np.asarray(
                            [
                                np.nan
                                if sample.vendor_timestamp_unix is None
                                else sample.vendor_timestamp_unix
                                for sample in batch
                            ],
                            dtype=np.float64,
                        ),
                        "host_receipt_timestamp": np.asarray(
                            [
                                np.nan
                                if sample.host_receipt_timestamp is None
                                else sample.host_receipt_timestamp
                                for sample in batch
                            ],
                            dtype=np.float64,
                        ),
                    }
                    old_size = len(file["timestamp"])
                    new_size = old_size + len(batch)
                    for name, array in values.items():
                        dataset = file[name]
                        dataset.resize(new_size, axis=0)
                        dataset[old_size:new_size] = array
                    last_timestamp = float(timestamps[-1])
                file.flush()
        except BaseException as exc:
            self._set_failure(exc)
            self._ready.set()


class _LiveDiagnostics:
    """Small synchronized counter set shared with MindLink callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.scene_received = 0
        self.scene_processed = 0
        self.gaze_received = 0
        self.valid_gaze_received = 0
        self.scene_queue_drops = 0
        self.gaze_queue_drops = 0
        self.adapter_frame_drops = 0

    def received_scene(self) -> None:
        with self._lock:
            self.scene_received += 1

    def processed_scene(self) -> None:
        with self._lock:
            self.scene_processed += 1

    def received_gaze(self, valid: bool) -> None:
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

    def frame_metadata(self, dropped: int) -> None:
        with self._lock:
            self.adapter_frame_drops = max(self.adapter_frame_drops, int(dropped))

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "scene_received": self.scene_received,
                "scene_processed": self.scene_processed,
                "gaze_received": self.gaze_received,
                "valid_gaze_received": self.valid_gaze_received,
                "scene_queue_drops": self.scene_queue_drops,
                "gaze_queue_drops": self.gaze_queue_drops,
                "adapter_frame_drops": self.adapter_frame_drops,
            }


def _overlay_status(image_rgb: Any, lines: list[str]) -> Any:
    """Draw a readable practice-style status panel without resizing the frame."""

    if not lines:
        return image_rgb
    line_height = 22
    panel_height = min(image_rgb.shape[0], 12 + line_height * len(lines))
    overlay = image_rgb.copy()
    cv2.rectangle(overlay, (0, 0), (image_rgb.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image_rgb, 0.38, 0.0, image_rgb)
    for index, line in enumerate(lines):
        cv2.putText(
            image_rgb,
            line,
            (8, 20 + line_height * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return image_rgb


def _overlay_gaze_indicator(image_rgb: Any, gaze: GazeSample | None) -> Any:
    if gaze is None or not gaze.valid:
        return image_rgb
    if gaze.x_normalized is None or gaze.y_normalized is None:
        return image_rgb
    x = int(round(float(gaze.x_normalized) * max(0, image_rgb.shape[1] - 1)))
    y = int(round(float(gaze.y_normalized) * max(0, image_rgb.shape[0] - 1)))
    cv2.circle(image_rgb, (x, y), 10, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(image_rgb, (x, y), 2, (255, 255, 0), -1, cv2.LINE_AA)
    return image_rgb


@dataclass(frozen=True)
class _SessionBinding:
    subject_directory: Path
    lineage_directory: Path
    completed_paths: tuple[Path, ...]
    session_number: int
    session_id: str
    attempt_id: str
    assignment: ModelAssignment
    assignment_path: Path
    policy: FrozenSessionPolicy
    policy_sha256: str
    policy_path: Path


class _DeferredAttemptClock:
    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.clock: Any | None = None

    @property
    def started(self) -> bool:
        return self.clock is not None

    def start(self) -> float:
        if self.clock is not None:
            raise RuntimeError("attempt clock has already started")
        candidate = self.factory()
        if not callable(getattr(candidate, "now", None)):
            raise TypeError("clock_factory must return an object with now()")
        self.clock = candidate
        return float(candidate.now())

    def now(self) -> float:
        return 0.0 if self.clock is None else float(self.clock.now())


class _AttemptState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.failure: BaseException | None = None
        self.stop_reason: str | None = None

    def fail(self, reason: str, error: BaseException) -> None:
        with self._lock:
            if self.failure is None:
                self.failure = error
                self.stop_reason = reason

    def stop(self, reason: str) -> None:
        with self._lock:
            if self.stop_reason is None:
                self.stop_reason = reason


class _OperatorKeys:
    """Poll the OpenCV window and, when available, the launching terminal."""

    def __init__(
        self,
        *,
        window_enabled: bool,
        injected: Callable[[], int] | None,
    ) -> None:
        self.window_enabled = window_enabled
        self.injected = injected
        self._terminal_state: Any | None = None

    def open(self) -> None:
        if self.injected is not None or os.name == "nt" or not sys.stdin.isatty():
            return
        import termios
        import tty

        descriptor = sys.stdin.fileno()
        self._terminal_state = termios.tcgetattr(descriptor)
        tty.setcbreak(descriptor)

    def poll(self) -> int:
        if self.injected is not None:
            return int(self.injected())
        if self.window_enabled:
            key = cv2.waitKeyEx(1)
            if key >= 0:
                return key
        if os.name == "nt":
            import msvcrt

            return ord(msvcrt.getwch()) if msvcrt.kbhit() else -1
        if self._terminal_state is None:
            return -1
        import select

        readable, _, _ = select.select([sys.stdin.fileno()], [], [], 0.0)
        return ord(sys.stdin.read(1)) if readable else -1

    def close(self) -> None:
        if self._terminal_state is None:
            return
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._terminal_state)
        self._terminal_state = None


class _LiveDisplay:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        clock: Callable[[], float],
        terminal: Callable[[float, str], None],
        status_lines: Callable[[], list[str]],
    ) -> None:
        self.enabled = bool(config["enabled"])
        self.window_name = config["window_name"]
        self.selection_banner_seconds = float(config["selection_banner_seconds"])
        self.feedback_banner_seconds = float(config["feedback_banner_seconds"])
        self.clock = clock
        self.terminal = terminal
        self.status_lines = status_lines
        self.frame: SceneFrame | None = None
        self.scene: SceneUpdate | None = None
        self.gaze: GazeSample | None = None
        self.interaction: InteractionUpdate | None = None
        self.intent_score: float | None = None
        self.decision_reason: str | None = None
        self.selection_label: str | None = None
        self.selection_until = 0.0
        self.feedback_until = 0.0
        self.opened = False

    def open(self) -> None:
        if self.enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.opened = True

    def update_scene(self, frame: SceneFrame, update: SceneUpdate) -> None:
        self.frame = frame
        self.scene = update

    def update_gaze(self, update: IntegratedGazeUpdate) -> None:
        self.gaze = update.interaction.gaze
        self.interaction = update.interaction
        self.intent_score = update.applied_intent_score
        self.decision_reason = None if update.decision is None else update.decision.reason

    def present(self, trigger: DwellTrigger, interaction: InteractionUpdate) -> float:
        label = next(
            (
                item.label
                for item in (() if self.scene is None else self.scene.tracks)
                if item.track_id == trigger.track_id
            ),
            f"object #{trigger.track_id}",
        )
        self.selection_label = label
        self.selection_until = self.clock() + self.selection_banner_seconds
        self.gaze = interaction.gaze
        self.interaction = interaction
        self.render()
        presented = max(float(trigger.timestamp), self.clock())
        self.terminal(
            presented,
            f"SELECTION: {label} (track {trigger.track_id})",
        )
        return presented

    def mark_feedback(self) -> None:
        self.feedback_until = self.clock() + self.feedback_banner_seconds

    def render(self) -> None:
        if not self.enabled or self.frame is None:
            return
        image = render_diagnostic(
            self.frame,
            tracks=(() if self.scene is None else self.scene.tracks),
            gaze=self.gaze,
            candidate=(None if self.interaction is None else self.interaction.candidate),
            dwell_state=(None if self.interaction is None else self.interaction.dwell_state),
            intent_score=self.intent_score,
        )
        image = _overlay_gaze_indicator(image, self.gaze)
        image = _overlay_status(image, self.status_lines())
        if self.selection_label is not None and self.clock() <= self.selection_until:
            cv2.putText(
                image,
                f"SELECTED: {self.selection_label}",
                (8, max(24, image.shape[0] - 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 215, 0),
                2,
                cv2.LINE_AA,
            )
        if self.clock() <= self.feedback_until:
            cv2.putText(
                image,
                "FEEDBACK BUTTON PRESSED",
                (8, max(24, image.shape[0] - 50)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.opened:
            cv2.destroyWindow(self.window_name)
            self.opened = False

    def window_was_closed(self) -> bool:
        return bool(
            self.opened
            and cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        )


def _positive_number(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_subject_id(value: Any) -> str:
    if not isinstance(value, str) or _SUBJECT_ID.fullmatch(value) is None:
        raise ValueError(
            "subject_id must be 1..64 characters using letters, digits, '.', '_', or '-'"
        )
    return value


def _new_attempt_directory(output_root: str, subject_id: str) -> tuple[Path, Path, str]:
    root = _project_path(output_root, name="output_root")
    assert root is not None
    subject = root / "subjects" / subject_id
    attempts = subject / "attempts"
    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-"
        + uuid4().hex[:8]
    )
    attempt = attempts / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    return subject, attempt, attempt_id


def _resolve_subsystem(entry: dict[str, Any], name: str) -> dict[str, Any]:
    default = _project_path(entry.get("default"), name=f"subsystem_configs.{name}.default")
    override = _project_path(
        entry.get("override"),
        name=f"subsystem_configs.{name}.override",
        allow_none=True,
    )
    assert default is not None
    return load_resolved_config(default, override)


def _load_subsystems(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    entries = _mapping(config, "subsystem_configs")
    return tuple(
        _resolve_subsystem(_mapping(entries, name), name)
        for name in (
            "mindlink",
            "gaze_interaction",
            "eeg_pipeline",
            "experiment_learning",
        )
    )


def _validate_config(
    config: dict[str, Any],
    mindlink: dict[str, Any],
    gaze: dict[str, Any],
    eeg: dict[str, Any],
    learning: dict[str, Any],
) -> tuple[str, Condition]:
    if not isinstance(config.get("output_root"), str) or not config["output_root"]:
        raise ValueError("output_root must be a non-empty path string")
    runtime = _mapping(config, "runtime")
    subject_id = _validate_subject_id(runtime.get("subject_id"))
    try:
        active = Condition(runtime.get("active_model"))
    except ValueError as exc:
        raise ValueError("runtime.active_model must be G or E") from exc
    selection = _mapping(config, "model_selection")
    if selection.get("source") != "cli" or _mapping(selection, "csv").get("enabled") is not False:
        raise ValueError("the prototype main runner currently requires CLI model selection")
    if _mapping(config, "terminal").get("concise_decisions") is not True:
        raise ValueError("the main runner requires terminal.concise_decisions: true")
    controlled = _mapping(config, "controlled_intention")
    if controlled.get("enabled") is not False:
        raise ValueError("controlled-intention trials are placeholders and are not implemented")
    duration = _positive_number(
        "session.maximum_duration_seconds",
        _mapping(config, "session").get("maximum_duration_seconds"),
    )
    processing = _mapping(config, "processing")
    _positive_int("processing.scene_queue_size", processing.get("scene_queue_size"))
    _positive_int("processing.gaze_queue_size", processing.get("gaze_queue_size"))
    _positive_int("processing.gaze_batch_size", processing.get("gaze_batch_size"))
    _positive_number("processing.idle_sleep_seconds", processing.get("idle_sleep_seconds"))
    recording = _mapping(config, "recording")
    if not isinstance(recording.get("glasses_enabled"), bool):
        raise ValueError("recording.glasses_enabled must be a bool")
    if recording.get("eeg_enabled") is not True:
        raise ValueError("the practice-style experiment still requires EEG recording")
    display = _mapping(config, "display")
    if not isinstance(display.get("enabled"), bool):
        raise ValueError("display.enabled must be a bool")
    if not isinstance(display.get("window_name"), str) or not display["window_name"]:
        raise ValueError("display.window_name must be non-empty")
    _positive_number("display.selection_banner_seconds", display.get("selection_banner_seconds"))
    _positive_number("display.feedback_banner_seconds", display.get("feedback_banner_seconds"))
    _positive_number("display.no_frame_warning_seconds", display.get("no_frame_warning_seconds"))
    feedback = _mapping(config, "feedback")
    key = feedback.get("key_code")
    if isinstance(key, bool) or not isinstance(key, int) or key < 0:
        raise ValueError("feedback.key_code must be a non-negative integer")
    stop_keys = feedback.get("stop_key_codes")
    if not isinstance(stop_keys, list) or not stop_keys or any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
        for item in stop_keys
    ):
        raise ValueError("feedback.stop_key_codes must be a non-empty list of byte codes")
    source = _mapping(eeg, "source")
    if source.get("mode") != "live":
        raise ValueError("the main experiment requires EEG source.mode: live")
    guardian = _mapping(source, "guardian")
    impedance = _mapping(guardian, "impedance")
    if impedance.get("enabled") is not True:
        raise ValueError("live Guardian fitting requires impedance.enabled: true")
    _positive_number("Guardian impedance.max_ohms", impedance.get("max_ohms"))
    if impedance.get("mains_frequency_hz") not in {50, 60}:
        raise ValueError("Guardian impedance.mains_frequency_hz must be 50 or 60")
    feedback_timeout = _positive_number(
        "learning.timing.feedback_timeout_s",
        _mapping(learning, "timing").get("feedback_timeout_s"),
    )
    recording_seconds = guardian.get("recording_seconds")
    if (
        isinstance(recording_seconds, bool)
        or not isinstance(recording_seconds, int)
        or recording_seconds <= duration + feedback_timeout
    ):
        raise ValueError(
            "Guardian recording_seconds must exceed session duration plus feedback grace"
        )
    if _mapping(eeg, "recording").get("enabled") is not True:
        raise ValueError("the main experiment requires EEG pipeline recording.enabled: true")
    _mapping(mindlink, "connection")
    _mapping(mindlink, "calibration")
    _mapping(mindlink, "capture")
    _mapping(gaze, "detector")
    if _mapping(config, "analysis").get("enabled") is not True:
        raise ValueError("the main experiment requires analysis.enabled: true")
    return subject_id, active


def _prepare_session_binding(
    *,
    config: dict[str, Any],
    learning_config: dict[str, Any],
    subject_directory: Path,
    attempt_id: str,
    subject_id: str,
    active: Condition,
) -> _SessionBinding:
    lineage = subject_directory / "lineage"
    lineage.mkdir(parents=True, exist_ok=True)
    provisional = cli_model_assignment(subject_id, 1, active)
    completed = _completed_inputs(
        lineage,
        participant_id=subject_id,
        sequence_id=provisional.binding_id,
        schedule_sha256=provisional.binding_sha256,
    )
    session_number = len(completed) + 1
    assignment = cli_model_assignment(subject_id, session_number, active)
    assignment_path = lineage / f"assignment_session_{session_number:03d}.json"
    if assignment_path.exists() and load_model_assignment(assignment_path) != assignment:
        existing = load_model_assignment(assignment_path)
        raise ValueError(
            "retry active model differs from the initiated session assignment: "
            f"expected {existing.active_condition.value}, received {active.value}"
        )
    policy_path = lineage / f"policy_session_{session_number:03d}.json"
    if policy_path.exists() or session_number > 1:
        policy, policy_sha256, policy_path = _ensure_policy(
            directory=lineage,
            participant_id=subject_id,
            sequence_id=assignment.binding_id,
            schedule_sha256=assignment.binding_sha256,
            session_number=session_number,
            completed_paths=completed,
            learning_config=learning_config,
        )
    else:
        cold = _mapping(learning_config, "cold_start_policy")
        policy = create_cold_start_policy(
            participant_id=subject_id,
            schedule_sequence_id=assignment.binding_id,
            schedule_sha256=assignment.binding_sha256,
            base_threshold_s=cold["base_threshold_s"],
            minimum_e_threshold_s=cold["minimum_e_threshold_s"],
            base_search_min_s=cold["base_search_min_s"],
            base_search_max_s=cold["base_search_max_s"],
            base_search_step_s=cold["base_search_step_s"],
            maximum_allowed_reduction_fraction=cold[
                "maximum_allowed_reduction_fraction"
            ],
        )
        policy_sha256 = artifact_digest(policy.to_payload())
    prefix = _mapping(config, "session")["id_prefix"]
    return _SessionBinding(
        subject_directory=subject_directory,
        lineage_directory=lineage,
        completed_paths=completed,
        session_number=session_number,
        session_id=f"{prefix}-{subject_id}-{session_number:03d}",
        attempt_id=attempt_id,
        assignment=assignment,
        assignment_path=assignment_path,
        policy=policy,
        policy_sha256=policy_sha256,
        policy_path=policy_path,
    )


def _build_gaze_pipeline(
    config: dict[str, Any],
    *,
    policy: FrozenSessionPolicy,
    active: Condition,
    detector: Any | None,
    tracker: Any | None,
) -> GazeInteractionPipeline:
    detector_config = _mapping(config, "detector")
    tracker_config = _mapping(config, "tracker")
    association = _mapping(config, "association")
    episode = _mapping(config, "episode")
    dwell = _mapping(config, "dwell")
    policy_dwell = policy.dwell_parameters(active)
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
            baseline_seconds=policy_dwell.baseline_seconds,
            minimum_seconds=policy_dwell.minimum_seconds,
            maximum_seconds=policy_dwell.maximum_seconds,
            maximum_reduction_fraction=policy_dwell.maximum_reduction_fraction,
            max_sample_gap_seconds=dwell["max_sample_gap_seconds"],
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def run_live_experiment(
    config: dict[str, Any],
    *,
    detector: Any | None = None,
    tracker: Any | None = None,
    mindlink_factory: Callable[..., Any] = MindLinkAdapter,
    guardian_factory: Callable[..., Any] = GuardianAdapter,
    clock_factory: Callable[[], Any] = MonotonicClock,
    start_gate: Callable[[], bool] | None = None,
    poll_key: Callable[[], int] | None = None,
    guardian_token_loader: Callable[..., str] = load_guardian_api_token,
) -> Path:
    """Run one live attempt; only a clean duration completion advances training."""

    resolved = deepcopy(config)
    mindlink_config, gaze_config, eeg_config, learning_config = _load_subsystems(resolved)
    subject_id, active = _validate_config(
        resolved, mindlink_config, gaze_config, eeg_config, learning_config
    )
    subject_directory, run_directory, attempt_id = _new_attempt_directory(
        resolved["output_root"], subject_id
    )
    binding = _prepare_session_binding(
        config=resolved,
        learning_config=learning_config,
        subject_directory=subject_directory,
        attempt_id=attempt_id,
        subject_id=subject_id,
        active=active,
    )
    save_resolved_config(resolved, run_directory / "resolved_experiment_config.json")
    for name, value in (
        ("mindlink", mindlink_config),
        ("gaze_interaction", gaze_config),
        ("eeg_pipeline", eeg_config),
        ("experiment_learning", learning_config),
    ):
        save_resolved_config(value, run_directory / f"resolved_{name}_config.json")
    events = _BufferedEventLogger(run_directory / "events.jsonl")
    clock = _DeferredAttemptClock(clock_factory)
    state = _AttemptState()
    duration = float(resolved["session"]["maximum_duration_seconds"])
    feedback_timeout = float(learning_config["timing"]["feedback_timeout_s"])

    def terminal(timestamp: float, message: str) -> None:
        print(f"[experiment {timestamp:8.3f}s] {message}", flush=True)

    def setup(message: str) -> None:
        print(f"[experiment setup] {message}", flush=True)

    def fail(source: str, error: BaseException) -> None:
        state.fail(source, error)
        events.log(
            Event(
                clock.now(),
                "experiment_attempt_error",
                {"phase": "attempt" if clock.started else "setup", "source": source,
                 "error": type(error).__name__},
            )
        )

    gaze_pipeline = _build_gaze_pipeline(
        gaze_config,
        policy=binding.policy,
        active=active,
        detector=detector,
        tracker=tracker,
    )
    eeg_pipeline = _build_eeg_pipeline(eeg_config)
    experiment = ExperimentController(
        policy=binding.policy,
        policy_sha256=binding.policy_sha256,
        session_id=binding.session_id,
        session_number=binding.session_number,
        attempt_id=binding.attempt_id,
        active_condition=active,
        schedule_binding=binding.assignment.schedule_binding,
        minimum_prediction_elapsed_s=learning_config["timing"][
            "minimum_prediction_elapsed_s"
        ],
        eeg_window_s=learning_config["timing"]["eeg_window_s"],
        feedback_timeout_s=feedback_timeout,
        event_logger=events,
    )
    feedback = ContextualFeedbackDriver(event_logger=events, session_id=binding.session_id)
    diagnostics = _LiveDiagnostics()
    scene_queue: queue.Queue[SceneFrame] = queue.Queue(
        maxsize=resolved["processing"]["scene_queue_size"]
    )
    gaze_queue: queue.Queue[GazeSample] = queue.Queue(
        maxsize=resolved["processing"]["gaze_queue_size"]
    )

    mindlink: Any | None = None
    guardian: Any | None = None
    eeg_source: GuardianEEGFeatureSource | None = None
    glasses_recorder: HDF5Recorder | None = None
    eeg_recorder: _AsyncBatchEEGRecorder | None = None
    orchestrator: IntegratedExperimentOrchestrator | None = None
    guardian_impedance_started = False
    guardian_started = False
    capture_started = False
    assignment_saved = False
    normal_duration_completion = False
    completed_timestamp = 0.0
    latest_frame: SceneFrame | None = None
    latest_input_timestamp = 0.0
    setup_aborted = False
    intentional_close = threading.Event()
    battery_percent: float | None = None
    impedance_ohms: float | None = None
    capture_started_at = 0.0
    glasses_lock = threading.Lock()

    def status_lines() -> list[str]:
        snapshot = diagnostics.snapshot()
        now = clock.now()
        elapsed = max(now - capture_started_at, 1e-9)
        gaze_validity = (
            snapshot["valid_gaze_received"] / snapshot["gaze_received"]
            if snapshot["gaze_received"]
            else 0.0
        )
        gaze = None if display is None else display.gaze
        interaction = None if display is None else display.interaction
        candidate = None if interaction is None else interaction.candidate
        dwell = None if interaction is None else interaction.dwell_state
        lines = [
            (
                f"EXPERIMENT | subject {subject_id} | session {binding.session_number} | "
                f"active {active.value} / shadow {binding.assignment.shadow_condition.value}"
            ),
            (
                f"elapsed {now:.1f}/{duration:.0f}s | scene rx/proc "
                f"{snapshot['scene_received'] / elapsed:.1f}/"
                f"{snapshot['scene_processed'] / elapsed:.1f} Hz | "
                f"gaze {snapshot['gaze_received'] / elapsed:.1f} Hz | valid {gaze_validity:.1%}"
            ),
            (
                "drops adapter/scene-q/gaze-q: "
                f"{snapshot['adapter_frame_drops']}/"
                f"{snapshot['scene_queue_drops']}/"
                f"{snapshot['gaze_queue_drops']} | "
                f"raw glasses {'ON' if glasses_recorder is not None else 'OFF'}"
            ),
        ]
        if gaze is None:
            lines.append("gaze: waiting for sample")
        elif gaze.valid:
            lines.append(
                f"gaze: VALID x={gaze.x_normalized:.3f} y={gaze.y_normalized:.3f}"
            )
        else:
            lines.append("gaze: INVALID / outside image")
        candidate_text = "none" if candidate is None else (
            f"track {candidate.track_id} {candidate.label or ''}".strip()
        )
        dwell_text = "--" if dwell is None else (
            f"{dwell.accumulated_seconds:.3f}/{dwell.required_seconds:.3f}s"
        )
        score_text = "--" if display is None or display.intent_score is None else (
            f"{display.intent_score:.3f}"
        )
        reason_text = "--" if display is None or display.decision_reason is None else (
            display.decision_reason
        )
        lines.append(
            f"candidate: {candidate_text} | dwell {dwell_text} | "
            f"intent {score_text} | decision {reason_text}"
        )
        if guardian is None:
            lines.append("EEG: waiting for Guardian")
        else:
            lines.append(
                "EEG: recording | lost blocks/samples "
                f"{getattr(guardian, 'lost_block_count', 0)}/"
                f"{getattr(guardian, 'lost_sample_count', 0)} | "
                f"battery {'--' if battery_percent is None else f'{battery_percent:.0f}%'}"
            )
        lines.append("PRESENTER: feedback | Q/Esc: terminate")
        return lines

    display: _LiveDisplay | None = _LiveDisplay(
        config=resolved["display"],
        clock=clock.now,
        terminal=terminal,
        status_lines=status_lines,
    )
    keys = _OperatorKeys(
        window_enabled=resolved["display"]["enabled"], injected=poll_key
    )

    def handle_key(key: int, timestamp: float) -> None:
        if key < 0 or key == 255:
            return
        if key == resolved["feedback"]["key_code"]:
            feedback.submit_press(timestamp)
            display.mark_feedback()
        elif key in resolved["feedback"]["stop_key_codes"] or key == 3:
            state.stop("operator_terminated")

    def on_disconnect(error: Any) -> None:
        if not intentional_close.is_set():
            fail("mindlink_disconnect", RuntimeError(f"MindLink disconnected: {error}"))

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
        count = diagnostics.queue_drop(stream)
        events.log(
            Event(
                float(sample.timestamp),
                "experiment_practice_style_queue_drop",
                {"stream": stream, "count": count},
            )
        )
        target.put_nowait(sample)

    def on_scene(frame: SceneFrame) -> None:
        diagnostics.received_scene()
        record_glasses(frame)
        enqueue_latest(scene_queue, frame, "scene")

    def on_gaze(gaze: GazeSample) -> None:
        diagnostics.received_gaze(gaze.valid)
        record_glasses(gaze)
        enqueue_latest(gaze_queue, gaze, "gaze")

    def on_frame_metadata(metadata: FrameMetadata) -> None:
        diagnostics.frame_metadata(metadata.dropped_frame_count)
        if resolved["diagnostics"]["write_mindlink_metadata"]:
            events.log(
                Event(
                    metadata.timestamp,
                    "mindlink_frame_metadata",
                    {
                        "host_receipt_timestamp": metadata.host_receipt_timestamp,
                        "vendor_frame_timestamp": (
                            None
                            if metadata.vendor_frame_timestamp is None
                            else metadata.vendor_frame_timestamp.isoformat()
                        ),
                        "tracker_timestamp": metadata.tracker_timestamp,
                        "dropped_frame_count": metadata.dropped_frame_count,
                    },
                )
            )

    def on_gaze_metadata(metadata: GazeMetadata) -> None:
        if resolved["diagnostics"]["write_mindlink_metadata"]:
            events.log(
                Event(
                    metadata.timestamp,
                    "mindlink_gaze_metadata",
                    {
                        "host_receipt_timestamp": metadata.host_receipt_timestamp,
                        "vendor_timestamp": metadata.vendor_timestamp,
                    },
                )
            )

    try:
        events.log(
            Event(
                0.0,
                "experiment_attempt_prepared",
                {
                    "participant_id": subject_id,
                    "session_id": binding.session_id,
                    "session_number": binding.session_number,
                    "attempt_id": attempt_id,
                    "active_condition": active.value,
                    "shadow_condition": binding.assignment.shadow_condition.value,
                    "model_selection_source": binding.assignment.source.value,
                    "policy_path": str(binding.policy_path),
                    "policy_sha256": binding.policy_sha256,
                },
            )
        )
        capture = mindlink_config["capture"]
        mindlink = mindlink_factory(
            clock=clock.now,
            frame_queue_size=capture["frame_queue_size"],
            on_disconnect=on_disconnect,
        )
        setup("connecting to MindLink")
        mindlink.connect(**mindlink_config["connection"])
        setup("starting MindLink calibration")
        calibration_result = mindlink.calibrate(**mindlink_config["calibration"])
        if state.failure is not None:
            raise state.failure
        events.log(
            Event(
                0.0,
                "experiment_mindlink_calibrated",
                {"phase": "setup", "result": repr(calibration_result)},
            )
        )
        setup("MindLink calibration complete; acquisition remains OFF")

        guardian_config = eeg_config["source"]["guardian"]
        token = guardian_token_loader(
            environment_variable=guardian_config["api_token_env"],
            token_file=guardian_config["api_token_file"],
            base_directory=PROJECT_ROOT,
        )
        guardian = guardian_factory(
            clock=clock.now,
            address=guardian_config["address"],
            api_token=token,
            debug=guardian_config["debug"],
            queue_capacity_samples=guardian_config["queue_capacity_samples"],
        )
        setup("connecting to Guardian after gaze calibration")
        guardian.connect()
        battery_percent = guardian.check_battery()
        events.log(
            Event(
                0.0,
                "experiment_guardian_battery_checked",
                {"phase": "setup", "battery_percent": battery_percent},
            )
        )
        impedance_config = guardian_config["impedance"]
        guardian.start_impedance(
            mains_frequency_hz=impedance_config["mains_frequency_hz"]
        )
        guardian_impedance_started = True
        events.log(
            Event(
                0.0,
                "experiment_guardian_impedance_started",
                {
                    "phase": "setup",
                    "mains_frequency_hz": impedance_config["mains_frequency_hz"],
                },
            )
        )

        def fitting_status() -> str:
            if state.failure is not None:
                raise state.failure
            guardian.check_health()
            return (
                "Guardian fitting; raw EEG is OFF | "
                f"battery {battery_percent:.0f}% | "
                f"impedance {format_impedance(guardian.latest_impedance())} | "
                "press SPACE to start; Q, Esc, or Ctrl-C to abort"
            )

        try:
            should_start = (
                wait_for_space_or_abort(status=fitting_status, emit=setup)
                if start_gate is None
                else start_gate()
            )
        finally:
            guardian.stop_impedance()
            guardian_impedance_started = False
        impedance_ohms = guardian.latest_impedance()
        events.log(
            Event(
                0.0,
                "experiment_guardian_impedance_stopped",
                {"phase": "setup", "impedance_ohms": impedance_ohms},
            )
        )
        if not isinstance(should_start, bool):
            raise TypeError("start_gate must return a bool")
        if state.failure is not None:
            raise state.failure
        guardian.check_health()
        if not should_start:
            setup_aborted = True
            state.stop("operator_abort_before_start")
            events.log(
                Event(0.0, "experiment_setup_aborted", {"reason": state.stop_reason})
            )
        else:
            if impedance_ohms is None:
                raise RuntimeError("Guardian fitting ended before an impedance reading arrived")
            if impedance_ohms >= impedance_config["max_ohms"]:
                raise RuntimeError(
                    f"Guardian impedance {impedance_ohms:.0f} ohm is not below configured "
                    f"{impedance_config['max_ohms']:.0f} ohm threshold"
                )
            save_model_assignment(binding.assignment_path, binding.assignment)
            assignment_saved = True
            if not binding.policy_path.exists():
                persisted_policy_digest = save_frozen_policy(
                    binding.policy_path, binding.policy
                )
                if persisted_policy_digest != binding.policy_sha256:
                    raise RuntimeError("persisted cold-start policy digest changed")
            started_at = clock.start()
            if resolved["recording"]["glasses_enabled"]:
                glasses_recorder = HDF5Recorder(run_directory / "raw_glasses.h5")
            eeg_recorder = _AsyncBatchEEGRecorder(
                run_directory / "raw_eeg.h5",
                sample_rate_hz=eeg_config["signal"]["sample_rate_hz"],
            )
            eeg_source = GuardianEEGFeatureSource(
                guardian=guardian,
                pipeline=eeg_pipeline,
                recorder=eeg_recorder,
            )
            guardian_started = True
            guardian.start(recording_seconds=guardian_config["recording_seconds"])
            events.log(
                Event(
                    started_at,
                    "experiment_guardian_recording_started",
                    {"phase": "attempt"},
                )
            )
            display.open()
            keys.open()

            def present_action(
                trigger: DwellTrigger, interaction: InteractionUpdate
            ) -> float:
                presented = display.present(trigger, interaction)
                handle_key(keys.poll(), clock.now())
                return max(presented, clock.now())

            orchestrator = IntegratedExperimentOrchestrator(
                gaze_pipeline=gaze_pipeline,
                eeg_source=eeg_source,
                experiment=experiment,
                feedback=feedback,
                event_logger=events,
                session_id=binding.session_id,
                present_action=present_action,
            )
            mindlink.start_capture(
                on_scene_frame=on_scene,
                on_gaze_sample=on_gaze,
                on_frame_metadata=on_frame_metadata,
                on_gaze_metadata=on_gaze_metadata,
            )
            capture_started = True
            capture_started_at = clock.now()
            events.log(
                Event(
                    started_at,
                    "experiment_session_started",
                    {
                        "participant_id": subject_id,
                        "session_id": binding.session_id,
                        "session_number": binding.session_number,
                        "attempt_id": attempt_id,
                        "active_condition": active.value,
                        "shadow_condition": binding.assignment.shadow_condition.value,
                        "model_selection_source": "cli",
                        "maximum_duration_seconds": duration,
                    },
                )
            )
            terminal(
                started_at,
                "session started; presenter gives feedback, Q or Esc terminates",
            )
            warned_no_frame = False

            def take_practice_batch(*, drain_all_gaze: bool = False) -> list[tuple[str, Any]]:
                batch: list[tuple[str, Any]] = []
                try:
                    batch.append(("scene", scene_queue.get_nowait()))
                except queue.Empty:
                    pass
                limit = (
                    gaze_queue.qsize()
                    if drain_all_gaze
                    else resolved["processing"]["gaze_batch_size"]
                )
                for _ in range(limit):
                    try:
                        batch.append(("gaze", gaze_queue.get_nowait()))
                    except queue.Empty:
                        break
                # Practice uses independent queues.  Sorting only this small live
                # batch keeps the experiment controller's scientific clock
                # monotonic without rebuilding the lag-prone global merger.
                batch.sort(
                    key=lambda item: (
                        float(item[1].timestamp),
                        0 if item[0] == "scene" else 1,
                    )
                )
                return batch

            def process_practice_batch(batch: list[tuple[str, Any]]) -> bool:
                nonlocal latest_frame, latest_input_timestamp
                processed_any = False
                for stream, sample in batch:
                    timestamp = float(sample.timestamp)
                    if timestamp > duration + 1e-12:
                        continue
                    if timestamp + 1e-12 < orchestrator.latest_processed_scientific_timestamp:
                        events.log(
                            Event(
                                timestamp,
                                "experiment_practice_style_late_processing_drop",
                                {
                                    "stream": stream,
                                    "latest_processed_timestamp": (
                                        orchestrator.latest_processed_scientific_timestamp
                                    ),
                                },
                            )
                        )
                        continue
                    latest_input_timestamp = max(latest_input_timestamp, timestamp)
                    if stream == "scene":
                        latest_frame = sample
                        display.update_scene(sample, orchestrator.process_scene(sample))
                        diagnostics.processed_scene()
                    else:
                        display.update_gaze(orchestrator.process_gaze(sample))
                    processed_any = True
                return processed_any

            while state.failure is None and state.stop_reason is None:
                now = clock.now()
                eeg_recorder.check_health()
                if now >= duration:
                    state.stop("duration_reached")
                    break
                processed = process_practice_batch(take_practice_batch())
                if not processed:
                    eeg_source.drain_through(
                        orchestrator.latest_processed_scientific_timestamp
                    )
                    if scene_queue.empty() and gaze_queue.empty():
                        orchestrator.advance_time(
                            orchestrator.latest_processed_scientific_timestamp
                        )
                if guardian.recording_done and now < duration - 1e-9:
                    raise RuntimeError("Guardian recording ended before the session deadline")
                display.render()
                handle_key(keys.poll(), now)
                if display.window_was_closed():
                    state.stop("operator_terminated")
                if (
                    latest_frame is None
                    and not warned_no_frame
                    and now >= resolved["display"]["no_frame_warning_seconds"]
                ):
                    warned_no_frame = True
                    terminal(now, "WARNING: no scene frame received")
                if not processed:
                    time.sleep(resolved["processing"]["idle_sleep_seconds"])

            if state.failure is None and state.stop_reason == "duration_reached":
                mindlink.stop_capture()
                capture_started = False
                process_practice_batch(take_practice_batch(drain_all_gaze=True))
                deadline = orchestrator.begin_deadline(duration)
                grace_deadline = max(deadline, clock.now()) + feedback_timeout
                while (
                    experiment.pending_feedback_episode_id is not None
                    and clock.now() < grace_deadline
                    and state.failure is None
                ):
                    now = clock.now()
                    handle_key(keys.poll(), now)
                    orchestrator.advance_time(now)
                    display.render()
                    time.sleep(resolved["processing"]["idle_sleep_seconds"])
                if experiment.pending_feedback_episode_id is not None:
                    orchestrator.advance_time(grace_deadline)
                orchestrator.assert_ready_to_complete()
                completed_timestamp = max(deadline, clock.now())
                normal_duration_completion = True
    except KeyboardInterrupt:
        state.stop("operator_terminated")
    except BaseException as exc:
        fail("runtime_error", exc)

    intentional_close.set()
    cleanup_steps: list[tuple[str, Callable[[], Any]]] = []
    if mindlink is not None:
        if capture_started:
            cleanup_steps.append(("mindlink_stop_capture", mindlink.stop_capture))
        cleanup_steps.append(("mindlink_close", mindlink.close))
    if guardian is not None:
        if guardian_impedance_started:
            cleanup_steps.append(("guardian_stop_impedance", guardian.stop_impedance))
        if guardian_started:
            cleanup_steps.append(("guardian_stop", guardian.stop))
        if eeg_source is not None:
            cleanup_steps.append(("eeg_drain_remaining", eeg_source.drain_remaining))
        cleanup_steps.append(("guardian_close", guardian.close))
    for name, operation in cleanup_steps:
        try:
            operation()
            events.log(Event(clock.now(), "experiment_cleanup_completed", {"step": name}))
        except BaseException as exc:
            fail(name, exc)
    for name, recorder in (("glasses_recorder", glasses_recorder), ("eeg_recorder", eeg_recorder)):
        if recorder is None:
            continue
        try:
            recorder.close()
        except BaseException as exc:
            fail(name, exc)
    try:
        display.close()
    except BaseException as exc:
        fail("display_close", exc)
    try:
        keys.close()
    except BaseException as exc:
        fail("key_reader_close", exc)

    successful = normal_duration_completion and state.failure is None
    summary: dict[str, Any] = {
        "schema": "neurotech.experiment_attempt_summary.v1",
        "participant_id": subject_id,
        "session_id": binding.session_id,
        "session_number": binding.session_number,
        "attempt_id": attempt_id,
        "active_condition": active.value,
        "shadow_condition": binding.assignment.shadow_condition.value,
        "model_selection_source": binding.assignment.source.value,
        "assignment_saved": assignment_saved,
        "attempt_started": clock.started,
        "setup_aborted": setup_aborted,
        "stop_reason": state.stop_reason,
        "successful": successful,
        "latest_input_timestamp": latest_input_timestamp,
        "completed_timestamp": completed_timestamp if successful else None,
        "scene_queue_drops": diagnostics.snapshot()["scene_queue_drops"],
        "gaze_queue_drops": diagnostics.snapshot()["gaze_queue_drops"],
        "practice_style_runner": True,
        "raw_glasses_recording_enabled": glasses_recorder is not None,
        "mindlink_adapter_frame_drops": int(
            getattr(mindlink, "dropped_frame_count", 0)
        ),
        "mindlink_frame_timestamp_drops": int(
            getattr(mindlink, "dropped_frame_timestamp_count", 0)
        ),
        "mindlink_gaze_timestamp_drops": int(
            getattr(mindlink, "dropped_gaze_timestamp_count", 0)
        ),
        "guardian_battery_percent": battery_percent,
        "guardian_impedance_ohms": impedance_ohms,
        "guardian_recording_id": getattr(guardian, "recording_id", None),
        "guardian_lost_sample_count": int(getattr(guardian, "lost_sample_count", 0)),
        "guardian_lost_block_count": int(getattr(guardian, "lost_block_count", 0)),
        "error": None if state.failure is None else type(state.failure).__name__,
    }

    if successful:
        try:
            completed = experiment.completed_session(completed_timestamp)
            run_completed_path = run_directory / "completed_session.json"
            run_digest = save_completed_session(run_completed_path, completed)
            events.flush()
            generate_analysis(run_directory / "events.jsonl", run_directory)
            generate_participant_analysis(
                (*binding.completed_paths, run_completed_path),
                run_directory,
            )

            staged_policy_path = (
                run_directory
                / f"staged_policy_session_{binding.session_number + 1:03d}.json"
            )
            training = train_next_session_policy(
                (*binding.completed_paths, run_completed_path),
                binding.policy,
                staged_policy_path,
                TrainerConfig.from_mapping(learning_config["trainer"]),
            )

            participant_completed_path = (
                binding.lineage_directory
                / f"completed_session_{binding.session_number:03d}.json"
            )
            participant_digest = save_completed_session(
                participant_completed_path, completed
            )
            if run_digest != participant_digest:
                raise RuntimeError("run and participant completed-session artifacts differ")
            next_policy_path = (
                binding.lineage_directory
                / f"policy_session_{binding.session_number + 1:03d}.json"
            )
            published_policy_digest = immutable_write_json(
                next_policy_path, load_json_object(staged_policy_path)
            )
            if published_policy_digest != training.policy_sha256:
                raise RuntimeError("staged and participant policy artifacts differ")
            staged_report_path = staged_policy_path.with_name(
                staged_policy_path.stem + ".training_report.json"
            )
            participant_report_path = next_policy_path.with_name(
                next_policy_path.stem + ".training_report.json"
            )
            immutable_write_json(
                participant_report_path, load_json_object(staged_report_path)
            )
            generate_participant_analysis(
                (*binding.completed_paths, participant_completed_path),
                binding.lineage_directory,
            )
            summary.update(
                {
                    "completed_session_sha256": participant_digest,
                    "next_policy_path": str(next_policy_path),
                    "next_policy_sha256": published_policy_digest,
                    "training_status": training.report["status"],
                }
            )
            events.log(
                Event(
                    completed_timestamp,
                    "experiment_session_completed",
                    {
                        "participant_id": subject_id,
                        "session_number": binding.session_number,
                        "attempt_id": attempt_id,
                        "active_condition": active.value,
                        "shadow_condition": binding.assignment.shadow_condition.value,
                        "model_selection_source": "cli",
                        "next_policy_sha256": published_policy_digest,
                        "training_status": training.report["status"],
                    },
                )
            )
            terminal(
                completed_timestamp,
                f"session {binding.session_number} completed and trained",
            )
        except BaseException as exc:
            fail("session_finalization", exc)
            successful = False
            summary["successful"] = False
            summary["completed_timestamp"] = None
            summary["error"] = type(exc).__name__
            summary["stop_reason"] = "session_finalization"
    if not successful:
        events.log(
            Event(
                clock.now(),
                "experiment_session_incomplete",
                {
                    "participant_id": subject_id,
                    "session_number": binding.session_number,
                    "attempt_id": attempt_id,
                    "attempt_started": clock.started,
                    "reason": state.stop_reason,
                    "error": summary["error"],
                },
            )
        )
    events.close()
    _write_json(run_directory / "attempt_summary.json", summary)
    if state.failure is not None:
        raise state.failure
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="partial YAML configuration overriding configs/experiment.yaml",
    )
    parser.add_argument(
        "--subject-id",
        required=True,
        help="pseudonymous subject identifier used for recording and policy history",
    )
    parser.add_argument(
        "--active-model",
        required=True,
        choices=("G", "E"),
        help="model controlling visible actions; the other model is shadow",
    )
    parser.add_argument(
        "--record-glasses",
        action="store_true",
        help=(
            "opt in to raw scene/gaze HDF5 recording; disabled by default to match "
            "the smooth practice configuration"
        ),
    )
    args = parser.parse_args()
    default_config = PROJECT_ROOT / "configs" / "experiment.yaml"
    try:
        config = load_resolved_config(default_config, args.config)
        config["runtime"]["subject_id"] = args.subject_id
        config["runtime"]["active_model"] = args.active_model
        config["recording"]["glasses_enabled"] = bool(args.record_glasses)
        run_directory = run_live_experiment(config)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(run_directory)
    summary_path = run_directory / "attempt_summary.json"
    if summary_path.is_file():
        print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
