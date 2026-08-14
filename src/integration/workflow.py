"""Configured end-to-end experiment workflow with live Guardian support."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from eeg_pipeline.buffer import EEGBuffer
from eeg_pipeline.contracts import EEGSample
from eeg_pipeline.credentials import load_guardian_api_token
from eeg_pipeline.guardian import GuardianAdapter
from eeg_pipeline.pipeline import EEGPipeline
from eeg_pipeline.processing import EEGFeatureExtractor, EEGPreprocessor, EEGQualityGate
from eeg_pipeline.recording import EEGHDF5Recorder, EEGHDF5Replay
from eeg_pipeline.synthetic import synthetic_eeg_samples
from experiment_learning.contracts import Condition
from experiment_learning.guardian_source import GuardianEEGFeatureSource
from experiment_learning.policy import (
    FrozenSessionPolicy,
    create_cold_start_policy,
    load_frozen_policy,
    save_frozen_policy,
)
from experiment_learning.schedule import (
    ScheduleBinding,
    load_condition_schedule,
    resolve_scheduled_condition,
)
from experiment_learning.sessions import load_completed_session, save_completed_session
from experiment_learning.state_machine import ExperimentController
from experiment_learning.trainer import TrainerConfig, train_next_session_policy
from foundations.config import load_resolved_config, save_resolved_config
from foundations.contracts import GazeSample, SceneFrame
from foundations.events import Event, JsonlEventLogger
from foundations.operator_gate import format_impedance, wait_for_space_or_abort
from foundations.recording import HDF5Recorder, HDF5Replay
from foundations.timebase import MonotonicClock
from foundations.virtual_glasses import VirtualGlasses
from gaze_interaction.association import GazeAssociator
from gaze_interaction.detector import YOLOEDetector
from gaze_interaction.dwell import DwellController
from gaze_interaction.episodes import EpisodeTracker
from gaze_interaction.pipeline import GazeInteractionPipeline
from gaze_interaction.tracker import ByteTrackAdapter
from gaze_interaction.visualization import close_windows, render_diagnostic, save_rgb_image
from test_harness.gaze import GazeCsvSource, MouseGazeSource
from test_harness.video import VideoSceneSource

from integration.analysis import generate_analysis
from integration.orchestrator import (
    IntegratedExperimentOrchestrator,
    KeyboardFeedbackDriver,
    RecordedEEGFeatureSource,
    ScheduledFeedbackPress,
    SyntheticFeedbackDriver,
    TimedFeedbackDriver,
)
from integration.vision import SyntheticVisionAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAZE_DEFAULT = PROJECT_ROOT / "configs" / "gaze_interaction.yaml"
EEG_DEFAULT = PROJECT_ROOT / "configs" / "eeg_pipeline.yaml"
LEARNING_DEFAULT = PROJECT_ROOT / "configs" / "experiment_learning.yaml"
VIDEO_WINDOW = "NeuroTech integrated experiment"


class OperatorAbort(RuntimeError):
    """Raised when the operator aborts at the live-attempt SPACE gate."""


@dataclass(frozen=True)
class _AttemptBinding:
    participant_directory: Path
    completed_paths: tuple[Path, ...]
    session_number: int
    session_id: str
    attempt_id: str
    condition: Condition
    schedule_binding: ScheduleBinding
    policy: FrozenSessionPolicy
    policy_sha256: str
    policy_path: Path


class _DeferredAttemptClock:
    """Expose a clock callable whose origin is the operator start signal."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        if not callable(factory):
            raise TypeError("clock_factory must be callable")
        self._factory = factory
        self._clock: Any | None = None

    def start(self) -> float:
        if self._clock is not None:
            raise RuntimeError("attempt clock has already started")
        clock = self._factory()
        if not callable(getattr(clock, "now", None)):
            raise TypeError("clock_factory must return an object with now()")
        self._clock = clock
        return float(clock.now())

    def now(self) -> float:
        return 0.0 if self._clock is None else float(self._clock.now())


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


def _project_path(value: Any, *, name: str, allow_none: bool = False) -> Path | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _new_run_directory(output_root: str) -> Path:
    root = _project_path(output_root, name="output_root")
    assert root is not None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / "integration" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _load_subsystem_configs(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    overrides = _mapping(config, "subsystem_config_overrides")

    def resolved(default: Path, key: str) -> dict[str, Any]:
        override = _project_path(
            overrides.get(key),
            name=f"subsystem_config_overrides.{key}",
            allow_none=True,
        )
        return load_resolved_config(default, override)

    return (
        resolved(GAZE_DEFAULT, "gaze_interaction"),
        resolved(EEG_DEFAULT, "eeg_pipeline"),
        resolved(LEARNING_DEFAULT, "experiment_learning"),
    )


def _validate_integration_config(config: dict[str, Any]) -> None:
    if not isinstance(config.get("output_root"), str) or not config["output_root"]:
        raise ValueError("output_root must be a non-empty path string")
    participant = _mapping(config, "participant")
    for key in ("id", "sequence_id", "artifact_directory"):
        if not isinstance(participant.get(key), str) or not participant[key]:
            raise ValueError(f"participant.{key} must be a non-empty string")
    session = _mapping(config, "session")
    if not isinstance(session.get("id_prefix"), str) or not session["id_prefix"]:
        raise ValueError("session.id_prefix must be a non-empty string")
    _positive_number(
        "session.maximum_duration_seconds", session.get("maximum_duration_seconds")
    )
    input_config = _mapping(config, "input")
    if input_config.get("mode") not in {"synthetic", "replay", "video"}:
        raise ValueError("input.mode must be 'synthetic', 'replay', or 'video'")
    if not isinstance(input_config.get("record_glasses"), bool):
        raise ValueError("input.record_glasses must be a bool")
    feedback = _mapping(config, "feedback")
    if feedback.get("mode") not in {"synthetic", "replay", "keyboard"}:
        raise ValueError("feedback.mode must be 'synthetic', 'replay', or 'keyboard'")
    if not isinstance(_mapping(config, "analysis").get("enabled"), bool):
        raise ValueError("analysis.enabled must be a bool")
    vision = _mapping(config, "synthetic_vision")
    SyntheticVisionAdapter(
        warmup_seconds=vision.get("warmup_seconds"),
        visible_seconds=vision.get("visible_seconds"),
        blank_seconds=vision.get("blank_seconds"),
        label=vision.get("label"),
    )


def _validate_resolved_inputs(
    config: dict[str, Any],
    gaze_config: dict[str, Any],
    eeg_config: dict[str, Any],
    learning_config: dict[str, Any],
) -> None:
    input_config = _mapping(config, "input")
    input_mode = input_config["mode"]
    gaze_source = _mapping(gaze_config, "source")
    if input_mode == "synthetic":
        virtual = _mapping(gaze_source, "virtual")
        VirtualGlasses(
            seed=virtual["seed"],
            duration_seconds=virtual["duration_seconds"],
            scene_width=virtual["scene_width"],
            scene_height=virtual["scene_height"],
            scene_rate_hz=virtual["scene_rate_hz"],
            gaze_rate_hz=virtual["gaze_rate_hz"],
            scene_dropout_probability=virtual["scene_dropout_probability"],
            gaze_dropout_probability=virtual["gaze_dropout_probability"],
            gaze_invalid_probability=virtual["gaze_invalid_probability"],
        )
    elif input_mode == "replay":
        path = _project_path(gaze_source.get("recording_path"), name="gaze source.recording_path")
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(path)
        if not isinstance(gaze_source.get("replay_paced"), bool):
            raise ValueError("gaze source.replay_paced must be a bool")
    else:
        video = _mapping(input_config, "video")
        path = _project_path(video.get("path"), name="input.video.path")
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(path)
        if video.get("gaze_mode") not in {"mouse", "file"}:
            raise ValueError("input.video.gaze_mode must be 'mouse' or 'file'")
        if video["gaze_mode"] == "file":
            gaze_path = _project_path(video.get("gaze_csv_path"), name="input.video.gaze_csv_path")
            assert gaze_path is not None
            if not gaze_path.is_file():
                raise FileNotFoundError(gaze_path)
        if video["gaze_mode"] == "mouse" and video.get("show_window") is not True:
            raise ValueError("input.video.show_window must be true for mouse gaze")
        for key in ("paced", "show_window", "save_frames"):
            if not isinstance(video.get(key), bool):
                raise ValueError(f"input.video.{key} must be a bool")

    source = _mapping(eeg_config, "source")
    eeg_mode = source.get("mode")
    if eeg_mode not in {"synthetic", "replay", "live"}:
        raise ValueError("EEG source.mode must be 'synthetic', 'replay', or 'live'")
    if eeg_mode == "replay":
        path = _project_path(source.get("replay_path"), name="EEG source.replay_path")
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(path)
    if eeg_mode == "live":
        duration = config["session"]["maximum_duration_seconds"]
        guardian = _mapping(source, "guardian")
        impedance = _mapping(guardian, "impedance")
        if impedance.get("enabled") is not True:
            raise ValueError("live Guardian fitting requires impedance.enabled: true")
        _positive_number("Guardian impedance.max_ohms", impedance.get("max_ohms"))
        if impedance.get("mains_frequency_hz") not in {50, 60}:
            raise ValueError("Guardian impedance.mains_frequency_hz must be 50 or 60")
        recording_seconds = guardian.get("recording_seconds")
        if (
            isinstance(recording_seconds, bool)
            or not isinstance(recording_seconds, int)
            or recording_seconds <= duration
        ):
            raise ValueError(
                "live Guardian recording_seconds must exceed session.maximum_duration_seconds"
            )
        if input_mode == "synthetic":
            pass
        elif input_mode == "replay" and gaze_source.get("replay_paced") is not True:
            raise ValueError("live Guardian with replay input requires source.replay_paced: true")
        elif input_mode == "video" and input_config["video"].get("paced") is not True:
            raise ValueError("live Guardian with video input requires input.video.paced: true")

    feedback = _mapping(config, "feedback")
    timeout = float(_mapping(learning_config, "timing")["feedback_timeout_s"])
    if feedback["mode"] == "synthetic":
        synthetic = _mapping(feedback, "synthetic")
        cycle = synthetic.get("press_cycle")
        if not isinstance(cycle, list) or not cycle or any(not isinstance(item, bool) for item in cycle):
            raise ValueError("feedback.synthetic.press_cycle must be a non-empty bool list")
        delay = synthetic.get("press_delay_seconds")
        if (
            isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or delay < 0
            or delay >= timeout
        ):
            raise ValueError("feedback.synthetic.press_delay_seconds must satisfy 0 <= delay < timeout")
    elif feedback["mode"] == "replay":
        path = _project_path(
            _mapping(feedback, "replay").get("events_path"),
            name="feedback.replay.events_path",
        )
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(path)
        _replay_feedback(path)
    else:
        key_code = _mapping(feedback, "keyboard").get("key_code")
        if isinstance(key_code, bool) or not isinstance(key_code, int) or not 0 <= key_code <= 255:
            raise ValueError("feedback.keyboard.key_code must be an integer within [0, 255]")
        if input_mode != "video" or input_config["video"].get("show_window") is not True:
            raise ValueError("keyboard feedback requires video input with show_window: true")


def _build_gaze_pipeline(
    config: dict[str, Any],
    *,
    policy: FrozenSessionPolicy,
    condition: Condition,
    synthetic_vision: dict[str, Any] | None,
) -> GazeInteractionPipeline:
    detector_config = _mapping(config, "detector")
    tracker_config = _mapping(config, "tracker")
    association = _mapping(config, "association")
    episode = _mapping(config, "episode")
    gaze_dwell = _mapping(config, "dwell")
    policy_dwell = policy.dwell_parameters(condition)
    if synthetic_vision is None:
        detector = YOLOEDetector(
            detector_config["model"],
            confidence_threshold=detector_config["confidence_threshold"],
            image_size=detector_config["image_size"],
            device=detector_config["device"],
            category_filter=detector_config["category_filter"],
        )
        tracker = ByteTrackAdapter(
            activation_threshold=tracker_config["activation_threshold"],
            lost_track_buffer=tracker_config["lost_track_buffer"],
            matching_threshold=tracker_config["matching_threshold"],
            frame_rate=tracker_config["frame_rate"],
        )
    else:
        adapter = SyntheticVisionAdapter(
            warmup_seconds=synthetic_vision["warmup_seconds"],
            visible_seconds=synthetic_vision["visible_seconds"],
            blank_seconds=synthetic_vision["blank_seconds"],
            label=synthetic_vision["label"],
        )
        detector = adapter
        tracker = adapter
    return GazeInteractionPipeline(
        detector=detector,
        tracker=tracker,
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
            max_sample_gap_seconds=gaze_dwell["max_sample_gap_seconds"],
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


def _completed_inputs(
    directory: Path,
    *,
    participant_id: str,
    sequence_id: str,
    schedule_sha256: str,
) -> tuple[Path, ...]:
    paths = tuple(sorted(directory.glob("completed_session_*.json")))
    for expected_number, path in enumerate(paths, start=1):
        expected_name = f"completed_session_{expected_number:03d}.json"
        if path.name != expected_name:
            raise ValueError(
                "completed-session artifacts must be contiguous and named "
                f"completed_session_NNN.json; expected {expected_name}, found {path.name}"
            )
        session, _ = load_completed_session(path)
        if not session.successful:
            raise ValueError("participant artifact directory contains an incomplete session")
        if session.participant_id != participant_id or session.session_number != expected_number:
            raise ValueError("completed-session participant/session identity mismatch")
        if (
            session.schedule_sequence_id != sequence_id
            or session.schedule_sha256 != schedule_sha256
        ):
            raise ValueError("completed-session schedule binding differs from current schedule")
    return paths


def _ensure_policy(
    *,
    directory: Path,
    participant_id: str,
    sequence_id: str,
    schedule_sha256: str,
    session_number: int,
    completed_paths: tuple[Path, ...],
    learning_config: dict[str, Any],
) -> tuple[FrozenSessionPolicy, str, Path]:
    path = directory / f"policy_session_{session_number:03d}.json"
    if not path.exists():
        if session_number == 1:
            cold = _mapping(learning_config, "cold_start_policy")
            policy = create_cold_start_policy(
                participant_id=participant_id,
                schedule_sequence_id=sequence_id,
                schedule_sha256=schedule_sha256,
                base_threshold_s=cold["base_threshold_s"],
                minimum_e_threshold_s=cold["minimum_e_threshold_s"],
                base_search_min_s=cold["base_search_min_s"],
                base_search_max_s=cold["base_search_max_s"],
                base_search_step_s=cold["base_search_step_s"],
                maximum_allowed_reduction_fraction=cold[
                    "maximum_allowed_reduction_fraction"
                ],
            )
            save_frozen_policy(path, policy)
        else:
            prior, _, _ = _ensure_policy(
                directory=directory,
                participant_id=participant_id,
                sequence_id=sequence_id,
                schedule_sha256=schedule_sha256,
                session_number=session_number - 1,
                completed_paths=completed_paths[:-1],
                learning_config=learning_config,
            )
            if len(completed_paths) != session_number - 1:
                raise ValueError("cannot train a policy without completed sessions 1..N-1")
            train_next_session_policy(
                completed_paths,
                prior,
                path,
                TrainerConfig.from_mapping(_mapping(learning_config, "trainer")),
            )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    policy = load_frozen_policy(
        path,
        expected_participant_id=participant_id,
        expected_session=session_number,
        expected_sha256=digest,
    )
    if (
        policy.schedule_sequence_id != sequence_id
        or policy.schedule_sha256 != schedule_sha256
    ):
        raise ValueError("frozen policy schedule binding differs from current schedule")
    if len(policy.source_attempts) != len(completed_paths):
        raise ValueError("frozen policy lineage does not cover completed sessions 1..N-1")
    for source_attempt, completed_path in zip(
        policy.source_attempts, completed_paths, strict=True
    ):
        completed, completed_sha256 = load_completed_session(completed_path)
        if (
            source_attempt.session_number != completed.session_number
            or source_attempt.attempt_id != completed.attempt_id
            or source_attempt.artifact_sha256 != completed_sha256
        ):
            raise ValueError("frozen policy lineage differs from completed-session artifacts")
    return policy, digest, path


def _prepare_attempt_binding(
    config: dict[str, Any],
    learning_config: dict[str, Any],
    run_directory: Path,
) -> _AttemptBinding:
    participant = _mapping(config, "participant")
    participant_id = participant["id"]
    sequence_id = participant["sequence_id"]
    directory = _project_path(
        participant["artifact_directory"], name="participant.artifact_directory"
    )
    assert directory is not None
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = _project_path(
        learning_config.get("schedule_path"), name="experiment schedule_path"
    )
    assert schedule_path is not None
    schedule = load_condition_schedule(schedule_path)
    completed_paths = _completed_inputs(
        directory,
        participant_id=participant_id,
        sequence_id=sequence_id,
        schedule_sha256=schedule.sha256,
    )
    session_number = len(completed_paths) + 1
    scheduled = resolve_scheduled_condition(
        schedule,
        sequence_id,
        session_number,
        ScheduleBinding(sequence_id, schedule.sha256),
    )
    policy, policy_sha256, policy_path = _ensure_policy(
        directory=directory,
        participant_id=participant_id,
        sequence_id=sequence_id,
        schedule_sha256=schedule.sha256,
        session_number=session_number,
        completed_paths=completed_paths,
        learning_config=learning_config,
    )
    session_id = f"{config['session']['id_prefix']}-{session_number:03d}"
    attempt_id = f"{session_id}-{run_directory.name}"
    return _AttemptBinding(
        directory,
        completed_paths,
        session_number,
        session_id,
        attempt_id,
        scheduled.condition,
        scheduled.binding,
        policy,
        policy_sha256,
        policy_path,
    )


def _eeg_samples(config: dict[str, Any]) -> Iterator[EEGSample]:
    source = _mapping(config, "source")
    mode = source.get("mode")
    if mode == "synthetic":
        synthetic = _mapping(source, "synthetic")
        yield from synthetic_eeg_samples(
            sample_rate_hz=config["signal"]["sample_rate_hz"],
            duration_seconds=synthetic["duration_seconds"],
            tones=synthetic["tones"],
            noise_std_uv=synthetic["noise_std_uv"],
            seed=synthetic["seed"],
            gaps=synthetic["gaps"],
            invalid_intervals=synthetic["invalid_intervals"],
        )
        return
    if mode == "replay":
        path = _project_path(source.get("replay_path"), name="EEG source.replay_path")
        assert path is not None
        yield from EEGHDF5Replay(path).samples()
        return
    raise ValueError("iterable EEG source.mode must be 'synthetic' or 'replay'")


def _glasses_samples(
    input_mode: str, gaze_config: dict[str, Any], events: JsonlEventLogger
) -> Iterator[SceneFrame | GazeSample]:
    source = _mapping(gaze_config, "source")
    if input_mode == "synthetic":
        virtual = _mapping(source, "virtual")
        glasses = VirtualGlasses(
            seed=virtual["seed"],
            duration_seconds=virtual["duration_seconds"],
            scene_width=virtual["scene_width"],
            scene_height=virtual["scene_height"],
            scene_rate_hz=virtual["scene_rate_hz"],
            gaze_rate_hz=virtual["gaze_rate_hz"],
            scene_dropout_probability=virtual["scene_dropout_probability"],
            gaze_dropout_probability=virtual["gaze_dropout_probability"],
            gaze_invalid_probability=virtual["gaze_invalid_probability"],
        )
        yield from glasses.samples(
            on_dropout=lambda stream, timestamp: events.log(
                Event(timestamp, "sensor_dropout", {"stream": stream})
            )
        )
        return
    if input_mode == "replay":
        path = _project_path(source.get("recording_path"), name="gaze source.recording_path")
        assert path is not None
        yield from HDF5Replay(path).replay(paced=bool(source["replay_paced"]))
        return
    raise ValueError("glasses samples are only available for synthetic/replay input")


def _replay_feedback(path: Path) -> list[ScheduledFeedbackPress]:
    raw_presses: list[ScheduledFeedbackPress] = []
    record_presses: list[ScheduledFeedbackPress] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid feedback JSONL on line {line_number}") from exc
            payload = event.get("payload", {})
            if event.get("name") == "integration_feedback_press":
                raw_presses.append(
                    ScheduledFeedbackPress(float(event["timestamp"]), int(payload["episode_id"]))
                )
            elif (
                event.get("name") == "experiment_episode_training_record"
                and payload.get("feedback_pressed")
            ):
                record_presses.append(
                    ScheduledFeedbackPress(
                        float(payload["feedback_resolution_timestamp"]),
                        int(payload["episode_id"]),
                    )
                )
    return raw_presses if raw_presses else record_presses


def _build_feedback_driver(
    config: dict[str, Any],
    *,
    event_logger: JsonlEventLogger,
    session_id: str,
    feedback_timeout_s: float,
):
    feedback = _mapping(config, "feedback")
    mode = feedback["mode"]
    if mode == "synthetic":
        synthetic = _mapping(feedback, "synthetic")
        if synthetic["press_delay_seconds"] >= feedback_timeout_s:
            raise ValueError("feedback.synthetic.press_delay_seconds must be less than timeout")
        return SyntheticFeedbackDriver(
            press_cycle=synthetic["press_cycle"],
            press_delay_seconds=synthetic["press_delay_seconds"],
            event_logger=event_logger,
            session_id=session_id,
        )
    if mode == "replay":
        path = _project_path(
            _mapping(feedback, "replay").get("events_path"),
            name="feedback.replay.events_path",
        )
        assert path is not None
        return TimedFeedbackDriver(
            _replay_feedback(path), event_logger=event_logger, session_id=session_id
        )
    return KeyboardFeedbackDriver(
        key_code=_mapping(feedback, "keyboard")["key_code"],
        event_logger=event_logger,
        session_id=session_id,
    )


class _Pacer:
    def __init__(self, paced: bool) -> None:
        if not isinstance(paced, bool):
            raise ValueError("paced must be a bool")
        self.paced = paced
        self.started_at: float | None = None

    def wait(self, timestamp: float) -> None:
        if not self.paced:
            return
        if self.started_at is None:
            self.started_at = time.monotonic() - timestamp
        delay = self.started_at + timestamp - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def _wait_for_start_signal(
    read_key: Callable[[], str] | None = None,
    *,
    status: Callable[[], str] | None = None,
    emit: Callable[[str], None] | None = None,
) -> bool:
    return wait_for_space_or_abort(
        read_key=read_key,
        status=status,
        emit=emit,
    )


def _show_rgb(image) -> None:
    import cv2

    cv2.imshow(VIDEO_WINDOW, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _render_video(
    *,
    frame: SceneFrame,
    tracks: tuple,
    gaze: GazeSample | None,
    integrated_update,
    config: dict[str, Any],
    run_directory: Path,
    image_index: int,
) -> None:
    interaction = integrated_update.interaction if integrated_update is not None else None
    image = render_diagnostic(
        frame,
        tracks=tracks,
        gaze=gaze,
        candidate=interaction.candidate if interaction else None,
        dwell_state=interaction.dwell_state if interaction else None,
        intent_score=(integrated_update.applied_intent_score if integrated_update else None),
    )
    if config["save_frames"]:
        save_rgb_image(run_directory / "diagnostics" / f"frame_{image_index:06d}.png", image)
    if config["show_window"]:
        _show_rgb(image)


def _run_stream_input(
    *,
    samples: Iterator[SceneFrame | GazeSample],
    orchestrator: IntegratedExperimentOrchestrator,
    glasses_recorder: HDF5Recorder | None,
    deadline: float,
    pacer: _Pacer,
    health_check: Callable[[float], None],
) -> tuple[float, float]:
    last_timestamp = 0.0
    deadline_reached = False
    for sample in samples:
        timestamp = float(sample.timestamp)
        if timestamp > deadline + 1e-12:
            deadline_reached = True
            break
        pacer.wait(timestamp)
        last_timestamp = max(last_timestamp, timestamp)
        if glasses_recorder is not None:
            glasses_recorder.record(sample)
        if isinstance(sample, SceneFrame):
            orchestrator.process_scene(sample)
        else:
            orchestrator.process_gaze(sample)
        health_check(timestamp)
        if timestamp >= deadline - 1e-12:
            deadline_reached = True
            break
    if deadline_reached:
        completed_at = orchestrator.cancel_at_deadline(deadline)
        return deadline, completed_at
    completed_at = orchestrator.finish(last_timestamp)
    return last_timestamp, completed_at


def _run_video_input(
    *,
    config: dict[str, Any],
    orchestrator: IntegratedExperimentOrchestrator,
    glasses_recorder: HDF5Recorder | None,
    run_directory: Path,
    deadline: float,
    health_check: Callable[[float], None],
) -> tuple[float, float]:
    video = _mapping(_mapping(config, "input"), "video")
    path = _project_path(video.get("path"), name="input.video.path")
    assert path is not None
    source = VideoSceneSource(path)
    gaze_mode = video["gaze_mode"]
    pacer = _Pacer(video["paced"])
    image_index = 0
    last_timestamp = 0.0
    deadline_reached = False

    if gaze_mode == "mouse":
        mouse = MouseGazeSource(window_name=VIDEO_WINDOW)
        for frame in source.frames():
            timestamp = float(frame.timestamp)
            if timestamp > deadline + 1e-12:
                deadline_reached = True
                break
            pacer.wait(timestamp)
            last_timestamp = max(last_timestamp, timestamp)
            scene_update = orchestrator.process_scene(frame)
            if glasses_recorder is not None:
                glasses_recorder.record(frame)
            mouse.set_scene_shape(*frame.image.shape[:2])
            _render_video(
                frame=frame,
                tracks=scene_update.tracks,
                gaze=None,
                integrated_update=None,
                config=video,
                run_directory=run_directory,
                image_index=image_index,
            )
            image_index += 1
            health_check(timestamp)
            if not mouse.window_is_open():
                break
            gaze = mouse.sample(timestamp)
            if glasses_recorder is not None:
                glasses_recorder.record(gaze)
            integrated = orchestrator.process_gaze(gaze)
            _render_video(
                frame=frame,
                tracks=scene_update.tracks,
                gaze=gaze,
                integrated_update=integrated,
                config=video,
                run_directory=run_directory,
                image_index=image_index,
            )
            image_index += 1
            health_check(timestamp)
            if timestamp >= deadline - 1e-12:
                deadline_reached = True
                break
    else:
        gaze_path = _project_path(video.get("gaze_csv_path"), name="input.video.gaze_csv_path")
        assert gaze_path is not None
        scene_iterator = iter(source.frames())
        gaze_iterator = iter(GazeCsvSource(gaze_path).samples())
        next_scene = next(scene_iterator, None)
        next_gaze = next(gaze_iterator, None)
        latest_frame: SceneFrame | None = None
        latest_tracks: tuple = ()
        while next_scene is not None or next_gaze is not None:
            scene_first = next_scene is not None and (
                next_gaze is None or next_scene.timestamp <= next_gaze.timestamp
            )
            item = next_scene if scene_first else next_gaze
            assert item is not None
            timestamp = float(item.timestamp)
            if timestamp > deadline + 1e-12:
                deadline_reached = True
                break
            pacer.wait(timestamp)
            last_timestamp = max(last_timestamp, timestamp)
            if scene_first:
                frame = next_scene
                assert frame is not None
                scene_update = orchestrator.process_scene(frame)
                latest_frame = frame
                latest_tracks = scene_update.tracks
                if glasses_recorder is not None:
                    glasses_recorder.record(frame)
                _render_video(
                    frame=frame,
                    tracks=scene_update.tracks,
                    gaze=None,
                    integrated_update=None,
                    config=video,
                    run_directory=run_directory,
                    image_index=image_index,
                )
                image_index += 1
                next_scene = next(scene_iterator, None)
            else:
                gaze = next_gaze
                assert gaze is not None
                if latest_frame is None:
                    raise ValueError("video gaze begins before the first video frame")
                if next_scene is None:
                    video_end = float(latest_frame.timestamp) + source.frame_period_seconds
                    if timestamp >= video_end + 1e-9:
                        raise ValueError(
                            f"gaze timestamp {timestamp:.6f}s is beyond the video timeline "
                            f"ending at {video_end:.6f}s"
                        )
                if glasses_recorder is not None:
                    glasses_recorder.record(gaze)
                integrated = orchestrator.process_gaze(gaze)
                _render_video(
                    frame=latest_frame,
                    tracks=latest_tracks,
                    gaze=gaze,
                    integrated_update=integrated,
                    config=video,
                    run_directory=run_directory,
                    image_index=image_index,
                )
                image_index += 1
                next_gaze = next(gaze_iterator, None)
            health_check(timestamp)
            if timestamp >= deadline - 1e-12:
                deadline_reached = True
                break

    if deadline_reached:
        completed_at = orchestrator.cancel_at_deadline(deadline)
        return deadline, completed_at
    completed_at = orchestrator.finish(last_timestamp)
    return last_timestamp, completed_at


def run_integrated_experiment(
    config: dict[str, Any],
    *,
    guardian_factory: Callable[..., Any] = GuardianAdapter,
    clock_factory: Callable[[], Any] = MonotonicClock,
    start_gate: Callable[[], bool] | None = None,
) -> Path:
    """Run one frozen-policy attempt and persist immutable successful inputs."""

    _validate_integration_config(config)
    resolved = deepcopy(config)
    gaze_config, eeg_config, learning_config = _load_subsystem_configs(resolved)
    _validate_resolved_inputs(resolved, gaze_config, eeg_config, learning_config)
    run_directory = _new_run_directory(resolved["output_root"])
    save_resolved_config(resolved, run_directory / "resolved_integration_config.json")
    save_resolved_config(gaze_config, run_directory / "resolved_gaze_interaction_config.json")
    save_resolved_config(eeg_config, run_directory / "resolved_eeg_pipeline_config.json")
    save_resolved_config(learning_config, run_directory / "resolved_experiment_learning_config.json")
    events = JsonlEventLogger(run_directory / "events.jsonl")
    binding = _prepare_attempt_binding(resolved, learning_config, run_directory)
    timing = _mapping(learning_config, "timing")
    deadline = float(resolved["session"]["maximum_duration_seconds"])
    eeg_mode = eeg_config["source"]["mode"]

    events.log(
        Event(
            0.0,
            "integration_attempt_prepared",
            {
                "participant_id": binding.policy.participant_id,
                "session_id": binding.session_id,
                "session_number": binding.session_number,
                "attempt_id": binding.attempt_id,
                "active_condition": binding.condition.value,
                "sequence_id": binding.schedule_binding.sequence_id,
                "schedule_sha256": binding.schedule_binding.csv_sha256,
                "policy_path": str(binding.policy_path),
                "policy_sha256": binding.policy_sha256,
                "eeg_mode": eeg_mode,
            },
        )
    )

    experiment = ExperimentController(
        policy=binding.policy,
        policy_sha256=binding.policy_sha256,
        session_id=binding.session_id,
        session_number=binding.session_number,
        attempt_id=binding.attempt_id,
        active_condition=binding.condition,
        schedule_binding=binding.schedule_binding,
        minimum_prediction_elapsed_s=timing["minimum_prediction_elapsed_s"],
        eeg_window_s=timing["eeg_window_s"],
        feedback_timeout_s=timing["feedback_timeout_s"],
        event_logger=events,
    )
    input_mode = resolved["input"]["mode"]
    gaze_pipeline = _build_gaze_pipeline(
        gaze_config,
        policy=binding.policy,
        condition=binding.condition,
        synthetic_vision=(
            resolved["synthetic_vision"] if input_mode in {"synthetic", "replay"} else None
        ),
    )
    eeg_pipeline = _build_eeg_pipeline(eeg_config)
    feedback = _build_feedback_driver(
        resolved,
        event_logger=events,
        session_id=binding.session_id,
        feedback_timeout_s=float(timing["feedback_timeout_s"]),
    )
    attempt_clock = _DeferredAttemptClock(clock_factory)
    guardian_adapter: Any | None = None
    guardian_impedance_started = False
    guardian_started = False
    eeg_source: Any | None = None
    eeg_recorder: EEGHDF5Recorder | None = None
    glasses_recorder: HDF5Recorder | None = None
    orchestrator: IntegratedExperimentOrchestrator | None = None
    last_timestamp = 0.0
    completed_timestamp = 0.0
    attempt_started = False
    primary_error: BaseException | None = None

    try:
        if eeg_mode == "live":
            guardian = _mapping(_mapping(eeg_config, "source"), "guardian")
            api_token = load_guardian_api_token(
                environment_variable=guardian["api_token_env"],
                token_file=guardian["api_token_file"],
                base_directory=PROJECT_ROOT,
            )
            guardian_adapter = guardian_factory(
                clock=attempt_clock.now,
                address=guardian["address"],
                api_token=api_token,
                debug=guardian["debug"],
                queue_capacity_samples=guardian["queue_capacity_samples"],
            )
            impedance = _mapping(guardian, "impedance")
            events.log(
                Event(
                    0.0,
                    "integration_guardian_setup_started",
                    {"phase": "setup", "raw_eeg_active": False},
                )
            )
            guardian_adapter.connect()
            battery_percent = guardian_adapter.check_battery()
            events.log(
                Event(
                    0.0,
                    "integration_guardian_battery_checked",
                    {
                        "phase": "setup",
                        "raw_eeg_active": False,
                        "battery_percent": battery_percent,
                    },
                )
            )
            if impedance["enabled"] is not True:
                raise ValueError("live Guardian fitting requires impedance.enabled: true")
            guardian_adapter.start_impedance(
                mains_frequency_hz=impedance["mains_frequency_hz"]
            )
            guardian_impedance_started = True
            events.log(
                Event(
                    0.0,
                    "integration_guardian_impedance_started",
                    {
                        "phase": "setup",
                        "raw_eeg_active": False,
                        "mains_frequency_hz": impedance["mains_frequency_hz"],
                    },
                )
            )

            def fitting_status() -> str:
                return (
                    "Guardian fitting; raw EEG is OFF | "
                    f"battery {battery_percent:.0f}% | "
                    f"impedance {format_impedance(guardian_adapter.latest_impedance())} | "
                    "press SPACE to accept fit; Q, Esc, or Ctrl-C to abort"
                )

            try:
                if start_gate is None:
                    should_start = _wait_for_start_signal(
                        status=fitting_status,
                        emit=lambda message: print(
                            f"[integration setup] {message}", flush=True
                        ),
                    )
                else:
                    print(f"[integration setup] {fitting_status()}", flush=True)
                    should_start = start_gate()
            finally:
                guardian_adapter.stop_impedance()
                guardian_impedance_started = False
            impedance_ohms = guardian_adapter.latest_impedance()
            events.log(
                Event(
                    0.0,
                    "integration_guardian_impedance_stopped",
                    {
                        "phase": "setup",
                        "raw_eeg_active": False,
                        "impedance_ohms": impedance_ohms,
                    },
                )
            )
            if not isinstance(should_start, bool):
                raise TypeError("start_gate must return a bool")
            if not should_start:
                events.log(
                    Event(
                        0.0,
                        "integration_setup_aborted",
                        {"phase": "setup", "reason": "operator_abort_before_start"},
                    )
                )
                raise OperatorAbort("operator aborted before acquisition start")
            if impedance_ohms is None:
                raise RuntimeError("Guardian fitting ended before an impedance reading arrived")
            if impedance_ohms >= impedance["max_ohms"]:
                raise RuntimeError(
                    f"Guardian impedance {impedance_ohms:.0f} ohm is not below "
                    f"configured {impedance['max_ohms']:.0f} ohm threshold"
                )
            started_at = attempt_clock.start()
            attempt_started = True
            if eeg_config["recording"]["enabled"]:
                eeg_recorder = EEGHDF5Recorder(
                    run_directory / "raw_eeg.h5",
                    sample_rate_hz=eeg_config["signal"]["sample_rate_hz"],
                )
            eeg_source = GuardianEEGFeatureSource(
                guardian=guardian_adapter,
                pipeline=eeg_pipeline,
                recorder=eeg_recorder,
            )
            guardian_started = True
            guardian_adapter.start(recording_seconds=guardian["recording_seconds"])
        else:
            started_at = 0.0
            attempt_started = True
            if eeg_config["recording"]["enabled"]:
                eeg_recorder = EEGHDF5Recorder(
                    run_directory / "raw_eeg.h5",
                    sample_rate_hz=eeg_config["signal"]["sample_rate_hz"],
                )
            eeg_source = RecordedEEGFeatureSource(
                _eeg_samples(eeg_config),
                pipeline=eeg_pipeline,
                recorder=eeg_recorder,
            )

        if resolved["input"]["record_glasses"]:
            glasses_recorder = HDF5Recorder(run_directory / "raw_glasses.h5")
        events.log(
            Event(
                started_at,
                "integration_session_started",
                {
                    "participant_id": binding.policy.participant_id,
                    "session_id": binding.session_id,
                    "session_number": binding.session_number,
                    "attempt_id": binding.attempt_id,
                    "active_condition": binding.condition.value,
                    "policy_sha256": binding.policy_sha256,
                    "maximum_duration_seconds": deadline,
                },
            )
        )
        orchestrator = IntegratedExperimentOrchestrator(
            gaze_pipeline=gaze_pipeline,
            eeg_source=eeg_source,
            experiment=experiment,
            feedback=feedback,
            event_logger=events,
            session_id=binding.session_id,
        )

        def health_check(timestamp: float) -> None:
            if guardian_adapter is not None:
                # PR #20 keeps this as the causal health/finalization hook.
                eeg_source.drain_through(timestamp)
                if guardian_adapter.recording_done and timestamp < deadline - 1e-9:
                    raise RuntimeError("Guardian recording ended before the attempt deadline")

        if input_mode == "video":
            last_timestamp, completed_timestamp = _run_video_input(
                config=resolved,
                orchestrator=orchestrator,
                glasses_recorder=glasses_recorder,
                run_directory=run_directory,
                deadline=deadline,
                health_check=health_check,
            )
        else:
            last_timestamp, completed_timestamp = _run_stream_input(
                samples=_glasses_samples(input_mode, gaze_config, events),
                orchestrator=orchestrator,
                glasses_recorder=glasses_recorder,
                deadline=deadline,
                pacer=_Pacer(eeg_mode == "live" and input_mode == "synthetic"),
                health_check=health_check,
            )
    except BaseException as exc:
        primary_error = exc

    # Guardian cleanup steps are intentionally independent. Failure in one does
    # not prevent the remaining queue from being recorded or the device closing.
    if guardian_adapter is not None:
        cleanup_operations: list[tuple[str, Callable[[], Any]]] = []
        if guardian_impedance_started:
            cleanup_operations.append(("stop_impedance", guardian_adapter.stop_impedance))
        if guardian_started:
            cleanup_operations.append(("stop", guardian_adapter.stop))
        if eeg_source is not None:
            cleanup_operations.append(("drain_remaining", eeg_source.drain_remaining))
        cleanup_operations.append(("close", guardian_adapter.close))
        for step, operation in cleanup_operations:
            try:
                operation()
            except BaseException as exc:
                events.log(
                    Event(
                        max(last_timestamp, attempt_clock.now()),
                        "integration_guardian_cleanup_failed",
                        {"step": step, "reason": type(exc).__name__},
                    )
                )
                if primary_error is None:
                    primary_error = exc

    for name, recorder in (("eeg", eeg_recorder), ("glasses", glasses_recorder)):
        if recorder is None:
            continue
        try:
            recorder.close()
        except BaseException as exc:
            events.log(
                Event(
                    max(last_timestamp, attempt_clock.now()),
                    "integration_recorder_close_failed",
                    {"recorder": name, "reason": type(exc).__name__},
                )
            )
            if primary_error is None:
                primary_error = exc
    if input_mode == "video" and resolved["input"]["video"]["show_window"]:
        close_windows()

    if primary_error is not None:
        events.log(
            Event(
                max(last_timestamp, attempt_clock.now()),
                "integration_session_incomplete",
                {
                    "participant_id": binding.policy.participant_id,
                    "session_id": binding.session_id,
                    "session_number": binding.session_number,
                    "attempt_id": binding.attempt_id,
                    "active_condition": binding.condition.value,
                    "attempt_started": attempt_started,
                    "reason": type(primary_error).__name__,
                    "guardian_lost_sample_count": (
                        int(getattr(guardian_adapter, "lost_sample_count", 0))
                        if guardian_adapter is not None
                        else 0
                    ),
                    "guardian_lost_block_count": (
                        int(getattr(guardian_adapter, "lost_block_count", 0))
                        if guardian_adapter is not None
                        else 0
                    ),
                },
            )
        )
        raise primary_error

    completed = experiment.completed_session(completed_timestamp)
    run_completed_path = run_directory / "completed_session.json"
    run_completed_sha256 = save_completed_session(run_completed_path, completed)
    participant_completed_path = (
        binding.participant_directory
        / f"completed_session_{binding.session_number:03d}.json"
    )
    participant_completed_sha256 = save_completed_session(participant_completed_path, completed)
    if run_completed_sha256 != participant_completed_sha256:
        raise RuntimeError("run and participant completed-session artifacts differ")
    next_policy_path = (
        binding.participant_directory / f"policy_session_{binding.session_number + 1:03d}.json"
    )
    training = train_next_session_policy(
        (*binding.completed_paths, participant_completed_path),
        binding.policy,
        next_policy_path,
        TrainerConfig.from_mapping(_mapping(learning_config, "trainer")),
    )
    events.log(
        Event(
            completed_timestamp,
            "integration_session_completed",
            {
                "participant_id": binding.policy.participant_id,
                "session_id": binding.session_id,
                "session_number": binding.session_number,
                "attempt_id": binding.attempt_id,
                "active_condition": binding.condition.value,
                "policy_sha256": binding.policy_sha256,
                "completed_session_path": str(participant_completed_path),
                "completed_session_sha256": participant_completed_sha256,
                "episode_records": len(completed.records),
                "training_eligible_records": sum(
                    record.training_eligible for record in completed.records
                ),
                "next_policy_path": str(training.policy_path),
                "next_policy_sha256": training.policy_sha256,
                "training_status": training.report["status"],
                "guardian_recording_id": (
                    guardian_adapter.recording_id if guardian_adapter is not None else None
                ),
                "guardian_lost_sample_count": (
                    int(getattr(guardian_adapter, "lost_sample_count", 0))
                    if guardian_adapter is not None
                    else 0
                ),
                "guardian_lost_block_count": (
                    int(getattr(guardian_adapter, "lost_block_count", 0))
                    if guardian_adapter is not None
                    else 0
                ),
            },
        )
    )

    if resolved["analysis"]["enabled"]:
        generate_analysis(run_directory / "events.jsonl", run_directory)
    return run_directory


if __name__ == "__main__":
    print("Use scripts/run_integrated_experiment.py for the configured workflow.")
