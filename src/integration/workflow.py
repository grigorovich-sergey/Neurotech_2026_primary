"""Configured end-to-end pre-hardware experiment workflow."""

from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from eeg_pipeline.buffer import EEGBuffer
from eeg_pipeline.contracts import EEGSample
from eeg_pipeline.pipeline import EEGPipeline
from eeg_pipeline.processing import EEGFeatureExtractor, EEGPreprocessor, EEGQualityGate
from eeg_pipeline.recording import EEGHDF5Recorder, EEGHDF5Replay
from eeg_pipeline.synthetic import synthetic_eeg_samples
from experiment_learning.checkpoint import (
    ParticipantState,
    load_participant_checkpoint,
    save_participant_checkpoint,
)
from experiment_learning.models import ModelConfig
from experiment_learning.state_machine import ExperimentController
from foundations.config import load_resolved_config, save_resolved_config
from foundations.contracts import GazeSample, SceneFrame
from foundations.events import Event, JsonlEventLogger
from foundations.recording import HDF5Recorder, HDF5Replay
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
    ScheduledFeedbackPress,
    SyntheticFeedbackDriver,
    TimedEEGFeeder,
    TimedFeedbackDriver,
)
from integration.vision import SyntheticVisionAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAZE_DEFAULT = PROJECT_ROOT / "configs" / "gaze_interaction.yaml"
EEG_DEFAULT = PROJECT_ROOT / "configs" / "eeg_pipeline.yaml"
LEARNING_DEFAULT = PROJECT_ROOT / "configs" / "experiment_learning.yaml"
VIDEO_WINDOW = "NeuroTech integrated experiment"


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


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
        override = _project_path(overrides.get(key), name=f"subsystem_config_overrides.{key}", allow_none=True)
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
    if not isinstance(participant.get("id"), str) or not participant["id"]:
        raise ValueError("participant.id must be a non-empty pseudonymous identifier")
    sequence = participant.get("sequence_index")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("participant.sequence_index must be a non-negative integer")
    if not isinstance(participant.get("resume_checkpoint"), bool):
        raise ValueError("participant.resume_checkpoint must be a bool")
    _project_path(participant.get("checkpoint_path"), name="participant.checkpoint_path")
    session = _mapping(config, "session")
    if not isinstance(session.get("id_prefix"), str) or not session["id_prefix"]:
        raise ValueError("session.id_prefix must be a non-empty string")
    input_config = _mapping(config, "input")
    if input_config.get("mode") not in {"synthetic", "replay", "video"}:
        raise ValueError("input.mode must be 'synthetic', 'replay', or 'video'")
    if not isinstance(input_config.get("record_glasses"), bool):
        raise ValueError("input.record_glasses must be a bool")
    feedback = _mapping(config, "feedback")
    if feedback.get("mode") not in {"synthetic", "replay", "keyboard"}:
        raise ValueError("feedback.mode must be 'synthetic', 'replay', or 'keyboard'")
    analysis = _mapping(config, "analysis")
    if not isinstance(analysis.get("enabled"), bool):
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
    """Reject configuration/path errors before allocating a participant session slot."""

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

    eeg_source = _mapping(eeg_config, "source")
    eeg_mode = eeg_source.get("mode")
    if eeg_mode not in {"synthetic", "replay"}:
        raise ValueError("integrated pre-hardware EEG source.mode must be 'synthetic' or 'replay'")
    if eeg_mode == "replay":
        path = _project_path(eeg_source.get("replay_path"), name="EEG source.replay_path")
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(path)

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
    config: dict[str, Any], *, synthetic_vision: dict[str, Any] | None
) -> GazeInteractionPipeline:
    detector_config = _mapping(config, "detector")
    tracker_config = _mapping(config, "tracker")
    association = _mapping(config, "association")
    episode = _mapping(config, "episode")
    dwell = _mapping(config, "dwell")
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
        if not path.is_file():
            raise FileNotFoundError(path)
        yield from EEGHDF5Replay(path).samples()
        return
    raise ValueError("integrated pre-hardware EEG source.mode must be 'synthetic' or 'replay'")


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
        if not path.is_file():
            raise FileNotFoundError(path)
        yield from HDF5Replay(path).replay(paced=bool(source["replay_paced"]))
        return
    raise ValueError("glasses samples are only available for synthetic/replay input")


def _model_config(config: dict[str, Any]) -> ModelConfig:
    model = _mapping(config, "model")
    return ModelConfig(
        learning_rate=float(model["learning_rate"]),
        l2=float(model["l2"]),
        decision_threshold=float(model["decision_threshold"]),
    )


def _load_participant_state(
    config: dict[str, Any], model_config: ModelConfig
) -> tuple[ParticipantState, Path]:
    participant = _mapping(config, "participant")
    checkpoint = _project_path(participant["checkpoint_path"], name="participant.checkpoint_path")
    assert checkpoint is not None
    if checkpoint.exists():
        if not participant["resume_checkpoint"]:
            raise FileExistsError(
                f"participant checkpoint already exists and resume is disabled: {checkpoint}"
            )
        state = load_participant_checkpoint(
            checkpoint,
            expected_participant_id=participant["id"],
            expected_participant_sequence_index=participant["sequence_index"],
            expected_model_config=model_config,
        )
    else:
        state = ParticipantState.create(
            participant_id=participant["id"],
            participant_sequence_index=participant["sequence_index"],
            model_config=model_config,
        )
    return state, checkpoint


def _replay_feedback(path: Path) -> list[ScheduledFeedbackPress]:
    raw_presses: list[ScheduledFeedbackPress] = []
    result_presses: list[ScheduledFeedbackPress] = []
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
            elif event.get("name") == "experiment_episode_result" and payload.get("feedback_pressed"):
                result_presses.append(
                    ScheduledFeedbackPress(
                        float(payload["feedback_resolution_timestamp"]), int(payload["episode_id"])
                    )
                )
    return raw_presses if raw_presses else result_presses


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
        delay = synthetic["press_delay_seconds"]
        if delay >= feedback_timeout_s:
            raise ValueError("feedback.synthetic.press_delay_seconds must be less than timeout")
        return SyntheticFeedbackDriver(
            press_cycle=synthetic["press_cycle"],
            press_delay_seconds=delay,
            event_logger=event_logger,
            session_id=session_id,
        )
    if mode == "replay":
        replay = _mapping(feedback, "replay")
        path = _project_path(replay.get("events_path"), name="feedback.replay.events_path")
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(path)
        return TimedFeedbackDriver(
            _replay_feedback(path), event_logger=event_logger, session_id=session_id
        )
    keyboard = _mapping(feedback, "keyboard")
    return KeyboardFeedbackDriver(
        key_code=keyboard["key_code"], event_logger=event_logger, session_id=session_id
    )


class _Pacer:
    def __init__(self, paced: bool) -> None:
        if not isinstance(paced, bool):
            raise ValueError("input.video.paced must be a bool")
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
) -> float:
    last_timestamp = 0.0
    for sample in samples:
        last_timestamp = max(last_timestamp, float(sample.timestamp))
        if glasses_recorder is not None:
            glasses_recorder.record(sample)
        if isinstance(sample, SceneFrame):
            orchestrator.eeg_feeder.feed_through(float(sample.timestamp))
            orchestrator.process_scene(sample)
        else:
            orchestrator.process_gaze(sample)
    orchestrator.eeg_feeder.feed_through(last_timestamp)
    orchestrator.finish(last_timestamp)
    return last_timestamp


def _run_video_input(
    *,
    config: dict[str, Any],
    orchestrator: IntegratedExperimentOrchestrator,
    glasses_recorder: HDF5Recorder | None,
    run_directory: Path,
) -> float:
    video = _mapping(_mapping(config, "input"), "video")
    path = _project_path(video.get("path"), name="input.video.path")
    assert path is not None
    source = VideoSceneSource(path)
    gaze_mode = video.get("gaze_mode")
    if gaze_mode not in {"mouse", "file"}:
        raise ValueError("input.video.gaze_mode must be 'mouse' or 'file'")
    if gaze_mode == "mouse" and video.get("show_window") is not True:
        raise ValueError("input.video.show_window must be true for mouse gaze")
    if config["feedback"]["mode"] == "keyboard" and video.get("show_window") is not True:
        raise ValueError("input.video.show_window must be true for keyboard feedback")
    pacer = _Pacer(video["paced"])
    image_index = 0
    last_timestamp = 0.0

    if gaze_mode == "mouse":
        mouse = MouseGazeSource(window_name=VIDEO_WINDOW)
        for frame in source.frames():
            timestamp = float(frame.timestamp)
            last_timestamp = max(last_timestamp, timestamp)
            pacer.wait(timestamp)
            orchestrator.eeg_feeder.feed_through(timestamp)
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
            if scene_first:
                frame = next_scene
                timestamp = float(frame.timestamp)
                last_timestamp = max(last_timestamp, timestamp)
                pacer.wait(timestamp)
                orchestrator.eeg_feeder.feed_through(timestamp)
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
                continue
            assert next_gaze is not None
            if latest_frame is None:
                raise ValueError("video gaze begins before the first video frame")
            gaze = next_gaze
            timestamp = float(gaze.timestamp)
            if next_scene is None:
                video_end = float(latest_frame.timestamp) + source.frame_period_seconds
                if timestamp >= video_end + 1e-9:
                    raise ValueError(
                        f"gaze timestamp {timestamp:.6f}s is beyond the video timeline "
                        f"ending at {video_end:.6f}s"
                    )
            last_timestamp = max(last_timestamp, timestamp)
            pacer.wait(timestamp)
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

    orchestrator.eeg_feeder.feed_through(last_timestamp)
    orchestrator.finish(last_timestamp)
    return last_timestamp


def run_integrated_experiment(config: dict[str, Any]) -> Path:
    """Run one counterbalanced session and persist reproducible scientific outputs."""

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

    model_config = _model_config(learning_config)
    state, checkpoint_path = _load_participant_state(resolved, model_config)
    session_index, condition = state.schedule.allocate_next()
    save_participant_checkpoint(checkpoint_path, state)
    session_id = f"{resolved['session']['id_prefix']}-{session_index + 1:03d}"
    counts_before = {
        "G": state.learners.g_model.training_count,
        "E": state.learners.e_model.training_count,
    }
    events.log(
        Event(
            0.0,
            "integration_session_started",
            {
                "participant_id": state.participant_id,
                "session_id": session_id,
                "session_index": session_index,
                "active_condition": condition.value,
                "training_counts_before": counts_before,
            },
        )
    )

    timing = _mapping(learning_config, "timing")
    experiment = ExperimentController(
        participant_state=state,
        session_id=session_id,
        active_condition=condition,
        minimum_prediction_elapsed_s=timing["minimum_prediction_elapsed_s"],
        eeg_window_s=timing["eeg_window_s"],
        feedback_timeout_s=timing["feedback_timeout_s"],
        checkpoint_path=checkpoint_path,
        event_logger=events,
    )
    input_mode = resolved["input"]["mode"]
    gaze_pipeline = _build_gaze_pipeline(
        gaze_config,
        synthetic_vision=(resolved["synthetic_vision"] if input_mode in {"synthetic", "replay"} else None),
    )
    eeg_pipeline = _build_eeg_pipeline(eeg_config)
    eeg_recording = _mapping(eeg_config, "recording")
    eeg_recorder = (
        EEGHDF5Recorder(
            run_directory / "raw_eeg.h5",
            sample_rate_hz=eeg_config["signal"]["sample_rate_hz"],
        )
        if eeg_recording["enabled"]
        else None
    )
    glasses_recorder = (
        HDF5Recorder(run_directory / "raw_glasses.h5")
        if resolved["input"]["record_glasses"]
        else None
    )
    eeg_feeder = TimedEEGFeeder(
        _eeg_samples(eeg_config), pipeline=eeg_pipeline, recorder=eeg_recorder
    )
    feedback = _build_feedback_driver(
        resolved,
        event_logger=events,
        session_id=session_id,
        feedback_timeout_s=float(timing["feedback_timeout_s"]),
    )
    orchestrator = IntegratedExperimentOrchestrator(
        gaze_pipeline=gaze_pipeline,
        eeg_feeder=eeg_feeder,
        experiment=experiment,
        feedback=feedback,
        event_logger=events,
        session_id=session_id,
    )

    last_timestamp = 0.0
    completed = False
    try:
        if input_mode == "video":
            last_timestamp = _run_video_input(
                config=resolved,
                orchestrator=orchestrator,
                glasses_recorder=glasses_recorder,
                run_directory=run_directory,
            )
        else:
            last_timestamp = _run_stream_input(
                samples=_glasses_samples(input_mode, gaze_config, events),
                orchestrator=orchestrator,
                glasses_recorder=glasses_recorder,
            )

        results_count = len(experiment.results)
        counts_after = {
            "G": state.learners.g_model.training_count,
            "E": state.learners.e_model.training_count,
        }
        if counts_after["G"] - counts_before["G"] != results_count:
            raise RuntimeError("G training-count delta differs from persisted session results")
        if counts_after["E"] - counts_before["E"] != results_count:
            raise RuntimeError("E training-count delta differs from persisted session results")
        if counts_after["G"] != counts_after["E"]:
            raise RuntimeError("paired G/E participant training counts diverged")
        experiment.save_session_checkpoint()
        events.log(
            Event(
                last_timestamp + float(timing["feedback_timeout_s"]),
                "integration_session_completed",
                {
                    "participant_id": state.participant_id,
                    "session_id": session_id,
                    "session_index": session_index,
                    "active_condition": condition.value,
                    "episode_results": results_count,
                    "training_counts_after": counts_after,
                },
            )
        )
        completed = True
    except BaseException as exc:
        incomplete_timestamp = max(last_timestamp, orchestrator.last_gaze_timestamp)
        events.log(
            Event(
                incomplete_timestamp,
                "integration_session_incomplete",
                {
                    "participant_id": state.participant_id,
                    "session_id": session_id,
                    "session_index": session_index,
                    "active_condition": condition.value,
                    "reason": type(exc).__name__,
                },
            )
        )
        raise
    finally:
        if eeg_recorder is not None:
            eeg_recorder.close()
        if glasses_recorder is not None:
            glasses_recorder.close()
        if input_mode == "video" and resolved["input"]["video"]["show_window"]:
            close_windows()

    if completed and resolved["analysis"]["enabled"]:
        generate_analysis(run_directory / "events.jsonl", run_directory)
    return run_directory


if __name__ == "__main__":
    print("Use scripts/run_integrated_experiment.py for the configured workflow.")
