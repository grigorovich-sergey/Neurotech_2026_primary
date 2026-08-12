"""Run detector -> tracker -> gaze association -> episode -> dwell diagnostics."""

import argparse
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from foundations.config import load_resolved_config, save_resolved_config
from foundations.contracts import GazeSample, SceneFrame
from foundations.events import Event, JsonlEventLogger
from foundations.recording import HDF5Replay
from foundations.virtual_glasses import VirtualGlasses
from gaze_interaction.association import GazeAssociator
from gaze_interaction.detector import YOLOEDetector
from gaze_interaction.dwell import DwellController
from gaze_interaction.episodes import CandidateEpisode, EpisodeTracker
from gaze_interaction.pipeline import GazeInteractionPipeline
from gaze_interaction.tracker import ByteTrackAdapter
from gaze_interaction.visualization import close_windows, render_diagnostic, save_rgb_image, show_rgb_image


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gaze_interaction.yaml"
Sample = SceneFrame | GazeSample


def _new_run_directory(output_root: str) -> Path:
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root must be a non-empty path string")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(output_root) / "gaze_interaction" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _source_samples(config: dict[str, Any], events: JsonlEventLogger) -> Iterator[Sample]:
    source = config["source"]
    mode = source["mode"]
    if mode == "virtual":
        virtual = source["virtual"]
        generator = VirtualGlasses(
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
        yield from generator.samples(
            on_dropout=lambda stream, timestamp: events.log(
                Event(timestamp, "sensor_dropout", {"stream": stream})
            )
        )
        return
    if mode == "replay":
        recording_path = source["recording_path"]
        if not isinstance(recording_path, str) or not recording_path:
            raise ValueError("source.recording_path is required in replay mode")
        if not Path(recording_path).is_file():
            raise FileNotFoundError(recording_path)
        yield from HDF5Replay(recording_path).replay(paced=source["replay_paced"])
        return
    raise ValueError("source.mode must be 'virtual' or 'replay'")


def _build_pipeline(config: dict[str, Any]) -> GazeInteractionPipeline:
    detector = config["detector"]
    tracker = config["tracker"]
    association = config["association"]
    episode = config["episode"]
    dwell = config["dwell"]
    return GazeInteractionPipeline(
        detector=YOLOEDetector(
            detector["model"],
            confidence_threshold=detector["confidence_threshold"],
            image_size=detector["image_size"],
            device=detector["device"],
            category_filter=detector["category_filter"],
        ),
        tracker=ByteTrackAdapter(
            activation_threshold=tracker["activation_threshold"],
            lost_track_buffer=tracker["lost_track_buffer"],
            matching_threshold=tracker["matching_threshold"],
            frame_rate=tracker["frame_rate"],
        ),
        associator=GazeAssociator(
            box_margin_normalized=association["box_margin_normalized"],
            max_scene_age_seconds=association["max_scene_age_seconds"],
        ),
        episode_tracker=EpisodeTracker(
            gap_grace_seconds=episode["gap_grace_seconds"]
        ),
        dwell_controller=DwellController(
            baseline_seconds=dwell["baseline_seconds"],
            minimum_seconds=dwell["minimum_seconds"],
            maximum_seconds=dwell["maximum_seconds"],
            maximum_reduction_fraction=dwell["maximum_reduction_fraction"],
            max_sample_gap_seconds=dwell["max_sample_gap_seconds"],
        ),
    )


def _episode_payload(episode: CandidateEpisode) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "track_id": episode.track_id,
        "label": episode.label,
        "start_timestamp": episode.start_timestamp,
        "last_match_timestamp": episode.last_match_timestamp,
        "end_timestamp": episode.end_timestamp,
        "end_reason": episode.end_reason.value if episode.end_reason is not None else None,
    }


def run_gaze_interaction(config: dict[str, Any]) -> Path:
    run_directory = _new_run_directory(config["output_root"])
    save_resolved_config(config, run_directory / "resolved_config.json")
    events = JsonlEventLogger(run_directory / "events.jsonl")
    events.log(Event(0.0, "gaze_interaction_started"))
    pipeline = _build_pipeline(config)
    intent_score = config["demo"]["intent_score"]
    visualization = config["visualization"]
    if intent_score is not None:
        # Validate immediately rather than waiting for the first gaze sample.
        pipeline.dwell_controller.required_seconds(intent_score)
    if not isinstance(visualization["save_frames"], bool) or not isinstance(
        visualization["show_window"], bool
    ):
        raise ValueError("visualization switches must be bool values")

    frames: dict[float, SceneFrame] = {}
    tracks_by_timestamp: dict[float, tuple] = {}
    counts = {"scene": 0, "gaze": 0, "triggers": 0, "episodes_ended": 0}
    last_timestamp = 0.0
    try:
        for sample in _source_samples(config, events):
            last_timestamp = max(last_timestamp, float(sample.timestamp))
            if isinstance(sample, SceneFrame):
                update = pipeline.process_scene(sample)
                frames[float(sample.timestamp)] = sample
                tracks_by_timestamp[float(sample.timestamp)] = update.tracks
                counts["scene"] += 1
                events.log(
                    Event(
                        sample.timestamp,
                        "scene_processed",
                        {"detections": len(update.detections), "tracks": len(update.tracks)},
                    )
                )
                continue

            update = pipeline.process_gaze(sample, intent_score=intent_score)
            counts["gaze"] += 1
            if update.ended_episode is not None:
                counts["episodes_ended"] += 1
                events.log(
                    Event(
                        (
                            update.ended_episode.end_timestamp
                            if update.ended_episode.end_timestamp is not None
                            else sample.timestamp
                        ),
                        "candidate_episode_ended",
                        _episode_payload(update.ended_episode),
                    )
                )
            if update.dwell_trigger is not None:
                counts["triggers"] += 1
                events.log(
                    Event(
                        update.dwell_trigger.timestamp,
                        "dwell_trigger",
                        {
                            "episode_id": update.dwell_trigger.episode_id,
                            "track_id": update.dwell_trigger.track_id,
                            "required_seconds": update.dwell_trigger.required_seconds,
                        },
                    )
                )
            events.log(
                Event(
                    sample.timestamp,
                    "gaze_processed",
                    {
                        "valid": sample.valid,
                        "scene_timestamp": update.scene_timestamp,
                        "candidate_track_id": (
                            update.candidate.track_id if update.candidate else None
                        ),
                        "episode_id": (
                            update.active_episode.episode_id if update.active_episode else None
                        ),
                        "dwell_accumulated_seconds": update.dwell_state.accumulated_seconds,
                        "dwell_required_seconds": update.dwell_state.required_seconds,
                        "dwell_triggered": update.dwell_state.triggered,
                    },
                )
            )

            if update.scene_timestamp is not None:
                frame = frames.get(update.scene_timestamp)
                scene_tracks = tracks_by_timestamp.get(update.scene_timestamp)
                if frame is not None and scene_tracks is not None:
                    image = render_diagnostic(
                        frame,
                        tracks=scene_tracks,
                        gaze=sample,
                        candidate=update.candidate,
                        dwell_state=update.dwell_state,
                        intent_score=intent_score,
                    )
                    if visualization["save_frames"]:
                        save_rgb_image(
                            run_directory
                            / "diagnostics"
                            / f"gaze_{counts['gaze']:06d}.png",
                            image,
                        )
                    if visualization["show_window"]:
                        show_rgb_image(image)

            cutoff = float(sample.timestamp) - config["association"]["max_scene_age_seconds"]
            for timestamp in [value for value in frames if value < cutoff]:
                frames.pop(timestamp, None)
                tracks_by_timestamp.pop(timestamp, None)
    finally:
        if visualization["show_window"]:
            close_windows()

    ended = pipeline.finish(last_timestamp)
    if ended is not None:
        counts["episodes_ended"] += 1
        events.log(
            Event(
                ended.end_timestamp if ended.end_timestamp is not None else last_timestamp,
                "candidate_episode_ended",
                _episode_payload(ended),
            )
        )
    events.log(Event(last_timestamp, "gaze_interaction_finished", counts))
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="partial YAML configuration overriding the project default",
    )
    args = parser.parse_args()
    try:
        config = load_resolved_config(DEFAULT_CONFIG, args.config)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(run_gaze_interaction(config))


if __name__ == "__main__":
    main()
