"""Run prerecorded video with mouse or CSV gaze through gaze_interaction."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from foundations.config import load_resolved_config, save_resolved_config
from foundations.contracts import GazeSample, SceneFrame
from gaze_interaction.association import GazeAssociator
from gaze_interaction.detector import YOLOEDetector
from gaze_interaction.dwell import DwellController
from gaze_interaction.episodes import CandidateEpisode, EpisodeTracker
from gaze_interaction.pipeline import GazeInteractionPipeline, InteractionUpdate
from gaze_interaction.tracker import ByteTrackAdapter
from gaze_interaction.visualization import (
    close_windows,
    render_diagnostic,
    save_rgb_image,
    show_rgb_image,
)
from test_harness.gaze import GazeCsvSource, GazeCsvWriter, MouseGazeSource
from test_harness.video import VideoSceneSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "test_harness.yaml"
GAZE_INTERACTION_CONFIG = PROJECT_ROOT / "configs" / "gaze_interaction.yaml"
WINDOW_NAME = "NeuroTech video + gaze harness"


class _PlaybackPacer:
    def __init__(self, *, paced: bool) -> None:
        if not isinstance(paced, bool):
            raise TypeError("playback.paced must be a bool")
        self._paced = paced
        self._started_at: float | None = None

    def wait_until(self, timestamp: float) -> None:
        if not self._paced:
            return
        if self._started_at is None:
            self._started_at = time.monotonic() - timestamp
        delay = self._started_at + timestamp - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)


def _new_run_directory(output_root: str) -> Path:
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output_root must be a non-empty path string")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(output_root) / "test_harness" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


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
        episode_tracker=EpisodeTracker(gap_grace_seconds=episode["gap_grace_seconds"]),
        dwell_controller=DwellController(
            baseline_seconds=dwell["baseline_seconds"],
            minimum_seconds=dwell["minimum_seconds"],
            maximum_seconds=dwell["maximum_seconds"],
            maximum_reduction_fraction=dwell["maximum_reduction_fraction"],
            max_sample_gap_seconds=dwell["max_sample_gap_seconds"],
        ),
    )


def _validate_harness_config(config: dict[str, Any]) -> None:
    video_path = config["video_path"]
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("video_path must be set to a prerecorded video file")
    mode = config["gaze"]["mode"]
    if mode not in {"mouse", "file"}:
        raise ValueError("gaze.mode must be 'mouse' or 'file'")
    input_path = config["gaze"]["csv_input_path"]
    output_path = config["gaze"]["mouse_csv_output_path"]
    if mode == "file" and (not isinstance(input_path, str) or not input_path):
        raise ValueError("gaze.csv_input_path is required in file mode")
    if mode == "mouse" and input_path is not None:
        raise ValueError("gaze.csv_input_path must be null in mouse mode")
    if mode == "file" and output_path is not None:
        raise ValueError("gaze.mouse_csv_output_path is only valid in mouse mode")
    if output_path is not None and (not isinstance(output_path, str) or not output_path):
        raise ValueError("gaze.mouse_csv_output_path must be a path string or null")
    if mode == "mouse" and config["visualization"]["show_window"] is not True:
        raise ValueError("visualization.show_window must be true in mouse mode")
    for key in ("show_window", "save_frames"):
        if not isinstance(config["visualization"][key], bool):
            raise TypeError(f"visualization.{key} must be a bool")


def _report_interaction(update: InteractionUpdate) -> None:
    if update.ended_episode is not None:
        episode = update.ended_episode
        reason = episode.end_reason.value if episode.end_reason is not None else "unknown"
        print(f"episode {episode.episode_id} ended: track={episode.track_id} reason={reason}")
    if update.dwell_trigger is not None:
        trigger = update.dwell_trigger
        print(
            f"dwell trigger: episode={trigger.episode_id} track={trigger.track_id} "
            f"timestamp={trigger.timestamp:.3f}s"
        )


def _render(
    *,
    frame: SceneFrame,
    tracks: tuple,
    gaze: GazeSample | None,
    update: InteractionUpdate | None,
    visualization: dict[str, Any],
    run_directory: Path,
    image_index: int,
) -> None:
    image = render_diagnostic(
        frame,
        tracks=tracks,
        gaze=gaze,
        candidate=update.candidate if update is not None else None,
        dwell_state=update.dwell_state if update is not None else None,
        intent_score=None,
    )
    if visualization["save_frames"]:
        save_rgb_image(
            run_directory / "diagnostics" / f"frame_{image_index:06d}.png", image
        )
    if visualization["show_window"]:
        show_rgb_image(image, window_name=WINDOW_NAME)


def _run_mouse_mode(
    *,
    source: VideoSceneSource,
    pipeline: GazeInteractionPipeline,
    config: dict[str, Any],
    run_directory: Path,
) -> tuple[int, float]:
    pacer = _PlaybackPacer(paced=config["playback"]["paced"])
    mouse = MouseGazeSource(window_name=WINDOW_NAME)
    output_path = config["gaze"]["mouse_csv_output_path"]
    writer_context = GazeCsvWriter(output_path) if output_path is not None else nullcontext()
    processed = 0
    last_timestamp = 0.0
    with writer_context as writer:
        for frame in source.frames():
            last_timestamp = float(frame.timestamp)
            pacer.wait_until(last_timestamp)
            scene_update = pipeline.process_scene(frame)
            mouse.set_scene_shape(*frame.image.shape[:2])
            _render(
                frame=frame,
                tracks=scene_update.tracks,
                gaze=None,
                update=None,
                visualization=config["visualization"],
                run_directory=run_directory,
                image_index=processed * 2,
            )
            if not mouse.window_is_open():
                break
            gaze = mouse.sample(last_timestamp)
            if writer is not None:
                writer.write(gaze)
            interaction = pipeline.process_gaze(gaze, intent_score=None)
            _report_interaction(interaction)
            _render(
                frame=frame,
                tracks=scene_update.tracks,
                gaze=gaze,
                update=interaction,
                visualization=config["visualization"],
                run_directory=run_directory,
                image_index=processed * 2 + 1,
            )
            processed += 1
    return processed, last_timestamp


def _run_file_mode(
    *,
    source: VideoSceneSource,
    gaze_source: GazeCsvSource,
    pipeline: GazeInteractionPipeline,
    config: dict[str, Any],
    interaction_config: dict[str, Any],
    run_directory: Path,
) -> tuple[int, float]:
    pacer = _PlaybackPacer(paced=config["playback"]["paced"])
    scene_iterator = iter(source.frames())
    gaze_iterator = iter(gaze_source.samples())
    next_scene = next(scene_iterator, None)
    next_gaze = next(gaze_iterator, None)
    frames: dict[float, SceneFrame] = {}
    tracks_by_timestamp: dict[float, tuple] = {}
    processed_gaze = 0
    image_index = 0
    last_scene_timestamp: float | None = None
    last_timestamp = 0.0

    while next_scene is not None or next_gaze is not None:
        scene_first = next_scene is not None and (
            next_gaze is None or next_scene.timestamp <= next_gaze.timestamp
        )
        if scene_first:
            frame = next_scene
            pacer.wait_until(float(frame.timestamp))
            scene_update = pipeline.process_scene(frame)
            timestamp = float(frame.timestamp)
            frames[timestamp] = frame
            tracks_by_timestamp[timestamp] = scene_update.tracks
            last_scene_timestamp = timestamp
            last_timestamp = max(last_timestamp, timestamp)
            _render(
                frame=frame,
                tracks=scene_update.tracks,
                gaze=None,
                update=None,
                visualization=config["visualization"],
                run_directory=run_directory,
                image_index=image_index,
            )
            image_index += 1
            next_scene = next(scene_iterator, None)
            continue

        assert next_gaze is not None
        gaze = next_gaze
        if last_scene_timestamp is None:
            raise ValueError("gaze CSV begins before the first video frame")
        if next_scene is None:
            video_end = last_scene_timestamp + source.frame_period_seconds
            if float(gaze.timestamp) >= video_end + 1e-9:
                raise ValueError(
                    f"gaze timestamp {gaze.timestamp:.6f}s is beyond the video timeline "
                    f"ending at {video_end:.6f}s"
                )
        pacer.wait_until(float(gaze.timestamp))
        interaction = pipeline.process_gaze(gaze, intent_score=None)
        _report_interaction(interaction)
        processed_gaze += 1
        last_timestamp = max(last_timestamp, float(gaze.timestamp))
        if interaction.scene_timestamp is not None:
            frame = frames.get(interaction.scene_timestamp)
            tracks = tracks_by_timestamp.get(interaction.scene_timestamp)
            if frame is not None and tracks is not None:
                _render(
                    frame=frame,
                    tracks=tracks,
                    gaze=gaze,
                    update=interaction,
                    visualization=config["visualization"],
                    run_directory=run_directory,
                    image_index=image_index,
                )
                image_index += 1

        cutoff = float(gaze.timestamp) - interaction_config["association"][
            "max_scene_age_seconds"
        ]
        for timestamp in [value for value in frames if value < cutoff]:
            frames.pop(timestamp, None)
            tracks_by_timestamp.pop(timestamp, None)
        next_gaze = next(gaze_iterator, None)

    return processed_gaze, last_timestamp


def _finish_pipeline(
    pipeline: GazeInteractionPipeline, timestamp: float
) -> CandidateEpisode | None:
    ended = pipeline.finish(timestamp)
    if ended is not None:
        reason = ended.end_reason.value if ended.end_reason is not None else "unknown"
        print(f"episode {ended.episode_id} ended: track={ended.track_id} reason={reason}")
    return ended


def run_test_harness(config: dict[str, Any]) -> Path:
    _validate_harness_config(config)
    interaction_config = load_resolved_config(
        GAZE_INTERACTION_CONFIG, config["gaze_interaction_config"]
    )
    run_directory = _new_run_directory(config["output_root"])
    save_resolved_config(config, run_directory / "resolved_config.json")
    save_resolved_config(
        interaction_config, run_directory / "resolved_gaze_interaction_config.json"
    )
    source = VideoSceneSource(config["video_path"])
    pipeline = _build_pipeline(interaction_config)

    try:
        if config["gaze"]["mode"] == "mouse":
            sample_count, last_timestamp = _run_mouse_mode(
                source=source,
                pipeline=pipeline,
                config=config,
                run_directory=run_directory,
            )
        else:
            gaze_source = GazeCsvSource(config["gaze"]["csv_input_path"])
            sample_count, last_timestamp = _run_file_mode(
                source=source,
                gaze_source=gaze_source,
                pipeline=pipeline,
                config=config,
                interaction_config=interaction_config,
                run_directory=run_directory,
            )
        _finish_pipeline(pipeline, last_timestamp)
    finally:
        if config["visualization"]["show_window"]:
            close_windows()

    print(f"processed {sample_count} gaze samples")
    return run_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="partial YAML configuration overriding the test-harness default",
    )
    args = parser.parse_args()
    try:
        config = load_resolved_config(DEFAULT_CONFIG, args.config)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(run_test_harness(config))


if __name__ == "__main__":
    main()
