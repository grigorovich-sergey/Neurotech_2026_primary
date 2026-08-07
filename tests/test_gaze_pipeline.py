from collections.abc import Sequence

import numpy as np

from foundations.contracts import GazeSample, SceneFrame
from gaze_interaction.association import GazeAssociator
from gaze_interaction.contracts import BoundingBox, Detection, TrackedObject, TrackedScene
from gaze_interaction.dwell import DwellController
from gaze_interaction.episodes import EpisodeTracker
from gaze_interaction.pipeline import GazeInteractionPipeline


class _Detector:
    def detect(self, frame: SceneFrame) -> tuple[Detection, ...]:
        return (Detection(BoundingBox(0.2, 0.2, 0.8, 0.8), "object", 0.9),)


class _Tracker:
    def update(
        self, frame: SceneFrame, detections: Sequence[Detection]
    ) -> TrackedScene:
        detection = detections[0]
        tracked = TrackedObject(
            5,
            detection.box,
            detection.label,
            detection.confidence,
            frame.timestamp,
        )
        return TrackedScene(frame.timestamp, (tracked,))


def _run_sequence() -> list[tuple]:
    pipeline = GazeInteractionPipeline(
        detector=_Detector(),
        tracker=_Tracker(),
        associator=GazeAssociator(max_scene_age_seconds=0.25),
        episode_tracker=EpisodeTracker(gap_grace_seconds=0.15),
        dwell_controller=DwellController(
            baseline_seconds=0.2,
            minimum_seconds=0.1,
            maximum_seconds=0.2,
            maximum_reduction_fraction=0.5,
            max_sample_gap_seconds=0.1,
        ),
    )
    states = []
    stream = [
        SceneFrame(0.0, np.zeros((10, 10, 3), dtype=np.uint8)),
        GazeSample(0.0, 0.5, 0.5, True, 1.0),
        GazeSample(0.05, 0.5, 0.5, True, 1.0),
        GazeSample(0.1, 0.5, 0.5, False, 0.2),
        SceneFrame(0.1, np.zeros((10, 10, 3), dtype=np.uint8)),
        GazeSample(0.15, 0.5, 0.5, True, 1.0),
    ]
    for sample in stream:
        if isinstance(sample, SceneFrame):
            pipeline.process_scene(sample)
            continue
        update = pipeline.process_gaze(sample, intent_score=0.5)
        states.append(
            (
                sample.timestamp,
                update.scene_timestamp,
                update.candidate.track_id if update.candidate else None,
                update.active_episode.episode_id if update.active_episode else None,
                update.dwell_state.accumulated_seconds,
                update.dwell_trigger is not None,
            )
        )
    return states


def test_fixed_ordered_inputs_produce_identical_interaction_state_sequence() -> None:
    assert _run_sequence() == _run_sequence()
