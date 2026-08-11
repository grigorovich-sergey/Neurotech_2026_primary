from collections.abc import Sequence

import numpy as np
import pytest

from foundations.contracts import GazeSample, SceneFrame
from gaze_interaction.association import GazeAssociator
from gaze_interaction.contracts import BoundingBox, Detection, TrackedObject, TrackedScene
from gaze_interaction.dwell import DwellController
from gaze_interaction.episodes import EpisodeEndReason, EpisodeTracker
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


def _pipeline() -> GazeInteractionPipeline:
    return GazeInteractionPipeline(
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


def _run_sequence() -> list[tuple]:
    pipeline = _pipeline()
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


def test_pipeline_gates_then_releases_pending_trigger_on_confirmed_gaze() -> None:
    pipeline = _pipeline()
    pipeline.process_scene(SceneFrame(0.0, np.zeros((10, 10, 3), dtype=np.uint8)))

    for timestamp in (0.0, 0.1):
        update = pipeline.process_gaze(
            GazeSample(timestamp, 0.5, 0.5, True, 1.0),
            intent_score=None,
            trigger_gate_open=False,
        )
        assert update.dwell_trigger is None

    pending = pipeline.process_gaze(
        GazeSample(0.2, 0.5, 0.5, True, 1.0),
        intent_score=None,
        trigger_gate_open=False,
    )
    assert pending.dwell_state.trigger_pending
    assert pending.dwell_trigger is None

    no_match = pipeline.process_gaze(
        GazeSample(0.21, 0.5, 0.5, False, 0.0),
        intent_score=None,
        trigger_gate_open=True,
    )
    assert no_match.dwell_state.trigger_pending
    assert no_match.dwell_trigger is None

    released = pipeline.process_gaze(
        GazeSample(0.22, 0.5, 0.5, True, 1.0),
        intent_score=None,
        trigger_gate_open=True,
    )
    assert not released.dwell_state.trigger_pending
    assert released.dwell_state.triggered
    assert released.dwell_trigger is not None
    assert released.dwell_trigger.timestamp == 0.22


@pytest.mark.parametrize(
    "reason",
    [
        EpisodeEndReason.FEEDBACK_INTERRUPTION,
        EpisodeEndReason.SESSION_DURATION_REACHED,
    ],
)
def test_pipeline_cancellation_clears_candidate_dwell_and_pending_trigger(
    reason: EpisodeEndReason,
) -> None:
    pipeline = _pipeline()
    pipeline.process_scene(SceneFrame(0.0, np.zeros((10, 10, 3), dtype=np.uint8)))

    first_episode_id = None
    for timestamp in (0.0, 0.1, 0.2):
        update = pipeline.process_gaze(
            GazeSample(timestamp, 0.5, 0.5, True, 1.0),
            trigger_gate_open=False,
        )
        assert update.active_episode is not None
        first_episode_id = update.active_episode.episode_id
    assert update.dwell_state.trigger_pending

    cancellation = pipeline.cancel(0.21, reason)

    assert cancellation.reason == reason
    assert cancellation.ended_episode is not None
    assert cancellation.ended_episode.end_reason == reason
    assert cancellation.discarded_pending_trigger
    assert cancellation.dwell_state.episode_id is None
    assert cancellation.dwell_state.accumulated_seconds == 0.0
    assert not cancellation.dwell_state.trigger_pending
    assert not cancellation.dwell_state.triggered

    restarted = pipeline.process_gaze(
        GazeSample(0.22, 0.5, 0.5, True, 1.0),
        trigger_gate_open=True,
    )
    assert restarted.active_episode is not None
    assert restarted.active_episode.episode_id != first_episode_id
    assert restarted.dwell_state.accumulated_seconds == 0.0
    assert restarted.dwell_trigger is None
