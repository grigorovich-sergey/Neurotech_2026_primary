"""Core scene/gaze orchestration independent of CLI and physical glasses."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from foundations.contracts import GazeSample, SceneFrame
from gaze_interaction.association import GazeAssociator
from gaze_interaction.contracts import Detection, TrackedObject, TrackedScene
from gaze_interaction.dwell import DwellController, DwellState, DwellTrigger
from gaze_interaction.episodes import CandidateEpisode, EpisodeEndReason, EpisodeTracker


class Detector(Protocol):
    def detect(self, frame: SceneFrame) -> Sequence[Detection]: ...


class Tracker(Protocol):
    def update(
        self, frame: SceneFrame, detections: Sequence[Detection]
    ) -> TrackedScene: ...


@dataclass(frozen=True)
class SceneUpdate:
    timestamp: float
    detections: tuple[Detection, ...]
    tracks: tuple[TrackedObject, ...]


@dataclass(frozen=True)
class InteractionUpdate:
    gaze: GazeSample
    scene_timestamp: float | None
    candidate: TrackedObject | None
    active_episode: CandidateEpisode | None
    ended_episode: CandidateEpisode | None
    dwell_state: DwellState
    dwell_trigger: DwellTrigger | None


@dataclass(frozen=True)
class InteractionCancellation:
    timestamp: float
    reason: EpisodeEndReason
    ended_episode: CandidateEpisode | None
    dwell_state: DwellState
    discarded_pending_trigger: bool


class GazeInteractionPipeline:
    def __init__(
        self,
        *,
        detector: Detector,
        tracker: Tracker,
        associator: GazeAssociator,
        episode_tracker: EpisodeTracker,
        dwell_controller: DwellController,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.associator = associator
        self.episode_tracker = episode_tracker
        self.dwell_controller = dwell_controller

    def process_scene(self, frame: SceneFrame) -> SceneUpdate:
        detections = tuple(self.detector.detect(frame))
        tracked_scene = self.tracker.update(frame, detections)
        if tracked_scene.timestamp != frame.timestamp:
            raise ValueError("tracker returned a scene timestamp that differs from input")
        self.associator.add_scene(tracked_scene)
        return SceneUpdate(frame.timestamp, detections, tracked_scene.objects)

    def process_gaze(
        self,
        gaze: GazeSample,
        *,
        intent_score: float | None = None,
        trigger_gate_open: bool = True,
    ) -> InteractionUpdate:
        association = self.associator.associate(gaze)
        episode_update = self.episode_tracker.update(
            association.candidate, float(gaze.timestamp)
        )
        active = episode_update.active_episode
        matched = (
            association.candidate is not None
            and active is not None
            and association.candidate.track_id == active.track_id
        )
        dwell_state, dwell_trigger = self.dwell_controller.advance(
            active,
            matched=matched,
            timestamp=float(gaze.timestamp),
            intent_score=intent_score,
            trigger_gate_open=trigger_gate_open,
        )
        return InteractionUpdate(
            gaze=gaze,
            scene_timestamp=(association.scene.timestamp if association.scene else None),
            candidate=association.candidate,
            active_episode=active,
            ended_episode=episode_update.ended_episode,
            dwell_state=dwell_state,
            dwell_trigger=dwell_trigger,
        )

    def cancel(
        self, timestamp: float, reason: EpisodeEndReason
    ) -> InteractionCancellation:
        """Cancel current interaction state without emitting a dwell trigger."""

        ended_episode = self.episode_tracker.cancel(timestamp, reason)
        dwell_state, discarded_pending_trigger = self.dwell_controller._cancel(
            timestamp
        )
        return InteractionCancellation(
            timestamp=float(timestamp),
            reason=reason,
            ended_episode=ended_episode,
            dwell_state=dwell_state,
            discarded_pending_trigger=discarded_pending_trigger,
        )

    def finish(self, timestamp: float) -> CandidateEpisode | None:
        ended = self.episode_tracker.finish(timestamp)
        self.dwell_controller.advance(
            None, matched=False, timestamp=timestamp, intent_score=None
        )
        return ended


if __name__ == "__main__":
    from gaze_interaction.contracts import BoundingBox

    class _SmokeDetector:
        def detect(self, frame: SceneFrame) -> tuple[Detection, ...]:
            return (Detection(BoundingBox(0.2, 0.2, 0.8, 0.8), "object", 0.9),)

    class _SmokeTracker:
        def update(
            self, frame: SceneFrame, detections: Sequence[Detection]
        ) -> TrackedScene:
            detection = detections[0]
            tracked = TrackedObject(
                1, detection.box, detection.label, detection.confidence, frame.timestamp
            )
            return TrackedScene(frame.timestamp, (tracked,))

    pipeline = GazeInteractionPipeline(
        detector=_SmokeDetector(),
        tracker=_SmokeTracker(),
        associator=GazeAssociator(max_scene_age_seconds=0.25),
        episode_tracker=EpisodeTracker(gap_grace_seconds=0.15),
        dwell_controller=DwellController(
            baseline_seconds=1.0,
            minimum_seconds=0.35,
            maximum_seconds=1.0,
            maximum_reduction_fraction=0.5,
            max_sample_gap_seconds=0.1,
        ),
    )
    pipeline.process_scene(SceneFrame(0.0, np.zeros((4, 4, 3), dtype=np.uint8)))
    print(pipeline.process_gaze(GazeSample(0.0, 0.5, 0.5, True, 1.0)))
