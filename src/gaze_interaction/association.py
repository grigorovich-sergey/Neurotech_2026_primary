"""Deterministic gaze-to-tracked-object association."""

from bisect import bisect_right
from dataclasses import dataclass
import math

from foundations.contracts import GazeSample
from gaze_interaction.contracts import BoundingBox, TrackedObject, TrackedScene


@dataclass(frozen=True)
class GazeAssociation:
    scene: TrackedScene | None
    candidate: TrackedObject | None


class GazeAssociator:
    """Associate gaze with the newest non-future tracked scene snapshot."""

    def __init__(
        self, *, box_margin_normalized: float = 0.0, max_scene_age_seconds: float
    ) -> None:
        if (
            not math.isfinite(box_margin_normalized)
            or not 0.0 <= box_margin_normalized <= 1.0
        ):
            raise ValueError("box_margin_normalized must be within [0, 1]")
        if not math.isfinite(max_scene_age_seconds) or max_scene_age_seconds < 0.0:
            raise ValueError("max_scene_age_seconds must be finite and non-negative")
        self.box_margin_normalized = float(box_margin_normalized)
        self.max_scene_age_seconds = float(max_scene_age_seconds)
        self._timestamps: list[float] = []
        self._scenes: list[TrackedScene] = []

    def add_scene(self, scene: TrackedScene) -> None:
        if not isinstance(scene, TrackedScene):
            raise TypeError("scene must be a TrackedScene")
        if self._timestamps and scene.timestamp < self._timestamps[-1]:
            raise ValueError("tracked scene timestamps must be non-decreasing")
        self._timestamps.append(scene.timestamp)
        self._scenes.append(scene)

    def associate(self, gaze: GazeSample) -> GazeAssociation:
        if not isinstance(gaze, GazeSample):
            raise TypeError("gaze must be a foundations.contracts.GazeSample")
        index = bisect_right(self._timestamps, float(gaze.timestamp)) - 1
        if index < 0:
            return GazeAssociation(None, None)
        scene = self._scenes[index]
        if gaze.timestamp - scene.timestamp > self.max_scene_age_seconds:
            return GazeAssociation(None, None)
        if not gaze.valid or gaze.x_normalized is None or gaze.y_normalized is None:
            return GazeAssociation(scene, None)

        containing = [
            tracked_object
            for tracked_object in scene.objects
            if tracked_object.box.contains(
                gaze.x_normalized,
                gaze.y_normalized,
                margin=self.box_margin_normalized,
            )
        ]
        if not containing:
            return GazeAssociation(scene, None)
        candidate = min(
            containing,
            key=lambda item: (item.box.area, -item.confidence, item.track_id),
        )
        return GazeAssociation(scene, candidate)


if __name__ == "__main__":
    tracked = TrackedObject(1, BoundingBox(0.2, 0.2, 0.8, 0.8), "object", 0.9, 0.0)
    associator = GazeAssociator(max_scene_age_seconds=0.25)
    associator.add_scene(TrackedScene(0.0, (tracked,)))
    print(associator.associate(GazeSample(0.1, 0.5, 0.5, True, 1.0)))
