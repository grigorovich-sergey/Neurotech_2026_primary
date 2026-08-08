"""Deterministic verification-only vision adapter for virtual/replayed glasses."""

import math
from collections.abc import Sequence

from foundations.contracts import SceneFrame
from gaze_interaction.contracts import (
    BoundingBox,
    Detection,
    TrackedObject,
    TrackedScene,
)


class SyntheticVisionAdapter:
    """Expose one full-frame object in deterministic visible/blank intervals.

    Foundation virtual frames intentionally contain random pixels, so they cannot
    exercise candidate/dwell logic through YOLOE reliably.  This adapter replaces
    only detection/tracking for hardware-free verification.  Association, episode
    tracking, dwell, EEG, learning, and feedback remain the real subsystem paths.
    """

    def __init__(
        self,
        *,
        warmup_seconds: float,
        visible_seconds: float,
        blank_seconds: float,
        label: str = "synthetic-object",
    ) -> None:
        for name, value in (
            ("warmup_seconds", warmup_seconds),
            ("visible_seconds", visible_seconds),
            ("blank_seconds", blank_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (value < 0 if name == "warmup_seconds" else value <= 0)
            ):
                qualifier = "non-negative" if name == "warmup_seconds" else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string")
        self.warmup_seconds = float(warmup_seconds)
        self.visible_seconds = float(visible_seconds)
        self.blank_seconds = float(blank_seconds)
        self.label = label
        self._box = BoundingBox(0.0, 0.0, 1.0, 1.0)

    def _slot(self, timestamp: float) -> int | None:
        if timestamp < self.warmup_seconds:
            return None
        cycle_seconds = self.visible_seconds + self.blank_seconds
        elapsed = timestamp - self.warmup_seconds
        slot = int(math.floor(elapsed / cycle_seconds))
        phase = elapsed - slot * cycle_seconds
        return slot if phase < self.visible_seconds else None

    def detect(self, frame: SceneFrame) -> tuple[Detection, ...]:
        if not isinstance(frame, SceneFrame):
            raise TypeError("frame must be a foundations.contracts.SceneFrame")
        if self._slot(float(frame.timestamp)) is None:
            return ()
        return (Detection(self._box, self.label, 1.0),)

    def update(
        self, frame: SceneFrame, detections: Sequence[Detection]
    ) -> TrackedScene:
        if not isinstance(frame, SceneFrame):
            raise TypeError("frame must be a foundations.contracts.SceneFrame")
        slot = self._slot(float(frame.timestamp))
        if slot is None:
            if detections:
                raise ValueError("synthetic detector/tracker visibility disagrees")
            return TrackedScene(float(frame.timestamp), ())
        if len(detections) != 1:
            raise ValueError("synthetic visible interval requires exactly one detection")
        detection = detections[0]
        tracked = TrackedObject(
            track_id=slot + 1,
            box=detection.box,
            label=detection.label,
            confidence=detection.confidence,
            scene_timestamp=float(frame.timestamp),
        )
        return TrackedScene(float(frame.timestamp), (tracked,))


if __name__ == "__main__":
    import numpy as np

    adapter = SyntheticVisionAdapter(
        warmup_seconds=1.0, visible_seconds=1.0, blank_seconds=1.0
    )
    for timestamp in (0.5, 1.0, 2.0, 3.0):
        frame = SceneFrame(timestamp, np.zeros((2, 2, 3), dtype=np.uint8))
        detections = adapter.detect(frame)
        print(timestamp, adapter.update(frame, detections).objects)
