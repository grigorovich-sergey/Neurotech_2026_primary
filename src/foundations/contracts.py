"""Canonical scene and gaze data contracts."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


def _validate_timestamp(timestamp: float) -> None:
    if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
        raise TypeError("timestamp must be a real number")
    if not math.isfinite(float(timestamp)) or timestamp < 0:
        raise ValueError("timestamp must be finite and non-negative")


def _validate_unit_value(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True)
class SceneFrame:
    """One RGB scene-camera frame at run-relative capture time."""

    timestamp: float
    image: np.ndarray

    def __post_init__(self) -> None:
        _validate_timestamp(self.timestamp)
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if self.image.dtype != np.uint8:
            raise ValueError("image must have dtype uint8")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3)")
        if self.image.shape[0] == 0 or self.image.shape[1] == 0:
            raise ValueError("image height and width must be positive")


@dataclass(frozen=True)
class GazeSample:
    """One normalized gaze observation at run-relative observation time."""

    timestamp: float
    x_normalized: float | None
    y_normalized: float | None
    valid: bool
    confidence: float | None = None

    def __post_init__(self) -> None:
        _validate_timestamp(self.timestamp)
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        _validate_unit_value("x_normalized", self.x_normalized)
        _validate_unit_value("y_normalized", self.y_normalized)
        _validate_unit_value("confidence", self.confidence)
        if self.valid and (self.x_normalized is None or self.y_normalized is None):
            raise ValueError("valid gaze samples require both coordinates")


if __name__ == "__main__":
    frame = SceneFrame(0.0, np.zeros((2, 3, 3), dtype=np.uint8))
    gaze = GazeSample(0.0, 0.5, 0.5, True, 1.0)
    print(frame.image.shape, gaze)
