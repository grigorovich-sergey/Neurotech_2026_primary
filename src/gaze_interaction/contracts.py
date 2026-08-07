"""Subsystem-local detection and tracking records with normalized geometry."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence


def _unit_value(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return result


def _timestamp(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timestamp must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("timestamp must be finite and non-negative")
    return result


@dataclass(frozen=True)
class BoundingBox:
    """Normalized top-left-origin box: +x right, +y down."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            object.__setattr__(self, name, _unit_value(name, getattr(self, name)))
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must have positive width and height")

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def contains(self, x: float, y: float, *, margin: float = 0.0) -> bool:
        x_value = _unit_value("x", x)
        y_value = _unit_value("y", y)
        margin_value = _unit_value("margin", margin)
        return (
            max(0.0, self.x_min - margin_value)
            <= x_value
            <= min(1.0, self.x_max + margin_value)
            and max(0.0, self.y_min - margin_value)
            <= y_value
            <= min(1.0, self.y_max + margin_value)
        )


@dataclass(frozen=True)
class Detection:
    box: BoundingBox
    label: str | None
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.box, BoundingBox):
            raise TypeError("box must be a BoundingBox")
        if self.label is not None and (not isinstance(self.label, str) or not self.label):
            raise ValueError("label must be a non-empty string or None")
        object.__setattr__(self, "confidence", _unit_value("confidence", self.confidence))


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    box: BoundingBox
    label: str | None
    confidence: float
    scene_timestamp: float

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise TypeError("track_id must be an integer")
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        if not isinstance(self.box, BoundingBox):
            raise TypeError("box must be a BoundingBox")
        if self.label is not None and (not isinstance(self.label, str) or not self.label):
            raise ValueError("label must be a non-empty string or None")
        object.__setattr__(self, "confidence", _unit_value("confidence", self.confidence))
        object.__setattr__(self, "scene_timestamp", _timestamp(self.scene_timestamp))


@dataclass(frozen=True)
class TrackedScene:
    timestamp: float
    objects: tuple[TrackedObject, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        if not isinstance(self.objects, tuple):
            raise TypeError("objects must be a tuple")
        track_ids: set[int] = set()
        for tracked_object in self.objects:
            if not isinstance(tracked_object, TrackedObject):
                raise TypeError("objects must contain TrackedObject values")
            if tracked_object.scene_timestamp != self.timestamp:
                raise ValueError("tracked-object timestamp must match its scene")
            if tracked_object.track_id in track_ids:
                raise ValueError("track IDs must be unique within one tracked scene")
            track_ids.add(tracked_object.track_id)


def normalize_pixel_box(
    xyxy: Sequence[float], *, image_width: int, image_height: int
) -> BoundingBox | None:
    """Clip one pixel box to the image and normalize it; discard malformed boxes."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(xyxy) != 4:
        raise ValueError("xyxy must contain exactly four coordinates")
    values = [float(value) for value in xyxy]
    if not all(math.isfinite(value) for value in values):
        return None
    x_min = min(max(values[0], 0.0), float(image_width))
    y_min = min(max(values[1], 0.0), float(image_height))
    x_max = min(max(values[2], 0.0), float(image_width))
    y_max = min(max(values[3], 0.0), float(image_height))
    if x_max <= x_min or y_max <= y_min:
        return None
    return BoundingBox(
        x_min / image_width,
        y_min / image_height,
        x_max / image_width,
        y_max / image_height,
    )


def denormalize_box(
    box: BoundingBox, *, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    return (
        box.x_min * image_width,
        box.y_min * image_height,
        box.x_max * image_width,
        box.y_max * image_height,
    )


if __name__ == "__main__":
    smoke = normalize_pixel_box((-2.0, 1.0, 8.0, 5.0), image_width=10, image_height=10)
    print(smoke, smoke.area if smoke is not None else None)
