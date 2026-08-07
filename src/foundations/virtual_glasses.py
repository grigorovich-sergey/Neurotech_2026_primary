"""Deterministic scene and gaze source for hardware-independent development."""

from collections.abc import Callable, Iterator
import math

import numpy as np

from foundations.contracts import GazeSample, SceneFrame

Sample = SceneFrame | GazeSample
DropoutCallback = Callable[[str, float], None]


class VirtualGlasses:
    def __init__(
        self,
        *,
        seed: int,
        duration_seconds: float,
        scene_width: int,
        scene_height: int,
        scene_rate_hz: float,
        gaze_rate_hz: float,
        scene_dropout_probability: float = 0.0,
        gaze_dropout_probability: float = 0.0,
        gaze_invalid_probability: float = 0.0,
    ) -> None:
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if (
            isinstance(scene_width, bool)
            or not isinstance(scene_width, int)
            or scene_width <= 0
            or isinstance(scene_height, bool)
            or not isinstance(scene_height, int)
            or scene_height <= 0
        ):
            raise ValueError("scene dimensions must be positive integers")
        if (
            not math.isfinite(scene_rate_hz)
            or scene_rate_hz <= 0
            or not math.isfinite(gaze_rate_hz)
            or gaze_rate_hz <= 0
        ):
            raise ValueError("sample rates must be finite and positive")
        for name, probability in (
            ("scene_dropout_probability", scene_dropout_probability),
            ("gaze_dropout_probability", gaze_dropout_probability),
            ("gaze_invalid_probability", gaze_invalid_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

        self.seed = seed
        self.duration_seconds = float(duration_seconds)
        self.scene_width = scene_width
        self.scene_height = scene_height
        self.scene_rate_hz = float(scene_rate_hz)
        self.gaze_rate_hz = float(gaze_rate_hz)
        self.scene_dropout_probability = scene_dropout_probability
        self.gaze_dropout_probability = gaze_dropout_probability
        self.gaze_invalid_probability = gaze_invalid_probability

    def samples(self, on_dropout: DropoutCallback | None = None) -> Iterator[Sample]:
        rng = np.random.default_rng(self.seed)
        scene_index = 0
        gaze_index = 0

        while True:
            scene_time = scene_index / self.scene_rate_hz
            gaze_time = gaze_index / self.gaze_rate_hz
            next_time = min(scene_time, gaze_time)
            if next_time >= self.duration_seconds:
                break

            if scene_time <= gaze_time:
                scene_index += 1
                if rng.random() < self.scene_dropout_probability:
                    if on_dropout is not None:
                        on_dropout("scene", scene_time)
                    continue
                image = rng.integers(
                    0,
                    256,
                    size=(self.scene_height, self.scene_width, 3),
                    dtype=np.uint8,
                )
                yield SceneFrame(timestamp=scene_time, image=image)
            else:
                gaze_index += 1
                if rng.random() < self.gaze_dropout_probability:
                    if on_dropout is not None:
                        on_dropout("gaze", gaze_time)
                    continue
                x = float(rng.random())
                y = float(rng.random())
                valid = bool(rng.random() >= self.gaze_invalid_probability)
                confidence = float(rng.random())
                yield GazeSample(
                    timestamp=gaze_time,
                    x_normalized=x,
                    y_normalized=y,
                    valid=valid,
                    confidence=confidence,
                )


if __name__ == "__main__":
    source = VirtualGlasses(
        seed=1,
        duration_seconds=0.2,
        scene_width=4,
        scene_height=3,
        scene_rate_hz=5,
        gaze_rate_hz=10,
    )
    print(list(source.samples()))
