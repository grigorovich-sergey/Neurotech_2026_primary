"""Small deterministic data fixtures shared across subsystem development."""

import numpy as np

from foundations.contracts import GazeSample, SceneFrame


def scene_frame(timestamp: float = 0.0, width: int = 4, height: int = 3) -> SceneFrame:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return SceneFrame(timestamp=timestamp, image=image)


def gaze_sample(timestamp: float = 0.0) -> GazeSample:
    return GazeSample(
        timestamp=timestamp,
        x_normalized=0.5,
        y_normalized=0.5,
        valid=True,
        confidence=1.0,
    )


if __name__ == "__main__":
    print(scene_frame(), gaze_sample())
