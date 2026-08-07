import numpy as np
import pytest

from foundations.contracts import GazeSample, SceneFrame


def test_scene_frame_requires_canonical_rgb_uint8() -> None:
    with pytest.raises(ValueError, match="dtype uint8"):
        SceneFrame(0.0, np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        SceneFrame(0.0, np.zeros((2, 2), dtype=np.uint8))


def test_gaze_values_are_rejected_not_clipped() -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        GazeSample(0.0, 1.1, 0.5, True)


def test_valid_gaze_requires_coordinates_but_invalid_can_retain_them() -> None:
    with pytest.raises(ValueError, match="require both coordinates"):
        GazeSample(0.0, None, 0.5, True)

    sample = GazeSample(0.0, 0.25, 0.75, False, 0.2)
    assert sample.x_normalized == 0.25
    assert sample.y_normalized == 0.75
    assert sample.valid is False
