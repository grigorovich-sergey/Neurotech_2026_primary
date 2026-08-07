import numpy as np

from foundations.contracts import GazeSample, SceneFrame
from foundations.virtual_glasses import VirtualGlasses


def _source(**overrides: object) -> VirtualGlasses:
    values = {
        "seed": 7,
        "duration_seconds": 0.5,
        "scene_width": 4,
        "scene_height": 3,
        "scene_rate_hz": 4.0,
        "gaze_rate_hz": 8.0,
        "scene_dropout_probability": 0.0,
        "gaze_dropout_probability": 0.0,
        "gaze_invalid_probability": 0.2,
    }
    values.update(overrides)
    return VirtualGlasses(**values)


def test_seeded_generation_is_identical() -> None:
    first = list(_source().samples())
    second = list(_source().samples())

    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert type(left) is type(right)
        assert left.timestamp == right.timestamp
        if isinstance(left, SceneFrame):
            assert isinstance(right, SceneFrame)
            np.testing.assert_array_equal(left.image, right.image)
        else:
            assert isinstance(left, GazeSample) and isinstance(right, GazeSample)
            assert left == right


def test_dropouts_create_gaps_and_callbacks_not_placeholder_samples() -> None:
    dropouts: list[tuple[str, float]] = []
    samples = list(
        _source(scene_dropout_probability=1.0).samples(
            on_dropout=lambda stream, timestamp: dropouts.append((stream, timestamp))
        )
    )

    assert not any(isinstance(sample, SceneFrame) for sample in samples)
    assert dropouts == [("scene", 0.0), ("scene", 0.25)]
