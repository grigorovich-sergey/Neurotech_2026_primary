import pytest

from foundations.contracts import GazeSample
from gaze_interaction.association import GazeAssociator
from gaze_interaction.contracts import BoundingBox, TrackedObject, TrackedScene, normalize_pixel_box


def _tracked(
    track_id: int,
    box: BoundingBox,
    *,
    confidence: float = 0.8,
    timestamp: float = 0.0,
) -> TrackedObject:
    return TrackedObject(track_id, box, f"object-{track_id}", confidence, timestamp)


def test_pixel_boxes_are_clipped_normalized_and_degenerate_boxes_discarded() -> None:
    assert normalize_pixel_box(
        (-5.0, 10.0, 120.0, 80.0), image_width=100, image_height=100
    ) == BoundingBox(0.0, 0.1, 1.0, 0.8)
    assert (
        normalize_pixel_box((10, 10, 10, 20), image_width=100, image_height=100)
        is None
    )


def test_overlap_prefers_smallest_then_confidence_then_lowest_track_id() -> None:
    large = _tracked(9, BoundingBox(0.1, 0.1, 0.9, 0.9), confidence=0.99)
    small_low = _tracked(7, BoundingBox(0.3, 0.3, 0.7, 0.7), confidence=0.7)
    small_high_id = _tracked(5, BoundingBox(0.3, 0.3, 0.7, 0.7), confidence=0.9)
    small_high_low_id = _tracked(3, BoundingBox(0.3, 0.3, 0.7, 0.7), confidence=0.9)
    associator = GazeAssociator(max_scene_age_seconds=0.25)
    associator.add_scene(
        TrackedScene(0.0, (large, small_low, small_high_id, small_high_low_id))
    )

    result = associator.associate(GazeSample(0.1, 0.5, 0.5, True, 1.0))

    assert result.candidate is not None
    assert result.candidate.track_id == 3


def test_invalid_no_match_stale_and_future_scene_never_fabricate_candidate() -> None:
    tracked = _tracked(1, BoundingBox(0.2, 0.2, 0.8, 0.8), timestamp=0.1)
    future = _tracked(2, BoundingBox(0.2, 0.2, 0.8, 0.8), timestamp=0.4)
    associator = GazeAssociator(max_scene_age_seconds=0.25)
    associator.add_scene(TrackedScene(0.1, (tracked,)))
    associator.add_scene(TrackedScene(0.4, (future,)))

    invalid = associator.associate(GazeSample(0.2, 0.5, 0.5, False, 0.2))
    no_match = associator.associate(GazeSample(0.2, 0.05, 0.05, True, 1.0))
    stale = associator.associate(GazeSample(0.7, 0.5, 0.5, True, 1.0))

    assert invalid.candidate is None
    assert invalid.scene is not None and invalid.scene.timestamp == pytest.approx(0.1)
    assert no_match.candidate is None
    assert stale.candidate is None
    assert stale.scene is None
