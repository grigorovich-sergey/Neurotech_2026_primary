"""Thin adapter around Supervision's ByteTrack implementation."""

from collections.abc import Sequence

import numpy as np

from foundations.contracts import SceneFrame
from gaze_interaction.contracts import (
    Detection,
    TrackedObject,
    TrackedScene,
    denormalize_box,
    normalize_pixel_box,
)


class ByteTrackAdapter:
    """Track local detections; IDs are meaningful only within this adapter/run."""

    def __init__(
        self,
        *,
        activation_threshold: float,
        lost_track_buffer: int,
        matching_threshold: float,
        frame_rate: int,
    ) -> None:
        if not 0.0 <= activation_threshold <= 1.0:
            raise ValueError("activation_threshold must be within [0, 1]")
        if isinstance(lost_track_buffer, bool) or lost_track_buffer < 0:
            raise ValueError("lost_track_buffer must be a non-negative integer")
        if not 0.0 <= matching_threshold <= 1.0:
            raise ValueError("matching_threshold must be within [0, 1]")
        if isinstance(frame_rate, bool) or frame_rate <= 0:
            raise ValueError("frame_rate must be a positive integer")

        import supervision as sv

        self._sv = sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=float(activation_threshold),
            lost_track_buffer=int(lost_track_buffer),
            minimum_matching_threshold=float(matching_threshold),
            frame_rate=int(frame_rate),
        )
        self._label_to_class_id: dict[str, int] = {}
        self._class_id_to_label: dict[int, str] = {}

    def update(
        self, frame: SceneFrame, detections: Sequence[Detection]
    ) -> TrackedScene:
        if not isinstance(frame, SceneFrame):
            raise TypeError("frame must be a foundations.contracts.SceneFrame")
        height, width = frame.image.shape[:2]
        xyxy = np.asarray(
            [
                denormalize_box(
                    detection.box, image_width=width, image_height=height
                )
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape((-1, 4))
        confidence = np.asarray(
            [detection.confidence for detection in detections], dtype=np.float32
        )
        class_id = np.asarray(
            [self._class_id(detection.label) for detection in detections], dtype=int
        )
        local = self._sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        tracked = self._tracker.update_with_detections(local)
        if len(tracked) == 0:
            return TrackedScene(float(frame.timestamp), ())
        if tracked.tracker_id is None or tracked.confidence is None:
            raise RuntimeError("ByteTrack output omitted required tracker/confidence data")

        returned_class_ids = (
            tracked.class_id
            if tracked.class_id is not None
            else np.full(len(tracked), -1, dtype=int)
        )
        objects: list[TrackedObject] = []
        for coordinates, returned_confidence, returned_class_id, track_id in zip(
            tracked.xyxy,
            tracked.confidence,
            returned_class_ids,
            tracked.tracker_id,
            strict=True,
        ):
            box = normalize_pixel_box(
                coordinates, image_width=width, image_height=height
            )
            if box is None:
                continue
            objects.append(
                TrackedObject(
                    track_id=int(track_id),
                    box=box,
                    label=self._class_id_to_label.get(int(returned_class_id)),
                    confidence=float(returned_confidence),
                    scene_timestamp=float(frame.timestamp),
                )
            )
        objects.sort(key=lambda item: item.track_id)
        return TrackedScene(float(frame.timestamp), tuple(objects))

    def _class_id(self, label: str | None) -> int:
        if label is None:
            return -1
        if label not in self._label_to_class_id:
            class_id = len(self._label_to_class_id)
            self._label_to_class_id[label] = class_id
            self._class_id_to_label[class_id] = label
        return self._label_to_class_id[label]


if __name__ == "__main__":
    frame = SceneFrame(0.0, np.zeros((20, 20, 3), dtype=np.uint8))
    box = normalize_pixel_box((2, 2, 10, 10), image_width=20, image_height=20)
    assert box is not None
    detection = Detection(
        box,
        "object",
        0.9,
    )
    smoke = ByteTrackAdapter(
        activation_threshold=0.25,
        lost_track_buffer=3,
        matching_threshold=0.8,
        frame_rate=10,
    )
    print(smoke.update(frame, [detection]))
