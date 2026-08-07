"""Lightweight RGB diagnostic rendering for gaze-interaction verification."""

from pathlib import Path

import numpy as np

from foundations.contracts import GazeSample, SceneFrame
from gaze_interaction.contracts import TrackedObject
from gaze_interaction.dwell import DwellState


def render_diagnostic(
    frame: SceneFrame,
    *,
    tracks: tuple[TrackedObject, ...],
    gaze: GazeSample | None,
    candidate: TrackedObject | None,
    dwell_state: DwellState | None,
    intent_score: float | None,
) -> np.ndarray:
    """Return an annotated RGB image without mutating the canonical SceneFrame."""

    import cv2

    image = frame.image.copy()
    height, width = image.shape[:2]
    candidate_track_id = candidate.track_id if candidate is not None else None
    for tracked_object in tracks:
        box = tracked_object.box
        p1 = (int(round(box.x_min * (width - 1))), int(round(box.y_min * (height - 1))))
        p2 = (int(round(box.x_max * (width - 1))), int(round(box.y_max * (height - 1))))
        selected = tracked_object.track_id == candidate_track_id
        color_rgb = (255, 215, 0) if selected else (0, 220, 90)
        thickness = 3 if selected else 1
        cv2.rectangle(image, p1, p2, color_rgb, thickness)
        label = tracked_object.label or "object"
        cv2.putText(
            image,
            f"{label} #{tracked_object.track_id}",
            (p1[0], max(12, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color_rgb,
            1,
            cv2.LINE_AA,
        )

    if gaze is not None and gaze.x_normalized is not None and gaze.y_normalized is not None:
        point = (
            int(round(gaze.x_normalized * (width - 1))),
            int(round(gaze.y_normalized * (height - 1))),
        )
        gaze_color_rgb = (40, 130, 255) if gaze.valid else (255, 70, 70)
        cv2.drawMarker(
            image,
            point,
            gaze_color_rgb,
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )

    if dwell_state is not None:
        status = (
            f"dwell {dwell_state.accumulated_seconds:.2f}/"
            f"{dwell_state.required_seconds:.2f}s"
        )
        if dwell_state.triggered:
            status += " TRIGGERED"
        cv2.putText(
            image,
            status,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if intent_score is not None:
        cv2.putText(
            image,
            f"intent {intent_score:.2f}",
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def save_rgb_image(path: str | Path, image_rgb: np.ndarray) -> None:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"could not write diagnostic image: {path}")


def show_rgb_image(image_rgb: np.ndarray, *, window_name: str = "gaze interaction") -> None:
    import cv2

    cv2.imshow(window_name, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def close_windows() -> None:
    import cv2

    cv2.destroyAllWindows()


if __name__ == "__main__":
    smoke_frame = SceneFrame(0.0, np.zeros((80, 120, 3), dtype=np.uint8))
    print(
        render_diagnostic(
            smoke_frame,
            tracks=(),
            gaze=GazeSample(0.0, 0.5, 0.5, True, 1.0),
            candidate=None,
            dwell_state=None,
            intent_score=None,
        ).shape
    )
