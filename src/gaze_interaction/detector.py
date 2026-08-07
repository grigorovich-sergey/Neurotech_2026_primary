"""Thin adapter around the Ultralytics prompt-free YOLOE detector."""

from pathlib import Path
from typing import Any

import numpy as np

from foundations.contracts import SceneFrame
from gaze_interaction.contracts import Detection, normalize_pixel_box


class YOLOEDetector:
    """Load YOLOE explicitly and expose only normalized local Detection records."""

    def __init__(
        self,
        model: str | Path,
        *,
        confidence_threshold: float,
        image_size: int,
        device: str,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")

        # Keep imports and possible weight resolution/download out of module import/smoke paths.
        from ultralytics import YOLOE

        self.model_source = str(model)
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.device = device
        self._model = YOLOE(self.model_source)

    def detect(self, frame: SceneFrame) -> tuple[Detection, ...]:
        if not isinstance(frame, SceneFrame):
            raise TypeError("frame must be a foundations.contracts.SceneFrame")

        # Foundation frames are RGB; Ultralytics treats ndarray sources as BGR.
        bgr_image = np.ascontiguousarray(frame.image[:, :, ::-1])
        results = self._model.predict(
            source=bgr_image,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return ()
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return ()

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        height, width = frame.image.shape[:2]
        detections: list[Detection] = []
        for coordinates, confidence, class_id in zip(
            xyxy, confidences, class_ids, strict=True
        ):
            normalized_box = normalize_pixel_box(
                coordinates,
                image_width=width,
                image_height=height,
            )
            if normalized_box is None:
                continue
            detections.append(
                Detection(
                    box=normalized_box,
                    label=_class_name(result.names, int(class_id)),
                    confidence=float(confidence),
                )
            )
        return tuple(detections)


def _class_name(names: Any, class_id: int) -> str | None:
    if isinstance(names, dict):
        value = names.get(class_id)
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        value = names[class_id]
    else:
        value = None
    return str(value) if value is not None else None


if __name__ == "__main__":
    from gaze_interaction.contracts import normalize_pixel_box

    print(normalize_pixel_box((-1, 2, 11, 8), image_width=10, image_height=10))
