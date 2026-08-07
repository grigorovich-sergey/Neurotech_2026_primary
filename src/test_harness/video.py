"""Prerecorded-video source producing canonical scene frames."""

import math
from pathlib import Path
from collections.abc import Iterator

from foundations.contracts import SceneFrame


def _frame_timestamp(frame_index: int, frames_per_second: float) -> float:
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame_index must be a non-negative integer")
    if not math.isfinite(frames_per_second) or frames_per_second <= 0.0:
        raise ValueError("video frame rate must be finite and positive")
    return frame_index / frames_per_second


class VideoSceneSource:
    """Decode an ordinary constant-frame-rate video on a run-relative timeline."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        import cv2

        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise ValueError(f"could not open video: {self.path}")
            frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(frames_per_second) or frames_per_second <= 0.0:
                raise ValueError(
                    f"video does not report a valid constant frame rate: {self.path}"
                )
            self.frames_per_second = frames_per_second
        finally:
            capture.release()

    @property
    def frame_period_seconds(self) -> float:
        return 1.0 / self.frames_per_second

    def frames(self) -> Iterator[SceneFrame]:
        import cv2

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"could not open video: {self.path}")

        frame_index = 0
        try:
            while True:
                ok, image_bgr = capture.read()
                if not ok:
                    break
                timestamp = _frame_timestamp(frame_index, self.frames_per_second)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                yield SceneFrame(timestamp, image_rgb)
                frame_index += 1
            if frame_index == 0:
                raise ValueError(f"video contains no decodable frames: {self.path}")
        finally:
            capture.release()


if __name__ == "__main__":
    print(_frame_timestamp(3, 30.0))
