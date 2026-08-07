"""Mouse and CSV sources for canonical gaze samples."""

import csv
from pathlib import Path
from collections.abc import Iterator

from foundations.contracts import GazeSample


CSV_COLUMNS = ("timestamp", "x", "y", "validity")


def _normalize_scene_position(
    x: int, y: int, *, width: int, height: int
) -> tuple[float, float] | None:
    if height <= 0 or width <= 0:
        raise ValueError("scene dimensions must be positive")
    if not (0 <= x < width and 0 <= y < height):
        return None
    x_normalized = 0.0 if width == 1 else x / (width - 1)
    y_normalized = 0.0 if height == 1 else y / (height - 1)
    return x_normalized, y_normalized


def _parse_optional_float(value: str, *, name: str, line_number: int) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid {name} on gaze CSV line {line_number}: {value!r}"
        ) from exc


def _parse_validity(value: str, *, line_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(
        f"invalid validity on gaze CSV line {line_number}: expected true/false or 1/0"
    )


class GazeCsvSource:
    """Replay canonical gaze samples from the harness CSV schema."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def samples(self) -> Iterator[GazeSample]:
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(CSV_COLUMNS):
                raise ValueError(
                    "gaze CSV header must be exactly: " + ",".join(CSV_COLUMNS)
                )

            previous_timestamp: float | None = None
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(row.get(column) is None for column in CSV_COLUMNS):
                    raise ValueError(f"malformed gaze CSV row on line {line_number}")
                timestamp = _parse_optional_float(
                    row["timestamp"], name="timestamp", line_number=line_number
                )
                if timestamp is None:
                    raise ValueError(
                        f"missing timestamp on gaze CSV line {line_number}"
                    )
                x = _parse_optional_float(row["x"], name="x", line_number=line_number)
                y = _parse_optional_float(row["y"], name="y", line_number=line_number)
                valid = _parse_validity(row["validity"], line_number=line_number)
                try:
                    sample = GazeSample(timestamp, x, y, valid, 1.0 if valid else None)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid gaze CSV sample on line {line_number}: {exc}"
                    ) from exc
                if (
                    previous_timestamp is not None
                    and sample.timestamp < previous_timestamp
                ):
                    raise ValueError(
                        f"gaze CSV timestamps must be non-decreasing (line {line_number})"
                    )
                previous_timestamp = float(sample.timestamp)
                yield sample


class GazeCsvWriter:
    """Record exactly the canonical gaze samples supplied to the pipeline."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(CSV_COLUMNS)

    def write(self, sample: GazeSample) -> None:
        if not isinstance(sample, GazeSample):
            raise TypeError("sample must be a foundations.contracts.GazeSample")
        self._writer.writerow(
            (
                format(float(sample.timestamp), ".17g"),
                "" if sample.x_normalized is None else format(sample.x_normalized, ".17g"),
                "" if sample.y_normalized is None else format(sample.y_normalized, ".17g"),
                "true" if sample.valid else "false",
            )
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "GazeCsvWriter":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


class MouseGazeSource:
    """Convert the cursor in an OpenCV scene window to normalized gaze."""

    def __init__(self, *, window_name: str) -> None:
        if not window_name:
            raise ValueError("window_name must be non-empty")
        self.window_name = window_name
        self._width: int | None = None
        self._height: int | None = None
        self._cursor_xy: tuple[int, int] | None = None

        import cv2

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self._on_mouse)

    def set_scene_shape(self, height: int, width: int) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("scene dimensions must be positive")
        self._height = height
        self._width = width

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        import cv2

        if event == cv2.EVENT_MOUSEMOVE:
            self._cursor_xy = (x, y)

    def sample(self, timestamp: float) -> GazeSample:
        if self._width is None or self._height is None or self._cursor_xy is None:
            return GazeSample(timestamp, None, None, False, None)
        x, y = self._cursor_xy
        normalized = _normalize_scene_position(
            x, y, width=self._width, height=self._height
        )
        if normalized is None:
            return GazeSample(timestamp, None, None, False, None)
        x_normalized, y_normalized = normalized
        return GazeSample(timestamp, x_normalized, y_normalized, True, 1.0)

    def window_is_open(self) -> bool:
        import cv2

        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1.0
        except cv2.error:
            return False


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "gaze.csv"
        with GazeCsvWriter(path) as writer:
            writer.write(GazeSample(0.0, 0.25, 0.75, True, 1.0))
            writer.write(GazeSample(0.1, None, None, False, None))
        print(list(GazeCsvSource(path).samples()))
    print(_normalize_scene_position(9, 4, width=10, height=5))
