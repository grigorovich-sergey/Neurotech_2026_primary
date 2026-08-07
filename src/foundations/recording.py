"""HDF5 recording and exact-order replay for scene/gaze samples."""

from collections.abc import Callable, Iterator
from pathlib import Path
import time

import h5py
import numpy as np

from foundations.contracts import GazeSample, SceneFrame

Sample = SceneFrame | GazeSample
SCHEMA_VERSION = 1
SCENE_STREAM = np.uint8(0)
GAZE_STREAM = np.uint8(1)


def _append(dataset: h5py.Dataset, value: object) -> None:
    index = len(dataset)
    dataset.resize(index + 1, axis=0)
    dataset[index] = value


class HDF5Recorder:
    """Incrementally record canonical samples and their arrival order."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._last_scene_timestamp: float | None = None
        self._last_gaze_timestamp: float | None = None

        scene = self._file.create_group("scene")
        scene.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)

        gaze = self._file.create_group("gaze")
        for name in ("timestamp", "x_normalized", "y_normalized", "confidence"):
            gaze.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.float64)
        for name in ("has_x", "has_y", "valid", "has_confidence"):
            gaze.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.bool_)

        order = self._file.create_group("order")
        order.create_dataset("stream", shape=(0,), maxshape=(None,), dtype=np.uint8)
        order.create_dataset("index", shape=(0,), maxshape=(None,), dtype=np.int64)

    def __enter__(self) -> "HDF5Recorder":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._file:
            self._file.close()

    def record(self, sample: Sample) -> None:
        if isinstance(sample, SceneFrame):
            self._record_scene(sample)
        elif isinstance(sample, GazeSample):
            self._record_gaze(sample)
        else:
            raise TypeError(f"unsupported sample type: {type(sample).__name__}")

    def _record_scene(self, sample: SceneFrame) -> None:
        if (
            self._last_scene_timestamp is not None
            and sample.timestamp < self._last_scene_timestamp
        ):
            raise ValueError("scene timestamps must be non-decreasing")

        scene = self._file["scene"]
        image_dataset = scene.get("image")
        if image_dataset is None:
            image_dataset = scene.create_dataset(
                "image",
                shape=(0, *sample.image.shape),
                maxshape=(None, *sample.image.shape),
                dtype=np.uint8,
            )
        elif image_dataset.shape[1:] != sample.image.shape:
            raise ValueError("scene frame shape changed within one recording")

        index = len(scene["timestamp"])
        _append(scene["timestamp"], sample.timestamp)
        _append(image_dataset, sample.image)
        self._append_order(SCENE_STREAM, index)
        self._last_scene_timestamp = float(sample.timestamp)

    def _record_gaze(self, sample: GazeSample) -> None:
        if (
            self._last_gaze_timestamp is not None
            and sample.timestamp < self._last_gaze_timestamp
        ):
            raise ValueError("gaze timestamps must be non-decreasing")

        gaze = self._file["gaze"]
        index = len(gaze["timestamp"])
        _append(gaze["timestamp"], sample.timestamp)
        _append(
            gaze["x_normalized"],
            np.nan if sample.x_normalized is None else sample.x_normalized,
        )
        _append(
            gaze["y_normalized"],
            np.nan if sample.y_normalized is None else sample.y_normalized,
        )
        _append(gaze["confidence"], np.nan if sample.confidence is None else sample.confidence)
        _append(gaze["has_x"], sample.x_normalized is not None)
        _append(gaze["has_y"], sample.y_normalized is not None)
        _append(gaze["valid"], sample.valid)
        _append(gaze["has_confidence"], sample.confidence is not None)
        self._append_order(GAZE_STREAM, index)
        self._last_gaze_timestamp = float(sample.timestamp)

    def _append_order(self, stream: np.uint8, index: int) -> None:
        order = self._file["order"]
        _append(order["stream"], stream)
        _append(order["index"], index)


class HDF5Replay:
    """Read samples with original timestamps and arrival interleaving."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def samples(self) -> Iterator[Sample]:
        with h5py.File(self.path, "r") as file:
            if file.attrs.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported recording schema version")
            streams = file["order/stream"]
            indexes = file["order/index"]
            for stream, index in zip(streams, indexes, strict=True):
                if stream == SCENE_STREAM:
                    yield self._read_scene(file, int(index))
                elif stream == GAZE_STREAM:
                    yield self._read_gaze(file, int(index))
                else:
                    raise ValueError(f"unknown stream id in recording: {stream}")

    def replay(
        self,
        *,
        paced: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Iterator[Sample]:
        max_timestamp: float | None = None
        for sample in self.samples():
            if paced and max_timestamp is not None and sample.timestamp > max_timestamp:
                sleep(sample.timestamp - max_timestamp)
            if max_timestamp is None or sample.timestamp > max_timestamp:
                max_timestamp = float(sample.timestamp)
            yield sample

    @staticmethod
    def _read_scene(file: h5py.File, index: int) -> SceneFrame:
        return SceneFrame(
            timestamp=float(file["scene/timestamp"][index]),
            image=np.asarray(file["scene/image"][index], dtype=np.uint8),
        )

    @staticmethod
    def _read_gaze(file: h5py.File, index: int) -> GazeSample:
        gaze = file["gaze"]
        x = float(gaze["x_normalized"][index]) if gaze["has_x"][index] else None
        y = float(gaze["y_normalized"][index]) if gaze["has_y"][index] else None
        confidence = (
            float(gaze["confidence"][index]) if gaze["has_confidence"][index] else None
        )
        return GazeSample(
            timestamp=float(gaze["timestamp"][index]),
            x_normalized=x,
            y_normalized=y,
            valid=bool(gaze["valid"][index]),
            confidence=confidence,
        )


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    from foundations.fixtures import gaze_sample, scene_frame

    with TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.h5"
        with HDF5Recorder(path) as recorder:
            recorder.record(scene_frame())
            recorder.record(gaze_sample())
        print(list(HDF5Replay(path).samples()))
