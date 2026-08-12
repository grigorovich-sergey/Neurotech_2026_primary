"""Subsystem-local HDF5 raw EEG recording and deterministic replay."""

from collections.abc import Callable, Iterator
import math
from numbers import Real
from pathlib import Path
import time

import h5py
import numpy as np

from eeg_pipeline.contracts import EEGSample


SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
TIMEBASE = "run_relative_seconds"
VALUE_UNITS = "microvolts"


def _append(dataset: h5py.Dataset, value: object) -> None:
    index = len(dataset)
    dataset.resize(index + 1, axis=0)
    dataset[index] = value


class EEGHDF5Recorder:
    """Persist the incoming raw stream before any preprocessing."""

    def __init__(self, path: str | Path, *, sample_rate_hz: float) -> None:
        if (
            isinstance(sample_rate_hz, bool)
            or not isinstance(sample_rate_hz, Real)
            or not math.isfinite(float(sample_rate_hz))
            or sample_rate_hz <= 0
        ):
            raise ValueError("sample_rate_hz must be a positive finite number")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: h5py.File | None = h5py.File(self.path, "w")
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.attrs["sample_rate_hz"] = float(sample_rate_hz)
        self._file.attrs["timebase"] = TIMEBASE
        self._file.attrs["value_units"] = VALUE_UNITS
        self._file.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)
        self._file.create_dataset("value_uv", shape=(0,), maxshape=(None,), dtype=np.float64)
        self._file.create_dataset("valid", shape=(0,), maxshape=(None,), dtype=np.bool_)
        self._file.create_dataset(
            "vendor_timestamp_unix", shape=(0,), maxshape=(None,), dtype=np.float64
        )
        self._file.create_dataset(
            "host_receipt_timestamp", shape=(0,), maxshape=(None,), dtype=np.float64
        )
        self._last_timestamp: float | None = None

    def __enter__(self) -> "EEGHDF5Recorder":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def record(self, sample: EEGSample) -> None:
        if not isinstance(sample, EEGSample):
            raise TypeError("sample must be an EEGSample")
        if self._file is None:
            raise RuntimeError("recorder is closed")
        if self._last_timestamp is not None and sample.timestamp < self._last_timestamp:
            raise ValueError("EEG timestamps must be non-decreasing")
        _append(self._file["timestamp"], sample.timestamp)
        _append(self._file["value_uv"], sample.value_uv)
        _append(self._file["valid"], sample.valid)
        _append(
            self._file["vendor_timestamp_unix"],
            np.nan
            if sample.vendor_timestamp_unix is None
            else sample.vendor_timestamp_unix,
        )
        _append(
            self._file["host_receipt_timestamp"],
            np.nan
            if sample.host_receipt_timestamp is None
            else sample.host_receipt_timestamp,
        )
        self._last_timestamp = float(sample.timestamp)


class EEGHDF5Replay:
    """Replay raw EEG samples from current schema v2 or legacy schema v1."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def _optional_timestamp(
        dataset: h5py.Dataset, index: int, name: str
    ) -> float | None:
        value = float(dataset[index])
        if math.isnan(value):
            return None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"EEG recording {name} contains an invalid value")
        return value

    def samples(self) -> Iterator[EEGSample]:
        with h5py.File(self.path, "r") as file:
            schema_version = int(file.attrs.get("schema_version", -1))
            if schema_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
                raise ValueError("unsupported EEG recording schema version")
            if file.attrs.get("timebase") != TIMEBASE:
                raise ValueError("unsupported EEG recording timebase")
            timestamps = file["timestamp"]
            values = file["value_uv"]
            validity = file["valid"]
            datasets = [timestamps, values, validity]
            vendor_timestamps: h5py.Dataset | None = None
            receipt_timestamps: h5py.Dataset | None = None
            if schema_version == SCHEMA_VERSION:
                vendor_timestamps = file["vendor_timestamp_unix"]
                receipt_timestamps = file["host_receipt_timestamp"]
                datasets.extend((vendor_timestamps, receipt_timestamps))
            if len({len(dataset) for dataset in datasets}) != 1:
                raise ValueError("EEG recording datasets have inconsistent lengths")
            last_timestamp: float | None = None
            for index in range(len(timestamps)):
                vendor_timestamp = (
                    None
                    if vendor_timestamps is None
                    else self._optional_timestamp(
                        vendor_timestamps, index, "vendor_timestamp_unix"
                    )
                )
                receipt_timestamp = (
                    None
                    if receipt_timestamps is None
                    else self._optional_timestamp(
                        receipt_timestamps, index, "host_receipt_timestamp"
                    )
                )
                sample = EEGSample(
                    float(timestamps[index]),
                    float(values[index]),
                    bool(validity[index]),
                    vendor_timestamp_unix=vendor_timestamp,
                    host_receipt_timestamp=receipt_timestamp,
                )
                if last_timestamp is not None and sample.timestamp < last_timestamp:
                    raise ValueError("EEG recording timestamps are not non-decreasing")
                last_timestamp = float(sample.timestamp)
                yield sample

    def replay(
        self,
        *,
        paced: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Iterator[EEGSample]:
        previous_timestamp: float | None = None
        for sample in self.samples():
            if paced and previous_timestamp is not None and sample.timestamp > previous_timestamp:
                sleep(sample.timestamp - previous_timestamp)
            previous_timestamp = float(sample.timestamp)
            yield sample


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "eeg.h5"
        with EEGHDF5Recorder(path, sample_rate_hz=250.0) as recorder:
            recorder.record(EEGSample(0.0, 1.0))
            recorder.record(EEGSample(0.004, 2.0))
        print(list(EEGHDF5Replay(path).samples()))
