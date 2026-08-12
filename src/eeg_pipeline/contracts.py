"""Subsystem-local timestamped EEG data contracts."""

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

import numpy as np


def _finite_non_negative(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _optional_finite_non_negative(name: str, value: float | None) -> None:
    if value is not None:
        _finite_non_negative(name, value)


def _validate_bounds(start: float, end: float) -> None:
    _finite_non_negative("requested_start", start)
    _finite_non_negative("requested_end", end)
    if end < start:
        raise ValueError("requested_end must be greater than or equal to requested_start")


class WindowCompleteness(str, Enum):
    """Whether the retained buffer covers a requested interval."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    EMPTY = "empty"


class QualityState(str, Enum):
    """Downstream usability of an EEG window."""

    USABLE = "usable"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


@dataclass(frozen=True)
class EEGSample:
    """One raw single-channel EEG observation in microvolts."""

    timestamp: float
    value_uv: float
    valid: bool = True
    vendor_timestamp_unix: float | None = None
    host_receipt_timestamp: float | None = None

    def __post_init__(self) -> None:
        _finite_non_negative("timestamp", self.timestamp)
        if isinstance(self.value_uv, bool) or not isinstance(self.value_uv, Real):
            raise TypeError("value_uv must be a real number")
        if not math.isfinite(float(self.value_uv)):
            raise ValueError("value_uv must be finite")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        _optional_finite_non_negative(
            "vendor_timestamp_unix", self.vendor_timestamp_unix
        )
        _optional_finite_non_negative(
            "host_receipt_timestamp", self.host_receipt_timestamp
        )


@dataclass(frozen=True)
class EEGWindow:
    """Raw samples actually available inside one closed requested interval."""

    requested_start: float
    requested_end: float
    samples: tuple[EEGSample, ...]
    actual_start: float | None
    actual_end: float | None
    completeness: WindowCompleteness

    def __post_init__(self) -> None:
        _validate_bounds(self.requested_start, self.requested_end)
        if not isinstance(self.samples, tuple) or not all(
            isinstance(sample, EEGSample) for sample in self.samples
        ):
            raise TypeError("samples must be a tuple of EEGSample values")
        if (self.actual_start is None) != (self.actual_end is None):
            raise ValueError("actual_start and actual_end must both be present or absent")
        if self.samples:
            if self.actual_start is None or self.actual_end is None:
                raise ValueError("non-empty windows require actual bounds")
            if self.actual_start != self.samples[0].timestamp:
                raise ValueError("actual_start must match the first sample")
            if self.actual_end != self.samples[-1].timestamp:
                raise ValueError("actual_end must match the last sample")
            if any(
                sample.timestamp < self.requested_start
                or sample.timestamp > self.requested_end
                for sample in self.samples
            ):
                raise ValueError("window contains a sample outside requested bounds")
        elif self.actual_start is not None or self.actual_end is not None:
            raise ValueError("empty windows cannot have actual bounds")


@dataclass(frozen=True)
class EEGFeatureWindow:
    """Quality-aware, episode-agnostic feature contract for Instance 4."""

    requested_start: float
    requested_end: float
    actual_start: float | None
    actual_end: float | None
    sample_count: int
    completeness: WindowCompleteness
    quality_state: QualityState
    quality_reasons: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray | None

    def __post_init__(self) -> None:
        _validate_bounds(self.requested_start, self.requested_end)
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if (self.actual_start is None) != (self.actual_end is None):
            raise ValueError("actual_start and actual_end must both be present or absent")
        if not isinstance(self.quality_reasons, tuple) or not all(
            isinstance(reason, str) and reason for reason in self.quality_reasons
        ):
            raise TypeError("quality_reasons must be a tuple of non-empty strings")
        if not isinstance(self.feature_names, tuple) or not all(
            isinstance(name, str) and name for name in self.feature_names
        ):
            raise TypeError("feature_names must be a tuple of non-empty strings")
        if self.quality_state is QualityState.USABLE:
            if self.values is None:
                raise ValueError("usable feature windows require values")
            if not isinstance(self.values, np.ndarray):
                raise TypeError("values must be a numpy.ndarray or None")
            if self.values.dtype != np.float64 or self.values.ndim != 1:
                raise ValueError("values must be a one-dimensional float64 array")
            if len(self.values) != len(self.feature_names):
                raise ValueError("values must match feature_names")
            if not np.all(np.isfinite(self.values)):
                raise ValueError("feature values must be finite")
        elif self.values is not None:
            raise ValueError("unusable feature windows must expose values=None")


if __name__ == "__main__":
    sample = EEGSample(0.0, 1.0)
    print(EEGWindow(0.0, 0.0, (sample,), 0.0, 0.0, WindowCompleteness.COMPLETE))
