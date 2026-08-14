"""Mutable Guardian snapshot -> EEG feature source boundary for live attempts.

This module owns no SDK worker or device lifecycle. Its methods run synchronously
on the caller (Integration) thread, where finalized raw recording and feature
evaluation belong.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Protocol

from eeg_pipeline.contracts import EEGFeatureWindow, EEGSample, EEGWindow

_DEFAULT_RETENTION_SECONDS = 30.0
_GUARDIAN_SAMPLE_RATE_HZ = 250.0
_TIMESTAMP_TOLERANCE = 1e-9


class _GuardianWindowSource(Protocol):
    def check_health(self) -> None: ...

    def finalize_before(self, timestamp: float) -> tuple[EEGSample, ...]: ...

    def window(self, start: float, end: float) -> EEGWindow: ...


class _EEGPipeline(Protocol):
    def features_from_window(self, window: EEGWindow) -> EEGFeatureWindow: ...


class _EEGRecorder(Protocol):
    def record(self, sample: EEGSample) -> None: ...


def _timestamp(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_seconds(name: str, value: float) -> float:
    result = _timestamp(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


class GuardianEEGFeatureSource:
    """Evaluate replaceable Guardian windows and persist only finalized samples.

    ``drain_through`` remains the Integration health hook, but does not expose raw
    callback order to the append-only EEG pipeline. It checks acquisition health
    and finalizes only samples older than ``retention_seconds``. ``features`` then
    closes data before the requested start and evaluates the adapter's current
    ordered snapshot directly. Acquisition failures propagate unchanged.
    """

    def __init__(
        self,
        *,
        guardian: _GuardianWindowSource,
        pipeline: _EEGPipeline,
        recorder: _EEGRecorder | None = None,
        retention_seconds: float = _DEFAULT_RETENTION_SECONDS,
    ) -> None:
        for method_name in ("check_health", "finalize_before", "window"):
            if not callable(getattr(guardian, method_name, None)):
                raise TypeError(f"guardian must expose {method_name}()")
        if not callable(getattr(pipeline, "features_from_window", None)):
            raise TypeError("pipeline must expose features_from_window()")
        if recorder is not None and not callable(getattr(recorder, "record", None)):
            raise TypeError("recorder must expose record() or be None")
        self.guardian = guardian
        self.pipeline = pipeline
        self.recorder = recorder
        self.retention_seconds = _positive_seconds(
            "retention_seconds", retention_seconds
        )
        self.finalized_sample_count = 0
        self.last_finalized_timestamp: float | None = None
        self._finalized_before_timestamp = 0.0
        self._latest_cutoff_timestamp: float | None = None

    def _record_finalized(
        self,
        samples: tuple[EEGSample, ...],
        *,
        boundary: float,
    ) -> int:
        if not isinstance(samples, tuple) or not all(
            isinstance(sample, EEGSample) for sample in samples
        ):
            raise TypeError(
                "Guardian finalize_before() must return a tuple of EEGSample values"
            )
        if any(
            sample.timestamp >= boundary + _TIMESTAMP_TOLERANCE for sample in samples
        ):
            raise RuntimeError("Guardian finalized an EEG sample beyond its boundary")
        previous = self.last_finalized_timestamp
        for sample in samples:
            if (
                previous is not None
                and sample.timestamp < previous - _TIMESTAMP_TOLERANCE
            ):
                raise ValueError("finalized Guardian timestamps must be non-decreasing")
            previous = float(sample.timestamp)

        for sample in samples:
            if self.recorder is not None:
                self.recorder.record(sample)
            self.finalized_sample_count += 1
            self.last_finalized_timestamp = float(sample.timestamp)
        return len(samples)

    def _finalize_before(self, boundary: float) -> int:
        if boundary <= self._finalized_before_timestamp + _TIMESTAMP_TOLERANCE:
            return 0
        samples = self.guardian.finalize_before(boundary)
        count = self._record_finalized(samples, boundary=boundary)
        self._finalized_before_timestamp = boundary
        return count

    def drain_through(self, cutoff_timestamp: float) -> int:
        """Check health and finalize data older than the retained live horizon."""

        cutoff = _timestamp("cutoff_timestamp", cutoff_timestamp)
        if (
            self._latest_cutoff_timestamp is not None
            and cutoff < self._latest_cutoff_timestamp - _TIMESTAMP_TOLERANCE
        ):
            raise ValueError("Guardian scientific cutoffs must be non-decreasing")
        self.guardian.check_health()
        self._latest_cutoff_timestamp = cutoff
        stable_boundary = max(0.0, cutoff - self.retention_seconds)
        return self._finalize_before(stable_boundary)

    def features(self, start: float, end: float) -> EEGFeatureWindow:
        """Evaluate the current ordered snapshot for the closed requested window."""

        window_start = _timestamp("start", start)
        window_end = _timestamp("end", end)
        if window_end < window_start:
            raise ValueError("EEG feature bounds must satisfy 0 <= start <= end")
        self.drain_through(window_end)
        if window_start < self._finalized_before_timestamp - _TIMESTAMP_TOLERANCE:
            raise ValueError("EEG feature start is older than retained Guardian data")
        self._finalize_before(window_start)
        raw_window = self.guardian.window(window_start, window_end)
        if not isinstance(raw_window, EEGWindow):
            raise TypeError("Guardian window() must return an EEGWindow")
        if not math.isclose(
            raw_window.requested_start,
            window_start,
            rel_tol=0.0,
            abs_tol=_TIMESTAMP_TOLERANCE,
        ) or not math.isclose(
            raw_window.requested_end,
            window_end,
            rel_tol=0.0,
            abs_tol=_TIMESTAMP_TOLERANCE,
        ):
            raise RuntimeError(
                "Guardian returned a different EEG window than requested"
            )
        return self.pipeline.features_from_window(raw_window)

    def drain_remaining(self) -> int:
        """Finalize through the latest processed cutoff after recording stops."""

        self.guardian.check_health()
        if self._latest_cutoff_timestamp is None:
            return 0
        final_slot = math.floor(
            self._latest_cutoff_timestamp * _GUARDIAN_SAMPLE_RATE_HZ
            + _TIMESTAMP_TOLERANCE
        )
        return self._finalize_before((final_slot + 1) / _GUARDIAN_SAMPLE_RATE_HZ)


if __name__ == "__main__":
    from eeg_pipeline.contracts import WindowCompleteness

    class _Guardian:
        def check_health(self) -> None:
            pass

        def finalize_before(self, timestamp):
            return ()

        def window(self, start, end):
            sample = EEGSample(end, 1.0)
            return EEGWindow(
                start,
                end,
                (sample,),
                sample.timestamp,
                sample.timestamp,
                WindowCompleteness.PARTIAL,
            )

    class _Pipeline:
        def features_from_window(self, window):
            return window.requested_start, window.requested_end, len(window.samples)

    demo = GuardianEEGFeatureSource(guardian=_Guardian(), pipeline=_Pipeline())
    print(demo.features(0.0, 0.5))
