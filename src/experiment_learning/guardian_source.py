"""Causal Guardian queue -> EEG feature source boundary for live attempts.

This module owns no worker or lifecycle.  Its methods run synchronously on the
caller (Integration) thread, where raw recording and EEGPipeline mutation belong.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Protocol

from eeg_pipeline.contracts import EEGFeatureWindow, EEGSample


class _GuardianDrain(Protocol):
    def drain(
        self, *, cutoff_timestamp: float | None = None
    ) -> tuple[EEGSample, ...]: ...


class _EEGPipeline(Protocol):
    def add_sample(self, sample: EEGSample) -> None: ...

    def features(self, start: float, end: float) -> EEGFeatureWindow: ...


class _EEGRecorder(Protocol):
    def record(self, sample: EEGSample) -> None: ...


def _timestamp(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


class GuardianEEGFeatureSource:
    """Drain live Guardian samples causally before computing EEG features.

    The optional recorder is written before each sample reaches the processing
    pipeline.  Guardian acquisition errors, including queue overflow, propagate
    unchanged so the attempt can fail rather than silently use incomplete EEG.
    """

    def __init__(
        self,
        *,
        guardian: _GuardianDrain,
        pipeline: _EEGPipeline,
        recorder: _EEGRecorder | None = None,
    ) -> None:
        if not callable(getattr(guardian, "drain", None)):
            raise TypeError("guardian must expose drain(cutoff_timestamp=...)")
        if not callable(getattr(pipeline, "add_sample", None)) or not callable(
            getattr(pipeline, "features", None)
        ):
            raise TypeError("pipeline must expose add_sample() and features()")
        if recorder is not None and not callable(getattr(recorder, "record", None)):
            raise TypeError("recorder must expose record() or be None")
        self.guardian = guardian
        self.pipeline = pipeline
        self.recorder = recorder
        self.ingested_sample_count = 0
        self.last_ingested_timestamp: float | None = None

    def _ingest(
        self,
        samples: tuple[EEGSample, ...],
        *,
        cutoff_timestamp: float | None,
    ) -> int:
        if not isinstance(samples, tuple) or not all(
            isinstance(sample, EEGSample) for sample in samples
        ):
            raise TypeError("Guardian drain must return a tuple of EEGSample values")
        if cutoff_timestamp is not None and any(
            sample.timestamp > cutoff_timestamp for sample in samples
        ):
            raise RuntimeError("Guardian drain returned an EEG sample after its closed cutoff")
        previous = self.last_ingested_timestamp
        for sample in samples:
            if previous is not None and sample.timestamp < previous:
                raise ValueError("drained Guardian EEG timestamps must be non-decreasing")
            previous = float(sample.timestamp)

        for sample in samples:
            if self.recorder is not None:
                self.recorder.record(sample)
            self.pipeline.add_sample(sample)
            self.ingested_sample_count += 1
            self.last_ingested_timestamp = float(sample.timestamp)
        return len(samples)

    def drain_through(self, cutoff_timestamp: float) -> int:
        """Ingest queued samples through one closed scientific-time cutoff."""

        cutoff = _timestamp("cutoff_timestamp", cutoff_timestamp)
        samples = self.guardian.drain(cutoff_timestamp=cutoff)
        return self._ingest(samples, cutoff_timestamp=cutoff)

    def features(self, start: float, end: float) -> EEGFeatureWindow:
        """Drain exactly through ``end``, then return the closed feature window."""

        window_start = _timestamp("start", start)
        window_end = _timestamp("end", end)
        if window_end < window_start:
            raise ValueError("EEG feature bounds must satisfy 0 <= start <= end")
        self.drain_through(window_end)
        return self.pipeline.features(window_start, window_end)

    def drain_remaining(self) -> int:
        """Ingest all queued samples after Guardian recording has stopped."""

        samples = self.guardian.drain()
        return self._ingest(samples, cutoff_timestamp=None)


if __name__ == "__main__":
    class _Guardian:
        def __init__(self) -> None:
            self.samples = [EEGSample(0.5, 1.0)]

        def drain(self, *, cutoff_timestamp=None):
            ready = tuple(
                sample
                for sample in self.samples
                if cutoff_timestamp is None or sample.timestamp <= cutoff_timestamp
            )
            self.samples = [sample for sample in self.samples if sample not in ready]
            return ready

    class _Pipeline:
        def __init__(self) -> None:
            self.samples = []

        def add_sample(self, sample):
            self.samples.append(sample)

        def features(self, start, end):
            return start, end, len(self.samples)

    demo = GuardianEEGFeatureSource(guardian=_Guardian(), pipeline=_Pipeline())
    print(demo.features(0.0, 0.5))
