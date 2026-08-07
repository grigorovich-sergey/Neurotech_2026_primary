"""Bounded timestamp-addressable raw EEG buffering."""

from collections import deque
import math
from numbers import Real

from eeg_pipeline.contracts import EEGSample, EEGWindow, WindowCompleteness


class EEGBuffer:
    """Retain recent ordered raw samples and return closed time windows."""

    def __init__(self, history_seconds: float = 30.0) -> None:
        if (
            isinstance(history_seconds, bool)
            or not isinstance(history_seconds, Real)
            or not math.isfinite(float(history_seconds))
            or history_seconds <= 0
        ):
            raise ValueError("history_seconds must be a positive finite number")
        self.history_seconds = float(history_seconds)
        self._samples: deque[EEGSample] = deque()

    def add(self, sample: EEGSample) -> None:
        if not isinstance(sample, EEGSample):
            raise TypeError("sample must be an EEGSample")
        if self._samples and sample.timestamp < self._samples[-1].timestamp:
            raise ValueError("EEG timestamps must be non-decreasing")
        self._samples.append(sample)
        cutoff = float(sample.timestamp) - self.history_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def window(self, start: float, end: float) -> EEGWindow:
        if isinstance(start, bool) or not isinstance(start, Real) or not math.isfinite(float(start)):
            raise ValueError("window start must be a finite real number")
        if isinstance(end, bool) or not isinstance(end, Real) or not math.isfinite(float(end)):
            raise ValueError("window end must be a finite real number")
        if start < 0 or end < start:
            raise ValueError("window bounds must satisfy 0 <= start <= end")

        selected = tuple(
            sample for sample in self._samples if start <= sample.timestamp <= end
        )
        if not selected:
            completeness = WindowCompleteness.EMPTY
            actual_start = None
            actual_end = None
        else:
            actual_start = float(selected[0].timestamp)
            actual_end = float(selected[-1].timestamp)
            covers_request = bool(
                self._samples
                and self._samples[0].timestamp <= start
                and self._samples[-1].timestamp >= end
            )
            completeness = (
                WindowCompleteness.COMPLETE
                if covers_request
                else WindowCompleteness.PARTIAL
            )
        return EEGWindow(
            requested_start=float(start),
            requested_end=float(end),
            samples=selected,
            actual_start=actual_start,
            actual_end=actual_end,
            completeness=completeness,
        )

    def __len__(self) -> int:
        return len(self._samples)


if __name__ == "__main__":
    buffer = EEGBuffer(1.0)
    for timestamp in (0.0, 0.5, 1.0):
        buffer.add(EEGSample(timestamp, timestamp))
    print(buffer.window(0.0, 0.5))
