"""Run-relative clocks used by live and deterministic workflows."""

from dataclasses import dataclass, field
import math
import time


@dataclass
class MonotonicClock:
    """Clock whose zero is construction time and source is perf_counter()."""

    _origin: float = field(default_factory=time.perf_counter)

    def now(self) -> float:
        return time.perf_counter() - self._origin


@dataclass
class ManualClock:
    """Explicit clock for deterministic simulation and tests."""

    _time: float = 0.0

    def __post_init__(self) -> None:
        self.set(self._time)

    def now(self) -> float:
        return self._time

    def set(self, timestamp: float) -> None:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if timestamp < self._time:
            raise ValueError("manual clock cannot move backwards")
        self._time = timestamp

    def advance(self, seconds: float) -> None:
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("advance must be finite and non-negative")
        self._time += seconds


if __name__ == "__main__":
    clock = ManualClock()
    clock.advance(0.25)
    print(clock.now())
