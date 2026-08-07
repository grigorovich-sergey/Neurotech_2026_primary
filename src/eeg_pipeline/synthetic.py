"""Deterministic synthetic EEG used only to exercise the pipeline."""

from collections.abc import Iterable, Iterator, Sequence
import math
from numbers import Real

import numpy as np

from eeg_pipeline.contracts import EEGSample


def _validate_intervals(
    name: str, intervals: Sequence[Sequence[float]], duration_seconds: float
) -> tuple[tuple[float, float], ...]:
    validated: list[tuple[float, float]] = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError(f"{name} intervals must contain [start, end]")
        start, end = interval
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, Real)
            or not isinstance(end, Real)
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or start < 0
            or end < start
            or end > duration_seconds
        ):
            raise ValueError(f"{name} intervals must satisfy 0 <= start <= end <= duration")
        validated.append((float(start), float(end)))
    return tuple(validated)


def _inside(timestamp: float, intervals: Iterable[tuple[float, float]]) -> bool:
    return any(start <= timestamp <= end for start, end in intervals)


def synthetic_eeg_samples(
    *,
    sample_rate_hz: float = 250.0,
    duration_seconds: float = 2.0,
    tones: Sequence[dict[str, float]] = ({"frequency_hz": 10.0, "amplitude_uv": 20.0},),
    noise_std_uv: float = 0.0,
    seed: int = 42,
    gaps: Sequence[Sequence[float]] = (),
    invalid_intervals: Sequence[Sequence[float]] = (),
) -> Iterator[EEGSample]:
    """Yield simple tones with optional explicit gaps and invalid intervals."""

    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, Real)
        or not math.isfinite(float(sample_rate_hz))
        or sample_rate_hz <= 0
    ):
        raise ValueError("sample_rate_hz must be positive and finite")
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, Real)
        or not math.isfinite(float(duration_seconds))
        or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be positive and finite")
    if (
        isinstance(noise_std_uv, bool)
        or not isinstance(noise_std_uv, Real)
        or not math.isfinite(float(noise_std_uv))
        or noise_std_uv < 0
    ):
        raise ValueError("noise_std_uv must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    parsed_tones: list[tuple[float, float]] = []
    for tone in tones:
        if not isinstance(tone, dict) or set(tone) != {"frequency_hz", "amplitude_uv"}:
            raise ValueError("each tone must contain frequency_hz and amplitude_uv")
        frequency_hz = tone["frequency_hz"]
        amplitude_uv = tone["amplitude_uv"]
        if (
            isinstance(frequency_hz, bool)
            or isinstance(amplitude_uv, bool)
            or not isinstance(frequency_hz, Real)
            or not isinstance(amplitude_uv, Real)
            or not math.isfinite(float(frequency_hz))
            or not math.isfinite(float(amplitude_uv))
            or frequency_hz <= 0
            or frequency_hz >= float(sample_rate_hz) / 2.0
        ):
            raise ValueError("tone frequency/amplitude values are invalid")
        parsed_tones.append((float(frequency_hz), float(amplitude_uv)))

    parsed_gaps = _validate_intervals("gap", gaps, float(duration_seconds))
    parsed_invalid = _validate_intervals(
        "invalid", invalid_intervals, float(duration_seconds)
    )
    rng = np.random.default_rng(seed)
    count = int(math.floor(float(duration_seconds) * float(sample_rate_hz) + 1e-9)) + 1
    for index in range(count):
        timestamp = index / float(sample_rate_hz)
        if _inside(timestamp, parsed_gaps):
            continue
        value = sum(
            amplitude * math.sin(2.0 * math.pi * frequency * timestamp)
            for frequency, amplitude in parsed_tones
        )
        if noise_std_uv:
            value += float(rng.normal(0.0, float(noise_std_uv)))
        yield EEGSample(
            timestamp=timestamp,
            value_uv=value,
            valid=not _inside(timestamp, parsed_invalid),
        )


if __name__ == "__main__":
    samples = list(synthetic_eeg_samples(duration_seconds=0.02))
    print(samples)
