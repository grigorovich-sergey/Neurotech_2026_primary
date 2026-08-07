"""Configurable preprocessing, quality gating, and generic EEG features."""

import math
from numbers import Real

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch

from eeg_pipeline.contracts import EEGWindow, QualityState, WindowCompleteness


FEATURE_NAMES = (
    "std_uv",
    "peak_to_peak_uv",
    "mean_abs_diff_uv",
    "delta_power_1_4_hz",
    "theta_power_4_8_hz",
    "alpha_power_8_13_hz",
    "beta_power_13_30_hz",
    "low_gamma_power_30_40_hz",
)

BANDS = (
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 40.0),
)


def _positive(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


class EEGPreprocessor:
    """Apply a zero-phase Butterworth band-pass to one already-bounded window."""

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        low_hz: float = 1.0,
        high_hz: float = 40.0,
        order: int = 4,
    ) -> None:
        self.sample_rate_hz = _positive("sample_rate_hz", sample_rate_hz)
        self.low_hz = _positive("low_hz", low_hz)
        self.high_hz = _positive("high_hz", high_hz)
        if self.low_hz >= self.high_hz:
            raise ValueError("low_hz must be less than high_hz")
        if self.high_hz >= self.sample_rate_hz / 2.0:
            raise ValueError("high_hz must be below the Nyquist frequency")
        if isinstance(order, bool) or not isinstance(order, int) or order <= 0:
            raise ValueError("order must be a positive integer")
        self.order = order
        self._sos = butter(
            self.order,
            [self.low_hz, self.high_hz],
            btype="bandpass",
            fs=self.sample_rate_hz,
            output="sos",
        )

    def process(self, window: EEGWindow) -> np.ndarray:
        values = np.asarray([sample.value_uv for sample in window.samples], dtype=np.float64)
        if values.size == 0:
            raise ValueError("cannot preprocess an empty EEG window")
        try:
            return np.asarray(sosfiltfilt(self._sos, values), dtype=np.float64)
        except ValueError as exc:
            raise ValueError("EEG window is too short for configured zero-phase filtering") from exc


class EEGQualityGate:
    """Small engineering gate for availability and gross single-channel artifacts."""

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        min_duration_seconds: float = 1.0,
        min_coverage: float = 0.95,
        max_gap_seconds: float = 0.006,
        min_std_uv: float = 0.05,
        max_peak_to_peak_uv: float = 1000.0,
    ) -> None:
        self.sample_rate_hz = _positive("sample_rate_hz", sample_rate_hz)
        self.min_duration_seconds = _positive("min_duration_seconds", min_duration_seconds)
        if (
            isinstance(min_coverage, bool)
            or not isinstance(min_coverage, Real)
            or not 0.0 <= min_coverage <= 1.0
        ):
            raise ValueError("min_coverage must be within [0, 1]")
        self.min_coverage = float(min_coverage)
        self.max_gap_seconds = _positive("max_gap_seconds", max_gap_seconds)
        self.min_std_uv = _positive("min_std_uv", min_std_uv)
        self.max_peak_to_peak_uv = _positive("max_peak_to_peak_uv", max_peak_to_peak_uv)

    def evaluate(self, window: EEGWindow) -> tuple[QualityState, tuple[str, ...]]:
        unavailable: list[str] = []
        rejected: list[str] = []
        duration = float(window.requested_end - window.requested_start)
        samples = window.samples

        if window.completeness is WindowCompleteness.EMPTY:
            unavailable.append("window_empty")
        if duration < self.min_duration_seconds:
            unavailable.append("duration_below_minimum")

        expected_count = max(1, int(round(duration * self.sample_rate_hz)) + 1)
        coverage = len(samples) / expected_count
        if coverage < self.min_coverage:
            unavailable.append("coverage_below_minimum")

        if len(samples) >= 2:
            timestamps = np.asarray([sample.timestamp for sample in samples], dtype=np.float64)
            if float(np.max(np.diff(timestamps))) > self.max_gap_seconds:
                unavailable.append("gap_exceeds_maximum")
        if any(not sample.valid for sample in samples):
            rejected.append("explicit_invalid_sample")

        if samples:
            values = np.asarray([sample.value_uv for sample in samples], dtype=np.float64)
            if float(np.std(values)) < self.min_std_uv:
                rejected.append("flatline")
            if float(np.ptp(values)) > self.max_peak_to_peak_uv:
                rejected.append("peak_to_peak_exceeds_maximum")

        reasons = tuple(unavailable + rejected)
        if rejected:
            return QualityState.REJECTED, reasons
        if unavailable:
            return QualityState.UNAVAILABLE, reasons
        return QualityState.USABLE, ()


class EEGFeatureExtractor:
    """Compute fixed-order time statistics and absolute Welch band powers."""

    feature_names = FEATURE_NAMES

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        welch_nperseg_samples: int = 250,
        welch_noverlap_samples: int = 125,
    ) -> None:
        self.sample_rate_hz = _positive("sample_rate_hz", sample_rate_hz)
        if (
            isinstance(welch_nperseg_samples, bool)
            or not isinstance(welch_nperseg_samples, int)
            or welch_nperseg_samples <= 1
        ):
            raise ValueError("welch_nperseg_samples must be an integer greater than one")
        if (
            isinstance(welch_noverlap_samples, bool)
            or not isinstance(welch_noverlap_samples, int)
            or welch_noverlap_samples < 0
            or welch_noverlap_samples >= welch_nperseg_samples
        ):
            raise ValueError("welch_noverlap_samples must satisfy 0 <= overlap < nperseg")
        self.welch_nperseg_samples = welch_nperseg_samples
        self.welch_noverlap_samples = welch_noverlap_samples

    def extract(self, processed_values: np.ndarray) -> np.ndarray:
        values = np.asarray(processed_values, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError("processed_values must contain at least two finite samples")

        nperseg = min(self.welch_nperseg_samples, len(values))
        noverlap = min(self.welch_noverlap_samples, nperseg - 1)
        frequencies, psd = welch(
            values,
            fs=self.sample_rate_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
        band_powers = []
        for low_hz, high_hz in BANDS:
            mask = (frequencies >= low_hz) & (frequencies <= high_hz)
            band_powers.append(float(trapezoid(psd[mask], frequencies[mask])))

        features = np.asarray(
            [
                float(np.std(values)),
                float(np.ptp(values)),
                float(np.mean(np.abs(np.diff(values)))),
                *band_powers,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(features)):
            raise ValueError("feature extraction produced a non-finite value")
        return features


if __name__ == "__main__":
    time = np.arange(251, dtype=np.float64) / 250.0
    values = np.sin(2.0 * np.pi * 10.0 * time)
    print(dict(zip(FEATURE_NAMES, EEGFeatureExtractor(sample_rate_hz=250.0).extract(values))))
