"""Approved one-dimensional EEG indicator used by the experimental policy.

This deliberately small module is the single place to change the scientific
formula if the project later replaces the engagement-index hypothesis.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Mapping


ENGAGEMENT_INDEX_ID = "engagement_beta_over_alpha_plus_theta_v1"
ENGAGEMENT_INDEX_FORMULA = "beta_power_13_30_hz / (alpha_power_8_13_hz + theta_power_4_8_hz)"
ENGAGEMENT_INDEX_INPUTS = (
    "beta_power_13_30_hz",
    "alpha_power_8_13_hz",
    "theta_power_4_8_hz",
)


class InvalidEEGIndicator(ValueError):
    """The supplied EEG features cannot produce the approved indicator."""


def engagement_index(eeg_features: Mapping[str, float]) -> float:
    """Return beta / (alpha + theta), with no hidden epsilon or clipping."""

    try:
        beta = eeg_features["beta_power_13_30_hz"]
        alpha = eeg_features["alpha_power_8_13_hz"]
        theta = eeg_features["theta_power_4_8_hz"]
    except KeyError as exc:
        raise InvalidEEGIndicator(f"missing EEG feature: {exc.args[0]}") from exc
    values = (beta, alpha, theta)
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise InvalidEEGIndicator("engagement-index inputs must be finite real numbers")
    denominator = float(alpha) + float(theta)
    if denominator <= 0.0:
        raise InvalidEEGIndicator("alpha power + theta power must be positive")
    result = float(beta) / denominator
    if not math.isfinite(result) or result < 0.0:
        raise InvalidEEGIndicator("engagement index must be finite and non-negative")
    return result


if __name__ == "__main__":
    print(
        engagement_index(
            {
                "beta_power_13_30_hz": 6.0,
                "alpha_power_8_13_hz": 2.0,
                "theta_power_4_8_hz": 1.0,
            }
        )
    )
