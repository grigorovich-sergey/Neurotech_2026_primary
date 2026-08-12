"""Immutable participant/session policy artifacts and runtime evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from eeg_pipeline.processing import FEATURE_NAMES as EEG_FEATURE_NAMES

from experiment_learning.artifacts import immutable_write_json, load_json_object
from experiment_learning.contracts import Condition, DwellPolicyParameters
from experiment_learning.eeg_indicator import (
    ENGAGEMENT_INDEX_FORMULA,
    ENGAGEMENT_INDEX_ID,
)


POLICY_SCHEMA = "experiment_policy_v1"


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(name: str, value: str) -> str:
    _non_empty(name, value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(name: str, value: int, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


@dataclass(frozen=True)
class SourceAttempt:
    session_number: int
    attempt_id: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _positive_int("session_number", self.session_number)
        _non_empty("attempt_id", self.attempt_id)
        _sha256("artifact_sha256", self.artifact_sha256)

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_number": self.session_number,
            "attempt_id": self.attempt_id,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True)
class FrozenSessionPolicy:
    """All policy values are fixed for exactly one participant session."""

    participant_id: str
    policy_for_session: int
    trained_through_session: int
    schedule_sequence_id: str
    schedule_sha256: str
    source_attempts: tuple[SourceAttempt, ...]
    g_base_threshold_s: float
    minimum_e_threshold_s: float
    maximum_eeg_reduction_fraction: float
    engagement_mean: float
    engagement_scale: float
    logistic_intercept: float
    logistic_coefficient: float
    base_search_min_s: float
    base_search_max_s: float
    base_search_step_s: float
    maximum_allowed_reduction_fraction: float
    cold_start_status: str
    fitted_example_count: int
    eeg_feature_signature: tuple[str, ...] = EEG_FEATURE_NAMES
    eeg_indicator_id: str = ENGAGEMENT_INDEX_ID
    eeg_indicator_formula: str = ENGAGEMENT_INDEX_FORMULA

    def __post_init__(self) -> None:
        _non_empty("participant_id", self.participant_id)
        _positive_int("policy_for_session", self.policy_for_session)
        _positive_int("trained_through_session", self.trained_through_session, allow_zero=True)
        if self.trained_through_session != self.policy_for_session - 1:
            raise ValueError("policy must be trained through the preceding session")
        _non_empty("schedule_sequence_id", self.schedule_sequence_id)
        _sha256("schedule_sha256", self.schedule_sha256)
        if not isinstance(self.source_attempts, tuple) or not all(
            isinstance(item, SourceAttempt) for item in self.source_attempts
        ):
            raise TypeError("source_attempts must be a tuple of SourceAttempt values")
        if tuple(item.session_number for item in self.source_attempts) != tuple(
            range(1, self.trained_through_session + 1)
        ):
            raise ValueError("source attempts must cover successful sessions 1..N exactly")
        for name in (
            "g_base_threshold_s",
            "minimum_e_threshold_s",
            "engagement_scale",
            "base_search_min_s",
            "base_search_max_s",
            "base_search_step_s",
        ):
            value = _finite(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("engagement_mean", "logistic_intercept", "logistic_coefficient"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        for name in (
            "maximum_eeg_reduction_fraction",
            "maximum_allowed_reduction_fraction",
        ):
            value = _finite(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)
        if self.maximum_eeg_reduction_fraction > self.maximum_allowed_reduction_fraction:
            raise ValueError("selected EEG reduction exceeds the artifact bound")
        if not self.base_search_min_s <= self.g_base_threshold_s <= self.base_search_max_s:
            raise ValueError("G threshold lies outside the artifact search bounds")
        if self.minimum_e_threshold_s > self.g_base_threshold_s:
            raise ValueError("minimum E threshold cannot exceed the G base")
        _non_empty("cold_start_status", self.cold_start_status)
        _positive_int("fitted_example_count", self.fitted_example_count, allow_zero=True)
        if tuple(self.eeg_feature_signature) != EEG_FEATURE_NAMES:
            raise ValueError("EEG feature signature differs from Instance 3")
        if self.eeg_indicator_id != ENGAGEMENT_INDEX_ID:
            raise ValueError("EEG indicator identifier is incompatible")
        if self.eeg_indicator_formula != ENGAGEMENT_INDEX_FORMULA:
            raise ValueError("EEG indicator formula is incompatible")

    def eeg_probability(self, engagement_value: float) -> float:
        value = _finite("engagement_value", engagement_value)
        if value < 0.0:
            raise ValueError("engagement_value must be non-negative")
        standardized = (value - self.engagement_mean) / self.engagement_scale
        linear = self.logistic_intercept + self.logistic_coefficient * standardized
        if linear >= 0.0:
            return 1.0 / (1.0 + math.exp(-linear))
        exponential = math.exp(linear)
        return exponential / (1.0 + exponential)

    @staticmethod
    def positive_eeg_evidence(probability: float) -> float:
        value = _finite("probability", probability)
        if not 0.0 <= value <= 1.0:
            raise ValueError("probability must be within [0, 1]")
        return max(0.0, 2.0 * value - 1.0)

    def e_required_dwell(self, eeg_evidence: float) -> float:
        evidence = _finite("eeg_evidence", eeg_evidence)
        if not 0.0 <= evidence <= 1.0:
            raise ValueError("eeg_evidence must be within [0, 1]")
        return max(
            self.minimum_e_threshold_s,
            self.g_base_threshold_s
            * (1.0 - self.maximum_eeg_reduction_fraction * evidence),
        )

    def dwell_parameters(self, condition: Condition) -> DwellPolicyParameters:
        if not isinstance(condition, Condition):
            raise TypeError("condition must be a Condition")
        return DwellPolicyParameters(
            baseline_seconds=self.g_base_threshold_s,
            minimum_seconds=self.minimum_e_threshold_s,
            maximum_seconds=self.g_base_threshold_s,
            maximum_reduction_fraction=(
                self.maximum_eeg_reduction_fraction if condition is Condition.E else 0.0
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": POLICY_SCHEMA,
            "participant_id": self.participant_id,
            "policy_for_session": self.policy_for_session,
            "trained_through_session": self.trained_through_session,
            "schedule_binding": {
                "sequence_id": self.schedule_sequence_id,
                "csv_sha256": self.schedule_sha256,
            },
            "source_attempts": [item.to_payload() for item in self.source_attempts],
            "gaze_policy": {"base_threshold_s": self.g_base_threshold_s},
            "eeg_policy": {
                "feature_signature": list(self.eeg_feature_signature),
                "indicator": {
                    "id": self.eeg_indicator_id,
                    "formula": self.eeg_indicator_formula,
                },
                "normalization": {
                    "mean": self.engagement_mean,
                    "scale": self.engagement_scale,
                },
                "logistic": {
                    "intercept": self.logistic_intercept,
                    "coefficient": self.logistic_coefficient,
                },
                "adjustment": {
                    "evidence_formula": "max(0, 2 * P_EEG - 1)",
                    "threshold_formula": "max(minimum_e_threshold_s, g_base_threshold_s * (1 - reduction_fraction * evidence))",
                    "minimum_e_threshold_s": self.minimum_e_threshold_s,
                    "reduction_fraction": self.maximum_eeg_reduction_fraction,
                },
            },
            "bounds": {
                "base_search_min_s": self.base_search_min_s,
                "base_search_max_s": self.base_search_max_s,
                "base_search_step_s": self.base_search_step_s,
                "maximum_allowed_reduction_fraction": self.maximum_allowed_reduction_fraction,
            },
            "training_state": {
                "cold_start_status": self.cold_start_status,
                "fitted_example_count": self.fitted_example_count,
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenSessionPolicy":
        if payload.get("schema") != POLICY_SCHEMA:
            raise ValueError("policy schema is incompatible")
        schedule = payload["schedule_binding"]
        gaze = payload["gaze_policy"]
        eeg = payload["eeg_policy"]
        indicator = eeg["indicator"]
        normalization = eeg["normalization"]
        logistic = eeg["logistic"]
        adjustment = eeg["adjustment"]
        bounds = payload["bounds"]
        training = payload["training_state"]
        return cls(
            participant_id=payload["participant_id"],
            policy_for_session=payload["policy_for_session"],
            trained_through_session=payload["trained_through_session"],
            schedule_sequence_id=schedule["sequence_id"],
            schedule_sha256=schedule["csv_sha256"],
            source_attempts=tuple(SourceAttempt(**item) for item in payload["source_attempts"]),
            g_base_threshold_s=gaze["base_threshold_s"],
            minimum_e_threshold_s=adjustment["minimum_e_threshold_s"],
            maximum_eeg_reduction_fraction=adjustment["reduction_fraction"],
            engagement_mean=normalization["mean"],
            engagement_scale=normalization["scale"],
            logistic_intercept=logistic["intercept"],
            logistic_coefficient=logistic["coefficient"],
            base_search_min_s=bounds["base_search_min_s"],
            base_search_max_s=bounds["base_search_max_s"],
            base_search_step_s=bounds["base_search_step_s"],
            maximum_allowed_reduction_fraction=bounds[
                "maximum_allowed_reduction_fraction"
            ],
            cold_start_status=training["cold_start_status"],
            fitted_example_count=training["fitted_example_count"],
            eeg_feature_signature=tuple(eeg["feature_signature"]),
            eeg_indicator_id=indicator["id"],
            eeg_indicator_formula=indicator["formula"],
        )


def create_cold_start_policy(
    *,
    participant_id: str,
    schedule_sequence_id: str,
    schedule_sha256: str,
    base_threshold_s: float = 1.0,
    minimum_e_threshold_s: float = 0.35,
    base_search_min_s: float = 0.5,
    base_search_max_s: float = 1.5,
    base_search_step_s: float = 0.05,
    maximum_allowed_reduction_fraction: float = 0.5,
) -> FrozenSessionPolicy:
    return FrozenSessionPolicy(
        participant_id=participant_id,
        policy_for_session=1,
        trained_through_session=0,
        schedule_sequence_id=schedule_sequence_id,
        schedule_sha256=schedule_sha256,
        source_attempts=(),
        g_base_threshold_s=base_threshold_s,
        minimum_e_threshold_s=minimum_e_threshold_s,
        maximum_eeg_reduction_fraction=0.0,
        engagement_mean=0.0,
        engagement_scale=1.0,
        logistic_intercept=0.0,
        logistic_coefficient=0.0,
        base_search_min_s=base_search_min_s,
        base_search_max_s=base_search_max_s,
        base_search_step_s=base_search_step_s,
        maximum_allowed_reduction_fraction=maximum_allowed_reduction_fraction,
        cold_start_status="initial_session_no_eeg_reduction",
        fitted_example_count=0,
    )


def save_frozen_policy(path: str | Path, policy: FrozenSessionPolicy) -> str:
    return immutable_write_json(path, policy.to_payload())


def load_frozen_policy(
    path: str | Path,
    *,
    expected_participant_id: str,
    expected_session: int,
    expected_sha256: str | None = None,
) -> FrozenSessionPolicy:
    source = Path(path)
    content = source.read_bytes()
    if source.suffix.lower() in {".pkl", ".pickle"}:
        raise ValueError("legacy River pickle checkpoints are not supported")
    if expected_sha256 is not None:
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected_sha256:
            raise ValueError("policy artifact digest differs from the persisted attempt binding")
    policy = FrozenSessionPolicy.from_payload(load_json_object(source))
    if policy.participant_id != expected_participant_id:
        raise ValueError("policy participant does not match the requested participant")
    if policy.policy_for_session != expected_session:
        raise ValueError("policy was not created for the requested session")
    return policy


if __name__ == "__main__":
    policy = create_cold_start_policy(
        participant_id="P001", schedule_sequence_id="seq-a", schedule_sha256="0" * 64
    )
    print(policy.e_required_dwell(policy.positive_eeg_evidence(0.75)))
