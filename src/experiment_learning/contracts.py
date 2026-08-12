"""Project-owned contracts for frozen-session experimental learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any, Mapping

from gaze_interaction.contracts import BoundingBox


EPISODE_RECORD_SCHEMA = "experiment_episode_training_record_v1"
COMPLETED_SESSION_SCHEMA = "experiment_completed_session_v1"


class Condition(str, Enum):
    """Policy allowed to control visible dwell behavior in one session."""

    G = "G"
    E = "E"


class OutcomeStatus(str, Enum):
    ACTION = "action"
    NO_ACTION = "no_action"
    COUNTERFACTUAL_CENSORED = "counterfactual_censored"
    NOT_SCORED = "not_scored"


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(name: str, value: str) -> str:
    _non_empty(name, value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _timestamp(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _optional_timestamp(name: str, value: float | None) -> float | None:
    return None if value is None else _timestamp(name, value)


def _positive(name: str, value: float) -> float:
    result = _timestamp(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _unit(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    result = _timestamp(name, value)
    if result > 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _binary_optional(name: str, value: int | None) -> int | None:
    if value is not None and (isinstance(value, bool) or value not in (0, 1)):
        raise ValueError(f"{name} must be 0, 1, or None")
    return value


@dataclass(frozen=True)
class GazeContextObservation:
    """Current-match context available at an episode update."""

    episode_id: int
    track_id: int
    timestamp: float
    matched_dwell_s: float
    gaze_x_normalized: float
    gaze_y_normalized: float
    candidate_box: BoundingBox

    def __post_init__(self) -> None:
        _positive_int("episode_id", self.episode_id)
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        object.__setattr__(self, "timestamp", _timestamp("timestamp", self.timestamp))
        object.__setattr__(
            self, "matched_dwell_s", _timestamp("matched_dwell_s", self.matched_dwell_s)
        )
        object.__setattr__(
            self, "gaze_x_normalized", _unit("gaze_x_normalized", self.gaze_x_normalized)
        )
        object.__setattr__(
            self, "gaze_y_normalized", _unit("gaze_y_normalized", self.gaze_y_normalized)
        )
        if not isinstance(self.candidate_box, BoundingBox):
            raise TypeError("candidate_box must be a BoundingBox")


@dataclass(frozen=True)
class DwellPolicyParameters:
    """Values passed unchanged into Instance 2's DwellController."""

    baseline_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    maximum_reduction_fraction: float

    def __post_init__(self) -> None:
        for name in ("baseline_seconds", "minimum_seconds", "maximum_seconds"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        if not self.minimum_seconds <= self.baseline_seconds <= self.maximum_seconds:
            raise ValueError("minimum <= baseline <= maximum is required")
        reduction = _unit("maximum_reduction_fraction", self.maximum_reduction_fraction)
        assert reduction is not None
        object.__setattr__(self, "maximum_reduction_fraction", reduction)


@dataclass(frozen=True)
class PolicyDecision:
    """Frozen episode decision consumed by adaptive dwell on later updates."""

    episode_id: int
    intent_score: float | None
    required_dwell_s: float
    eeg_probability: float | None
    engagement_index: float | None
    training_eligible: bool
    reason: str
    newly_frozen: bool = False

    def __post_init__(self) -> None:
        _positive_int("episode_id", self.episode_id)
        object.__setattr__(self, "intent_score", _unit("intent_score", self.intent_score))
        object.__setattr__(
            self, "eeg_probability", _unit("eeg_probability", self.eeg_probability)
        )
        object.__setattr__(self, "required_dwell_s", _positive("required_dwell_s", self.required_dwell_s))
        if self.engagement_index is not None:
            value = _timestamp("engagement_index", self.engagement_index)
            object.__setattr__(self, "engagement_index", value)
        if not isinstance(self.training_eligible, bool) or not isinstance(self.newly_frozen, bool):
            raise TypeError("eligibility/freeze fields must be bool values")
        _non_empty("reason", self.reason)


@dataclass(frozen=True)
class TrajectoryPoint:
    timestamp: float
    accumulated_matched_dwell_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _timestamp("timestamp", self.timestamp))
        object.__setattr__(
            self,
            "accumulated_matched_dwell_s",
            _timestamp("accumulated_matched_dwell_s", self.accumulated_matched_dwell_s),
        )

    def to_payload(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TrajectoryPoint":
        return cls(
            timestamp=payload["timestamp"],
            accumulated_matched_dwell_s=payload["accumulated_matched_dwell_s"],
        )


@dataclass(frozen=True)
class ModelOutcome:
    status: OutcomeStatus
    crossing_timestamp: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, OutcomeStatus):
            raise TypeError("status must be an OutcomeStatus")
        object.__setattr__(
            self,
            "crossing_timestamp",
            _optional_timestamp("crossing_timestamp", self.crossing_timestamp),
        )
        if (self.status is OutcomeStatus.ACTION) != (self.crossing_timestamp is not None):
            raise ValueError("only action outcomes have a crossing timestamp")

    def to_payload(self) -> dict[str, Any]:
        return {"status": self.status.value, "crossing_timestamp": self.crossing_timestamp}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ModelOutcome":
        return cls(OutcomeStatus(payload["status"]), payload.get("crossing_timestamp"))


@dataclass(frozen=True)
class EpisodeTrainingRecord:
    """One resolved episode containing all deterministic trainer inputs."""

    participant_id: str
    session_id: str
    session_number: int
    attempt_id: str
    episode_id: int
    track_id: int
    active_condition: Condition
    policy_sha256: str
    episode_start_timestamp: float
    prediction_cutoff_timestamp: float | None
    eeg_window_start: float | None
    eeg_window_end: float | None
    eeg_quality_state: str
    eeg_quality_reasons: tuple[str, ...]
    eeg_feature_names: tuple[str, ...]
    eeg_feature_values: tuple[float, ...] | None
    eeg_indicator_id: str
    eeg_indicator_formula: str
    engagement_index: float | None
    eeg_probability: float | None
    eeg_evidence: float | None
    g_required_dwell_s: float
    e_required_dwell_s: float
    trajectory: tuple[TrajectoryPoint, ...]
    action_occurred: bool | None
    action_timestamp: float | None
    natural_endpoint_timestamp: float | None
    feedback_window_open: float | None
    feedback_deadline: float | None
    feedback_pressed: bool | None
    feedback_resolution_timestamp: float | None
    common_label: int | None
    instructed_intention: int | None
    g_outcome: ModelOutcome
    e_outcome: ModelOutcome
    training_eligible: bool
    exclusion_reasons: tuple[str, ...]
    canceled: bool
    cancellation_reason: str | None

    def __post_init__(self) -> None:
        _non_empty("participant_id", self.participant_id)
        _non_empty("session_id", self.session_id)
        _non_empty("attempt_id", self.attempt_id)
        _positive_int("session_number", self.session_number)
        _positive_int("episode_id", self.episode_id)
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if not isinstance(self.active_condition, Condition):
            raise TypeError("active_condition must be a Condition")
        _sha256("policy_sha256", self.policy_sha256)
        object.__setattr__(
            self,
            "episode_start_timestamp",
            _timestamp("episode_start_timestamp", self.episode_start_timestamp),
        )
        for name in (
            "prediction_cutoff_timestamp",
            "eeg_window_start",
            "eeg_window_end",
            "action_timestamp",
            "natural_endpoint_timestamp",
            "feedback_window_open",
            "feedback_deadline",
            "feedback_resolution_timestamp",
        ):
            object.__setattr__(self, name, _optional_timestamp(name, getattr(self, name)))
        if self.eeg_window_start is not None and self.eeg_window_end is not None:
            if self.eeg_window_end < self.eeg_window_start:
                raise ValueError("EEG window end cannot precede its start")
            if (
                self.prediction_cutoff_timestamp is not None
                and self.eeg_window_end > self.prediction_cutoff_timestamp
            ):
                raise ValueError("EEG window cannot extend beyond its causal cutoff")
        _non_empty("eeg_quality_state", self.eeg_quality_state)
        if not isinstance(self.eeg_quality_reasons, tuple) or not all(
            isinstance(reason, str) and reason for reason in self.eeg_quality_reasons
        ):
            raise TypeError("eeg_quality_reasons must contain non-empty strings")
        if not isinstance(self.eeg_feature_names, tuple) or not all(
            isinstance(name, str) and name for name in self.eeg_feature_names
        ):
            raise TypeError("eeg_feature_names must contain non-empty strings")
        if self.eeg_feature_values is not None:
            if len(self.eeg_feature_values) != len(self.eeg_feature_names):
                raise ValueError("EEG feature names and values differ in length")
            if not all(math.isfinite(float(value)) for value in self.eeg_feature_values):
                raise ValueError("EEG feature values must be finite")
        _non_empty("eeg_indicator_id", self.eeg_indicator_id)
        _non_empty("eeg_indicator_formula", self.eeg_indicator_formula)
        if self.engagement_index is not None:
            object.__setattr__(
                self, "engagement_index", _timestamp("engagement_index", self.engagement_index)
            )
        object.__setattr__(self, "eeg_probability", _unit("eeg_probability", self.eeg_probability))
        object.__setattr__(self, "eeg_evidence", _unit("eeg_evidence", self.eeg_evidence))
        object.__setattr__(
            self, "g_required_dwell_s", _positive("g_required_dwell_s", self.g_required_dwell_s)
        )
        object.__setattr__(
            self, "e_required_dwell_s", _positive("e_required_dwell_s", self.e_required_dwell_s)
        )
        if not isinstance(self.trajectory, tuple) or not all(
            isinstance(point, TrajectoryPoint) for point in self.trajectory
        ):
            raise TypeError("trajectory must be a tuple of TrajectoryPoint values")
        if any(
            later.timestamp < earlier.timestamp
            or later.accumulated_matched_dwell_s < earlier.accumulated_matched_dwell_s
            for earlier, later in zip(self.trajectory, self.trajectory[1:])
        ):
            raise ValueError("trajectory must be monotonic")
        if self.action_occurred is not None and not isinstance(self.action_occurred, bool):
            raise TypeError("action_occurred must be bool or None")
        if self.feedback_pressed is not None and not isinstance(self.feedback_pressed, bool):
            raise TypeError("feedback_pressed must be bool or None")
        _binary_optional("common_label", self.common_label)
        _binary_optional("instructed_intention", self.instructed_intention)
        if not isinstance(self.g_outcome, ModelOutcome) or not isinstance(self.e_outcome, ModelOutcome):
            raise TypeError("G/E outcomes must be ModelOutcome values")
        if not isinstance(self.training_eligible, bool) or not isinstance(self.canceled, bool):
            raise TypeError("training_eligible and canceled must be bool values")
        if not isinstance(self.exclusion_reasons, tuple) or not all(
            isinstance(reason, str) and reason for reason in self.exclusion_reasons
        ):
            raise TypeError("exclusion_reasons must contain non-empty strings")
        if self.training_eligible and (self.common_label is None or self.exclusion_reasons):
            raise ValueError("training-eligible records require a label and no exclusions")
        if self.canceled != (self.cancellation_reason is not None):
            raise ValueError("cancellation flag and reason must agree")

    @property
    def identity(self) -> tuple[int, str, int]:
        return (self.session_number, self.attempt_id, self.episode_id)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = EPISODE_RECORD_SCHEMA
        payload["active_condition"] = self.active_condition.value
        payload["eeg_quality_reasons"] = list(self.eeg_quality_reasons)
        payload["eeg_feature_names"] = list(self.eeg_feature_names)
        payload["eeg_feature_values"] = (
            None if self.eeg_feature_values is None else list(self.eeg_feature_values)
        )
        payload["trajectory"] = [point.to_payload() for point in self.trajectory]
        payload["g_outcome"] = self.g_outcome.to_payload()
        payload["e_outcome"] = self.e_outcome.to_payload()
        payload["exclusion_reasons"] = list(self.exclusion_reasons)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeTrainingRecord":
        if payload.get("schema") != EPISODE_RECORD_SCHEMA:
            raise ValueError("episode record schema is incompatible")
        values = dict(payload)
        values.pop("schema", None)
        values["active_condition"] = Condition(values["active_condition"])
        for name in ("eeg_quality_reasons", "eeg_feature_names", "exclusion_reasons"):
            values[name] = tuple(values[name])
        if values["eeg_feature_values"] is not None:
            values["eeg_feature_values"] = tuple(values["eeg_feature_values"])
        values["trajectory"] = tuple(
            TrajectoryPoint.from_payload(point) for point in values["trajectory"]
        )
        values["g_outcome"] = ModelOutcome.from_payload(values["g_outcome"])
        values["e_outcome"] = ModelOutcome.from_payload(values["e_outcome"])
        return cls(**values)


@dataclass(frozen=True)
class CancellationInstruction:
    episode_id: int
    timestamp: float
    reason: str = "feedback_accepted"

    def __post_init__(self) -> None:
        _positive_int("episode_id", self.episode_id)
        object.__setattr__(self, "timestamp", _timestamp("timestamp", self.timestamp))
        _non_empty("reason", self.reason)


@dataclass(frozen=True)
class FeedbackResolution:
    record: EpisodeTrainingRecord
    cancellation_instructions: tuple[CancellationInstruction, ...]


@dataclass(frozen=True)
class CompletedSession:
    """Immutable trainer input created only after a successful session close."""

    participant_id: str
    session_id: str
    session_number: int
    attempt_id: str
    active_condition: Condition
    policy_sha256: str
    schedule_sequence_id: str
    schedule_sha256: str
    completed_timestamp: float
    successful: bool
    records: tuple[EpisodeTrainingRecord, ...]

    def __post_init__(self) -> None:
        _non_empty("participant_id", self.participant_id)
        _non_empty("session_id", self.session_id)
        _positive_int("session_number", self.session_number)
        _non_empty("attempt_id", self.attempt_id)
        if not isinstance(self.active_condition, Condition):
            raise TypeError("active_condition must be a Condition")
        _sha256("policy_sha256", self.policy_sha256)
        _non_empty("schedule_sequence_id", self.schedule_sequence_id)
        _sha256("schedule_sha256", self.schedule_sha256)
        object.__setattr__(
            self, "completed_timestamp", _timestamp("completed_timestamp", self.completed_timestamp)
        )
        if not isinstance(self.successful, bool):
            raise TypeError("successful must be a bool")
        if not isinstance(self.records, tuple) or not all(
            isinstance(record, EpisodeTrainingRecord) for record in self.records
        ):
            raise TypeError("records must be a tuple of EpisodeTrainingRecord values")
        if any(record.participant_id != self.participant_id for record in self.records):
            raise ValueError("session contains a cross-participant episode record")
        if any(
            record.session_number != self.session_number
            or record.attempt_id != self.attempt_id
            or record.session_id != self.session_id
            for record in self.records
        ):
            raise ValueError("session and episode record identities differ")
        if any(
            record.policy_sha256 != self.policy_sha256
            or record.active_condition is not self.active_condition
            for record in self.records
        ):
            raise ValueError("session and episode policy/condition bindings differ")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": COMPLETED_SESSION_SCHEMA,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "session_number": self.session_number,
            "attempt_id": self.attempt_id,
            "active_condition": self.active_condition.value,
            "policy_sha256": self.policy_sha256,
            "schedule_sequence_id": self.schedule_sequence_id,
            "schedule_sha256": self.schedule_sha256,
            "completed_timestamp": self.completed_timestamp,
            "successful": self.successful,
            "records": [record.to_payload() for record in self.records],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CompletedSession":
        if payload.get("schema") != COMPLETED_SESSION_SCHEMA:
            raise ValueError("completed-session schema is incompatible")
        return cls(
            participant_id=payload["participant_id"],
            session_id=payload["session_id"],
            session_number=payload["session_number"],
            attempt_id=payload["attempt_id"],
            active_condition=Condition(payload["active_condition"]),
            policy_sha256=payload["policy_sha256"],
            schedule_sequence_id=payload["schedule_sequence_id"],
            schedule_sha256=payload["schedule_sha256"],
            completed_timestamp=payload["completed_timestamp"],
            successful=payload["successful"],
            records=tuple(EpisodeTrainingRecord.from_payload(item) for item in payload["records"]),
        )


if __name__ == "__main__":
    box = BoundingBox(0.2, 0.2, 0.8, 0.8)
    print(GazeContextObservation(1, 1, 1.0, 0.25, 0.5, 0.5, box))
