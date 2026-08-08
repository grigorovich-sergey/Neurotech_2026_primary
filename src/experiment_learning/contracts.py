"""Project-owned contracts for Instance 4 experimental learning."""

from dataclasses import asdict, dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any

from gaze_interaction.contracts import BoundingBox


class Condition(str, Enum):
    """Model allowed to control user-visible dwell behavior in one session."""

    G = "G"
    E = "E"


def _non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _timestamp(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _unit(name: str, value: float) -> float:
    result = _timestamp(name, value)
    if result > 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


def _probability(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _unit(name, value)


def _binary_optional(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or value not in (0, 1)):
        raise ValueError(f"{name} must be 0, 1, or None")


@dataclass(frozen=True)
class GazeContextObservation:
    """Current-match context available at a candidate prediction cutoff."""

    episode_id: int
    track_id: int
    timestamp: float
    matched_dwell_s: float
    gaze_x_normalized: float
    gaze_y_normalized: float
    candidate_box: BoundingBox

    def __post_init__(self) -> None:
        if isinstance(self.episode_id, bool) or not isinstance(self.episode_id, int):
            raise TypeError("episode_id must be an integer")
        if self.episode_id <= 0:
            raise ValueError("episode_id must be positive")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise TypeError("track_id must be an integer")
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        object.__setattr__(self, "timestamp", _timestamp("timestamp", self.timestamp))
        object.__setattr__(
            self, "matched_dwell_s", _timestamp("matched_dwell_s", self.matched_dwell_s)
        )
        object.__setattr__(
            self,
            "gaze_x_normalized",
            _unit("gaze_x_normalized", self.gaze_x_normalized),
        )
        object.__setattr__(
            self,
            "gaze_y_normalized",
            _unit("gaze_y_normalized", self.gaze_y_normalized),
        )
        if not isinstance(self.candidate_box, BoundingBox):
            raise TypeError("candidate_box must be a BoundingBox")


@dataclass(frozen=True)
class PredictionRecord:
    """Frozen paired prediction, or explicit paired unavailability, at one cutoff."""

    participant_id: str
    session_id: str
    episode_id: int
    track_id: int
    active_condition: Condition
    cutoff_timestamp: float
    g_probability: float | None
    e_probability: float | None
    eeg_window_start: float
    eeg_window_end: float
    eeg_quality_state: str
    eeg_quality_reasons: tuple[str, ...]
    active_intent_score: float | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        _non_empty("participant_id", self.participant_id)
        _non_empty("session_id", self.session_id)
        if isinstance(self.episode_id, bool) or not isinstance(self.episode_id, int) or self.episode_id <= 0:
            raise ValueError("episode_id must be a positive integer")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if not isinstance(self.active_condition, Condition):
            raise TypeError("active_condition must be a Condition")
        object.__setattr__(
            self, "cutoff_timestamp", _timestamp("cutoff_timestamp", self.cutoff_timestamp)
        )
        object.__setattr__(
            self, "eeg_window_start", _timestamp("eeg_window_start", self.eeg_window_start)
        )
        object.__setattr__(
            self, "eeg_window_end", _timestamp("eeg_window_end", self.eeg_window_end)
        )
        if self.eeg_window_end < self.eeg_window_start:
            raise ValueError("eeg_window_end cannot precede eeg_window_start")
        if self.eeg_window_end > self.cutoff_timestamp:
            raise ValueError("EEG window cannot extend beyond the prediction cutoff")
        object.__setattr__(
            self, "g_probability", _probability("g_probability", self.g_probability)
        )
        object.__setattr__(
            self, "e_probability", _probability("e_probability", self.e_probability)
        )
        object.__setattr__(
            self,
            "active_intent_score",
            _probability("active_intent_score", self.active_intent_score),
        )
        if not isinstance(self.eeg_quality_state, str) or not self.eeg_quality_state:
            raise ValueError("eeg_quality_state must be a non-empty string")
        if not isinstance(self.eeg_quality_reasons, tuple) or not all(
            isinstance(reason, str) and reason for reason in self.eeg_quality_reasons
        ):
            raise TypeError("eeg_quality_reasons must be non-empty strings in a tuple")
        if self.unavailable_reason is not None and not self.unavailable_reason:
            raise ValueError("unavailable_reason must be non-empty or None")
        if self.unavailable_reason is None:
            if self.g_probability is None or self.e_probability is None:
                raise ValueError("available paired predictions require both probabilities")
            expected_active = (
                self.g_probability
                if self.active_condition is Condition.G
                else self.e_probability
            )
            if self.active_intent_score != expected_active:
                raise ValueError("active_intent_score must equal only the active model output")
        elif any(
            value is not None
            for value in (self.g_probability, self.e_probability, self.active_intent_score)
        ):
            raise ValueError("unavailable paired predictions cannot expose model scores")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_condition"] = self.active_condition.value
        payload["eeg_quality_reasons"] = list(self.eeg_quality_reasons)
        return payload


@dataclass(frozen=True)
class EpisodeResultRecord:
    """Resolved feedback label plus both frozen pre-update scoring results."""

    participant_id: str
    session_id: str
    episode_id: int
    active_condition: Condition
    action_occurred: bool
    outcome_timestamp: float
    feedback_window_open: float
    feedback_deadline: float
    feedback_pressed: bool
    feedback_resolution_timestamp: float
    common_label: int
    g_probability: float
    e_probability: float
    g_predicted_label: int
    e_predicted_label: int
    g_correct: bool
    e_correct: bool
    update_applied: bool
    update_reason: str
    instructed_intention: int | None

    def __post_init__(self) -> None:
        _non_empty("participant_id", self.participant_id)
        _non_empty("session_id", self.session_id)
        if isinstance(self.episode_id, bool) or not isinstance(self.episode_id, int) or self.episode_id <= 0:
            raise ValueError("episode_id must be a positive integer")
        if not isinstance(self.active_condition, Condition):
            raise TypeError("active_condition must be a Condition")
        if not isinstance(self.action_occurred, bool):
            raise TypeError("action_occurred must be a bool")
        if not isinstance(self.feedback_pressed, bool):
            raise TypeError("feedback_pressed must be a bool")
        if not isinstance(self.update_applied, bool):
            raise TypeError("update_applied must be a bool")
        for field_name in (
            "outcome_timestamp",
            "feedback_window_open",
            "feedback_deadline",
            "feedback_resolution_timestamp",
        ):
            object.__setattr__(self, field_name, _timestamp(field_name, getattr(self, field_name)))
        if self.feedback_deadline <= self.feedback_window_open:
            raise ValueError("feedback_deadline must follow feedback_window_open")
        if self.outcome_timestamp != self.feedback_window_open:
            raise ValueError("feedback window must open at the visible outcome timestamp")
        if self.feedback_resolution_timestamp < self.feedback_window_open:
            raise ValueError("feedback cannot resolve before its window opens")
        if self.feedback_pressed and self.feedback_resolution_timestamp >= self.feedback_deadline:
            raise ValueError("feedback press must occur before the half-open deadline")
        if not self.feedback_pressed and self.feedback_resolution_timestamp != self.feedback_deadline:
            raise ValueError("feedback timeout must resolve exactly at the deadline")
        object.__setattr__(
            self, "g_probability", _probability("g_probability", self.g_probability)
        )
        object.__setattr__(
            self, "e_probability", _probability("e_probability", self.e_probability)
        )
        if self.g_probability is None or self.e_probability is None:
            raise ValueError("episode results require both frozen probabilities")
        _binary_optional("common_label", self.common_label)
        _binary_optional("g_predicted_label", self.g_predicted_label)
        _binary_optional("e_predicted_label", self.e_predicted_label)
        if self.common_label is None:
            raise ValueError("common_label is required")
        if self.g_predicted_label is None or self.e_predicted_label is None:
            raise ValueError("both predicted labels are required")
        _binary_optional("instructed_intention", self.instructed_intention)
        if not isinstance(self.g_correct, bool) or not isinstance(self.e_correct, bool):
            raise TypeError("model correctness fields must be bool values")
        if self.g_correct != (self.g_predicted_label == self.common_label):
            raise ValueError("g_correct is inconsistent with the stored labels")
        if self.e_correct != (self.e_predicted_label == self.common_label):
            raise ValueError("e_correct is inconsistent with the stored labels")
        _non_empty("update_reason", self.update_reason)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_condition"] = self.active_condition.value
        return payload


@dataclass(frozen=True)
class PredictionDecision:
    """Value Instance 5 can pass to adaptive dwell for the current gaze update."""

    intent_score: float | None
    record: PredictionRecord | None
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_score", _probability("intent_score", self.intent_score))
        _non_empty("reason", self.reason)


if __name__ == "__main__":
    box = BoundingBox(0.2, 0.2, 0.8, 0.8)
    print(GazeContextObservation(1, 1, 1.0, 0.25, 0.5, 0.5, box))
