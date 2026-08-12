"""Frozen-policy episode, feedback, cancellation, and record state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Protocol

from eeg_pipeline.contracts import EEGFeatureWindow, QualityState
from eeg_pipeline.processing import FEATURE_NAMES as EEG_FEATURE_NAMES
from foundations.events import Event, JsonlEventLogger
from gaze_interaction.episodes import CandidateEpisode

from experiment_learning.contracts import (
    CancellationInstruction,
    CompletedSession,
    Condition,
    EpisodeTrainingRecord,
    FeedbackResolution,
    GazeContextObservation,
    ModelOutcome,
    OutcomeStatus,
    PolicyDecision,
    TrajectoryPoint,
)
from experiment_learning.artifacts import artifact_digest
from experiment_learning.eeg_indicator import (
    ENGAGEMENT_INDEX_FORMULA,
    ENGAGEMENT_INDEX_ID,
    InvalidEEGIndicator,
    engagement_index,
)
from experiment_learning.features import eeg_feature_mapping
from experiment_learning.policy import FrozenSessionPolicy
from experiment_learning.schedule import ScheduleBinding


class EEGFeatureSource(Protocol):
    def features(self, start: float, end: float) -> EEGFeatureWindow: ...


@dataclass
class _EpisodeState:
    episode_id: int
    track_id: int
    start_timestamp: float
    instructed_intention: int | None
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    eeg_attempted: bool = False
    prediction_cutoff_timestamp: float | None = None
    eeg_window_start: float | None = None
    eeg_window_end: float | None = None
    eeg_quality_state: str = "not_evaluated"
    eeg_quality_reasons: tuple[str, ...] = ()
    eeg_feature_names: tuple[str, ...] = EEG_FEATURE_NAMES
    eeg_feature_values: tuple[float, ...] | None = None
    engagement_index: float | None = None
    eeg_probability: float | None = None
    eeg_evidence: float | None = None
    exclusion_reasons: list[str] = field(default_factory=list)
    action_occurred: bool | None = None
    action_timestamp: float | None = None
    natural_endpoint_timestamp: float | None = None
    feedback_window_open: float | None = None
    feedback_deadline: float | None = None
    feedback_pressed: bool | None = None
    feedback_resolution_timestamp: float | None = None
    common_label: int | None = None
    canceled: bool = False
    cancellation_reason: str | None = None
    completed: bool = False


@dataclass(frozen=True)
class _FeedbackWindow:
    episode_id: int
    action_occurred: bool
    opened_at: float
    deadline: float


def derive_common_label(*, action_occurred: bool, feedback_pressed: bool) -> int:
    """Fixed contextual one-button truth table."""

    if not isinstance(action_occurred, bool) or not isinstance(feedback_pressed, bool):
        raise TypeError("action_occurred and feedback_pressed must be bool values")
    return int(action_occurred == (not feedback_pressed))


def _timestamp(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive(name: str, value: float) -> float:
    result = _timestamp(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _binary_optional(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or value not in (0, 1)):
        raise ValueError(f"{name} must be 0, 1, or None")


class ExperimentController:
    """Evaluate a frozen policy and create immutable between-session trainer inputs."""

    def __init__(
        self,
        *,
        policy: FrozenSessionPolicy,
        policy_sha256: str,
        session_id: str,
        session_number: int,
        attempt_id: str,
        active_condition: Condition,
        schedule_binding: ScheduleBinding,
        minimum_prediction_elapsed_s: float = 0.25,
        eeg_window_s: float = 1.0,
        feedback_timeout_s: float = 1.5,
        event_logger: JsonlEventLogger | None = None,
    ) -> None:
        if not session_id or not attempt_id or not policy_sha256:
            raise ValueError("session_id, attempt_id, and policy_sha256 must be non-empty")
        if policy_sha256 != artifact_digest(policy.to_payload()):
            raise ValueError("policy_sha256 does not identify the supplied frozen policy")
        if isinstance(session_number, bool) or not isinstance(session_number, int) or session_number <= 0:
            raise ValueError("session_number must be a positive integer")
        if policy.policy_for_session != session_number:
            raise ValueError("frozen policy was not created for this session")
        if (
            policy.schedule_sequence_id != schedule_binding.sequence_id
            or policy.schedule_sha256 != schedule_binding.csv_sha256
        ):
            raise ValueError("policy and attempt schedule bindings differ")
        if not isinstance(active_condition, Condition):
            raise TypeError("active_condition must be a Condition")
        self.policy = policy
        self.policy_sha256 = policy_sha256
        self.participant_id = policy.participant_id
        self.session_id = session_id
        self.session_number = session_number
        self.attempt_id = attempt_id
        self.active_condition = active_condition
        self.schedule_binding = schedule_binding
        self.minimum_prediction_elapsed_s = _positive(
            "minimum_prediction_elapsed_s", minimum_prediction_elapsed_s
        )
        self.eeg_window_s = _positive("eeg_window_s", eeg_window_s)
        self.feedback_timeout_s = _positive("feedback_timeout_s", feedback_timeout_s)
        self.event_logger = event_logger
        self._episodes: dict[int, _EpisodeState] = {}
        self._feedback: _FeedbackWindow | None = None
        self._records: list[EpisodeTrainingRecord] = []

    @property
    def pending_feedback_episode_id(self) -> int | None:
        return None if self._feedback is None else self._feedback.episode_id

    @property
    def action_gate_open(self) -> bool:
        return self._feedback is None

    @property
    def records(self) -> tuple[EpisodeTrainingRecord, ...]:
        return tuple(self._records)

    def evaluate_update(
        self,
        episode: CandidateEpisode,
        observation: GazeContextObservation,
        eeg_source: EEGFeatureSource,
        *,
        instructed_intention: int | None = None,
    ) -> PolicyDecision:
        """Record dwell progress and freeze one causal engagement-index decision."""

        _binary_optional("instructed_intention", instructed_intention)
        if not episode.active:
            raise ValueError("cannot evaluate an ended CandidateEpisode")
        if observation.episode_id != episode.episode_id or observation.track_id != episode.track_id:
            raise ValueError("observation identity does not match CandidateEpisode")
        if observation.timestamp != episode.last_match_timestamp:
            raise ValueError("evaluation requires the current confirmed match")
        if observation.timestamp < episode.start_timestamp:
            raise ValueError("observation cannot precede episode start")
        self.advance_time(observation.timestamp)
        state = self._episodes.get(episode.episode_id)
        if state is None:
            state = _EpisodeState(
                episode_id=episode.episode_id,
                track_id=episode.track_id,
                start_timestamp=episode.start_timestamp,
                instructed_intention=instructed_intention,
            )
            self._episodes[episode.episode_id] = state
        elif (
            state.track_id != episode.track_id
            or state.start_timestamp != episode.start_timestamp
            or state.instructed_intention != instructed_intention
        ):
            raise ValueError("episode identity or instructed intention changed during evaluation")
        if state.completed:
            return self._decision(state, "episode_already_completed")
        self._append_trajectory(state, observation)
        if state.eeg_attempted:
            return self._decision(state, "frozen_policy_decision")

        earliest_cutoff = max(
            episode.start_timestamp + self.minimum_prediction_elapsed_s,
            self.eeg_window_s,
        )
        if observation.timestamp < earliest_cutoff and not math.isclose(
            observation.timestamp, earliest_cutoff, rel_tol=1e-9, abs_tol=1e-12
        ):
            return self._decision(state, "waiting_for_eeg_cutoff")

        cutoff = observation.timestamp
        eeg_start = cutoff - self.eeg_window_s
        window = eeg_source.features(eeg_start, cutoff)
        self._validate_eeg_interval(window, eeg_start=eeg_start, cutoff=cutoff)
        state.eeg_attempted = True
        state.prediction_cutoff_timestamp = cutoff
        state.eeg_window_start = eeg_start
        state.eeg_window_end = cutoff
        state.eeg_quality_state = window.quality_state.value
        state.eeg_quality_reasons = window.quality_reasons
        state.eeg_feature_names = tuple(window.feature_names)

        if window.quality_state is not QualityState.USABLE or window.values is None:
            self._exclude(state, f"eeg_{window.quality_state.value}")
            self._log_decision(state, cutoff, "paired_eeg_unusable")
            return self._decision(state, f"paired_eeg_{window.quality_state.value}", newly=True)
        state.eeg_feature_values = tuple(float(value) for value in window.values)
        try:
            mapping = eeg_feature_mapping(window)
        except ValueError:
            self._exclude(state, "eeg_feature_signature_invalid")
            self._log_decision(state, cutoff, "eeg_feature_signature_invalid")
            return self._decision(state, "eeg_feature_signature_invalid", newly=True)
        assert mapping is not None
        try:
            indicator = engagement_index(mapping)
        except InvalidEEGIndicator:
            self._exclude(state, "eeg_indicator_invalid")
            self._log_decision(state, cutoff, "eeg_indicator_invalid")
            return self._decision(state, "eeg_indicator_invalid", newly=True)
        state.engagement_index = indicator
        state.eeg_probability = self.policy.eeg_probability(indicator)
        state.eeg_evidence = self.policy.positive_eeg_evidence(state.eeg_probability)
        self._log_decision(state, cutoff, "policy_decision_frozen")
        return self._decision(state, "policy_decision_frozen", newly=True)

    def open_action_feedback(self, episode_id: int, display_timestamp: float) -> bool:
        """Open feedback after an announcement, including for excluded EEG episodes."""

        timestamp = _timestamp("display_timestamp", display_timestamp)
        self.advance_time(timestamp)
        state = self._require_episode(episode_id)
        if state.completed:
            return False
        if timestamp < state.start_timestamp or (
            state.trajectory and timestamp < state.trajectory[-1].timestamp
        ):
            raise ValueError("action display cannot precede observed episode data")
        if self._feedback is not None:
            raise RuntimeError("cannot open a second feedback window")
        state.action_occurred = True
        state.action_timestamp = timestamp
        self._open_feedback(state, timestamp)
        return True

    def open_no_action_feedback(self, episode: CandidateEpisode, timestamp: float) -> bool:
        """Open a silent-outcome target only for a paired eligible episode."""

        value = _timestamp("timestamp", timestamp)
        if episode.active or episode.end_timestamp is None:
            raise ValueError("no-action feedback requires an ended CandidateEpisode")
        if not math.isclose(value, episode.end_timestamp, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("no-action outcome timestamp must equal the episode endpoint")
        self.advance_time(value)
        state = self._episodes.get(episode.episode_id)
        if state is None:
            state = _EpisodeState(
                episode_id=episode.episode_id,
                track_id=episode.track_id,
                start_timestamp=episode.start_timestamp,
                instructed_intention=None,
            )
            self._episodes[episode.episode_id] = state
            self._exclude(state, "episode_ended_without_evaluation")
        if state.completed:
            return False
        state.action_occurred = False
        state.natural_endpoint_timestamp = value
        if state.engagement_index is None or state.exclusion_reasons:
            self._exclude(state, "ineligible_no_action_has_no_feedback_target")
            self._finalize_record(state)
            return False
        if self._feedback is not None:
            raise RuntimeError("cannot open a second feedback window")
        self._open_feedback(state, value)
        return True

    def accept_feedback(self, timestamp: float) -> FeedbackResolution | None:
        """Accept the exclusive earlier target and cancel every newer provisional episode."""

        value = _timestamp("timestamp", timestamp)
        feedback = self._feedback
        if feedback is None:
            self._log_aux(value, "experiment_feedback_ignored", {"reason": "no_open_window"})
            return None
        if value < feedback.opened_at:
            self._log_aux(value, "experiment_feedback_ignored", {"reason": "before_window"})
            return None
        if value >= feedback.deadline:
            self.advance_time(value)
            self._log_aux(value, "experiment_feedback_ignored", {"reason": "at_or_after_deadline"})
            return None
        target = self._resolve_feedback(feedback_pressed=True, resolution_timestamp=value)
        cancellations: list[CancellationInstruction] = []
        for state in sorted(self._episodes.values(), key=lambda item: (item.start_timestamp, item.episode_id)):
            if state.completed or state.episode_id == target.episode_id:
                continue
            if state.start_timestamp < feedback.opened_at:
                continue
            self._cancel_state(state, value, "feedback_accepted")
            cancellations.append(CancellationInstruction(state.episode_id, value))
        return FeedbackResolution(target, tuple(cancellations))

    def advance_time(self, timestamp: float) -> FeedbackResolution | None:
        """Resolve feedback timeout without canceling a still-active newer episode."""

        value = _timestamp("timestamp", timestamp)
        feedback = self._feedback
        if feedback is None or value < feedback.deadline:
            return None
        record = self._resolve_feedback(
            feedback_pressed=False, resolution_timestamp=feedback.deadline
        )
        return FeedbackResolution(record, ())

    def cancel_episode(
        self, episode_id: int, timestamp: float, reason: str
    ) -> EpisodeTrainingRecord | None:
        value = _timestamp("timestamp", timestamp)
        if not reason:
            raise ValueError("reason must be non-empty")
        state = self._episodes.get(episode_id)
        if state is None or state.completed:
            return None
        return self._cancel_state(state, value, reason)

    def completed_session(self, completed_timestamp: float) -> CompletedSession:
        if self._feedback is not None:
            raise RuntimeError("cannot close a session with unresolved feedback")
        unfinished = [state.episode_id for state in self._episodes.values() if not state.completed]
        if unfinished:
            raise RuntimeError(f"cannot close session with unfinished episodes: {unfinished}")
        return CompletedSession(
            participant_id=self.participant_id,
            session_id=self.session_id,
            session_number=self.session_number,
            attempt_id=self.attempt_id,
            active_condition=self.active_condition,
            policy_sha256=self.policy_sha256,
            schedule_sequence_id=self.schedule_binding.sequence_id,
            schedule_sha256=self.schedule_binding.csv_sha256,
            completed_timestamp=_timestamp("completed_timestamp", completed_timestamp),
            successful=True,
            records=tuple(sorted(self._records, key=lambda record: record.identity)),
        )

    def _decision(self, state: _EpisodeState, reason: str, *, newly: bool = False) -> PolicyDecision:
        usable = state.engagement_index is not None and not state.exclusion_reasons
        evidence = state.eeg_evidence if usable else None
        if self.active_condition is Condition.E and evidence is not None:
            intent_score = evidence
            required = self.policy.e_required_dwell(evidence)
        else:
            intent_score = None
            required = self.policy.g_base_threshold_s
        return PolicyDecision(
            episode_id=state.episode_id,
            intent_score=intent_score,
            required_dwell_s=required,
            eeg_probability=state.eeg_probability,
            engagement_index=state.engagement_index,
            training_eligible=usable,
            reason=reason,
            newly_frozen=newly,
        )

    def _append_trajectory(
        self, state: _EpisodeState, observation: GazeContextObservation
    ) -> None:
        point = TrajectoryPoint(observation.timestamp, observation.matched_dwell_s)
        if state.trajectory:
            previous = state.trajectory[-1]
            if point.timestamp < previous.timestamp:
                raise ValueError("episode trajectory timestamps must be non-decreasing")
            if point.accumulated_matched_dwell_s < previous.accumulated_matched_dwell_s:
                raise ValueError("episode dwell trajectory cannot decrease")
            if point.timestamp == previous.timestamp:
                if point != previous:
                    raise ValueError("one timestamp cannot have conflicting trajectory values")
                return
        state.trajectory.append(point)

    def _open_feedback(self, state: _EpisodeState, timestamp: float) -> None:
        state.feedback_window_open = timestamp
        state.feedback_deadline = timestamp + self.feedback_timeout_s
        self._feedback = _FeedbackWindow(
            state.episode_id,
            bool(state.action_occurred),
            timestamp,
            timestamp + self.feedback_timeout_s,
        )

    def _resolve_feedback(
        self, *, feedback_pressed: bool, resolution_timestamp: float
    ) -> EpisodeTrainingRecord:
        feedback = self._feedback
        if feedback is None:
            raise RuntimeError("no feedback window is open")
        state = self._episodes[feedback.episode_id]
        state.feedback_pressed = feedback_pressed
        state.feedback_resolution_timestamp = resolution_timestamp
        state.common_label = derive_common_label(
            action_occurred=feedback.action_occurred,
            feedback_pressed=feedback_pressed,
        )
        self._feedback = None
        return self._finalize_record(state)

    def _cancel_state(
        self, state: _EpisodeState, timestamp: float, reason: str
    ) -> EpisodeTrainingRecord:
        if self._feedback is not None and self._feedback.episode_id == state.episode_id:
            self._feedback = None
        state.canceled = True
        state.cancellation_reason = reason
        self._exclude(state, f"canceled:{reason}")
        if state.natural_endpoint_timestamp is None and state.action_timestamp is None:
            state.natural_endpoint_timestamp = timestamp
        return self._finalize_record(state)

    def _finalize_record(self, state: _EpisodeState) -> EpisodeTrainingRecord:
        if state.completed:
            raise RuntimeError("episode record has already been finalized")
        if state.instructed_intention is not None:
            self._exclude(state, "controlled_intention_trial")
        if state.common_label is None:
            self._exclude(state, "no_common_feedback_label")
        if state.engagement_index is None:
            self._exclude(state, "no_valid_engagement_index")
        g_required = self.policy.g_base_threshold_s
        e_required = (
            self.policy.e_required_dwell(state.eeg_evidence)
            if state.eeg_evidence is not None
            else g_required
        )
        g_outcome = self._model_outcome(state, g_required)
        e_outcome = self._model_outcome(state, e_required)
        exclusions = tuple(state.exclusion_reasons)
        record = EpisodeTrainingRecord(
            participant_id=self.participant_id,
            session_id=self.session_id,
            session_number=self.session_number,
            attempt_id=self.attempt_id,
            episode_id=state.episode_id,
            track_id=state.track_id,
            active_condition=self.active_condition,
            policy_sha256=self.policy_sha256,
            episode_start_timestamp=state.start_timestamp,
            prediction_cutoff_timestamp=state.prediction_cutoff_timestamp,
            eeg_window_start=state.eeg_window_start,
            eeg_window_end=state.eeg_window_end,
            eeg_quality_state=state.eeg_quality_state,
            eeg_quality_reasons=state.eeg_quality_reasons,
            eeg_feature_names=state.eeg_feature_names,
            eeg_feature_values=state.eeg_feature_values,
            eeg_indicator_id=ENGAGEMENT_INDEX_ID,
            eeg_indicator_formula=ENGAGEMENT_INDEX_FORMULA,
            engagement_index=state.engagement_index,
            eeg_probability=state.eeg_probability,
            eeg_evidence=state.eeg_evidence,
            g_required_dwell_s=g_required,
            e_required_dwell_s=e_required,
            trajectory=tuple(state.trajectory),
            action_occurred=state.action_occurred,
            action_timestamp=state.action_timestamp,
            natural_endpoint_timestamp=state.natural_endpoint_timestamp,
            feedback_window_open=state.feedback_window_open,
            feedback_deadline=state.feedback_deadline,
            feedback_pressed=state.feedback_pressed,
            feedback_resolution_timestamp=state.feedback_resolution_timestamp,
            common_label=state.common_label,
            instructed_intention=state.instructed_intention,
            g_outcome=g_outcome,
            e_outcome=e_outcome,
            training_eligible=not exclusions,
            exclusion_reasons=exclusions,
            canceled=state.canceled,
            cancellation_reason=state.cancellation_reason,
        )
        state.completed = True
        self._records.append(record)
        if self.event_logger is not None:
            if state.feedback_resolution_timestamp is not None:
                event_timestamp = state.feedback_resolution_timestamp
            elif state.action_timestamp is not None:
                event_timestamp = state.action_timestamp
            elif state.natural_endpoint_timestamp is not None:
                event_timestamp = state.natural_endpoint_timestamp
            elif state.trajectory:
                event_timestamp = state.trajectory[-1].timestamp
            else:
                event_timestamp = state.start_timestamp
            self.event_logger.log(
                Event(event_timestamp, "experiment_episode_training_record", record.to_payload())
            )
        return record

    @staticmethod
    def _model_outcome(state: _EpisodeState, threshold: float) -> ModelOutcome:
        if state.canceled or state.engagement_index is None:
            return ModelOutcome(OutcomeStatus.NOT_SCORED, None)
        crossing = next(
            (
                point.timestamp
                for point in state.trajectory
                if point.accumulated_matched_dwell_s >= threshold
            ),
            None,
        )
        if crossing is not None:
            return ModelOutcome(OutcomeStatus.ACTION, crossing)
        if state.action_occurred is True:
            return ModelOutcome(OutcomeStatus.COUNTERFACTUAL_CENSORED, None)
        if state.action_occurred is False and state.natural_endpoint_timestamp is not None:
            return ModelOutcome(OutcomeStatus.NO_ACTION, None)
        return ModelOutcome(OutcomeStatus.NOT_SCORED, None)

    @staticmethod
    def _validate_eeg_interval(
        window: EEGFeatureWindow, *, eeg_start: float, cutoff: float
    ) -> None:
        if not math.isclose(window.requested_start, eeg_start, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("EEG source returned a different requested start")
        if not math.isclose(window.requested_end, cutoff, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("EEG source returned a different requested cutoff")
        if window.requested_end > cutoff:
            raise ValueError("EEG source returned future data beyond prediction cutoff")
        if window.quality_state is QualityState.USABLE and window.values is None:
            raise ValueError("usable EEGFeatureWindow unexpectedly has no values")

    def _require_episode(self, episode_id: int) -> _EpisodeState:
        if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id <= 0:
            raise ValueError("episode_id must be a positive integer")
        try:
            return self._episodes[episode_id]
        except KeyError as exc:
            raise ValueError("episode has not been observed by evaluate_update") from exc

    @staticmethod
    def _exclude(state: _EpisodeState, reason: str) -> None:
        if reason not in state.exclusion_reasons:
            state.exclusion_reasons.append(reason)

    def _log_decision(self, state: _EpisodeState, timestamp: float, reason: str) -> None:
        self._log_aux(
            timestamp,
            "experiment_policy_decision",
            {
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "episode_id": state.episode_id,
                "active_condition": self.active_condition.value,
                "policy_sha256": self.policy_sha256,
                "engagement_index": state.engagement_index,
                "eeg_probability": state.eeg_probability,
                "eeg_evidence": state.eeg_evidence,
                "reason": reason,
            },
        )

    def _log_aux(self, timestamp: float, name: str, payload: dict[str, object]) -> None:
        if self.event_logger is not None:
            self.event_logger.log(Event(timestamp, name, payload))


if __name__ == "__main__":
    print(
        {
            (action, press): derive_common_label(
                action_occurred=action, feedback_pressed=press
            )
            for action in (False, True)
            for press in (False, True)
        }
    )
