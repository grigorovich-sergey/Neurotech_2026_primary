"""Scientific episode, feedback, score, and paired-update state machine."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol

from eeg_pipeline.contracts import EEGFeatureWindow, QualityState
from foundations.events import Event, JsonlEventLogger
from gaze_interaction.dwell import DwellTrigger
from gaze_interaction.episodes import CandidateEpisode

from experiment_learning.checkpoint import ParticipantState, save_participant_checkpoint
from experiment_learning.contracts import (
    Condition,
    EpisodeResultRecord,
    GazeContextObservation,
    PredictionDecision,
    PredictionRecord,
)
from experiment_learning.features import combined_features, eeg_features, gaze_features
from experiment_learning.models import FrozenPredictions


class EEGFeatureSource(Protocol):
    def features(self, start: float, end: float) -> EEGFeatureWindow: ...


@dataclass(frozen=True)
class _EpisodePrediction:
    record: PredictionRecord
    g_features: dict[str, float]
    e_features: dict[str, float]
    frozen: FrozenPredictions
    instructed_intention: int | None


@dataclass(frozen=True)
class _FeedbackWindow:
    episode_id: int
    action_occurred: bool
    outcome_timestamp: float
    opened_at: float
    deadline: float


def derive_common_label(*, action_occurred: bool, feedback_pressed: bool) -> int:
    """Fixed contextual one-button truth table."""

    if not isinstance(action_occurred, bool) or not isinstance(feedback_pressed, bool):
        raise TypeError("action_occurred and feedback_pressed must be bool values")
    return int(action_occurred == (not feedback_pressed))


def _binary_optional(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or value not in (0, 1)):
        raise ValueError(f"{name} must be 0, 1, or None")


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class ExperimentController:
    """Orchestrate frozen paired predictions and one unambiguous feedback window."""

    def __init__(
        self,
        *,
        participant_state: ParticipantState,
        session_id: str,
        active_condition: Condition,
        minimum_prediction_elapsed_s: float = 0.25,
        eeg_window_s: float = 1.0,
        feedback_timeout_s: float = 1.5,
        checkpoint_path: str | Path | None = None,
        event_logger: JsonlEventLogger | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(active_condition, Condition):
            raise TypeError("active_condition must be a Condition")
        self.participant_state = participant_state
        self.session_id = session_id
        self.active_condition = active_condition
        self.minimum_prediction_elapsed_s = _positive(
            "minimum_prediction_elapsed_s", minimum_prediction_elapsed_s
        )
        self.eeg_window_s = _positive("eeg_window_s", eeg_window_s)
        self.feedback_timeout_s = _positive("feedback_timeout_s", feedback_timeout_s)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.event_logger = event_logger
        self._predictions: dict[int, _EpisodePrediction] = {}
        self._skipped: dict[int, str] = {}
        self._unavailable_records: dict[int, PredictionRecord] = {}
        self._suppressed: set[int] = set()
        self._completed: set[int] = set()
        self._feedback: _FeedbackWindow | None = None
        self._results: list[EpisodeResultRecord] = []

    @property
    def pending_feedback_episode_id(self) -> int | None:
        return None if self._feedback is None else self._feedback.episode_id

    @property
    def results(self) -> tuple[EpisodeResultRecord, ...]:
        return tuple(self._results)

    def consider_prediction(
        self,
        episode: CandidateEpisode,
        observation: GazeContextObservation,
        eeg_source: EEGFeatureSource,
        *,
        instructed_intention: int | None = None,
    ) -> PredictionDecision:
        """Freeze the first eligible paired prediction, otherwise expose baseline dwell."""

        _binary_optional("instructed_intention", instructed_intention)
        if not episode.active:
            raise ValueError("cannot predict for an ended CandidateEpisode")
        if observation.episode_id != episode.episode_id or observation.track_id != episode.track_id:
            raise ValueError("observation identity does not match CandidateEpisode")
        if observation.timestamp != episode.last_match_timestamp:
            raise ValueError("prediction requires the current confirmed match observation")

        episode_id = episode.episode_id
        pending_before_advance = self._feedback
        if (
            pending_before_advance is not None
            and pending_before_advance.episode_id != episode_id
            and episode.start_timestamp < pending_before_advance.deadline
        ):
            if episode_id not in self._suppressed:
                self._suppressed.add(episode_id)
                self._log_aux(
                    observation.timestamp,
                    "experiment_episode_suppressed",
                    {
                        "episode_id": episode_id,
                        "reason": "feedback_pending_at_episode_start",
                    },
                )
        self.advance_time(observation.timestamp)
        existing = self._predictions.get(episode_id)
        if existing is not None:
            if existing.instructed_intention != instructed_intention:
                raise ValueError("instructed_intention changed after prediction freeze")
            return PredictionDecision(
                intent_score=existing.record.active_intent_score,
                record=existing.record,
                reason="frozen_prediction",
            )
        if episode_id in self._completed:
            return PredictionDecision(None, None, "episode_already_completed")
        if episode_id in self._skipped:
            return PredictionDecision(
                None,
                self._unavailable_records.get(episode_id),
                self._skipped[episode_id],
            )
        if episode_id in self._suppressed:
            return PredictionDecision(None, None, "feedback_pending_at_episode_start")
        if self._feedback is not None and self._feedback.episode_id != episode_id:
            self._suppressed.add(episode_id)
            self._log_aux(
                observation.timestamp,
                "experiment_episode_suppressed",
                {"episode_id": episode_id, "reason": "feedback_pending_at_episode_start"},
            )
            return PredictionDecision(None, None, "feedback_pending_at_episode_start")

        earliest_cutoff = max(
            episode.start_timestamp + self.minimum_prediction_elapsed_s,
            self.eeg_window_s,
        )
        if observation.timestamp < earliest_cutoff and not math.isclose(
            observation.timestamp, earliest_cutoff, rel_tol=1e-9, abs_tol=1e-12
        ):
            return PredictionDecision(None, None, "waiting_for_prediction_cutoff")

        cutoff = observation.timestamp
        eeg_start = cutoff - self.eeg_window_s
        feature_window = eeg_source.features(eeg_start, cutoff)
        self._validate_eeg_interval(feature_window, eeg_start=eeg_start, cutoff=cutoff)
        g_values = gaze_features(episode, observation)
        eeg_values = eeg_features(feature_window)
        if eeg_values is None:
            reason = f"paired_eeg_{feature_window.quality_state.value}"
            record = PredictionRecord(
                participant_id=self.participant_state.participant_id,
                session_id=self.session_id,
                episode_id=episode_id,
                track_id=episode.track_id,
                active_condition=self.active_condition,
                cutoff_timestamp=cutoff,
                g_probability=None,
                e_probability=None,
                eeg_window_start=eeg_start,
                eeg_window_end=cutoff,
                eeg_quality_state=feature_window.quality_state.value,
                eeg_quality_reasons=feature_window.quality_reasons,
                active_intent_score=None,
                unavailable_reason=reason,
            )
            self._skipped[episode_id] = reason
            self._unavailable_records[episode_id] = record
            self._log_prediction(record)
            return PredictionDecision(None, record, reason)

        e_values = combined_features(g_values, eeg_values)
        frozen = self.participant_state.learners.predict_pair(g_values, e_values)
        active_score = (
            frozen.g_probability
            if self.active_condition is Condition.G
            else frozen.e_probability
        )
        record = PredictionRecord(
            participant_id=self.participant_state.participant_id,
            session_id=self.session_id,
            episode_id=episode_id,
            track_id=episode.track_id,
            active_condition=self.active_condition,
            cutoff_timestamp=cutoff,
            g_probability=frozen.g_probability,
            e_probability=frozen.e_probability,
            eeg_window_start=eeg_start,
            eeg_window_end=cutoff,
            eeg_quality_state=feature_window.quality_state.value,
            eeg_quality_reasons=feature_window.quality_reasons,
            active_intent_score=active_score,
        )
        self._predictions[episode_id] = _EpisodePrediction(
            record=record,
            g_features=g_values,
            e_features=e_values,
            frozen=frozen,
            instructed_intention=instructed_intention,
        )
        self._log_prediction(record)
        return PredictionDecision(active_score, record, "prediction_frozen")

    def on_dwell_trigger(self, trigger: DwellTrigger) -> None:
        """Open action feedback only when a paired prediction already exists."""

        self.advance_time(trigger.timestamp)
        episode_id = trigger.episode_id
        if episode_id in self._completed:
            return
        if episode_id not in self._predictions:
            self._skipped.setdefault(episode_id, "action_before_prediction")
            self._log_aux(
                trigger.timestamp,
                "experiment_outcome_unscored",
                {"episode_id": episode_id, "reason": self._skipped[episode_id]},
            )
            return
        if self._feedback is not None:
            raise RuntimeError("cannot open a second feedback window")
        self._open_feedback(
            episode_id=episode_id,
            action_occurred=True,
            outcome_timestamp=trigger.timestamp,
        )

    def on_episode_end(self, episode: CandidateEpisode) -> None:
        """An eligible predicted episode with no trigger becomes a no-action outcome."""

        if episode.active or episode.end_timestamp is None:
            raise ValueError("on_episode_end requires an ended CandidateEpisode")
        episode_id = episode.episode_id
        self.advance_time(episode.end_timestamp)
        if episode_id in self._completed:
            return
        if self._feedback is not None and self._feedback.episode_id == episode_id:
            return
        if episode_id not in self._predictions:
            self._skipped.setdefault(episode_id, "episode_ended_without_prediction")
            return
        if self._feedback is not None:
            raise RuntimeError("cannot open no-action feedback while another window is pending")
        self._open_feedback(
            episode_id=episode_id,
            action_occurred=False,
            outcome_timestamp=episode.end_timestamp,
        )

    def button_press(self, timestamp: float) -> EpisodeResultRecord | None:
        """Resolve only the currently open half-open feedback window [open, deadline)."""

        value = self._timestamp(timestamp)
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
        return self._finalize_feedback(feedback_pressed=True, resolution_timestamp=value)

    def advance_time(self, timestamp: float) -> EpisodeResultRecord | None:
        """Resolve a pending window as timeout once its half-open deadline is reached."""

        value = self._timestamp(timestamp)
        feedback = self._feedback
        if feedback is None or value < feedback.deadline:
            return None
        return self._finalize_feedback(
            feedback_pressed=False, resolution_timestamp=feedback.deadline
        )

    def save_session_checkpoint(self) -> None:
        if self.checkpoint_path is not None:
            save_participant_checkpoint(self.checkpoint_path, self.participant_state)

    def _open_feedback(
        self, *, episode_id: int, action_occurred: bool, outcome_timestamp: float
    ) -> None:
        self._feedback = _FeedbackWindow(
            episode_id=episode_id,
            action_occurred=action_occurred,
            outcome_timestamp=outcome_timestamp,
            opened_at=outcome_timestamp,
            deadline=outcome_timestamp + self.feedback_timeout_s,
        )

    def _finalize_feedback(
        self, *, feedback_pressed: bool, resolution_timestamp: float
    ) -> EpisodeResultRecord:
        feedback = self._feedback
        if feedback is None:
            raise RuntimeError("no feedback window is open")
        prediction = self._predictions[feedback.episode_id]
        label = derive_common_label(
            action_occurred=feedback.action_occurred,
            feedback_pressed=feedback_pressed,
        )

        # This materialized score pair is deliberately created before either learner updates.
        scored = self.participant_state.learners.score_pair(prediction.frozen, label)
        result = EpisodeResultRecord(
            participant_id=self.participant_state.participant_id,
            session_id=self.session_id,
            episode_id=feedback.episode_id,
            active_condition=self.active_condition,
            action_occurred=feedback.action_occurred,
            outcome_timestamp=feedback.outcome_timestamp,
            feedback_window_open=feedback.opened_at,
            feedback_deadline=feedback.deadline,
            feedback_pressed=feedback_pressed,
            feedback_resolution_timestamp=resolution_timestamp,
            common_label=label,
            g_probability=scored.frozen.g_probability,
            e_probability=scored.frozen.e_probability,
            g_predicted_label=scored.g_predicted_label,
            e_predicted_label=scored.e_predicted_label,
            g_correct=scored.g_correct,
            e_correct=scored.e_correct,
            update_applied=True,
            update_reason="paired_common_feedback_label",
            instructed_intention=prediction.instructed_intention,
        )
        self.participant_state.learners.learn_scored(
            scored, prediction.g_features, prediction.e_features
        )
        self._feedback = None
        self._completed.add(feedback.episode_id)
        self._results.append(result)
        self.save_session_checkpoint()
        self._log_result(result)
        return result

    def _validate_eeg_interval(
        self, window: EEGFeatureWindow, *, eeg_start: float, cutoff: float
    ) -> None:
        if not math.isclose(window.requested_start, eeg_start, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("EEG source returned a different requested start")
        if not math.isclose(window.requested_end, cutoff, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("EEG source returned a different requested cutoff")
        if window.requested_end > cutoff:
            raise ValueError("EEG source returned future data beyond prediction cutoff")
        if window.quality_state is QualityState.USABLE and window.values is None:
            raise ValueError("usable EEGFeatureWindow unexpectedly has no values")

    @staticmethod
    def _timestamp(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("timestamp must be a real number")
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("timestamp must be finite and non-negative")
        return result

    def _log_prediction(self, record: PredictionRecord) -> None:
        if self.event_logger is not None:
            self.event_logger.log(
                Event(record.cutoff_timestamp, "experiment_prediction", record.to_payload())
            )

    def _log_result(self, record: EpisodeResultRecord) -> None:
        if self.event_logger is not None:
            self.event_logger.log(
                Event(
                    record.feedback_resolution_timestamp,
                    "experiment_episode_result",
                    record.to_payload(),
                )
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
