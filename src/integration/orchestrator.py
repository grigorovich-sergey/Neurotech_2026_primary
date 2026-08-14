"""Cross-subsystem scientific-time orchestration for one experiment attempt."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Callable, Protocol

from eeg_pipeline.contracts import EEGFeatureWindow, EEGSample
from eeg_pipeline.pipeline import EEGPipeline
from eeg_pipeline.recording import EEGHDF5Recorder
from experiment_learning.contracts import FeedbackResolution, PolicyDecision
from experiment_learning.features import observation_from_interaction
from experiment_learning.state_machine import ExperimentController
from foundations.contracts import GazeSample, SceneFrame
from foundations.events import Event, JsonlEventLogger
from gaze_interaction.dwell import DwellTrigger
from gaze_interaction.episodes import CandidateEpisode, EpisodeEndReason
from gaze_interaction.pipeline import GazeInteractionPipeline, InteractionUpdate, SceneUpdate


@dataclass(frozen=True)
class ScheduledFeedbackPress:
    timestamp: float
    episode_id: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(float(self.timestamp))
            or self.timestamp < 0
        ):
            raise ValueError("feedback timestamp must be finite and non-negative")
        if (
            isinstance(self.episode_id, bool)
            or not isinstance(self.episode_id, int)
            or self.episode_id <= 0
        ):
            raise ValueError("feedback episode_id must be a positive integer")


class EEGFeatureSource(Protocol):
    def drain_through(self, cutoff_timestamp: float) -> int: ...

    def features(self, start: float, end: float) -> EEGFeatureWindow: ...

    def drain_remaining(self) -> int: ...


class RecordedEEGFeatureSource:
    """Causally expose an iterable EEG source through closed time cutoffs."""

    def __init__(
        self,
        samples: Iterable[EEGSample],
        *,
        pipeline: EEGPipeline,
        recorder: EEGHDF5Recorder | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.recorder = recorder
        self._iterator = iter(samples)
        self._next = next(self._iterator, None)
        self.ingested_sample_count = 0
        self.last_ingested_timestamp: float | None = None

    def drain_through(self, cutoff_timestamp: float) -> int:
        cutoff = _timestamp("cutoff_timestamp", cutoff_timestamp)
        count = 0
        while self._next is not None and self._next.timestamp <= cutoff + 1e-12:
            sample = self._next
            if sample.timestamp > cutoff + 1e-12:
                raise RuntimeError("EEG source attempted to expose a future sample")
            self._ingest(sample)
            count += 1
            self._next = next(self._iterator, None)
        return count

    def features(self, start: float, end: float) -> EEGFeatureWindow:
        window_start = _timestamp("start", start)
        window_end = _timestamp("end", end)
        if window_end < window_start:
            raise ValueError("EEG feature bounds must satisfy 0 <= start <= end")
        self.drain_through(window_end)
        return self.pipeline.features(window_start, window_end)

    def drain_remaining(self) -> int:
        count = 0
        while self._next is not None:
            sample = self._next
            self._ingest(sample)
            count += 1
            self._next = next(self._iterator, None)
        return count

    def _ingest(self, sample: EEGSample) -> None:
        if not isinstance(sample, EEGSample):
            raise TypeError("EEG source must yield EEGSample values")
        if (
            self.last_ingested_timestamp is not None
            and sample.timestamp < self.last_ingested_timestamp
        ):
            raise ValueError("EEG source timestamps must be non-decreasing")
        # Raw recording precedes mutation of the scientific processing pipeline.
        if self.recorder is not None:
            self.recorder.record(sample)
        self.pipeline.add_sample(sample)
        self.ingested_sample_count += 1
        self.last_ingested_timestamp = float(sample.timestamp)


class FeedbackDriver(Protocol):
    def before_time(
        self, timestamp: float, controller: ExperimentController
    ) -> tuple[FeedbackResolution, ...]: ...

    def feedback_opened(
        self, *, episode_id: int, outcome_timestamp: float, controller: ExperimentController
    ) -> None: ...


class TimedFeedbackDriver:
    """Replay timestamped button presses with strict episode identity checks."""

    def __init__(
        self,
        presses: Iterable[ScheduledFeedbackPress],
        *,
        event_logger: JsonlEventLogger,
        session_id: str,
    ) -> None:
        self._presses = deque(sorted(presses, key=lambda item: item.timestamp))
        self.event_logger = event_logger
        self.session_id = session_id

    def before_time(
        self, timestamp: float, controller: ExperimentController
    ) -> tuple[FeedbackResolution, ...]:
        resolutions: list[FeedbackResolution] = []
        while self._presses and self._presses[0].timestamp <= timestamp + 1e-12:
            press = self._presses.popleft()
            pending = controller.pending_feedback_episode_id
            if pending != press.episode_id:
                raise RuntimeError(
                    "feedback replay identity mismatch: "
                    f"press targets episode {press.episode_id}, pending episode is {pending}"
                )
            result = controller.accept_feedback(press.timestamp)
            if result is None or result.record.episode_id != press.episode_id:
                raise RuntimeError("scheduled feedback press did not resolve its episode")
            resolutions.append(result)
            self.event_logger.log(
                Event(
                    press.timestamp,
                    "integration_feedback_press",
                    {"session_id": self.session_id, "episode_id": press.episode_id},
                )
            )
        timeout = controller.advance_time(timestamp)
        if timeout is not None:
            resolutions.append(timeout)
        return tuple(resolutions)

    def feedback_opened(
        self, *, episode_id: int, outcome_timestamp: float, controller: ExperimentController
    ) -> None:
        del episode_id, outcome_timestamp, controller

    def assert_consumed(self) -> None:
        if self._presses:
            next_press = self._presses[0]
            raise RuntimeError(
                "feedback replay/source ended before recorded press for "
                f"episode {next_press.episode_id} at {next_press.timestamp:.6f}s"
            )


class SyntheticFeedbackDriver(TimedFeedbackDriver):
    """Create a deterministic press/timeout schedule without deriving labels."""

    def __init__(
        self,
        *,
        press_cycle: list[bool],
        press_delay_seconds: float,
        event_logger: JsonlEventLogger,
        session_id: str,
    ) -> None:
        if not press_cycle or any(not isinstance(value, bool) for value in press_cycle):
            raise ValueError("feedback.synthetic.press_cycle must be a non-empty bool list")
        if (
            isinstance(press_delay_seconds, bool)
            or not isinstance(press_delay_seconds, (int, float))
            or not math.isfinite(float(press_delay_seconds))
            or press_delay_seconds < 0
        ):
            raise ValueError("feedback.synthetic.press_delay_seconds must be non-negative")
        super().__init__((), event_logger=event_logger, session_id=session_id)
        self.press_cycle = tuple(press_cycle)
        self.press_delay_seconds = float(press_delay_seconds)
        self._feedback_index = 0
        self._scheduled_episodes: set[int] = set()

    def feedback_opened(
        self, *, episode_id: int, outcome_timestamp: float, controller: ExperimentController
    ) -> None:
        if controller.pending_feedback_episode_id != episode_id:
            return
        if episode_id in self._scheduled_episodes:
            return
        should_press = self.press_cycle[self._feedback_index % len(self.press_cycle)]
        self._feedback_index += 1
        self._scheduled_episodes.add(episode_id)
        if not should_press:
            return
        press_timestamp = outcome_timestamp + self.press_delay_seconds
        if press_timestamp >= outcome_timestamp + controller.feedback_timeout_s:
            raise ValueError("synthetic feedback press must precede the feedback deadline")
        self._presses.append(ScheduledFeedbackPress(press_timestamp, episode_id))


class KeyboardFeedbackDriver:
    """Treat one OpenCV key as the pre-hardware physical-button stand-in."""

    def __init__(
        self,
        *,
        key_code: int,
        event_logger: JsonlEventLogger,
        session_id: str,
        poll_key: Callable[[], int] | None = None,
    ) -> None:
        if isinstance(key_code, bool) or not isinstance(key_code, int) or not 0 <= key_code <= 255:
            raise ValueError("feedback.keyboard.key_code must be an integer within [0, 255]")
        self.key_code = key_code
        self.event_logger = event_logger
        self.session_id = session_id
        self._poll_key = poll_key

    def before_time(
        self, timestamp: float, controller: ExperimentController
    ) -> tuple[FeedbackResolution, ...]:
        poll = self._poll_key
        if poll is None:
            import cv2

            poll = lambda: cv2.waitKey(1) & 0xFF
        resolutions: list[FeedbackResolution] = []
        if poll() == self.key_code:
            pending = controller.pending_feedback_episode_id
            result = controller.accept_feedback(timestamp)
            if result is None:
                self.event_logger.log(
                    Event(
                        timestamp,
                        "integration_feedback_press_ignored",
                        {"session_id": self.session_id, "reason": "no_open_window"},
                    )
                )
            else:
                resolutions.append(result)
                self.event_logger.log(
                    Event(
                        timestamp,
                        "integration_feedback_press",
                        {"session_id": self.session_id, "episode_id": pending},
                    )
                )
        timeout = controller.advance_time(timestamp)
        if timeout is not None:
            resolutions.append(timeout)
        return tuple(resolutions)

    def feedback_opened(
        self, *, episode_id: int, outcome_timestamp: float, controller: ExperimentController
    ) -> None:
        del episode_id, outcome_timestamp, controller


@dataclass(frozen=True)
class IntegratedGazeUpdate:
    interaction: InteractionUpdate
    decision: PolicyDecision | None
    applied_intent_score: float | None


class IntegratedExperimentOrchestrator:
    """Connect gaze, causal EEG, frozen policy, feedback, and event logging."""

    def __init__(
        self,
        *,
        gaze_pipeline: GazeInteractionPipeline,
        eeg_source: EEGFeatureSource,
        experiment: ExperimentController,
        feedback: FeedbackDriver,
        event_logger: JsonlEventLogger,
        session_id: str,
        present_action: Callable[[DwellTrigger, InteractionUpdate], float] | None = None,
    ) -> None:
        self.gaze_pipeline = gaze_pipeline
        self.eeg_source = eeg_source
        self.experiment = experiment
        self.feedback = feedback
        self.event_logger = event_logger
        self.session_id = session_id
        self.present_action = present_action
        self._held_score_episode_id: int | None = None
        self._held_score: float | None = None
        self._seen_episode_ids: set[int] = set()
        self.latest_processed_scientific_timestamp = 0.0

    def process_scene(self, frame: SceneFrame) -> SceneUpdate:
        self._through(float(frame.timestamp))
        update = self.gaze_pipeline.process_scene(frame)
        self.event_logger.log(
            Event(
                frame.timestamp,
                "integration_scene_processed",
                {"detections": len(update.detections), "tracks": len(update.tracks)},
            )
        )
        return update

    def process_gaze(self, gaze: GazeSample) -> IntegratedGazeUpdate:
        timestamp = float(gaze.timestamp)
        self._through(timestamp)
        applied_score = self._score_for_this_update(gaze)
        interaction = self.gaze_pipeline.process_gaze(
            gaze,
            intent_score=applied_score,
            trigger_gate_open=self.experiment.action_gate_open,
        )
        self._record_episode_start(interaction.active_episode)

        if interaction.ended_episode is not None:
            self._clear_held_if(interaction.ended_episode.episode_id)
            self._handle_ended_episode(interaction.ended_episode)
            # A retrospective gap endpoint can open feedback before the current
            # sample. Resolve it through the actual current cutoff before scoring.
            self._apply_feedback_resolutions(
                self.feedback.before_time(timestamp, self.experiment)
            )

        if interaction.dwell_trigger is not None:
            trigger = interaction.dwell_trigger
            presentation_timestamp = (
                float(trigger.timestamp)
                if self.present_action is None
                else _timestamp(
                    "action presentation timestamp",
                    self.present_action(trigger, interaction),
                )
            )
            if presentation_timestamp < float(trigger.timestamp):
                raise ValueError("action presentation cannot precede its dwell trigger")
            opened = self.experiment.open_action_feedback(
                trigger.episode_id, presentation_timestamp
            )
            if not opened:
                raise RuntimeError("dwell trigger did not open its action feedback window")
            self.event_logger.log(
                Event(
                    trigger.timestamp,
                    "integration_dwell_trigger",
                    {
                        "episode_id": trigger.episode_id,
                        "track_id": trigger.track_id,
                        "required_seconds": trigger.required_seconds,
                        "presentation_timestamp": presentation_timestamp,
                    },
                )
            )
            self.event_logger.log(
                Event(
                    presentation_timestamp,
                    "integration_action_presented",
                    {
                        "episode_id": trigger.episode_id,
                        "track_id": trigger.track_id,
                        "trigger_timestamp": float(trigger.timestamp),
                    },
                )
            )
            self.feedback.feedback_opened(
                episode_id=trigger.episode_id,
                outcome_timestamp=presentation_timestamp,
                controller=self.experiment,
            )

        decision: PolicyDecision | None = None
        observation = observation_from_interaction(interaction)
        if observation is not None and interaction.active_episode is not None:
            decision = self.experiment.evaluate_update(
                interaction.active_episode,
                observation,
                self.eeg_source,
            )
            if decision.intent_score is None:
                self._clear_held_if(interaction.active_episode.episode_id)
            else:
                # A decision frozen on update N affects dwell only on update N+1.
                self._held_score_episode_id = interaction.active_episode.episode_id
                self._held_score = decision.intent_score

        self.event_logger.log(
            Event(
                timestamp,
                "integration_gaze_processed",
                {
                    "valid": gaze.valid,
                    "scene_timestamp": interaction.scene_timestamp,
                    "candidate_track_id": (
                        interaction.candidate.track_id if interaction.candidate else None
                    ),
                    "episode_id": (
                        interaction.active_episode.episode_id
                        if interaction.active_episode
                        else None
                    ),
                    "applied_intent_score": applied_score,
                    "decision_reason": decision.reason if decision else None,
                    "decision_newly_frozen": decision.newly_frozen if decision else None,
                    "dwell_accumulated_seconds": interaction.dwell_state.accumulated_seconds,
                    "dwell_required_seconds": interaction.dwell_state.required_seconds,
                    "dwell_triggered": interaction.dwell_state.triggered,
                    "action_gate_open": self.experiment.action_gate_open,
                },
            )
        )
        return IntegratedGazeUpdate(interaction, decision, applied_score)

    def finish(self, timestamp: float) -> float:
        value = max(float(timestamp), self.latest_processed_scientific_timestamp)
        self._through(value)
        ended = self.gaze_pipeline.finish(value)
        if ended is not None:
            self._clear_held_if(ended.episode_id)
            self._handle_ended_episode(ended)
        self._apply_feedback_resolutions(self.feedback.before_time(value, self.experiment))
        completed_at = value + self.experiment.feedback_timeout_s
        self._apply_feedback_resolutions(
            self.feedback.before_time(completed_at, self.experiment)
        )
        if self.experiment.pending_feedback_episode_id is not None:
            raise RuntimeError("feedback remained unresolved after source exhaustion")
        assert_consumed = getattr(self.feedback, "assert_consumed", None)
        if assert_consumed is not None:
            assert_consumed()
        return completed_at

    def cancel_at_deadline(self, timestamp: float) -> float:
        value = self.begin_deadline(timestamp)
        completed_at = value + self.experiment.feedback_timeout_s
        self.advance_time(completed_at)
        self.assert_ready_to_complete()
        return completed_at

    def begin_deadline(self, timestamp: float) -> float:
        """Stop scientific input at its deadline without skipping live feedback grace."""

        value = max(float(timestamp), self.latest_processed_scientific_timestamp)
        self._through(value)
        cancellation = self.gaze_pipeline.cancel(
            value, EpisodeEndReason.SESSION_DURATION_REACHED
        )
        ended = cancellation.ended_episode
        if ended is not None:
            self._clear_held_if(ended.episode_id)
            self._log_episode_end(ended)
            if self.experiment.pending_feedback_episode_id != ended.episode_id:
                self.experiment.cancel_episode(
                    ended.episode_id,
                    value,
                    EpisodeEndReason.SESSION_DURATION_REACHED.value,
                )
        return value

    def advance_time(self, timestamp: float) -> None:
        """Advance live feedback after scientific input has stopped."""

        value = _timestamp("feedback timestamp", timestamp)
        self._apply_feedback_resolutions(
            self.feedback.before_time(value, self.experiment)
        )

    def assert_ready_to_complete(self) -> None:
        """Verify that no feedback or replay input remains before session persistence."""

        if self.experiment.pending_feedback_episode_id is not None:
            raise RuntimeError("feedback remained unresolved after session deadline")
        assert_consumed = getattr(self.feedback, "assert_consumed", None)
        if assert_consumed is not None:
            assert_consumed()

    def _through(self, timestamp: float) -> None:
        value = _timestamp("scientific timestamp", timestamp)
        if value + 1e-12 < self.latest_processed_scientific_timestamp:
            raise ValueError("integration inputs must be processed in scientific-time order")
        self.eeg_source.drain_through(value)
        self._apply_feedback_resolutions(self.feedback.before_time(value, self.experiment))
        self.latest_processed_scientific_timestamp = max(
            self.latest_processed_scientific_timestamp, value
        )

    def _apply_feedback_resolutions(
        self, resolutions: tuple[FeedbackResolution, ...]
    ) -> None:
        for resolution in resolutions:
            for instruction in resolution.cancellation_instructions:
                active = self.gaze_pipeline.episode_tracker.active_episode
                if active is None or active.episode_id != instruction.episode_id:
                    continue
                cancellation = self.gaze_pipeline.cancel(
                    instruction.timestamp,
                    EpisodeEndReason.FEEDBACK_INTERRUPTION,
                )
                if cancellation.ended_episode is not None:
                    self._clear_held_if(cancellation.ended_episode.episode_id)
                    self._log_episode_end(cancellation.ended_episode)

    def _handle_ended_episode(self, episode: CandidateEpisode) -> None:
        self._log_episode_end(episode)
        pending = self.experiment.pending_feedback_episode_id
        if pending == episode.episode_id:
            return
        if pending is not None:
            canceled = self.experiment.cancel_episode(
                episode.episode_id,
                float(episode.end_timestamp),
                "feedback_pending_at_episode_end",
            )
            if canceled is not None:
                return
        opened = self.experiment.open_no_action_feedback(
            episode, float(episode.end_timestamp)
        )
        if opened:
            self.feedback.feedback_opened(
                episode_id=episode.episode_id,
                outcome_timestamp=float(episode.end_timestamp),
                controller=self.experiment,
            )

    def _score_for_this_update(self, gaze: GazeSample) -> float | None:
        active = self.gaze_pipeline.episode_tracker.active_episode
        if (
            active is None
            or self._held_score_episode_id != active.episode_id
            or self._held_score is None
        ):
            return None
        preview = self.gaze_pipeline.associator.associate(gaze)
        if preview.candidate is not None and preview.candidate.track_id != active.track_id:
            self._clear_held_if(active.episode_id)
            return None
        return self._held_score

    def _record_episode_start(self, episode: CandidateEpisode | None) -> None:
        if episode is None or episode.episode_id in self._seen_episode_ids:
            return
        self._seen_episode_ids.add(episode.episode_id)
        self.event_logger.log(
            Event(
                episode.start_timestamp,
                "integration_episode_started",
                {
                    "session_id": self.session_id,
                    "episode_id": episode.episode_id,
                    "track_id": episode.track_id,
                    "start_timestamp": episode.start_timestamp,
                },
            )
        )

    def _log_episode_end(self, episode: CandidateEpisode) -> None:
        if episode.end_timestamp is None:
            raise ValueError("cannot log an active episode as ended")
        self.event_logger.log(
            Event(
                float(episode.end_timestamp),
                "integration_episode_ended",
                {
                    "session_id": self.session_id,
                    "episode_id": episode.episode_id,
                    "track_id": episode.track_id,
                    "end_timestamp": episode.end_timestamp,
                    "end_reason": episode.end_reason.value if episode.end_reason else None,
                },
            )
        )

    def _clear_held_if(self, episode_id: int) -> None:
        if self._held_score_episode_id == episode_id:
            self._held_score_episode_id = None
            self._held_score = None


def _timestamp(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


if __name__ == "__main__":
    print("Use scripts/run_integrated_experiment.py for the configured workflow.")
