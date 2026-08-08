"""Cross-subsystem scientific-time orchestration for one experiment session."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Callable, Protocol

from eeg_pipeline.contracts import EEGSample
from eeg_pipeline.pipeline import EEGPipeline
from eeg_pipeline.recording import EEGHDF5Recorder
from experiment_learning.contracts import PredictionDecision
from experiment_learning.features import observation_from_interaction
from experiment_learning.state_machine import ExperimentController
from foundations.contracts import GazeSample, SceneFrame
from foundations.events import Event, JsonlEventLogger
from gaze_interaction.episodes import CandidateEpisode
from gaze_interaction.pipeline import (
    GazeInteractionPipeline,
    InteractionUpdate,
    SceneUpdate,
)


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


class FeedbackDriver(Protocol):
    def before_time(self, timestamp: float, controller: ExperimentController) -> None: ...

    def feedback_opened(
        self, *, episode_id: int, outcome_timestamp: float, controller: ExperimentController
    ) -> None: ...


class TimedFeedbackDriver:
    """Replay already timestamped button presses with strict episode identity checks."""

    def __init__(
        self,
        presses: Iterable[ScheduledFeedbackPress],
        *,
        event_logger: JsonlEventLogger,
        session_id: str,
    ) -> None:
        ordered = sorted(presses, key=lambda item: item.timestamp)
        self._presses = deque(ordered)
        self.event_logger = event_logger
        self.session_id = session_id

    def before_time(self, timestamp: float, controller: ExperimentController) -> None:
        while self._presses and self._presses[0].timestamp <= timestamp + 1e-12:
            press = self._presses.popleft()
            pending = controller.pending_feedback_episode_id
            if pending != press.episode_id:
                raise RuntimeError(
                    "feedback replay identity mismatch: "
                    f"press targets episode {press.episode_id}, pending episode is {pending}"
                )
            result = controller.button_press(press.timestamp)
            if result is None or result.episode_id != press.episode_id:
                raise RuntimeError("scheduled feedback press did not resolve its episode")
            self._log_press(press)
        controller.advance_time(timestamp)

    def feedback_opened(
        self, *, episode_id: int, outcome_timestamp: float, controller: ExperimentController
    ) -> None:
        del episode_id, outcome_timestamp, controller

    def _log_press(self, press: ScheduledFeedbackPress) -> None:
        self.event_logger.log(
            Event(
                press.timestamp,
                "integration_feedback_press",
                {"session_id": self.session_id, "episode_id": press.episode_id},
            )
        )

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

    def before_time(self, timestamp: float, controller: ExperimentController) -> None:
        poll = self._poll_key
        if poll is None:
            import cv2

            poll = lambda: cv2.waitKey(1) & 0xFF
        if poll() == self.key_code:
            pending = controller.pending_feedback_episode_id
            if pending is None:
                controller.button_press(timestamp)
                self.event_logger.log(
                    Event(
                        timestamp,
                        "integration_feedback_press_ignored",
                        {"session_id": self.session_id, "reason": "no_open_window"},
                    )
                )
            else:
                result = controller.button_press(timestamp)
                if result is not None:
                    self.event_logger.log(
                        Event(
                            timestamp,
                            "integration_feedback_press",
                            {"session_id": self.session_id, "episode_id": pending},
                        )
                    )
        controller.advance_time(timestamp)

    def feedback_opened(
        self, *, episode_id: int, outcome_timestamp: float, controller: ExperimentController
    ) -> None:
        del episode_id, outcome_timestamp, controller


class TimedEEGFeeder:
    """Feed raw EEG only through the current scientific-time cutoff."""

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
        self.last_ingested_timestamp: float | None = None

    def feed_through(self, timestamp: float) -> None:
        while self._next is not None and self._next.timestamp <= timestamp + 1e-12:
            sample = self._next
            if sample.timestamp > timestamp + 1e-12:
                raise RuntimeError("EEG feeder attempted to ingest a future sample")
            self.pipeline.add_sample(sample)
            if self.recorder is not None:
                self.recorder.record(sample)
            self.last_ingested_timestamp = float(sample.timestamp)
            self._next = next(self._iterator, None)


@dataclass(frozen=True)
class IntegratedGazeUpdate:
    interaction: InteractionUpdate
    prediction: PredictionDecision | None
    applied_intent_score: float | None


class IntegratedExperimentOrchestrator:
    """Connect gaze interaction, EEG, learning, feedback, and scientific logging."""

    def __init__(
        self,
        *,
        gaze_pipeline: GazeInteractionPipeline,
        eeg_feeder: TimedEEGFeeder,
        experiment: ExperimentController,
        feedback: FeedbackDriver,
        event_logger: JsonlEventLogger,
        session_id: str,
    ) -> None:
        self.gaze_pipeline = gaze_pipeline
        self.eeg_feeder = eeg_feeder
        self.experiment = experiment
        self.feedback = feedback
        self.event_logger = event_logger
        self.session_id = session_id
        self._held_score_episode_id: int | None = None
        self._held_score: float | None = None
        self._seen_episode_ids: set[int] = set()
        self.last_gaze_timestamp = 0.0

    def process_scene(self, frame: SceneFrame) -> SceneUpdate:
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
        self.last_gaze_timestamp = max(self.last_gaze_timestamp, timestamp)

        # Resolve press/timeout outcomes and expose raw EEG only through this cutoff.
        self.feedback.before_time(timestamp, self.experiment)
        self.eeg_feeder.feed_through(timestamp)

        applied_score = self._score_for_this_update(gaze)
        interaction = self.gaze_pipeline.process_gaze(
            gaze, intent_score=applied_score
        )
        self._record_episode_start(interaction.active_episode)

        if interaction.ended_episode is not None:
            ended = interaction.ended_episode
            self._clear_held_if(ended.episode_id)
            self._log_episode_end(ended)
            self.experiment.on_episode_end(ended)
            self.feedback.feedback_opened(
                episode_id=ended.episode_id,
                outcome_timestamp=float(ended.end_timestamp),
                controller=self.experiment,
            )
            # A gap timeout can end retrospectively.  Resolve anything due by the
            # current gaze before a new episode is allowed to predict.
            self.feedback.before_time(timestamp, self.experiment)

        if interaction.dwell_trigger is not None:
            trigger = interaction.dwell_trigger
            self.experiment.on_dwell_trigger(trigger)
            self.event_logger.log(
                Event(
                    trigger.timestamp,
                    "integration_dwell_trigger",
                    {
                        "episode_id": trigger.episode_id,
                        "track_id": trigger.track_id,
                        "required_seconds": trigger.required_seconds,
                    },
                )
            )
            self.feedback.feedback_opened(
                episode_id=trigger.episode_id,
                outcome_timestamp=float(trigger.timestamp),
                controller=self.experiment,
            )

        prediction: PredictionDecision | None = None
        observation = observation_from_interaction(interaction)
        if observation is not None and interaction.active_episode is not None:
            prediction = self.experiment.consider_prediction(
                interaction.active_episode, observation, self.eeg_feeder.pipeline
            )
            if prediction.intent_score is None:
                self._clear_held_if(interaction.active_episode.episode_id)
            else:
                self._held_score_episode_id = interaction.active_episode.episode_id
                self._held_score = prediction.intent_score

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
                    "prediction_reason": prediction.reason if prediction else None,
                    "dwell_accumulated_seconds": interaction.dwell_state.accumulated_seconds,
                    "dwell_required_seconds": interaction.dwell_state.required_seconds,
                    "dwell_triggered": interaction.dwell_state.triggered,
                },
            )
        )
        return IntegratedGazeUpdate(interaction, prediction, applied_score)

    def finish(self, timestamp: float) -> None:
        value = max(float(timestamp), self.last_gaze_timestamp)
        self.feedback.before_time(value, self.experiment)
        ended = self.gaze_pipeline.finish(value)
        if ended is not None:
            self._clear_held_if(ended.episode_id)
            self._log_episode_end(ended)
            self.experiment.on_episode_end(ended)
            self.feedback.feedback_opened(
                episode_id=ended.episode_id,
                outcome_timestamp=float(ended.end_timestamp),
                controller=self.experiment,
            )
        # Source exhaustion is a graceful close: honor a legitimate scheduled
        # press, otherwise let the authoritative state machine derive timeout.
        self.feedback.before_time(
            value + self.experiment.feedback_timeout_s, self.experiment
        )
        if self.experiment.pending_feedback_episode_id is not None:
            raise RuntimeError("feedback remained unresolved after graceful source exhaustion")
        assert_consumed = getattr(self.feedback, "assert_consumed", None)
        if assert_consumed is not None:
            assert_consumed()

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
            # A direct identity switch is known before pipeline mutation, so the
            # previous episode's score cannot leak into the new episode update.
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


if __name__ == "__main__":
    print("Use scripts/run_integrated_experiment.py for the configured workflow.")
