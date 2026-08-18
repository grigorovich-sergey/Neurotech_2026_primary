"""Observed gaze dwell accumulation with optional bounded intent adaptation."""

from dataclasses import dataclass
import math
from numbers import Real

from gaze_interaction.episodes import CandidateEpisode


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _intent_score(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("intent_score must be a real number or None")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("intent_score must be within [0, 1]")
    return result


@dataclass(frozen=True)
class DwellState:
    episode_id: int | None
    accumulated_seconds: float
    required_seconds: float
    triggered: bool
    trigger_pending: bool = False

    @property
    def progress(self) -> float:
        if self.required_seconds <= 0.0:
            return 0.0
        return min(1.0, self.accumulated_seconds / self.required_seconds)


@dataclass(frozen=True)
class DwellTrigger:
    """One-shot action request emitted at most once for a candidate episode."""

    episode_id: int
    track_id: int
    timestamp: float
    required_seconds: float


class DwellController:
    """Accumulate only consecutive, explicitly observed matches for one episode."""

    def __init__(
        self,
        *,
        baseline_seconds: float,
        minimum_seconds: float,
        maximum_seconds: float,
        maximum_reduction_fraction: float,
        max_sample_gap_seconds: float,
    ) -> None:
        self.baseline_seconds = _positive("baseline_seconds", baseline_seconds)
        self.minimum_seconds = _positive("minimum_seconds", minimum_seconds)
        self.maximum_seconds = _positive("maximum_seconds", maximum_seconds)
        self.max_sample_gap_seconds = _positive(
            "max_sample_gap_seconds", max_sample_gap_seconds
        )
        if not self.minimum_seconds <= self.baseline_seconds <= self.maximum_seconds:
            raise ValueError(
                "minimum_seconds <= baseline_seconds <= maximum_seconds is required"
            )
        if (
            isinstance(maximum_reduction_fraction, bool)
            or not isinstance(maximum_reduction_fraction, Real)
            or not math.isfinite(float(maximum_reduction_fraction))
            or not 0.0 <= float(maximum_reduction_fraction) <= 1.0
        ):
            raise ValueError("maximum_reduction_fraction must be within [0, 1]")
        self.maximum_reduction_fraction = float(maximum_reduction_fraction)
        self._episode_id: int | None = None
        self._accumulated_seconds = 0.0
        self._last_matched_timestamp: float | None = None
        self._previous_sample_matched = False
        self._triggered = False
        self._trigger_pending = False
        self._pending_required_seconds: float | None = None
        self._last_update_timestamp: float | None = None

    def required_seconds(self, intent_score: float | None) -> float:
        score = _intent_score(intent_score)
        if score is None:
            return self.baseline_seconds
        proposed = self.baseline_seconds * (
            1.0 - self.maximum_reduction_fraction * score
        )
        return min(self.maximum_seconds, max(self.minimum_seconds, proposed))

    def advance(
        self,
        episode: CandidateEpisode | None,
        *,
        matched: bool,
        timestamp: float,
        intent_score: float | None = None,
        trigger_gate_open: bool = True,
    ) -> tuple[DwellState, DwellTrigger | None]:
        if not isinstance(matched, bool):
            raise TypeError("matched must be a bool")
        if not isinstance(trigger_gate_open, bool):
            raise TypeError("trigger_gate_open must be a bool")
        if isinstance(timestamp, bool) or not isinstance(timestamp, Real):
            raise TypeError("timestamp must be a real number")
        timestamp_value = float(timestamp)
        if not math.isfinite(timestamp_value) or timestamp_value < 0.0:
            raise ValueError("timestamp must be finite and non-negative")
        if (
            self._last_update_timestamp is not None
            and timestamp_value < self._last_update_timestamp
        ):
            raise ValueError("dwell updates must have non-decreasing timestamps")
        self._last_update_timestamp = timestamp_value
        requirement = self.required_seconds(intent_score)

        if episode is None:
            if matched:
                raise ValueError("matched cannot be true without an active episode")
            self._reset_episode()
            return DwellState(None, 0.0, requirement, False), None
        if not episode.active:
            raise ValueError("dwell can only advance an active episode")

        if episode.episode_id != self._episode_id:
            self._episode_id = episode.episode_id
            self._accumulated_seconds = 0.0
            self._last_matched_timestamp = None
            self._previous_sample_matched = False
            self._triggered = False
            self._trigger_pending = False
            self._pending_required_seconds = None

        if not matched:
            self._previous_sample_matched = False
            return self._state(requirement), None

        if (
            self._previous_sample_matched
            and self._last_matched_timestamp is not None
        ):
            interval = timestamp_value - self._last_matched_timestamp
            if interval <= self.max_sample_gap_seconds or math.isclose(
                interval,
                self.max_sample_gap_seconds,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                self._accumulated_seconds += interval
        self._last_matched_timestamp = timestamp_value
        self._previous_sample_matched = True

        trigger: DwellTrigger | None = None
        trigger_requirement = requirement
        if self._trigger_pending and trigger_gate_open:
            if self._pending_required_seconds is None:
                raise RuntimeError("pending dwell trigger has no stored requirement")
            trigger_requirement = self._pending_required_seconds
            self._trigger_pending = False
            self._pending_required_seconds = None
            self._triggered = True
            trigger = DwellTrigger(
                episode_id=episode.episode_id,
                track_id=episode.track_id,
                timestamp=timestamp_value,
                required_seconds=trigger_requirement,
            )
        elif (
            not self._triggered
            and not self._trigger_pending
            and self._accumulated_seconds >= requirement
        ):
            if trigger_gate_open:
                self._triggered = True
                trigger = DwellTrigger(
                    episode_id=episode.episode_id,
                    track_id=episode.track_id,
                    timestamp=timestamp_value,
                    required_seconds=requirement,
                )
            else:
                self._trigger_pending = True
                self._pending_required_seconds = requirement
        return self._state(requirement), trigger

    def _state(self, requirement: float) -> DwellState:
        return DwellState(
            episode_id=self._episode_id,
            accumulated_seconds=self._accumulated_seconds,
            required_seconds=requirement,
            triggered=self._triggered,
            trigger_pending=self._trigger_pending,
        )

    def _cancel(self, timestamp: float) -> tuple[DwellState, bool]:
        """Clear dwell state without moving scientific time backwards."""

        timestamp_value = float(timestamp)

        if not math.isfinite(timestamp_value) or timestamp_value < 0.0:
            raise ValueError("timestamp must be finite and non-negative")

        # Feedback/key timestamps can be captured a few microseconds before
        # gaze samples that have already been processed. Cancellation is an
        # external interruption, so clamp it to the latest dwell update.
        if self._last_update_timestamp is not None:
            timestamp_value = max(
                timestamp_value,
                self._last_update_timestamp,
            )

        discarded_pending_trigger = self._trigger_pending

        state, trigger = self.advance(
            None,
            matched=False,
            timestamp=timestamp_value,
        )

        if trigger is not None:
            raise RuntimeError("dwell cancellation cannot emit a trigger")

        return state, discarded_pending_trigger

    def _reset_episode(self) -> None:
        self._episode_id = None
        self._accumulated_seconds = 0.0
        self._last_matched_timestamp = None
        self._previous_sample_matched = False
        self._triggered = False
        self._trigger_pending = False
        self._pending_required_seconds = None


if __name__ == "__main__":
    episode = CandidateEpisode(1, 7, "object", 0.0, 0.0, None, None)
    controller = DwellController(
        baseline_seconds=0.2,
        minimum_seconds=0.1,
        maximum_seconds=0.2,
        maximum_reduction_fraction=0.5,
        max_sample_gap_seconds=0.1,
    )
    for timestamp in (0.0, 0.05, 0.1, 0.15):
        print(controller.advance(episode, matched=True, timestamp=timestamp))
