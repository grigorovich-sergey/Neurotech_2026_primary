"""Candidate-episode lifecycle over deterministic tracked-object associations."""

from dataclasses import dataclass, replace
from enum import Enum
import math
from numbers import Real

from gaze_interaction.contracts import BoundingBox, TrackedObject


class EpisodeEndReason(str, Enum):
    CANDIDATE_CHANGE = "candidate_change"
    GAP_TIMEOUT = "gap_timeout"
    SOURCE_END = "source_end"
    FEEDBACK_INTERRUPTION = "feedback_interruption"
    SESSION_DURATION_REACHED = "session_duration_reached"


_CANCELLATION_REASONS = {
    EpisodeEndReason.FEEDBACK_INTERRUPTION,
    EpisodeEndReason.SESSION_DURATION_REACHED,
}


def _valid_timestamp(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class CandidateEpisode:
    """Downstream contract for one stable tracked-object gaze candidate."""

    episode_id: int
    track_id: int
    label: str | None
    start_timestamp: float
    last_match_timestamp: float
    end_timestamp: float | None
    end_reason: EpisodeEndReason | None

    def __post_init__(self) -> None:
        if isinstance(self.episode_id, bool) or not isinstance(self.episode_id, int):
            raise TypeError("episode_id must be an integer")
        if self.episode_id <= 0:
            raise ValueError("episode_id must be positive")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise TypeError("track_id must be an integer")
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        if self.label is not None and (not isinstance(self.label, str) or not self.label):
            raise ValueError("label must be a non-empty string or None")
        start = _valid_timestamp("start_timestamp", self.start_timestamp)
        last_match = _valid_timestamp("last_match_timestamp", self.last_match_timestamp)
        object.__setattr__(self, "start_timestamp", start)
        object.__setattr__(self, "last_match_timestamp", last_match)
        if last_match < start:
            raise ValueError("last_match_timestamp cannot precede start_timestamp")
        if self.end_timestamp is None:
            if self.end_reason is not None:
                raise ValueError("active episodes cannot have an end_reason")
        else:
            end = _valid_timestamp("end_timestamp", self.end_timestamp)
            object.__setattr__(self, "end_timestamp", end)
            if end < last_match:
                raise ValueError("end_timestamp cannot precede last_match_timestamp")
            if not isinstance(self.end_reason, EpisodeEndReason):
                raise TypeError("ended episodes require an EpisodeEndReason")

    @property
    def active(self) -> bool:
        return self.end_timestamp is None


@dataclass(frozen=True)
class EpisodeUpdate:
    active_episode: CandidateEpisode | None
    ended_episode: CandidateEpisode | None
    started_episode: CandidateEpisode | None


class EpisodeTracker:
    """Maintain one candidate episode with a bounded no-match grace interval."""

    def __init__(self, *, gap_grace_seconds: float) -> None:
        if not math.isfinite(gap_grace_seconds) or gap_grace_seconds < 0.0:
            raise ValueError("gap_grace_seconds must be finite and non-negative")
        self.gap_grace_seconds = float(gap_grace_seconds)
        self._active: CandidateEpisode | None = None
        self._next_episode_id = 1
        self._last_update_timestamp: float | None = None

    @property
    def active_episode(self) -> CandidateEpisode | None:
        return self._active

    def update(
        self, candidate: TrackedObject | None, timestamp: float
    ) -> EpisodeUpdate:
        if candidate is not None and not isinstance(candidate, TrackedObject):
            raise TypeError("candidate must be a TrackedObject or None")
        timestamp_value = self._check_order(timestamp)
        ended: CandidateEpisode | None = None
        started: CandidateEpisode | None = None

        if (
            self._active is not None
            and timestamp_value - self._active.last_match_timestamp
            > self.gap_grace_seconds
        ):
            ended = self._end_active(
                self._active.last_match_timestamp + self.gap_grace_seconds,
                EpisodeEndReason.GAP_TIMEOUT,
            )

        if candidate is None:
            return EpisodeUpdate(self._active, ended, None)

        if self._active is None:
            started = self._start(candidate, timestamp_value)
            return EpisodeUpdate(self._active, ended, started)

        if candidate.track_id == self._active.track_id:
            self._active = replace(self._active, last_match_timestamp=timestamp_value)
            return EpisodeUpdate(self._active, ended, None)

        ended = self._end_active(timestamp_value, EpisodeEndReason.CANDIDATE_CHANGE)
        started = self._start(candidate, timestamp_value)
        return EpisodeUpdate(self._active, ended, started)

    def finish(self, timestamp: float) -> CandidateEpisode | None:
        timestamp_value = self._check_order(timestamp)
        if self._active is None:
            return None
        if (
            timestamp_value - self._active.last_match_timestamp
            > self.gap_grace_seconds
        ):
            return self._end_active(
                self._active.last_match_timestamp + self.gap_grace_seconds,
                EpisodeEndReason.GAP_TIMEOUT,
            )
        return self._end_active(timestamp_value, EpisodeEndReason.SOURCE_END)

    def cancel(
        self, timestamp: float, reason: EpisodeEndReason
    ) -> CandidateEpisode | None:
        """End the active episode for an external interruption or deadline."""

        if not isinstance(reason, EpisodeEndReason):
            raise TypeError("reason must be an EpisodeEndReason")
        if reason not in _CANCELLATION_REASONS:
            raise ValueError(
                "cancellation reason must be feedback_interruption or "
                "session_duration_reached"
            )
        timestamp_value = self._check_order(timestamp)
        if self._active is None:
            return None
        return self._end_active(timestamp_value, reason)

    def _check_order(self, timestamp: float) -> float:
        value = _valid_timestamp("timestamp", timestamp)
        if self._last_update_timestamp is not None and value < self._last_update_timestamp:
            raise ValueError("episode updates must have non-decreasing timestamps")
        self._last_update_timestamp = value
        return value

    def _start(self, candidate: TrackedObject, timestamp: float) -> CandidateEpisode:
        self._active = CandidateEpisode(
            episode_id=self._next_episode_id,
            track_id=candidate.track_id,
            label=candidate.label,
            start_timestamp=timestamp,
            last_match_timestamp=timestamp,
            end_timestamp=None,
            end_reason=None,
        )
        self._next_episode_id += 1
        return self._active

    def _end_active(
        self, timestamp: float, reason: EpisodeEndReason
    ) -> CandidateEpisode:
        if self._active is None:
            raise RuntimeError("cannot end an episode when none is active")
        ended = replace(self._active, end_timestamp=timestamp, end_reason=reason)
        self._active = None
        return ended


if __name__ == "__main__":
    candidate = TrackedObject(1, BoundingBox(0.2, 0.2, 0.8, 0.8), "object", 0.9, 0.0)
    tracker = EpisodeTracker(gap_grace_seconds=0.15)
    print(tracker.update(candidate, 0.0))
    print(tracker.update(candidate, 0.1))
    print(tracker.finish(0.2))
