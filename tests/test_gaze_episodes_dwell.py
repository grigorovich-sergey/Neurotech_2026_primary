import pytest

from gaze_interaction.contracts import BoundingBox, TrackedObject
from gaze_interaction.dwell import DwellController
from gaze_interaction.episodes import CandidateEpisode, EpisodeEndReason, EpisodeTracker


def _candidate(track_id: int) -> TrackedObject:
    return TrackedObject(
        track_id,
        BoundingBox(0.2, 0.2, 0.8, 0.8),
        f"object-{track_id}",
        0.9,
        0.0,
    )


def _episode(episode_id: int, track_id: int) -> CandidateEpisode:
    return CandidateEpisode(episode_id, track_id, "object", 0.0, 0.0, None, None)


def _dwell(**overrides: float) -> DwellController:
    values = {
        "baseline_seconds": 0.2,
        "minimum_seconds": 0.1,
        "maximum_seconds": 0.2,
        "maximum_reduction_fraction": 0.5,
        "max_sample_gap_seconds": 0.1,
    }
    values.update(overrides)
    return DwellController(**values)


def test_episode_start_continue_switch_grace_resume_and_timeout() -> None:
    first = _candidate(1)
    second = _candidate(2)
    episodes = EpisodeTracker(gap_grace_seconds=0.15)

    started = episodes.update(first, 0.0)
    continued = episodes.update(first, 0.05)
    paused = episodes.update(None, 0.1)
    resumed = episodes.update(first, 0.15)
    switched = episodes.update(second, 0.2)

    assert started.started_episode is not None
    assert continued.active_episode is not None
    assert paused.active_episode is not None
    assert resumed.active_episode is not None
    assert resumed.active_episode.episode_id == started.started_episode.episode_id
    assert switched.ended_episode is not None
    assert switched.ended_episode.end_reason == EpisodeEndReason.CANDIDATE_CHANGE
    assert switched.started_episode is not None
    assert switched.started_episode.episode_id != started.started_episode.episode_id

    timed_out = episodes.update(None, 0.36)
    assert timed_out.active_episode is None
    assert timed_out.ended_episode is not None
    assert timed_out.ended_episode.end_reason == EpisodeEndReason.GAP_TIMEOUT
    assert timed_out.ended_episode.end_timestamp == pytest.approx(0.35)


def test_reentry_after_grace_is_a_new_episode_and_run_end_respects_expired_gap() -> None:
    candidate = _candidate(4)
    episodes = EpisodeTracker(gap_grace_seconds=0.15)
    first = episodes.update(candidate, 0.0).started_episode
    reentered = episodes.update(candidate, 0.2)

    assert first is not None
    assert reentered.ended_episode is not None
    assert reentered.ended_episode.end_reason == EpisodeEndReason.GAP_TIMEOUT
    assert reentered.started_episode is not None
    assert reentered.started_episode.episode_id != first.episode_id

    ended = episodes.finish(0.4)
    assert ended is not None
    assert ended.end_reason == EpisodeEndReason.GAP_TIMEOUT
    assert ended.end_timestamp == pytest.approx(0.35)


def test_dwell_pauses_on_observed_gap_ignores_long_sample_gap_and_triggers_once() -> None:
    dwell = _dwell(maximum_reduction_fraction=0.0)
    episode = _episode(1, 7)

    dwell.advance(episode, matched=True, timestamp=0.0)
    state, _ = dwell.advance(episode, matched=True, timestamp=0.05)
    assert state.accumulated_seconds == pytest.approx(0.05)
    dwell.advance(episode, matched=False, timestamp=0.06)
    state, _ = dwell.advance(episode, matched=True, timestamp=0.1)
    assert state.accumulated_seconds == pytest.approx(0.05)
    state, _ = dwell.advance(episode, matched=True, timestamp=0.25)
    assert state.accumulated_seconds == pytest.approx(0.05)

    state, _ = dwell.advance(episode, matched=True, timestamp=0.3)
    state, first_trigger = dwell.advance(episode, matched=True, timestamp=0.4)
    _, second_trigger = dwell.advance(episode, matched=True, timestamp=0.45)
    assert state.accumulated_seconds == pytest.approx(0.2)
    assert first_trigger is not None
    assert second_trigger is None


def test_intent_score_is_bounded_and_none_uses_baseline() -> None:
    dwell = DwellController(
        baseline_seconds=1.0,
        minimum_seconds=0.35,
        maximum_seconds=1.0,
        maximum_reduction_fraction=0.5,
        max_sample_gap_seconds=0.1,
    )

    assert dwell.required_seconds(None) == pytest.approx(1.0)
    assert dwell.required_seconds(0.75) == pytest.approx(0.625)
    assert dwell.required_seconds(1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="intent_score"):
        dwell.required_seconds(1.01)


def test_candidate_change_resets_dwell_state() -> None:
    dwell = _dwell(maximum_reduction_fraction=0.0)
    first = _episode(1, 1)
    second = _episode(2, 2)
    dwell.advance(first, matched=True, timestamp=0.0)
    dwell.advance(first, matched=True, timestamp=0.05)

    state, trigger = dwell.advance(second, matched=True, timestamp=0.06)

    assert state.episode_id == 2
    assert state.accumulated_seconds == 0.0
    assert trigger is None
