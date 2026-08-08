"""Stable, interpretable gaze/context and EEG feature assembly."""

import math

from eeg_pipeline.contracts import EEGFeatureWindow, QualityState
from eeg_pipeline.processing import FEATURE_NAMES as EEG_FEATURE_NAMES
from gaze_interaction.episodes import CandidateEpisode
from gaze_interaction.pipeline import InteractionUpdate

from experiment_learning.contracts import GazeContextObservation


GAZE_FEATURE_NAMES = (
    "candidate_elapsed_s",
    "matched_dwell_s",
    "gaze_center_dx_norm",
    "gaze_center_dy_norm",
    "candidate_width_norm",
    "candidate_height_norm",
    "candidate_area_norm",
)

E_FEATURE_NAMES = GAZE_FEATURE_NAMES + EEG_FEATURE_NAMES


def observation_from_interaction(update: InteractionUpdate) -> GazeContextObservation | None:
    """Build local context only for an explicitly matched current candidate."""

    episode = update.active_episode
    candidate = update.candidate
    if episode is None or candidate is None or candidate.track_id != episode.track_id:
        return None
    gaze = update.gaze
    if not gaze.valid or gaze.x_normalized is None or gaze.y_normalized is None:
        return None
    if update.dwell_state.episode_id != episode.episode_id:
        raise ValueError("dwell state episode does not match active candidate episode")
    return GazeContextObservation(
        episode_id=episode.episode_id,
        track_id=episode.track_id,
        timestamp=float(gaze.timestamp),
        matched_dwell_s=update.dwell_state.accumulated_seconds,
        gaze_x_normalized=gaze.x_normalized,
        gaze_y_normalized=gaze.y_normalized,
        candidate_box=candidate.box,
    )


def gaze_features(
    episode: CandidateEpisode, observation: GazeContextObservation
) -> dict[str, float]:
    """Return the fixed seven-feature G vector at the observation cutoff."""

    if not episode.active:
        raise ValueError("prediction features require an active CandidateEpisode")
    if observation.episode_id != episode.episode_id or observation.track_id != episode.track_id:
        raise ValueError("observation identity does not match CandidateEpisode")
    if observation.timestamp < episode.start_timestamp:
        raise ValueError("observation cannot precede episode start")
    if observation.timestamp != episode.last_match_timestamp:
        raise ValueError("observation must be the episode's current confirmed match")
    if observation.matched_dwell_s > observation.timestamp - episode.start_timestamp:
        raise ValueError("matched dwell cannot exceed elapsed candidate duration")

    box = observation.candidate_box
    width = box.x_max - box.x_min
    height = box.y_max - box.y_min
    center_x = (box.x_min + box.x_max) / 2.0
    center_y = (box.y_min + box.y_max) / 2.0
    values = (
        observation.timestamp - episode.start_timestamp,
        observation.matched_dwell_s,
        observation.gaze_x_normalized - center_x,
        observation.gaze_y_normalized - center_y,
        width,
        height,
        width * height,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("gaze/context features must be finite")
    return dict(zip(GAZE_FEATURE_NAMES, values, strict=True))


def eeg_features(feature_window: EEGFeatureWindow) -> dict[str, float] | None:
    """Use Instance 3 features as-is; unusable EEG remains explicitly unavailable."""

    if feature_window.quality_state is not QualityState.USABLE or feature_window.values is None:
        return None
    if feature_window.feature_names != EEG_FEATURE_NAMES:
        raise ValueError("EEG feature signature differs from the approved Instance 3 order")
    return {
        name: float(value)
        for name, value in zip(feature_window.feature_names, feature_window.values, strict=True)
    }


def combined_features(
    gaze: dict[str, float], eeg: dict[str, float]
) -> dict[str, float]:
    if tuple(gaze) != GAZE_FEATURE_NAMES:
        raise ValueError("gaze feature order/signature is not the approved G vector")
    if tuple(eeg) != EEG_FEATURE_NAMES:
        raise ValueError("EEG feature order/signature is not the approved Instance 3 vector")
    return {**gaze, **eeg}


if __name__ == "__main__":
    from gaze_interaction.contracts import BoundingBox

    episode = CandidateEpisode(1, 2, "object", 1.0, 1.25, None, None)
    observation = GazeContextObservation(
        1, 2, 1.25, 0.2, 0.55, 0.45, BoundingBox(0.2, 0.2, 0.8, 0.8)
    )
    print(gaze_features(episode, observation))
