"""Boundary helpers for gaze observations and auditable EEG feature values."""

from __future__ import annotations

import math

from eeg_pipeline.contracts import EEGFeatureWindow, QualityState
from eeg_pipeline.processing import FEATURE_NAMES as EEG_FEATURE_NAMES
from gaze_interaction.pipeline import InteractionUpdate

from experiment_learning.contracts import GazeContextObservation


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


def eeg_feature_mapping(feature_window: EEGFeatureWindow) -> dict[str, float] | None:
    """Return all original Instance 3 values for provenance and formula changes."""

    if feature_window.quality_state is not QualityState.USABLE or feature_window.values is None:
        return None
    if feature_window.feature_names != EEG_FEATURE_NAMES:
        raise ValueError("EEG feature signature differs from the approved Instance 3 order")
    values = {
        name: float(value)
        for name, value in zip(feature_window.feature_names, feature_window.values, strict=True)
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("EEG feature values must be finite")
    return values


if __name__ == "__main__":
    print(EEG_FEATURE_NAMES)
