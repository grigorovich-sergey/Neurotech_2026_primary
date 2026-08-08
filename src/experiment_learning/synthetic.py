"""Deterministic hardware-free stress workflow for the Instance 4 state machine."""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eeg_pipeline.contracts import EEGFeatureWindow, QualityState, WindowCompleteness
from eeg_pipeline.processing import FEATURE_NAMES as EEG_FEATURE_NAMES
from foundations.config import save_resolved_config
from foundations.events import JsonlEventLogger
from gaze_interaction.contracts import BoundingBox
from gaze_interaction.dwell import DwellTrigger
from gaze_interaction.episodes import CandidateEpisode, EpisodeEndReason

from experiment_learning.checkpoint import (
    ParticipantState,
    load_participant_checkpoint,
    save_participant_checkpoint,
)
from experiment_learning.contracts import GazeContextObservation
from experiment_learning.models import ModelConfig
from experiment_learning.state_machine import ExperimentController


FEEDBACK_CASES = {
    "action_no_press": (True, False),
    "action_press": (True, True),
    "no_action_no_press": (False, False),
    "no_action_press": (False, True),
}


@dataclass
class _SyntheticEEGSource:
    quality_state: QualityState
    values: np.ndarray

    def features(self, start: float, end: float) -> EEGFeatureWindow:
        values = self.values if self.quality_state is QualityState.USABLE else None
        has_samples = self.quality_state is not QualityState.UNAVAILABLE
        reasons = () if self.quality_state is QualityState.USABLE else (
            "synthetic_rejected"
            if self.quality_state is QualityState.REJECTED
            else "synthetic_unavailable",
        )
        return EEGFeatureWindow(
            requested_start=start,
            requested_end=end,
            actual_start=start if has_samples else None,
            actual_end=end if has_samples else None,
            sample_count=251 if has_samples else 0,
            completeness=(
                WindowCompleteness.COMPLETE
                if has_samples
                else WindowCompleteness.EMPTY
            ),
            quality_state=self.quality_state,
            quality_reasons=reasons,
            feature_names=EEG_FEATURE_NAMES,
            values=values,
        )


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _positive(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def validate_synthetic_config(config: dict[str, Any]) -> None:
    if not isinstance(config.get("output_root"), str) or not config["output_root"]:
        raise ValueError("output_root must be a non-empty path string")
    if (
        not isinstance(config.get("checkpoint_filename"), str)
        or not config["checkpoint_filename"]
    ):
        raise ValueError("checkpoint_filename must be a non-empty path string")
    if not isinstance(config.get("participant_id"), str) or not config["participant_id"]:
        raise ValueError("participant_id must be a non-empty string")
    for key in ("participant_sequence_index", "seed", "episodes", "sessions"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if config["episodes"] <= 0 or config["sessions"] <= 0:
        raise ValueError("episodes and sessions must be positive")
    if config["episodes"] < config["sessions"]:
        raise ValueError("episodes must be at least sessions")

    timing = _mapping(config, "timing")
    for key in (
        "minimum_prediction_elapsed_s",
        "eeg_window_s",
        "feedback_timeout_s",
        "episode_spacing_s",
        "outcome_delay_s",
        "button_delay_s",
    ):
        _positive(f"timing.{key}", timing.get(key))
    if timing["episode_spacing_s"] <= timing["feedback_timeout_s"] + timing["outcome_delay_s"]:
        raise ValueError("episode spacing must leave enough time for feedback timeout")

    model = _mapping(config, "model")
    ModelConfig(
        learning_rate=model.get("learning_rate"),
        l2=model.get("l2"),
        decision_threshold=model.get("decision_threshold"),
    )
    synthetic = _mapping(config, "synthetic")
    cases = synthetic.get("feedback_case_cycle")
    if not isinstance(cases, list) or not cases or any(case not in FEEDBACK_CASES for case in cases):
        raise ValueError("synthetic.feedback_case_cycle contains an unknown feedback case")
    quality_cycle = synthetic.get("eeg_quality_cycle")
    valid_quality = {state.value for state in QualityState}
    if (
        not isinstance(quality_cycle, list)
        or not quality_cycle
        or any(value not in valid_quality for value in quality_cycle)
    ):
        raise ValueError("synthetic.eeg_quality_cycle contains an unknown quality state")
    controlled_every = synthetic.get("controlled_trial_every_n")
    if isinstance(controlled_every, bool) or not isinstance(controlled_every, int) or controlled_every < 0:
        raise ValueError("synthetic.controlled_trial_every_n must be a non-negative integer")

    resume = _mapping(config, "resume_check")
    if not isinstance(resume.get("enabled"), bool):
        raise ValueError("resume_check.enabled must be a bool")
    after_session = resume.get("after_session")
    if isinstance(after_session, bool) or not isinstance(after_session, int) or after_session <= 0:
        raise ValueError("resume_check.after_session must be a positive integer")
    if resume["enabled"] and after_session >= config["sessions"]:
        raise ValueError("resume_check.after_session must be less than sessions")


def _new_run_directory(output_root: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(output_root) / "experiment_learning" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _model_config(config: dict[str, Any]) -> ModelConfig:
    values = config["model"]
    return ModelConfig(
        learning_rate=float(values["learning_rate"]),
        l2=float(values["l2"]),
        decision_threshold=float(values["decision_threshold"]),
    )


def run_synthetic_experiment(config: dict[str, Any]) -> Path:
    """Exercise paired prediction, feedback, learning, scheduling, and resume."""

    validate_synthetic_config(config)
    resolved = deepcopy(config)
    run_directory = _new_run_directory(resolved["output_root"])
    save_resolved_config(resolved, run_directory / "resolved_config.json")
    events = JsonlEventLogger(run_directory / "events.jsonl")
    checkpoint_path = run_directory / resolved["checkpoint_filename"]
    model_config = _model_config(resolved)
    state = ParticipantState.create(
        participant_id=resolved["participant_id"],
        participant_sequence_index=resolved["participant_sequence_index"],
        model_config=model_config,
    )
    rng = np.random.default_rng(resolved["seed"])
    timing = resolved["timing"]
    synthetic = resolved["synthetic"]
    resume = resolved["resume_check"]
    condition_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    feedback_counts: Counter[str] = Counter()
    predicted_count = 0
    unavailable_count = 0
    result_count = 0
    controlled_count = 0
    episode_index = 0

    per_session, remainder = divmod(resolved["episodes"], resolved["sessions"])
    for session_number in range(resolved["sessions"]):
        session_index, condition = state.schedule.allocate_next()
        if session_index != session_number:
            raise RuntimeError("synthetic session schedule did not resume deterministically")
        condition_counts[condition.value] += 1
        save_participant_checkpoint(checkpoint_path, state)
        controller = ExperimentController(
            participant_state=state,
            session_id=f"synthetic-session-{session_number + 1:03d}",
            active_condition=condition,
            minimum_prediction_elapsed_s=timing["minimum_prediction_elapsed_s"],
            eeg_window_s=timing["eeg_window_s"],
            feedback_timeout_s=timing["feedback_timeout_s"],
            checkpoint_path=checkpoint_path,
            event_logger=events,
        )
        session_episode_count = per_session + (1 if session_number < remainder else 0)
        for _ in range(session_episode_count):
            episode_id = episode_index + 1
            start = timing["eeg_window_s"] + 1.0 + episode_index * timing["episode_spacing_s"]
            cutoff = start + timing["minimum_prediction_elapsed_s"]
            box_x = float(rng.uniform(0.1, 0.45))
            box_y = float(rng.uniform(0.1, 0.45))
            width = float(rng.uniform(0.2, 0.4))
            height = float(rng.uniform(0.2, 0.4))
            box = BoundingBox(box_x, box_y, box_x + width, box_y + height)
            gaze_x = float(np.clip((box.x_min + box.x_max) / 2 + rng.normal(0, 0.03), box.x_min, box.x_max))
            gaze_y = float(np.clip((box.y_min + box.y_max) / 2 + rng.normal(0, 0.03), box.y_min, box.y_max))
            episode = CandidateEpisode(
                episode_id, episode_id, "synthetic-object", start, cutoff, None, None
            )
            observation = GazeContextObservation(
                episode_id=episode_id,
                track_id=episode_id,
                timestamp=cutoff,
                matched_dwell_s=timing["minimum_prediction_elapsed_s"],
                gaze_x_normalized=gaze_x,
                gaze_y_normalized=gaze_y,
                candidate_box=box,
            )
            quality_name = synthetic["eeg_quality_cycle"][
                episode_index % len(synthetic["eeg_quality_cycle"])
            ]
            quality = QualityState(quality_name)
            quality_counts[quality.value] += 1
            eeg_values = rng.normal(0.0, 1.0, len(EEG_FEATURE_NAMES)).astype(np.float64)
            eeg_source = _SyntheticEEGSource(quality, eeg_values)
            controlled_every = synthetic["controlled_trial_every_n"]
            instructed = (
                (episode_index // controlled_every) % 2
                if controlled_every and episode_index % controlled_every == 0
                else None
            )
            if instructed is not None:
                controlled_count += 1
            decision = controller.consider_prediction(
                episode, observation, eeg_source, instructed_intention=instructed
            )
            if decision.record is None or decision.record.unavailable_reason is not None:
                unavailable_count += 1
                episode_index += 1
                continue
            predicted_count += 1

            case_name = synthetic["feedback_case_cycle"][
                episode_index % len(synthetic["feedback_case_cycle"])
            ]
            feedback_counts[case_name] += 1
            action_occurred, feedback_pressed = FEEDBACK_CASES[case_name]
            outcome_time = cutoff + timing["outcome_delay_s"]
            if action_occurred:
                controller.on_dwell_trigger(
                    DwellTrigger(episode_id, episode_id, outcome_time, 0.5)
                )
            else:
                ended = CandidateEpisode(
                    episode_id,
                    episode_id,
                    "synthetic-object",
                    start,
                    cutoff,
                    outcome_time,
                    EpisodeEndReason.SOURCE_END,
                )
                controller.on_episode_end(ended)
            if feedback_pressed:
                result = controller.button_press(outcome_time + timing["button_delay_s"])
            else:
                result = controller.advance_time(outcome_time + timing["feedback_timeout_s"])
            if result is None:
                raise RuntimeError("synthetic feedback case failed to resolve")
            result_count += 1
            episode_index += 1

        controller.save_session_checkpoint()
        if resume["enabled"] and session_number + 1 == resume["after_session"]:
            state = load_participant_checkpoint(
                checkpoint_path,
                expected_participant_id=resolved["participant_id"],
                expected_participant_sequence_index=resolved["participant_sequence_index"],
                expected_model_config=model_config,
            )

    summary = {
        "episodes": resolved["episodes"],
        "paired_predictions": predicted_count,
        "paired_unavailable": unavailable_count,
        "trained_episode_results": result_count,
        "controlled_trials": controlled_count,
        "condition_sessions": dict(sorted(condition_counts.items())),
        "eeg_quality": dict(sorted(quality_counts.items())),
        "feedback_cases": dict(sorted(feedback_counts.items())),
        "training_counts": {
            "G": state.learners.g_model.training_count,
            "E": state.learners.e_model.training_count,
        },
        "next_session_index": state.schedule.next_session_index,
        "checkpoint_resume_exercised": bool(resume["enabled"]),
    }
    with (run_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return run_directory


if __name__ == "__main__":
    print("Use scripts/run_experiment_learning.py for the configured synthetic workflow.")
