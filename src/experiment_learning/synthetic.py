"""Deterministic hardware-free workflow for frozen policies and retraining."""

from __future__ import annotations

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
from gaze_interaction.episodes import CandidateEpisode, EpisodeEndReason

from experiment_learning.contracts import GazeContextObservation
from experiment_learning.policy import (
    create_cold_start_policy,
    load_frozen_policy,
    save_frozen_policy,
)
from experiment_learning.schedule import (
    ScheduleBinding,
    load_condition_schedule,
    resolve_scheduled_condition,
)
from experiment_learning.sessions import save_completed_session
from experiment_learning.state_machine import ExperimentController
from experiment_learning.trainer import TrainerConfig, train_next_session_policy


FEEDBACK_CASES = {
    "action_no_press": (True, False, 1),
    "action_press": (True, True, 0),
    "no_action_no_press": (False, False, 0),
    "no_action_press": (False, True, 1),
}


@dataclass
class _SyntheticEEGSource:
    quality_state: QualityState
    values: np.ndarray

    def features(self, start: float, end: float) -> EEGFeatureWindow:
        usable = self.quality_state is QualityState.USABLE
        has_samples = self.quality_state is not QualityState.UNAVAILABLE
        reasons = () if usable else (f"synthetic_{self.quality_state.value}",)
        return EEGFeatureWindow(
            requested_start=start,
            requested_end=end,
            actual_start=start if has_samples else None,
            actual_end=end if has_samples else None,
            sample_count=251 if has_samples else 0,
            completeness=(WindowCompleteness.COMPLETE if has_samples else WindowCompleteness.EMPTY),
            quality_state=self.quality_state,
            quality_reasons=reasons,
            feature_names=EEG_FEATURE_NAMES,
            values=self.values if usable else None,
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
    for key in ("output_root", "participant_id", "sequence_id", "schedule_path"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} must be a non-empty string")
    for key in ("seed", "episodes", "sessions"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["episodes"] < config["sessions"]:
        raise ValueError("episodes must be at least sessions")
    timing = _mapping(config, "timing")
    for key in (
        "minimum_prediction_elapsed_s",
        "eeg_window_s",
        "feedback_timeout_s",
        "episode_spacing_s",
        "button_delay_s",
        "trajectory_step_s",
    ):
        _positive(f"timing.{key}", timing.get(key))
    if timing["button_delay_s"] >= timing["feedback_timeout_s"]:
        raise ValueError("button delay must be inside the feedback window")
    if timing["episode_spacing_s"] < timing["feedback_timeout_s"] + 1.5:
        raise ValueError("episode spacing must leave feedback windows disjoint")
    cold = _mapping(config, "cold_start_policy")
    for key in (
        "base_threshold_s",
        "minimum_e_threshold_s",
        "base_search_min_s",
        "base_search_max_s",
        "base_search_step_s",
        "maximum_allowed_reduction_fraction",
    ):
        _positive(f"cold_start_policy.{key}", cold.get(key))
    TrainerConfig.from_mapping(_mapping(config, "trainer"))
    synthetic = _mapping(config, "synthetic")
    cases = synthetic.get("feedback_case_cycle")
    if not isinstance(cases, list) or not cases or any(case not in FEEDBACK_CASES for case in cases):
        raise ValueError("synthetic.feedback_case_cycle contains an unknown case")
    qualities = synthetic.get("eeg_quality_cycle")
    valid = {state.value for state in QualityState}
    if not isinstance(qualities, list) or not qualities or any(item not in valid for item in qualities):
        raise ValueError("synthetic.eeg_quality_cycle contains an unknown quality")
    controlled = synthetic.get("controlled_trial_every_n")
    if isinstance(controlled, bool) or not isinstance(controlled, int) or controlled < 0:
        raise ValueError("controlled_trial_every_n must be a non-negative integer")


def _new_run_directory(output_root: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(output_root) / "experiment_learning" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _eeg_values(label: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray([4.0, 12.0, 0.8, 2.0, 1.0, 2.0, 1.0, 0.4], dtype=np.float64)
    # Only beta/(alpha+theta) enters the model. Other values remain provenance.
    target_index = (1.8 if label else 0.25) + float(rng.normal(0.0, 0.08))
    values[6] = max(0.01, target_index * (values[4] + values[5]))
    return values


def _episode(
    episode_id: int,
    start: float,
    last_match: float,
    *,
    end: float | None = None,
) -> CandidateEpisode:
    return CandidateEpisode(
        episode_id,
        episode_id,
        "synthetic-object",
        start,
        last_match,
        end,
        None if end is None else EpisodeEndReason.CANDIDATE_CHANGE,
    )


def _observation(episode: CandidateEpisode, dwell: float) -> GazeContextObservation:
    return GazeContextObservation(
        episode_id=episode.episode_id,
        track_id=episode.track_id,
        timestamp=episode.last_match_timestamp,
        matched_dwell_s=dwell,
        gaze_x_normalized=0.52,
        gaze_y_normalized=0.48,
        candidate_box=BoundingBox(0.2, 0.2, 0.8, 0.8),
    )


def run_synthetic_experiment(config: dict[str, Any]) -> Path:
    """Exercise frozen runtime policies, feedback, artifacts, and retraining."""

    validate_synthetic_config(config)
    resolved = deepcopy(config)
    run_directory = _new_run_directory(resolved["output_root"])
    save_resolved_config(resolved, run_directory / "resolved_config.json")
    events = JsonlEventLogger(run_directory / "events.jsonl")
    schedule = load_condition_schedule(resolved["schedule_path"])
    binding = ScheduleBinding(resolved["sequence_id"], schedule.sha256)
    cold = resolved["cold_start_policy"]
    policy = create_cold_start_policy(
        participant_id=resolved["participant_id"],
        schedule_sequence_id=binding.sequence_id,
        schedule_sha256=binding.csv_sha256,
        base_threshold_s=cold["base_threshold_s"],
        minimum_e_threshold_s=cold["minimum_e_threshold_s"],
        base_search_min_s=cold["base_search_min_s"],
        base_search_max_s=cold["base_search_max_s"],
        base_search_step_s=cold["base_search_step_s"],
        maximum_allowed_reduction_fraction=cold["maximum_allowed_reduction_fraction"],
    )
    policy_path = run_directory / "policy_session_001.json"
    policy_digest = save_frozen_policy(policy_path, policy)
    trainer_config = TrainerConfig.from_mapping(resolved["trainer"])
    rng = np.random.default_rng(resolved["seed"])
    timing = resolved["timing"]
    synthetic = resolved["synthetic"]
    completed_paths: list[Path] = []
    condition_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    feedback_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    episode_index = 0
    per_session, remainder = divmod(resolved["episodes"], resolved["sessions"])

    for session_number in range(1, resolved["sessions"] + 1):
        scheduled = resolve_scheduled_condition(
            schedule, resolved["sequence_id"], session_number, binding
        )
        condition_counts[scheduled.condition.value] += 1
        policy = load_frozen_policy(
            policy_path,
            expected_participant_id=resolved["participant_id"],
            expected_session=session_number,
            expected_sha256=policy_digest,
        )
        controller = ExperimentController(
            policy=policy,
            policy_sha256=policy_digest,
            session_id=f"synthetic-session-{session_number:03d}",
            session_number=session_number,
            attempt_id=f"attempt-{session_number:03d}-01",
            active_condition=scheduled.condition,
            schedule_binding=scheduled.binding,
            minimum_prediction_elapsed_s=timing["minimum_prediction_elapsed_s"],
            eeg_window_s=timing["eeg_window_s"],
            feedback_timeout_s=timing["feedback_timeout_s"],
            event_logger=events,
        )
        session_count = per_session + (1 if session_number <= remainder else 0)
        last_resolution = 0.0
        for _ in range(session_count):
            episode_index += 1
            case_name = synthetic["feedback_case_cycle"][
                (episode_index - 1) % len(synthetic["feedback_case_cycle"])
            ]
            action, press, intended_label = FEEDBACK_CASES[case_name]
            quality = QualityState(
                synthetic["eeg_quality_cycle"][
                    (episode_index - 1) % len(synthetic["eeg_quality_cycle"])
                ]
            )
            quality_counts[quality.value] += 1
            feedback_counts[case_name] += 1
            start = 1.0 + (episode_index - 1) * timing["episode_spacing_s"]
            source = _SyntheticEEGSource(quality, _eeg_values(intended_label, rng))
            controlled_every = synthetic["controlled_trial_every_n"]
            instructed = intended_label if controlled_every and episode_index % controlled_every == 0 else None
            dwell = 0.0
            decision = None
            step = timing["trajectory_step_s"]
            while dwell <= timing["minimum_prediction_elapsed_s"] + 1e-12:
                timestamp = start + dwell
                active = _episode(episode_index, start, timestamp)
                decision = controller.evaluate_update(
                    active,
                    _observation(active, dwell),
                    source,
                    instructed_intention=instructed,
                )
                dwell = round(dwell + step, 12)
            assert decision is not None
            required = decision.required_dwell_s
            target_dwell = required if action else max(
                timing["minimum_prediction_elapsed_s"], required - step
            )
            while dwell <= target_dwell + 1e-12:
                timestamp = start + dwell
                active = _episode(episode_index, start, timestamp)
                decision = controller.evaluate_update(
                    active,
                    _observation(active, dwell),
                    source,
                    instructed_intention=instructed,
                )
                dwell = round(dwell + step, 12)
            last_match = start + min(target_dwell, dwell - step)
            if action:
                controller.open_action_feedback(episode_index, last_match)
                outcome_time = last_match
                opened = True
            else:
                outcome_time = last_match + step
                ended = _episode(episode_index, start, last_match, end=outcome_time)
                opened = controller.open_no_action_feedback(ended, outcome_time)
            if opened:
                if press:
                    controller.accept_feedback(outcome_time + timing["button_delay_s"])
                    last_resolution = outcome_time + timing["button_delay_s"]
                else:
                    controller.advance_time(outcome_time + timing["feedback_timeout_s"])
                    last_resolution = outcome_time + timing["feedback_timeout_s"]
            else:
                last_resolution = outcome_time

        session = controller.completed_session(last_resolution)
        for record in session.records:
            exclusion_counts.update(record.exclusion_reasons)
        session_path = run_directory / f"completed_session_{session_number:03d}.json"
        save_completed_session(session_path, session)
        completed_paths.append(session_path)
        next_path = run_directory / f"policy_session_{session_number + 1:03d}.json"
        trained = train_next_session_policy(
            completed_paths, policy, next_path, trainer_config
        )
        policy = trained.policy
        policy_path = trained.policy_path
        policy_digest = trained.policy_sha256

    summary = {
        "episodes": resolved["episodes"],
        "sessions": resolved["sessions"],
        "condition_sessions": dict(sorted(condition_counts.items())),
        "eeg_quality": dict(sorted(quality_counts.items())),
        "feedback_cases": dict(sorted(feedback_counts.items())),
        "exclusions": dict(sorted(exclusion_counts.items())),
        "completed_session_artifacts": len(completed_paths),
        "next_policy_session": policy.policy_for_session,
        "final_policy_sha256": policy_digest,
        "training_has_no_random_seed": True,
    }
    (run_directory / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return run_directory


if __name__ == "__main__":
    print("Use scripts/run_experiment_learning.py for the configured workflow.")
