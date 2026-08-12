"""Deterministic between-session trainer for frozen G/E policies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from experiment_learning.artifacts import artifact_digest, immutable_write_json
from experiment_learning.contracts import CompletedSession, EpisodeTrainingRecord
from experiment_learning.policy import FrozenSessionPolicy, SourceAttempt, save_frozen_policy
from experiment_learning.sessions import load_completed_session


TRAINING_REPORT_SCHEMA = "experiment_training_report_v1"


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class TrainerConfig:
    minimum_examples: int = 20
    minimum_per_class: int = 5
    l2: float = 0.0
    base_min_s: float = 0.5
    base_max_s: float = 1.5
    base_step_s: float = 0.05
    reduction_values: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    minimum_e_threshold_s: float = 0.35
    false_positive_weight: float = 2.0
    false_negative_weight: float = 1.0
    true_positive_latency_weight: float = 0.25
    optimizer_max_iterations: int = 1000
    optimizer_ftol: float = 1e-12

    def __post_init__(self) -> None:
        _positive_int("minimum_examples", self.minimum_examples)
        _positive_int("minimum_per_class", self.minimum_per_class)
        if self.minimum_examples < 2 * self.minimum_per_class:
            raise ValueError("minimum_examples must cover both minimum class counts")
        _positive_int("optimizer_max_iterations", self.optimizer_max_iterations)
        for name in (
            "l2",
            "base_min_s",
            "base_max_s",
            "base_step_s",
            "minimum_e_threshold_s",
            "false_positive_weight",
            "false_negative_weight",
            "true_positive_latency_weight",
            "optimizer_ftol",
        ):
            value = _finite(name, getattr(self, name))
            if value < 0.0 or name in {
                "base_min_s",
                "base_max_s",
                "base_step_s",
                "minimum_e_threshold_s",
                "optimizer_ftol",
            } and value <= 0.0:
                raise ValueError(f"{name} has an invalid value")
            object.__setattr__(self, name, value)
        if self.base_min_s > self.base_max_s:
            raise ValueError("base_min_s cannot exceed base_max_s")
        if self.minimum_e_threshold_s > self.base_min_s:
            raise ValueError("minimum E threshold cannot exceed the smallest G candidate")
        if not isinstance(self.reduction_values, tuple) or not self.reduction_values:
            raise ValueError("reduction_values must be a non-empty tuple")
        reductions = tuple(_finite("reduction value", item) for item in self.reduction_values)
        if any(not 0.0 <= item <= 0.5 for item in reductions):
            raise ValueError("reduction values must be within the approved [0, 0.5] bound")
        if tuple(sorted(set(reductions))) != reductions:
            raise ValueError("reduction_values must be unique and increasing")
        object.__setattr__(self, "reduction_values", reductions)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainerConfig":
        objective = values["objective"]
        optimizer = values["optimizer"]
        return cls(
            minimum_examples=values["minimum_examples"],
            minimum_per_class=values["minimum_per_class"],
            l2=values["l2"],
            base_min_s=values["base_min_s"],
            base_max_s=values["base_max_s"],
            base_step_s=values["base_step_s"],
            reduction_values=tuple(values["reduction_values"]),
            minimum_e_threshold_s=values["minimum_e_threshold_s"],
            false_positive_weight=objective["false_positive_weight"],
            false_negative_weight=objective["false_negative_weight"],
            true_positive_latency_weight=objective["true_positive_latency_weight"],
            optimizer_max_iterations=optimizer["max_iterations"],
            optimizer_ftol=optimizer["ftol"],
        )


@dataclass(frozen=True)
class TrainingResult:
    policy: FrozenSessionPolicy
    policy_sha256: str
    policy_path: Path
    report: Mapping[str, Any]
    report_path: Path


@dataclass(frozen=True)
class _SessionInput:
    session: CompletedSession
    digest: str


def _load_inputs(
    completed_sessions: Sequence[CompletedSession | str | Path],
) -> list[_SessionInput]:
    loaded: list[_SessionInput] = []
    for item in completed_sessions:
        if isinstance(item, CompletedSession):
            loaded.append(_SessionInput(item, artifact_digest(item.to_payload())))
        else:
            session, digest = load_completed_session(item)
            loaded.append(_SessionInput(session, digest))
    return sorted(
        loaded,
        key=lambda item: (
            item.session.session_number,
            item.session.attempt_id,
            item.session.session_id,
        ),
    )


def _base_candidates(config: TrainerConfig) -> tuple[float, ...]:
    count = int(round((config.base_max_s - config.base_min_s) / config.base_step_s))
    values = tuple(round(config.base_min_s + index * config.base_step_s, 12) for index in range(count + 1))
    if not math.isclose(values[-1], config.base_max_s, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("base threshold range must be divisible by base_step_s")
    return values


def _first_crossing(record: EpisodeTrainingRecord, threshold: float) -> float | None:
    return next(
        (
            point.timestamp
            for point in record.trajectory
            if point.accumulated_matched_dwell_s >= threshold
            or math.isclose(
                point.accumulated_matched_dwell_s,
                threshold,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ),
        None,
    )


def _candidate_metrics(
    records: Sequence[EpisodeTrainingRecord],
    thresholds: Sequence[float],
    config: TrainerConfig,
) -> dict[str, Any]:
    tp = fp = tn = fn = censored = 0
    latencies: list[float] = []
    positive_count = negative_count = 0
    for record, threshold in zip(records, thresholds, strict=True):
        crossing = _first_crossing(record, threshold)
        if crossing is None and record.action_occurred is True:
            censored += 1
            continue
        label = record.common_label
        assert label in (0, 1)
        prediction = int(crossing is not None)
        if label == 1:
            positive_count += 1
            if prediction:
                tp += 1
                latencies.append(crossing - record.episode_start_timestamp)
            else:
                fn += 1
        else:
            negative_count += 1
            if prediction:
                fp += 1
            else:
                tn += 1
    evaluated = tp + fp + tn + fn
    valid = (
        evaluated >= config.minimum_examples
        and positive_count >= config.minimum_per_class
        and negative_count >= config.minimum_per_class
    )
    latency = sum(latencies) / len(latencies) if latencies else 0.0
    objective = (
        config.false_positive_weight * fp
        + config.false_negative_weight * fn
        + config.true_positive_latency_weight * latency
        if valid
        else None
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "valid": valid,
        "evaluated": evaluated,
        "censored": censored,
        "positive_labels": positive_count,
        "negative_labels": negative_count,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "mean_true_positive_latency_s": latency,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weighted_loss": objective,
    }


def _fit_logistic(
    records: Sequence[EpisodeTrainingRecord], config: TrainerConfig
) -> tuple[float, float, float, float, Mapping[str, Any]] | None:
    indicators = np.asarray([record.engagement_index for record in records], dtype=np.float64)
    labels = np.asarray([record.common_label for record in records], dtype=np.float64)
    mean = float(np.mean(indicators))
    scale = float(np.std(indicators))
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    standardized = (indicators - mean) / scale
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    weights = np.where(
        labels == 1.0,
        labels.size / (2.0 * positives),
        labels.size / (2.0 * negatives),
    )

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, coefficient = parameters
        logits = intercept + coefficient * standardized
        probabilities = np.empty_like(logits)
        nonnegative = logits >= 0.0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
        exp_values = np.exp(logits[~nonnegative])
        probabilities[~nonnegative] = exp_values / (1.0 + exp_values)
        losses = np.logaddexp(0.0, logits) - labels * logits
        loss = float(np.mean(weights * losses) + 0.5 * config.l2 * coefficient**2)
        residual = weights * (probabilities - labels) / labels.size
        gradient = np.asarray(
            [
                np.sum(residual),
                np.dot(residual, standardized) + config.l2 * coefficient,
            ],
            dtype=np.float64,
        )
        return loss, gradient

    result = minimize(
        objective,
        x0=np.zeros(2, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": config.optimizer_max_iterations, "ftol": config.optimizer_ftol},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return None
    intercept, coefficient = (float(result.x[0]), float(result.x[1]))
    return (
        mean,
        scale,
        intercept,
        coefficient,
        {
            "method": "L-BFGS-B",
            "initial_parameters": [0.0, 0.0],
            "success": True,
            "iterations": int(result.nit),
            "final_loss": float(result.fun),
            "balanced_binary_cross_entropy": True,
            "l2": config.l2,
        },
    )


def _probability(value: float, mean: float, scale: float, intercept: float, coefficient: float) -> float:
    linear = intercept + coefficient * ((value - mean) / scale)
    if linear >= 0.0:
        return 1.0 / (1.0 + math.exp(-linear))
    exponential = math.exp(linear)
    return exponential / (1.0 + exponential)


def _choose_base(
    records: Sequence[EpisodeTrainingRecord], prior: FrozenSessionPolicy, config: TrainerConfig
) -> tuple[float, list[dict[str, Any]]] | None:
    table: list[dict[str, Any]] = []
    for threshold in _base_candidates(config):
        metrics = _candidate_metrics(records, [threshold] * len(records), config)
        table.append({"base_threshold_s": threshold, **metrics})
    valid = [row for row in table if row["valid"]]
    if not valid:
        return None
    chosen = min(
        valid,
        key=lambda row: (
            row["weighted_loss"],
            row["false_positives"],
            row["false_negatives"],
            row["mean_true_positive_latency_s"],
            abs(row["base_threshold_s"] - prior.g_base_threshold_s),
            -row["base_threshold_s"],
        ),
    )
    return float(chosen["base_threshold_s"]), table


def _choose_reduction(
    records: Sequence[EpisodeTrainingRecord],
    *,
    base: float,
    mean: float,
    scale: float,
    intercept: float,
    coefficient: float,
    prior: FrozenSessionPolicy,
    config: TrainerConfig,
) -> tuple[float, list[dict[str, Any]]] | None:
    evidence = [
        max(
            0.0,
            2.0
            * _probability(record.engagement_index, mean, scale, intercept, coefficient)
            - 1.0,
        )
        for record in records
    ]
    table: list[dict[str, Any]] = []
    for reduction in config.reduction_values:
        thresholds = [
            max(config.minimum_e_threshold_s, base * (1.0 - reduction * value))
            for value in evidence
        ]
        metrics = _candidate_metrics(records, thresholds, config)
        table.append({"reduction_fraction": reduction, **metrics})
    valid = [row for row in table if row["valid"]]
    if not valid:
        return None
    chosen = min(
        valid,
        key=lambda row: (
            row["weighted_loss"],
            row["false_positives"],
            row["false_negatives"],
            row["mean_true_positive_latency_s"],
            abs(row["reduction_fraction"] - prior.maximum_eeg_reduction_fraction),
            row["reduction_fraction"],
        ),
    )
    return float(chosen["reduction_fraction"]), table


def train_next_session_policy(
    completed_sessions: Sequence[CompletedSession | str | Path],
    prior_policy: FrozenSessionPolicy,
    output_path: str | Path,
    config: TrainerConfig,
) -> TrainingResult:
    """Train once after session N and immutably write the policy for N+1."""

    inputs = _load_inputs(completed_sessions)
    exclusions: Counter[str] = Counter()
    successful = [item for item in inputs if item.session.successful]
    incomplete_count = len(inputs) - len(successful)
    if incomplete_count:
        exclusions["incomplete_session"] = incomplete_count
    if any(item.session.participant_id != prior_policy.participant_id for item in successful):
        raise ValueError("completed sessions contain another participant")
    if any(
        item.session.schedule_sequence_id != prior_policy.schedule_sequence_id
        or item.session.schedule_sha256 != prior_policy.schedule_sha256
        for item in successful
    ):
        raise ValueError("completed session schedule binding differs from the prior policy")
    numbers = [item.session.session_number for item in successful]
    if len(numbers) != len(set(numbers)):
        raise ValueError("duplicate successful session numbers are not allowed")
    expected_numbers = list(range(1, prior_policy.policy_for_session + 1))
    if numbers != expected_numbers:
        raise ValueError("successful trainer inputs must cover sessions 1..N exactly")
    for source, item in zip(prior_policy.source_attempts, successful[:-1], strict=True):
        if (
            source.session_number != item.session.session_number
            or source.attempt_id != item.session.attempt_id
            or source.artifact_sha256 != item.digest
        ):
            raise ValueError("completed-session lineage differs from the prior policy")

    records: list[EpisodeTrainingRecord] = []
    identities: set[tuple[int, str, int]] = set()
    for item in successful:
        for record in sorted(item.session.records, key=lambda value: value.identity):
            if record.identity in identities:
                raise ValueError(f"duplicate episode record: {record.identity}")
            identities.add(record.identity)
            reason: str | None = None
            if record.instructed_intention is not None:
                reason = "controlled_intention_trial"
            elif record.canceled:
                reason = "canceled_episode"
            elif not record.training_eligible:
                reason = "record_marked_ineligible"
            elif record.eeg_quality_state != "usable" or record.engagement_index is None:
                reason = "unusable_eeg"
            elif record.common_label not in (0, 1):
                reason = "missing_common_label"
            if reason is not None:
                exclusions[reason] += 1
            else:
                records.append(record)
    records.sort(key=lambda record: (
        record.session_number,
        record.attempt_id,
        record.episode_id,
        record.prediction_cutoff_timestamp or 0.0,
    ))
    labels = Counter(record.common_label for record in records)
    enough = (
        len(records) >= config.minimum_examples
        and labels[0] >= config.minimum_per_class
        and labels[1] >= config.minimum_per_class
    )
    fit = _fit_logistic(records, config) if enough else None
    status = "trained_from_completed_sessions"
    base_table: list[dict[str, Any]] = []
    reduction_table: list[dict[str, Any]] = []
    if not enough:
        status = "carried_forward_insufficient_examples"
    elif fit is None:
        status = "carried_forward_unstable_or_failed_fit"

    if fit is not None:
        mean, scale, intercept, coefficient, optimizer_report = fit
        base_result = _choose_base(records, prior_policy, config)
        if base_result is None:
            status = "carried_forward_no_valid_base_candidate"
            fit = None
        else:
            base, base_table = base_result
            reduction_result = _choose_reduction(
                records,
                base=base,
                mean=mean,
                scale=scale,
                intercept=intercept,
                coefficient=coefficient,
                prior=prior_policy,
                config=config,
            )
            if reduction_result is None:
                status = "carried_forward_no_valid_e_candidate"
                fit = None
            else:
                reduction, reduction_table = reduction_result
    if fit is None:
        mean = prior_policy.engagement_mean
        scale = prior_policy.engagement_scale
        intercept = prior_policy.logistic_intercept
        coefficient = prior_policy.logistic_coefficient
        base = prior_policy.g_base_threshold_s
        reduction = prior_policy.maximum_eeg_reduction_fraction
        optimizer_report = {
            "method": "L-BFGS-B",
            "initial_parameters": [0.0, 0.0],
            "success": False,
            "reason": status,
            "balanced_binary_cross_entropy": True,
            "l2": config.l2,
        }

    sources = tuple(
        SourceAttempt(item.session.session_number, item.session.attempt_id, item.digest)
        for item in successful
    )
    next_policy = FrozenSessionPolicy(
        participant_id=prior_policy.participant_id,
        policy_for_session=prior_policy.policy_for_session + 1,
        trained_through_session=prior_policy.policy_for_session,
        schedule_sequence_id=prior_policy.schedule_sequence_id,
        schedule_sha256=prior_policy.schedule_sha256,
        source_attempts=sources,
        g_base_threshold_s=base,
        minimum_e_threshold_s=config.minimum_e_threshold_s,
        maximum_eeg_reduction_fraction=reduction,
        engagement_mean=mean,
        engagement_scale=scale,
        logistic_intercept=intercept,
        logistic_coefficient=coefficient,
        base_search_min_s=config.base_min_s,
        base_search_max_s=config.base_max_s,
        base_search_step_s=config.base_step_s,
        maximum_allowed_reduction_fraction=max(config.reduction_values),
        cold_start_status=status,
        fitted_example_count=len(records) if fit is not None else prior_policy.fitted_example_count,
    )
    output = Path(output_path)
    report_path = output.with_name(output.stem + ".training_report.json")
    report: dict[str, Any] = {
        "schema": TRAINING_REPORT_SCHEMA,
        "participant_id": prior_policy.participant_id,
        "policy_for_session": next_policy.policy_for_session,
        "trained_through_session": next_policy.trained_through_session,
        "status": status,
        "deterministic_order": [
            {
                "session_number": record.session_number,
                "attempt_id": record.attempt_id,
                "episode_id": record.episode_id,
                "cutoff_timestamp": record.prediction_cutoff_timestamp,
            }
            for record in records
        ],
        "input_artifacts": [source.to_payload() for source in sources],
        "counts": {
            "accepted": len(records),
            "label_0": labels[0],
            "label_1": labels[1],
            "excluded": dict(sorted(exclusions.items())),
        },
        "indicator": {
            "id": next_policy.eeg_indicator_id,
            "formula": next_policy.eeg_indicator_formula,
            "normalization_mean": mean,
            "normalization_scale": scale,
            "logistic_intercept": intercept,
            "logistic_coefficient": coefficient,
        },
        "optimizer": optimizer_report,
        "objective": {
            "formula": "false_positive_weight * FP + false_negative_weight * FN + true_positive_latency_weight * mean_TP_latency_s",
            "false_positive_weight": config.false_positive_weight,
            "false_negative_weight": config.false_negative_weight,
            "true_positive_latency_weight": config.true_positive_latency_weight,
        },
        "base_candidates": base_table,
        "reduction_candidates": reduction_table,
        "chosen": {
            "g_base_threshold_s": base,
            "maximum_eeg_reduction_fraction": reduction,
        },
        "tie_break_order": [
            "lower_weighted_loss",
            "fewer_false_positives",
            "fewer_false_negatives",
            "lower_true_positive_latency",
            "closest_to_prior",
            "safer_higher_base_or_smaller_reduction",
        ],
    }
    policy_digest = save_frozen_policy(output, next_policy)
    immutable_write_json(report_path, report)
    return TrainingResult(next_policy, policy_digest, output, report, report_path)


if __name__ == "__main__":
    print(TrainerConfig())
