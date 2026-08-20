#!/usr/bin/env python3
"""
Summarize one Neurotech_2026_primary participant/session from persisted run artifacts.

This script is deliberately standalone: it uses only the Python standard library
and does not import the Neurotech project package.

Expected layout (default --runs-root=runs):

runs/subjects/<participant>/
  attempts/<attempt_id>/
    events.jsonl
    resolved_experiment_config.json
    attempt_summary.json
    completed_session.json
    ...
  lineage/
    completed_session_NNN.json
    policy_session_NNN.json
    policy_session_NNN.training_report.json
    policy_session_(NNN+1).json
    policy_session_(NNN+1).training_report.json
    ...

Usage:
    python summarize_session.py P017 3
    python summarize_session.py P017 3 --runs-root /path/to/runs
    python summarize_session.py P017 3 --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


COMPLETED_SCHEMA = "experiment_completed_session_v1"
POLICY_SCHEMA = "experiment_policy_v1"
TRAINING_REPORT_SCHEMA = "experiment_training_report_v1"

SCORABLE_OUTCOMES = {"action", "no_action"}


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def optional_json(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.is_file() else None


def pct(value: int, denominator: int) -> float | None:
    return None if denominator == 0 else 100.0 * value / denominator


def outcome_prediction(record: dict[str, Any], model: str) -> int | None:
    outcome = record.get(f"{model.lower()}_outcome")
    if not isinstance(outcome, dict):
        return None
    status = outcome.get("status")
    if status == "action":
        return 1
    if status == "no_action":
        return 0
    return None


def model_metrics(records: Iterable[dict[str, Any]], model: str) -> dict[str, Any]:
    """
    Match the repository's integration.analysis semantics:
    use training-eligible records, then score only records having common_label
    and a non-censored action/no_action outcome for the requested model.
    """
    eligible = [r for r in records if r.get("training_eligible") is True]

    scored: list[tuple[int, int]] = []
    outcome_status_counts: dict[str, int] = {}
    for record in eligible:
        outcome = record.get(f"{model.lower()}_outcome")
        status = outcome.get("status") if isinstance(outcome, dict) else None
        key = str(status)
        outcome_status_counts[key] = outcome_status_counts.get(key, 0) + 1

        label = record.get("common_label")
        prediction = outcome_prediction(record, model)
        if label in (0, 1) and prediction is not None:
            scored.append((prediction, int(label)))

    tp = sum(pred == 1 and label == 1 for pred, label in scored)
    fp = sum(pred == 1 and label == 0 for pred, label in scored)
    tn = sum(pred == 0 and label == 0 for pred, label in scored)
    fn = sum(pred == 0 and label == 1 for pred, label in scored)
    n = len(scored)

    return {
        "scored_candidates": n,
        "successes_true_positive": tp,
        "success_pct_of_scored": pct(tp, n),
        "false_positives": fp,
        "false_positive_pct_of_scored": pct(fp, n),
        "false_negatives": fn,
        "false_negative_pct_of_scored": pct(fn, n),
        "true_negatives": tn,
        "true_negative_pct_of_scored": pct(tn, n),
        "correct_predictions": tp + tn,
        "accuracy_pct": pct(tp + tn, n),
        "outcome_status_counts_on_training_eligible_records": dict(
            sorted(outcome_status_counts.items())
        ),
    }


def extract_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(
            f"Unsupported policy schema: {policy.get('schema')!r}; expected {POLICY_SCHEMA!r}"
        )

    gaze = policy["gaze_policy"]
    eeg = policy["eeg_policy"]
    adjustment = eeg["adjustment"]
    normalization = eeg["normalization"]
    logistic = eeg["logistic"]

    base = float(gaze["base_threshold_s"])
    minimum_e = float(adjustment["minimum_e_threshold_s"])
    reduction = float(adjustment["reduction_fraction"])
    shortest_at_full_evidence = max(minimum_e, base * (1.0 - reduction))

    return {
        "policy_for_session": int(policy["policy_for_session"]),
        "trained_through_session": int(policy["trained_through_session"]),
        "training_state": policy.get("training_state"),
        "G": {
            "base_threshold_s": base,
        },
        "E": {
            # E uses the same G base and shortens it using EEG evidence.
            "base_threshold_s": base,
            "minimum_e_threshold_s": minimum_e,
            "reduction_fraction": reduction,
            "threshold_s_at_full_eeg_evidence": shortest_at_full_evidence,
            "engagement_normalization_mean": float(normalization["mean"]),
            "engagement_normalization_scale": float(normalization["scale"]),
            "logistic_intercept": float(logistic["intercept"]),
            "logistic_coefficient": float(logistic["coefficient"]),
        },
    }


def read_event_bounds(events_path: Path) -> dict[str, float | None]:
    if not events_path.is_file():
        return {
            "session_started_timestamp": None,
            "session_completed_timestamp": None,
            "actual_elapsed_start_to_complete_s": None,
        }

    start: float | None = None
    end: float | None = None
    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {events_path} at line {line_number}"
                ) from exc
            if not isinstance(event, dict):
                continue
            name = event.get("name")
            timestamp = event.get("timestamp")
            if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
                continue
            if name == "experiment_session_started" and start is None:
                start = float(timestamp)
            elif name == "experiment_session_completed":
                end = float(timestamp)

    elapsed = None if start is None or end is None else end - start
    return {
        "session_started_timestamp": start,
        "session_completed_timestamp": end,
        "actual_elapsed_start_to_complete_s": elapsed,
    }


def session_duration_info(
    completed: dict[str, Any],
    attempt_dir: Path,
) -> dict[str, Any]:
    resolved = optional_json(attempt_dir / "resolved_experiment_config.json")
    attempt_summary = optional_json(attempt_dir / "attempt_summary.json")
    event_bounds = read_event_bounds(attempt_dir / "events.jsonl")

    configured = None
    if resolved is not None:
        session = resolved.get("session")
        if isinstance(session, dict):
            value = session.get("maximum_duration_seconds")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                configured = float(value)

    completed_timestamp = completed.get("completed_timestamp")
    if not isinstance(completed_timestamp, (int, float)) or isinstance(
        completed_timestamp, bool
    ):
        completed_timestamp = None
    else:
        completed_timestamp = float(completed_timestamp)

    attempt_completed_timestamp = None
    if attempt_summary is not None:
        value = attempt_summary.get("completed_timestamp")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attempt_completed_timestamp = float(value)

    return {
        # This is the scientific session deadline configured for the attempt.
        "configured_scientific_duration_s": configured,
        # This can exceed the configured duration because completion may include
        # remaining feedback grace/finalization after the deadline.
        "completed_session_timestamp_s": completed_timestamp,
        "attempt_summary_completed_timestamp_s": attempt_completed_timestamp,
        **event_bounds,
    }


def training_report_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    if report.get("schema") != TRAINING_REPORT_SCHEMA:
        return {
            "schema": report.get("schema"),
            "warning": f"Expected {TRAINING_REPORT_SCHEMA}",
        }

    chosen = report.get("chosen")
    indicator = report.get("indicator")
    counts = report.get("counts")

    return {
        "status": report.get("status"),
        "counts": counts,
        "chosen": chosen,
        "indicator": indicator,
        "optimizer": report.get("optimizer"),
    }


def fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_policy_transition(initial: dict[str, Any], final: dict[str, Any]) -> None:
    print("\nAdjusted policy parameters")
    print(
        f"  Initial policy: session {initial['policy_for_session']} "
        f"(trained through {initial['trained_through_session']})"
    )
    print(
        f"  Final policy:   session {final['policy_for_session']} "
        f"(trained through {final['trained_through_session']})"
    )

    print("  G model")
    g0 = initial["G"]
    g1 = final["G"]
    print(
        "    base threshold: "
        f"{g0['base_threshold_s']:.6f} s -> {g1['base_threshold_s']:.6f} s "
        f"(delta {g1['base_threshold_s'] - g0['base_threshold_s']:+.6f} s)"
    )

    print("  E model")
    e0 = initial["E"]
    e1 = final["E"]
    fields = [
        ("base threshold", "base_threshold_s", "s"),
        ("minimum threshold", "minimum_e_threshold_s", "s"),
        ("reduction fraction", "reduction_fraction", ""),
        ("threshold at full EEG evidence", "threshold_s_at_full_eeg_evidence", "s"),
        ("engagement mean", "engagement_normalization_mean", ""),
        ("engagement scale", "engagement_normalization_scale", ""),
        ("logistic intercept", "logistic_intercept", ""),
        ("logistic coefficient", "logistic_coefficient", ""),
    ]
    for label, key, unit in fields:
        suffix = f" {unit}" if unit else ""
        print(
            f"    {label}: {e0[key]:.6f}{suffix} -> {e1[key]:.6f}{suffix} "
            f"(delta {e1[key] - e0[key]:+.6f}{suffix})"
        )


def print_human(summary: dict[str, Any]) -> None:
    print(
        f"Participant {summary['participant_id']} | "
        f"session {summary['session_number']} | "
        f"active model {summary['active_condition']}"
    )
    print(f"Attempt: {summary['attempt_id']}")
    print(f"Candidate events (episode records): {summary['candidate_events']}")
    print(f"Training-eligible candidate events: {summary['training_eligible_events']}")
    print(f"Excluded candidate events: {summary['excluded_events']}")

    duration = summary["duration"]
    print("\nSession duration")
    print(
        "  configured scientific duration: "
        f"{fmt_number(duration['configured_scientific_duration_s'])} s"
    )
    print(
        "  actual start->completion elapsed: "
        f"{fmt_number(duration['actual_elapsed_start_to_complete_s'])} s"
    )
    print(
        "  completed-session clock timestamp: "
        f"{fmt_number(duration['completed_session_timestamp_s'])} s"
    )

    for model in ("G", "E"):
        m = summary["models"][model]
        denom = m["scored_candidates"]
        print(f"\n{model} model outcomes (training-eligible, scorable candidates)")
        print(f"  scored candidates: {denom}")
        print(
            "  successes (true positives): "
            f"{m['successes_true_positive']} "
            f"({fmt_number(m['success_pct_of_scored'], 2)}%)"
        )
        print(
            "  false positives: "
            f"{m['false_positives']} "
            f"({fmt_number(m['false_positive_pct_of_scored'], 2)}%)"
        )
        print(
            "  false negatives: "
            f"{m['false_negatives']} "
            f"({fmt_number(m['false_negative_pct_of_scored'], 2)}%)"
        )
        print(
            "  true negatives: "
            f"{m['true_negatives']} "
            f"({fmt_number(m['true_negative_pct_of_scored'], 2)}%)"
        )
        print(f"  accuracy: {fmt_number(m['accuracy_pct'], 2)}%")

    if summary["initial_policy"] is not None and summary["final_policy"] is not None:
        print_policy_transition(summary["initial_policy"], summary["final_policy"])
    else:
        print("\nAdjusted policy parameters")
        print("  unavailable: initial or final policy artifact is missing")

    training = summary.get("training")
    if training is not None:
        print("\nTraining after this session")
        print(f"  status: {training.get('status')}")
        chosen = training.get("chosen")
        if isinstance(chosen, dict):
            print(
                "  chosen next G base threshold: "
                f"{fmt_number(chosen.get('g_base_threshold_s'), 6)} s"
            )
            print(
                "  chosen next E reduction fraction: "
                f"{fmt_number(chosen.get('maximum_eeg_reduction_fraction'), 6)}"
            )

    print("\nArtifacts")
    for key, value in summary["paths"].items():
        print(f"  {key}: {value if value is not None else 'not found'}")


def build_summary(runs_root: Path, participant: str, session_number: int) -> dict[str, Any]:
    subject_dir = runs_root / "subjects" / participant
    lineage_dir = subject_dir / "lineage"

    completed_path = lineage_dir / f"completed_session_{session_number:03d}.json"
    if not completed_path.is_file():
        raise FileNotFoundError(
            f"Completed session not found: {completed_path}\n"
            "Only successful sessions are published to lineage."
        )

    completed = load_json(completed_path)
    if completed.get("schema") != COMPLETED_SCHEMA:
        raise ValueError(
            f"Unsupported completed-session schema: {completed.get('schema')!r}; "
            f"expected {COMPLETED_SCHEMA!r}"
        )
    if completed.get("participant_id") != participant:
        raise ValueError(
            f"Participant mismatch in {completed_path}: {completed.get('participant_id')!r}"
        )
    if completed.get("session_number") != session_number:
        raise ValueError(
            f"Session mismatch in {completed_path}: {completed.get('session_number')!r}"
        )

    attempt_id = str(completed["attempt_id"])
    attempt_dir = subject_dir / "attempts" / attempt_id
    records = completed.get("records")
    if not isinstance(records, list):
        raise ValueError(f"'records' must be a list in {completed_path}")

    initial_policy_path = lineage_dir / f"policy_session_{session_number:03d}.json"
    final_policy_path = lineage_dir / f"policy_session_{session_number + 1:03d}.json"
    final_report_path = lineage_dir / (
        f"policy_session_{session_number + 1:03d}.training_report.json"
    )

    # If publication was interrupted after successful staging, the attempt-local
    # files are still useful for inspection.
    staged_final_policy_path = (
        attempt_dir / f"staged_policy_session_{session_number + 1:03d}.json"
    )
    staged_report_path = attempt_dir / (
        f"staged_policy_session_{session_number + 1:03d}.training_report.json"
    )
    resolved_final_policy_path = (
        final_policy_path
        if final_policy_path.is_file()
        else staged_final_policy_path
        if staged_final_policy_path.is_file()
        else None
    )
    resolved_report_path = (
        final_report_path
        if final_report_path.is_file()
        else staged_report_path
        if staged_report_path.is_file()
        else None
    )

    initial_policy = (
        extract_policy(load_json(initial_policy_path))
        if initial_policy_path.is_file()
        else None
    )
    final_policy = (
        extract_policy(load_json(resolved_final_policy_path))
        if resolved_final_policy_path is not None
        else None
    )
    report = (
        training_report_summary(load_json(resolved_report_path))
        if resolved_report_path is not None
        else None
    )

    candidate_events = len(records)
    eligible_count = sum(record.get("training_eligible") is True for record in records)

    return {
        "participant_id": participant,
        "session_number": session_number,
        "session_id": completed.get("session_id"),
        "attempt_id": attempt_id,
        "active_condition": completed.get("active_condition"),
        "successful": completed.get("successful"),
        "candidate_events": candidate_events,
        "training_eligible_events": eligible_count,
        "excluded_events": candidate_events - eligible_count,
        "models": {
            "G": model_metrics(records, "G"),
            "E": model_metrics(records, "E"),
        },
        "duration": session_duration_info(completed, attempt_dir),
        "initial_policy": initial_policy,
        "final_policy": final_policy,
        "training": report,
        "paths": {
            "completed_session": str(completed_path),
            "attempt_directory": str(attempt_dir) if attempt_dir.is_dir() else None,
            "initial_policy": str(initial_policy_path)
            if initial_policy_path.is_file()
            else None,
            "final_policy": str(resolved_final_policy_path)
            if resolved_final_policy_path is not None
            else None,
            "training_report": str(resolved_report_path)
            if resolved_report_path is not None
            else None,
            "events": str(attempt_dir / "events.jsonl")
            if (attempt_dir / "events.jsonl").is_file()
            else None,
            "attempt_summary": str(attempt_dir / "attempt_summary.json")
            if (attempt_dir / "attempt_summary.json").is_file()
            else None,
            "resolved_experiment_config": str(
                attempt_dir / "resolved_experiment_config.json"
            )
            if (attempt_dir / "resolved_experiment_config.json").is_file()
            else None,
        },
        "notes": {
            "success_definition": (
                "successes are true positives: model outcome=action and common_label=1"
            ),
            "percentage_denominator": (
                "per-model success/FP/FN/TN percentages use training-eligible "
                "records with common_label in {0,1} and scorable action/no_action "
                "outcomes, matching src/integration/analysis.py"
            ),
            "final_policy_meaning": (
                "final_policy is policy_session_(N+1), trained through session N; "
                "it did not control session N"
            ),
            "label_caveat": (
                "common_label is feedback-derived; silence is not independently "
                "observed ground truth"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("participant", help="Participant/subject ID, e.g. P017")
    parser.add_argument("session", type=int, help="1-based successful session number")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root containing subjects/<participant> (default: runs)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete summary as JSON instead of human-readable text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.session <= 0:
        raise SystemExit("session must be a positive integer")

    try:
        summary = build_summary(
            args.runs_root.expanduser(),
            args.participant,
            args.session,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        print_human(summary)


if __name__ == "__main__":
    main()
