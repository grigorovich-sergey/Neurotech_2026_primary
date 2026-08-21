#!/usr/bin/env python3
"""
Standalone participant/session report for Neurotech_2026_primary.

Purpose
-------
Report every VISIBLE/TRIGGERED selection independently of whether its episode
was training-eligible or whether paired EEG was usable.

This script intentionally does NOT import the Neurotech project package. It
reads persisted artifacts only, using Python's standard library.

Canonical input for a successful session:
    runs/subjects/<participant>/lineage/completed_session_NNN.json

The session's attempt directory is then resolved from the completed session's
attempt_id:
    runs/subjects/<participant>/attempts/<attempt_id>/

Triggered-event semantics (matching the repository state machine)
-----------------------------------------------------------------
A triggered/visible selection is an episode record with:
    action_occurred == True

For triggered actions, the contextual one-button feedback rule is:
    feedback_pressed == False / common_label == 1  -> true-positive trigger
    feedback_pressed == True  / common_label == 0  -> false-positive trigger

This classification does NOT require training_eligible == True and therefore
retains visible selections whose EEG was unavailable/rejected/otherwise
excluded from learning.

Important caveat
----------------
The repository only opens no-action feedback for paired eligible episodes.
Therefore all-triggered TP/FP reporting is comprehensive for visible actions,
but TN/FN across *all* candidate episodes cannot be reconstructed uniformly
when no-action episodes were ineligible and have no common feedback label.

Usage
-----
    python scripts/report_triggered_events.py P021 1
    python scripts/report_triggered_events.py P021 1 --runs-root /path/to/runs
    python scripts/report_triggered_events.py P021 1 --json
    python scripts/report_triggered_events.py P021 1 --csv triggered_P021_s001.csv
    python scripts/report_triggered_events.py P021 1 --details
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


COMPLETED_SCHEMA = "experiment_completed_session_v1"


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


def pct(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else 100.0 * float(numerator) / float(denominator)


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_s": None,
            "median_s": None,
            "minimum_s": None,
            "maximum_s": None,
        }
    return {
        "count": len(values),
        "mean_s": mean(values),
        "median_s": median(values),
        "minimum_s": min(values),
        "maximum_s": max(values),
    }


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
            if isinstance(item, dict):
                events.append(item)
    return events


def event_audit(events: list[dict[str, Any]]) -> dict[str, Any]:
    names = Counter(str(event.get("name")) for event in events)

    action_events = [
        event for event in events if event.get("name") == "integration_action_presented"
    ]
    dwell_events = [
        event for event in events if event.get("name") == "integration_dwell_trigger"
    ]
    feedback_events = [
        event for event in events if event.get("name") == "integration_feedback_press"
    ]
    ignored_feedback_events = [
        event
        for event in events
        if event.get("name") == "integration_feedback_press_ignored"
    ]

    action_episode_ids: list[int] = []
    duplicate_action_episode_ids: list[int] = []
    seen: set[int] = set()
    action_by_episode: dict[int, dict[str, Any]] = {}

    for event in action_events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        episode_id = payload.get("episode_id")
        if isinstance(episode_id, int) and not isinstance(episode_id, bool):
            action_episode_ids.append(episode_id)
            if episode_id in seen:
                duplicate_action_episode_ids.append(episode_id)
            seen.add(episode_id)
            action_by_episode[episode_id] = event

    return {
        "available": bool(events),
        "integration_action_presented": len(action_events),
        "integration_dwell_trigger": len(dwell_events),
        "integration_feedback_press": len(feedback_events),
        "integration_feedback_press_ignored": len(ignored_feedback_events),
        "unique_action_episode_ids": len(seen),
        "duplicate_action_episode_ids": sorted(set(duplicate_action_episode_ids)),
        "action_episode_ids": sorted(seen),
        "action_event_by_episode": action_by_episode,
        "event_name_counts": dict(sorted(names.items())),
    }


def triggered_classification(record: dict[str, Any]) -> str:
    """Classify one action_occurred=True record using persisted feedback fields."""
    if record.get("action_occurred") is not True:
        return "not_triggered"

    label = record.get("common_label")
    pressed = record.get("feedback_pressed")

    if label == 1:
        return "true_positive"
    if label == 0:
        return "false_positive"

    # Fallback to the equivalent feedback truth table if common_label is absent.
    if pressed is False:
        return "true_positive_feedback_inferred"
    if pressed is True:
        return "false_positive_feedback_inferred"
    return "unresolved_or_unlabeled"


def bool_prediction_from_action(record: dict[str, Any]) -> int | None:
    action = record.get("action_occurred")
    if action is True:
        return 1
    if action is False:
        return 0
    return None


def available_label_confusion(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """
    Confusion matrix using visible action_occurred as the prediction.

    This ignores training_eligible, but only records with an available common_label
    can be scored. Ineligible no-action episodes often have no common_label by
    design, so this is a partial all-record matrix, not a denominator over every
    candidate episode.
    """
    scored: list[tuple[int, int]] = []
    for record in records:
        prediction = bool_prediction_from_action(record)
        label = record.get("common_label")
        if prediction is not None and label in (0, 1):
            scored.append((prediction, int(label)))

    tp = sum(pred == 1 and label == 1 for pred, label in scored)
    fp = sum(pred == 1 and label == 0 for pred, label in scored)
    tn = sum(pred == 0 and label == 0 for pred, label in scored)
    fn = sum(pred == 0 and label == 1 for pred, label in scored)
    n = len(scored)

    return {
        "scored_feedback_labeled_records": n,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy_pct": pct(tp + tn, n),
        "precision_pct": pct(tp, tp + fp),
        "recall_pct": pct(tp, tp + fn),
        "note": (
            "Uses every record with action_occurred in {true,false} and common_label "
            "in {0,1}, regardless of training eligibility. No-action ineligible records "
            "often lack labels, so TN/FN coverage is not comprehensive."
        ),
    }


def session_duration_info(completed: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
    resolved = optional_json(attempt_dir / "resolved_experiment_config.json")
    attempt_summary = optional_json(attempt_dir / "attempt_summary.json")
    events = load_events(attempt_dir / "events.jsonl")

    configured = None
    if resolved is not None:
        session = resolved.get("session")
        if isinstance(session, dict):
            value = session.get("maximum_duration_seconds")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                configured = float(value)

    start = None
    completed_event = None
    for event in events:
        name = event.get("name")
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            continue
        if name == "experiment_session_started" and start is None:
            start = float(timestamp)
        elif name == "experiment_session_completed":
            completed_event = float(timestamp)

    completed_timestamp = completed.get("completed_timestamp")
    if not isinstance(completed_timestamp, (int, float)) or isinstance(
        completed_timestamp, bool
    ):
        completed_timestamp = None
    else:
        completed_timestamp = float(completed_timestamp)

    attempt_completed = None
    if attempt_summary is not None:
        value = attempt_summary.get("completed_timestamp")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attempt_completed = float(value)

    return {
        "configured_scientific_duration_s": configured,
        "completed_session_timestamp_s": completed_timestamp,
        "attempt_summary_completed_timestamp_s": attempt_completed,
        "session_started_timestamp_s": start,
        "session_completed_event_timestamp_s": completed_event,
        "actual_start_to_completion_elapsed_s": (
            None if start is None or completed_event is None else completed_event - start
        ),
    }


def triggered_rows(
    records: list[dict[str, Any]],
    action_event_by_episode: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("action_occurred") is not True:
            continue

        start = record.get("episode_start_timestamp")
        action = record.get("action_timestamp")
        latency = None
        if (
            isinstance(start, (int, float))
            and not isinstance(start, bool)
            and isinstance(action, (int, float))
            and not isinstance(action, bool)
        ):
            latency = float(action) - float(start)

        episode_id = record.get("episode_id")
        event = (
            action_event_by_episode.get(episode_id)
            if isinstance(episode_id, int) and not isinstance(episode_id, bool)
            else None
        )
        event_timestamp = None
        trigger_timestamp = None
        if isinstance(event, dict):
            ts = event.get("timestamp")
            if isinstance(ts, (int, float)) and not isinstance(ts, bool):
                event_timestamp = float(ts)
            payload = event.get("payload")
            if isinstance(payload, dict):
                ts = payload.get("trigger_timestamp")
                if isinstance(ts, (int, float)) and not isinstance(ts, bool):
                    trigger_timestamp = float(ts)

        exclusion_reasons = record.get("exclusion_reasons")
        if not isinstance(exclusion_reasons, list):
            exclusion_reasons = []

        rows.append(
            {
                "episode_id": episode_id,
                "track_id": record.get("track_id"),
                "episode_start_timestamp": start,
                "action_timestamp": action,
                "integration_action_presented_timestamp": event_timestamp,
                "dwell_trigger_timestamp": trigger_timestamp,
                "selection_latency_s": latency,
                "classification": triggered_classification(record),
                "feedback_pressed": record.get("feedback_pressed"),
                "feedback_resolution_timestamp": record.get(
                    "feedback_resolution_timestamp"
                ),
                "common_label": record.get("common_label"),
                "training_eligible": record.get("training_eligible"),
                "eeg_quality_state": record.get("eeg_quality_state"),
                "eeg_quality_reasons": record.get("eeg_quality_reasons"),
                "exclusion_reasons": exclusion_reasons,
                "g_required_dwell_s": record.get("g_required_dwell_s"),
                "e_required_dwell_s": record.get("e_required_dwell_s"),
                "engagement_index": record.get("engagement_index"),
                "eeg_probability": record.get("eeg_probability"),
                "eeg_evidence": record.get("eeg_evidence"),
            }
        )
    return rows


def build_summary(runs_root: Path, participant: str, session_number: int) -> dict[str, Any]:
    subject_dir = runs_root / "subjects" / participant
    lineage_dir = subject_dir / "lineage"
    completed_path = lineage_dir / f"completed_session_{session_number:03d}.json"

    if not completed_path.is_file():
        raise FileNotFoundError(
            f"Completed session not found: {completed_path}\n"
            "This report expects a successful session published to participant lineage."
        )

    completed = load_json(completed_path)
    schema = completed.get("schema")
    if schema != COMPLETED_SCHEMA:
        raise ValueError(
            f"Unsupported completed-session schema {schema!r}; expected {COMPLETED_SCHEMA!r}"
        )
    if completed.get("participant_id") != participant:
        raise ValueError(
            f"Participant mismatch in {completed_path}: {completed.get('participant_id')!r}"
        )
    if completed.get("session_number") != session_number:
        raise ValueError(
            f"Session mismatch in {completed_path}: {completed.get('session_number')!r}"
        )

    records = completed.get("records")
    if not isinstance(records, list):
        raise ValueError(f"'records' must be a list in {completed_path}")

    attempt_id = str(completed["attempt_id"])
    attempt_dir = subject_dir / "attempts" / attempt_id
    events_path = attempt_dir / "events.jsonl"
    events = load_events(events_path)
    audit = event_audit(events)
    action_event_by_episode = audit.pop("action_event_by_episode")

    trigger_rows = triggered_rows(records, action_event_by_episode)
    trigger_count = len(trigger_rows)

    tp = sum(row["classification"].startswith("true_positive") for row in trigger_rows)
    fp = sum(row["classification"].startswith("false_positive") for row in trigger_rows)
    unresolved = sum(
        row["classification"] == "unresolved_or_unlabeled" for row in trigger_rows
    )
    classified = tp + fp

    feedback_pressed = sum(row["feedback_pressed"] is True for row in trigger_rows)
    feedback_not_pressed = sum(row["feedback_pressed"] is False for row in trigger_rows)
    feedback_unknown = trigger_count - feedback_pressed - feedback_not_pressed

    eligible_triggers = sum(row["training_eligible"] is True for row in trigger_rows)
    excluded_triggers = trigger_count - eligible_triggers

    eeg_quality = Counter(str(row["eeg_quality_state"]) for row in trigger_rows)
    exclusion_reasons = Counter(
        reason
        for row in trigger_rows
        for reason in row["exclusion_reasons"]
        if isinstance(reason, str)
    )
    classifications = Counter(row["classification"] for row in trigger_rows)

    latencies = [
        float(row["selection_latency_s"])
        for row in trigger_rows
        if isinstance(row["selection_latency_s"], (int, float))
        and not isinstance(row["selection_latency_s"], bool)
    ]

    record_trigger_episode_ids = {
        int(row["episode_id"])
        for row in trigger_rows
        if isinstance(row["episode_id"], int) and not isinstance(row["episode_id"], bool)
    }
    event_trigger_episode_ids = set(audit["action_episode_ids"])

    mismatch = {
        "record_trigger_count": trigger_count,
        "integration_action_presented_count": audit["integration_action_presented"],
        "unique_integration_action_episode_count": audit["unique_action_episode_ids"],
        "record_trigger_episode_ids_missing_from_events": sorted(
            record_trigger_episode_ids - event_trigger_episode_ids
        ),
        "event_action_episode_ids_missing_from_records": sorted(
            event_trigger_episode_ids - record_trigger_episode_ids
        ),
        "counts_match": (
            not events
            or (
                trigger_count == audit["integration_action_presented"]
                and not (record_trigger_episode_ids - event_trigger_episode_ids)
                and not (event_trigger_episode_ids - record_trigger_episode_ids)
            )
        ),
    }

    all_eligible = sum(record.get("training_eligible") is True for record in records)
    action_false = sum(record.get("action_occurred") is False for record in records)
    action_unknown = len(records) - trigger_count - action_false

    return {
        "schema": "neurotech.triggered_event_report.v1",
        "participant_id": participant,
        "session_number": session_number,
        "session_id": completed.get("session_id"),
        "attempt_id": attempt_id,
        "active_condition": completed.get("active_condition"),
        "successful": completed.get("successful"),
        "session": {
            "candidate_episode_records": len(records),
            "action_occurred_true": trigger_count,
            "action_occurred_false": action_false,
            "action_occurred_unknown": action_unknown,
            "training_eligible_records": all_eligible,
            "excluded_records": len(records) - all_eligible,
            "duration": session_duration_info(completed, attempt_dir),
        },
        "triggered_events": {
            "total": trigger_count,
            "feedback_classified": classified,
            "true_positive": tp,
            "false_positive": fp,
            "unresolved_or_unlabeled": unresolved,
            "accepted_trigger_pct": pct(tp, classified),
            "false_positive_pct_of_classified_triggers": pct(fp, classified),
            "feedback_pressed": feedback_pressed,
            "feedback_not_pressed": feedback_not_pressed,
            "feedback_unknown": feedback_unknown,
            "training_eligible": eligible_triggers,
            "excluded_from_training": excluded_triggers,
            "classification_counts": dict(sorted(classifications.items())),
            "eeg_quality_counts": dict(sorted(eeg_quality.items())),
            "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
            "selection_latency": latency_summary(latencies),
        },
        "available_feedback_labeled_confusion": available_label_confusion(records),
        "event_log_audit": audit,
        "trigger_record_vs_event_audit": mismatch,
        "triggered_event_rows": trigger_rows,
        "paths": {
            "completed_session": str(completed_path),
            "attempt_directory": str(attempt_dir) if attempt_dir.is_dir() else None,
            "events": str(events_path) if events_path.is_file() else None,
            "attempt_summary": (
                str(attempt_dir / "attempt_summary.json")
                if (attempt_dir / "attempt_summary.json").is_file()
                else None
            ),
            "analysis_summary": (
                str(attempt_dir / "analysis_summary.json")
                if (attempt_dir / "analysis_summary.json").is_file()
                else None
            ),
            "participant_analysis_summary": (
                str(attempt_dir / "participant_analysis_summary.json")
                if (attempt_dir / "participant_analysis_summary.json").is_file()
                else None
            ),
        },
        "notes": {
            "primary_trigger_definition": "completed-session record action_occurred == true",
            "event_cross_check": (
                "events.jsonl integration_action_presented is used as an audit cross-check; "
                "the canonical successful-session record is completed_session_NNN.json"
            ),
            "false_positive_definition": (
                "triggered action with common_label == 0; under the repository's contextual "
                "one-button rule this corresponds to feedback_pressed == true"
            ),
            "true_positive_definition": (
                "triggered action with common_label == 1; under the repository's contextual "
                "one-button rule this corresponds to feedback_pressed == false (feedback timeout)"
            ),
            "training_independence": (
                "triggered-event TP/FP counts do not filter on training_eligible or EEG quality"
            ),
            "no_action_caveat": (
                "The repository does not open no-action feedback for ineligible episodes, so "
                "TN/FN coverage across every candidate is not complete."
            ),
        },
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_human(summary: dict[str, Any], *, details: bool) -> None:
    session = summary["session"]
    triggered = summary["triggered_events"]
    confusion = summary["available_feedback_labeled_confusion"]
    audit = summary["trigger_record_vs_event_audit"]

    print(
        f"Participant {summary['participant_id']} | session {summary['session_number']} | "
        f"active model {summary['active_condition']}"
    )
    print(f"Attempt: {summary['attempt_id']}")

    print("\nAll persisted episode records")
    print(f"  candidate episode records: {session['candidate_episode_records']}")
    print(f"  visible/triggered selections: {triggered['total']}")
    print(f"  no-action records: {session['action_occurred_false']}")
    print(f"  action state unknown/canceled before outcome: {session['action_occurred_unknown']}")
    print(f"  training-eligible records: {session['training_eligible_records']}")
    print(f"  excluded from training: {session['excluded_records']}")

    print("\nTriggered selections — ALL EEG qualities, independent of training eligibility")
    print(f"  total triggered selections: {triggered['total']}")
    print(f"  feedback-classified triggers: {triggered['feedback_classified']}")
    print(
        f"  true-positive / accepted triggers: {triggered['true_positive']} "
        f"({fmt(triggered['accepted_trigger_pct'])}%)"
    )
    print(
        f"  false-positive triggers: {triggered['false_positive']} "
        f"({fmt(triggered['false_positive_pct_of_classified_triggers'])}%)"
    )
    print(f"  unresolved/unlabeled triggers: {triggered['unresolved_or_unlabeled']}")
    print(f"  feedback button pressed on triggers: {triggered['feedback_pressed']}")
    print(f"  feedback not pressed (timeout) on triggers: {triggered['feedback_not_pressed']}")
    print(f"  training-eligible triggers: {triggered['training_eligible']}")
    print(f"  triggered but excluded from training: {triggered['excluded_from_training']}")

    latency = triggered["selection_latency"]
    print("\nSelection latency (episode start -> visible action)")
    print(f"  count: {latency['count']}")
    print(f"  mean: {fmt(latency['mean_s'], 3)} s")
    print(f"  median: {fmt(latency['median_s'], 3)} s")
    print(f"  min/max: {fmt(latency['minimum_s'], 3)} / {fmt(latency['maximum_s'], 3)} s")

    print("\nTriggered-event EEG quality distribution")
    if triggered["eeg_quality_counts"]:
        for key, value in triggered["eeg_quality_counts"].items():
            print(f"  {key}: {value}")
    else:
        print("  none")

    print("\nWhy triggered events were excluded from training")
    if triggered["exclusion_reason_counts"]:
        for key, value in triggered["exclusion_reason_counts"].items():
            print(f"  {key}: {value}")
    else:
        print("  none")

    print("\nAvailable feedback-labeled confusion matrix (all eligibility states)")
    print(f"  scored labeled records: {confusion['scored_feedback_labeled_records']}")
    print(f"  TP: {confusion['true_positive']}")
    print(f"  FP: {confusion['false_positive']}")
    print(f"  TN: {confusion['true_negative']}")
    print(f"  FN: {confusion['false_negative']}")
    print(f"  accuracy over available labels: {fmt(confusion['accuracy_pct'])}%")
    print("  NOTE: TN/FN do not cover ineligible no-action episodes without feedback labels.")

    print("\nEvent-log cross-check")
    print(f"  completed-session triggered records: {audit['record_trigger_count']}")
    print(f"  integration_action_presented events: {audit['integration_action_presented_count']}")
    print(f"  counts/episode IDs match: {audit['counts_match']}")
    if audit["record_trigger_episode_ids_missing_from_events"]:
        print(
            "  record triggers missing from event log: "
            + ", ".join(map(str, audit["record_trigger_episode_ids_missing_from_events"]))
        )
    if audit["event_action_episode_ids_missing_from_records"]:
        print(
            "  event actions missing from completed records: "
            + ", ".join(map(str, audit["event_action_episode_ids_missing_from_records"]))
        )

    duration = session["duration"]
    print("\nSession duration")
    print(f"  configured scientific duration: {fmt(duration['configured_scientific_duration_s'], 3)} s")
    print(f"  actual start->completion elapsed: {fmt(duration['actual_start_to_completion_elapsed_s'], 3)} s")
    print(f"  completed-session timestamp: {fmt(duration['completed_session_timestamp_s'], 3)} s")

    if details:
        print("\nTriggered event details")
        print(
            "  episode  track  class            feedback  eeg_quality      eligible  latency(s)"
        )
        for row in summary["triggered_event_rows"]:
            print(
                f"  {str(row['episode_id']):>7}  {str(row['track_id']):>5}  "
                f"{str(row['classification']):<15}  "
                f"{str(row['feedback_pressed']):<8}  "
                f"{str(row['eeg_quality_state']):<15}  "
                f"{str(row['training_eligible']):<8}  "
                f"{fmt(row['selection_latency_s'], 3):>10}"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "episode_id",
        "track_id",
        "episode_start_timestamp",
        "action_timestamp",
        "integration_action_presented_timestamp",
        "dwell_trigger_timestamp",
        "selection_latency_s",
        "classification",
        "feedback_pressed",
        "feedback_resolution_timestamp",
        "common_label",
        "training_eligible",
        "eeg_quality_state",
        "eeg_quality_reasons",
        "exclusion_reasons",
        "g_required_dwell_s",
        "e_required_dwell_s",
        "engagement_index",
        "eeg_probability",
        "eeg_evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["eeg_quality_reasons"] = json.dumps(
                out.get("eeg_quality_reasons"), separators=(",", ":")
            )
            out["exclusion_reasons"] = json.dumps(
                out.get("exclusion_reasons"), separators=(",", ":")
            )
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("participant", help="Participant/subject ID, e.g. P021")
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
        help="Emit the complete report as JSON instead of human-readable text",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print one row per triggered event after the summary",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write one CSV row per triggered event",
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

    if args.csv is not None:
        write_csv(args.csv.expanduser(), summary["triggered_event_rows"])

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    else:
        print_human(summary, details=args.details)
        if args.csv is not None:
            print(f"\nTriggered-event CSV: {args.csv.expanduser()}")


if __name__ == "__main__":
    main()
