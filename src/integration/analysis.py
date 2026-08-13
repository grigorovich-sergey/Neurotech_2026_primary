"""Regenerate descriptive G-versus-E summaries from persisted JSONL records."""

from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _prediction(record: dict[str, Any], prefix: str) -> int | None:
    status = record[f"{prefix}_outcome"]["status"]
    if status == "action":
        return 1
    if status == "no_action":
        return 0
    return None


def _model_metrics(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    scored = [
        (prediction, int(record["common_label"]))
        for record in records
        if record.get("common_label") in (0, 1)
        and (prediction := _prediction(record, prefix)) is not None
    ]
    tp = sum(prediction == 1 and label == 1 for prediction, label in scored)
    fp = sum(prediction == 1 and label == 0 for prediction, label in scored)
    tn = sum(prediction == 0 and label == 0 for prediction, label in scored)
    fn = sum(prediction == 0 and label == 1 for prediction, label in scored)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "scored": len(scored),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _ratio(tp + tn, len(scored)),
        "false_activations": fp,
        "missed_intentions": fn,
    }


def _latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL event on line {line_number}") from exc
            if not isinstance(event, dict) or not isinstance(event.get("payload"), dict):
                raise ValueError(f"invalid event structure on line {line_number}")
            events.append(event)
    return events


def _eligible(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("training_eligible") is True]


def _learning_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    for index, record in enumerate(_eligible(records), start=1):
        cumulative.append(record)
        g_metrics = _model_metrics(cumulative, "g")
        e_metrics = _model_metrics(cumulative, "e")
        rows.append(
            {
                "sample_index": index,
                "participant_id": record["participant_id"],
                "session_id": record["session_id"],
                "session_number": record["session_number"],
                "episode_id": record["episode_id"],
                "active_condition": record["active_condition"],
                "common_label": record["common_label"],
                "g_outcome": record["g_outcome"]["status"],
                "e_outcome": record["e_outcome"]["status"],
                "g_cumulative_accuracy": g_metrics["accuracy"],
                "e_cumulative_accuracy": e_metrics["accuracy"],
                "g_cumulative_f1": g_metrics["f1"],
                "e_cumulative_f1": e_metrics["f1"],
            }
        )
    return rows


def generate_analysis(events_path: str | Path, output_directory: str | Path) -> dict[str, Any]:
    """Write analysis_summary.json and learning_curve.csv from persisted events only."""

    source = Path(events_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    events = _load_events(source)
    decisions = [
        event["payload"] for event in events if event.get("name") == "experiment_policy_decision"
    ]
    records = [
        event["payload"]
        for event in events
        if event.get("name") == "experiment_episode_training_record"
    ]
    usable = _eligible(records)
    exclusions = Counter(
        reason
        for record in records
        if not record.get("training_eligible")
        for reason in record.get("exclusion_reasons", [])
    )
    label_counts = Counter(str(record["common_label"]) for record in usable)
    action_counts = Counter(
        "action" if record["action_occurred"] else "no_action"
        for record in usable
    )
    latencies = [
        float(record["action_timestamp"]) - float(record["episode_start_timestamp"])
        for record in usable
        if record.get("action_timestamp") is not None
    ]
    by_condition: dict[str, Any] = {}
    for condition in sorted({str(record["active_condition"]) for record in records}):
        subset = [record for record in records if record["active_condition"] == condition]
        by_condition[condition] = {
            "episode_records": len(subset),
            "training_eligible": len(_eligible(subset)),
            "G": _model_metrics(_eligible(subset), "g"),
            "E": _model_metrics(_eligible(subset), "e"),
        }

    summary = {
        "episode_records": len(records),
        "training_eligible_records": len(usable),
        "policy_decisions": len(decisions),
        "excluded_records": {
            "count": len(records) - len(usable),
            "reasons": dict(sorted(exclusions.items())),
        },
        "active_conditions": sorted({str(record["active_condition"]) for record in records}),
        "common_feedback_labels": dict(sorted(label_counts.items())),
        "outcomes": dict(sorted(action_counts.items())),
        "models": {
            "G": _model_metrics(usable, "g"),
            "E": _model_metrics(usable, "e"),
        },
        "by_active_condition": by_condition,
        "selection_latency_seconds": _latency_summary(latencies),
        "scientific_note": (
            "common_label is feedback-derived; feedback silence is not independently observed ground truth"
        ),
    }
    with (destination / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    columns = (
        "sample_index",
        "participant_id",
        "session_id",
        "session_number",
        "episode_id",
        "active_condition",
        "common_label",
        "g_outcome",
        "e_outcome",
        "g_cumulative_accuracy",
        "e_cumulative_accuracy",
        "g_cumulative_f1",
        "e_cumulative_f1",
    )
    with (destination / "learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(_learning_rows(records))
    return summary


if __name__ == "__main__":
    print("Use scripts/analyze_integrated_experiment.py with an integrated events.jsonl file.")
