"""Regenerate descriptive G-versus-E summaries from persisted JSONL records."""

from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _model_metrics(results: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    predicted_key = f"{prefix}_predicted_label"
    tp = sum(row[predicted_key] == 1 and row["common_label"] == 1 for row in results)
    fp = sum(row[predicted_key] == 1 and row["common_label"] == 0 for row in results)
    tn = sum(row[predicted_key] == 0 and row["common_label"] == 0 for row in results)
    fn = sum(row[predicted_key] == 0 and row["common_label"] == 1 for row in results)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _ratio(tp + tn, len(results)),
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


def _learning_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_results: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        cumulative_results.append(result)
        g_metrics = _model_metrics(cumulative_results, "g")
        e_metrics = _model_metrics(cumulative_results, "e")
        rows.append(
            {
                "sample_index": index,
                "participant_id": result["participant_id"],
                "session_id": result["session_id"],
                "episode_id": result["episode_id"],
                "active_condition": result["active_condition"],
                "common_label": result["common_label"],
                "g_correct": int(result["g_correct"]),
                "e_correct": int(result["e_correct"]),
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
    predictions = [
        event["payload"] for event in events if event.get("name") == "experiment_prediction"
    ]
    results = [
        event["payload"] for event in events if event.get("name") == "experiment_episode_result"
    ]
    if any(result.get("update_applied") is not True for result in results):
        raise ValueError("persisted episode result reports an unapplied paired update")

    episode_starts: dict[tuple[str, int], float] = {}
    for event in events:
        if event.get("name") != "integration_episode_started":
            continue
        payload = event["payload"]
        key = (str(payload["session_id"]), int(payload["episode_id"]))
        episode_starts[key] = float(payload["start_timestamp"])

    selection_latencies: list[float] = []
    for result in results:
        if not result["action_occurred"]:
            continue
        key = (str(result["session_id"]), int(result["episode_id"]))
        start = episode_starts.get(key)
        if start is None:
            continue
        latency = float(result["outcome_timestamp"]) - start
        if latency < -1e-12 or not math.isfinite(latency):
            raise ValueError("persisted selection latency is invalid")
        selection_latencies.append(max(0.0, latency))

    unavailable = [record for record in predictions if record.get("unavailable_reason")]
    skip_reasons = Counter(str(record["unavailable_reason"]) for record in unavailable)
    label_counts = Counter(str(result["common_label"]) for result in results)
    action_counts = Counter("action" if result["action_occurred"] else "no_action" for result in results)
    controlled = [result for result in results if result.get("instructed_intention") is not None]
    controlled_matches = sum(
        result["common_label"] == result["instructed_intention"] for result in controlled
    )

    by_condition: dict[str, Any] = {}
    for condition in sorted({str(result["active_condition"]) for result in results}):
        subset = [result for result in results if result["active_condition"] == condition]
        by_condition[condition] = {
            "episode_results": len(subset),
            "G": _model_metrics(subset, "g"),
            "E": _model_metrics(subset, "e"),
        }

    summary = {
        "episode_results": len(results),
        "paired_predictions_available": len(predictions) - len(unavailable),
        "paired_skips": {
            "count": len(unavailable),
            "reasons": dict(sorted(skip_reasons.items())),
        },
        "active_conditions": sorted({str(result["active_condition"]) for result in results}),
        "common_feedback_labels": dict(sorted(label_counts.items())),
        "outcomes": dict(sorted(action_counts.items())),
        "models": {
            "G": _model_metrics(results, "g"),
            "E": _model_metrics(results, "e"),
        },
        "by_active_condition": by_condition,
        "selection_latency_seconds": _latency_summary(selection_latencies),
        "controlled_intention_agreement": {
            "trials": len(controlled),
            "matches": controlled_matches,
            "fraction": _ratio(controlled_matches, len(controlled)),
        },
        "scientific_note": (
            "common_label is feedback-derived; feedback silence is not independently observed ground truth"
        ),
    }
    summary_path = destination / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    learning_path = destination / "learning_curve.csv"
    rows = _learning_rows(results)
    columns = (
        "sample_index",
        "participant_id",
        "session_id",
        "episode_id",
        "active_condition",
        "common_label",
        "g_correct",
        "e_correct",
        "g_cumulative_accuracy",
        "e_cumulative_accuracy",
        "g_cumulative_f1",
        "e_cumulative_f1",
    )
    with learning_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == "__main__":
    print("Use scripts/analyze_integrated_experiment.py with an integrated events.jsonl file.")
