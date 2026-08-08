import json
from pathlib import Path

import pytest

from foundations.config import load_resolved_config
from integration.analysis import generate_analysis
from integration.orchestrator import ScheduledFeedbackPress, TimedFeedbackDriver
from integration.workflow import run_integrated_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_CONFIG = PROJECT_ROOT / "configs" / "integration.yaml"


def _config(tmp_path: Path, *, participant_id: str = "test-P001") -> dict:
    config = load_resolved_config(INTEGRATION_CONFIG)
    config["output_root"] = str(tmp_path / "runs")
    config["participant"]["id"] = participant_id
    config["participant"]["checkpoint_path"] = str(tmp_path / f"{participant_id}.pkl")
    config["session"]["id_prefix"] = "test-session"
    return config


def _events(run_directory: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def _named(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event["name"] == name]


def test_synthetic_end_to_end_both_conditions_and_n_plus_one(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_run = run_integrated_experiment(config)
    second_run = run_integrated_experiment(config)

    first_events = _events(first_run)
    second_events = _events(second_run)
    assert _named(first_events, "integration_session_started")[0]["payload"]["active_condition"] == "G"
    assert _named(second_events, "integration_session_started")[0]["payload"]["active_condition"] == "E"

    predictions = _named(first_events, "experiment_prediction")
    results = _named(first_events, "experiment_episode_result")
    assert predictions and results
    assert all(
        event["payload"]["eeg_window_end"] <= event["payload"]["cutoff_timestamp"]
        for event in predictions
    )

    gaze_events = _named(first_events, "integration_gaze_processed")
    frozen = next(event for event in gaze_events if event["payload"]["prediction_reason"] == "prediction_frozen")
    episode_id = frozen["payload"]["episode_id"]
    assert frozen["payload"]["applied_intent_score"] is None
    later = next(
        event
        for event in gaze_events
        if event["timestamp"] > frozen["timestamp"]
        and event["payload"]["episode_id"] == episode_id
        and event["payload"]["applied_intent_score"] is not None
    )
    prediction = next(
        event["payload"] for event in predictions if event["payload"]["episode_id"] == episode_id
    )
    assert later["payload"]["applied_intent_score"] == pytest.approx(
        prediction["active_intent_score"]
    )

    completion = _named(second_events, "integration_session_completed")[0]["payload"]
    assert completion["training_counts_after"]["G"] == completion["training_counts_after"]["E"]
    assert completion["training_counts_after"]["G"] == len(results) + len(
        _named(second_events, "experiment_episode_result")
    )
    for run in (first_run, second_run):
        assert (run / "raw_glasses.h5").is_file()
        assert (run / "raw_eeg.h5").is_file()
        assert (run / "analysis_summary.json").is_file()
        assert (run / "learning_curve.csv").is_file()
        for name in (
            "resolved_integration_config.json",
            "resolved_gaze_interaction_config.json",
            "resolved_eeg_pipeline_config.json",
            "resolved_experiment_learning_config.json",
        ):
            assert (run / name).is_file()


def test_unusable_eeg_is_explicit_strict_paired_skip(tmp_path: Path) -> None:
    eeg_override = tmp_path / "bad_eeg.yaml"
    eeg_override.write_text(
        """source:\n  synthetic:\n    duration_seconds: 12.0\n    invalid_intervals:\n      - [0.0, 12.0]\n""",
        encoding="utf-8",
    )
    config = _config(tmp_path)
    config["subsystem_config_overrides"]["eeg_pipeline"] = str(eeg_override)
    run = run_integrated_experiment(config)
    summary = json.loads((run / "analysis_summary.json").read_text(encoding="utf-8"))
    completion = _named(_events(run), "integration_session_completed")[0]["payload"]

    assert summary["paired_skips"]["count"] > 0
    assert set(summary["paired_skips"]["reasons"]) == {"paired_eeg_rejected"}
    assert summary["episode_results"] == 0
    assert completion["training_counts_after"] == {"G": 0, "E": 0}


def test_invalid_gaze_does_not_fabricate_episode_prediction_or_label(tmp_path: Path) -> None:
    gaze_override = tmp_path / "invalid_gaze.yaml"
    gaze_override.write_text(
        """source:\n  virtual:\n    duration_seconds: 2.0\n    scene_width: 40\n    scene_height: 30\n    gaze_invalid_probability: 1.0\n""",
        encoding="utf-8",
    )
    config = _config(tmp_path)
    config["subsystem_config_overrides"]["gaze_interaction"] = str(gaze_override)
    run = run_integrated_experiment(config)
    events = _events(run)

    gaze_events = _named(events, "integration_gaze_processed")
    assert gaze_events and all(event["payload"]["valid"] is False for event in gaze_events)
    assert not _named(events, "integration_episode_started")
    assert not _named(events, "experiment_prediction")
    assert not _named(events, "experiment_episode_result")


def test_ended_outcome_is_routed_before_new_episode_prediction(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["synthetic_vision"]["visible_seconds"] = 0.3
    config["synthetic_vision"]["blank_seconds"] = 0.01
    config["feedback"]["synthetic"]["press_cycle"] = [False]
    run = run_integrated_experiment(config)
    events = _events(run)

    ended = _named(events, "integration_episode_ended")
    suppressed = [
        event
        for event in _named(events, "integration_gaze_processed")
        if event["payload"]["prediction_reason"] == "feedback_pending_at_episode_start"
    ]
    assert ended and suppressed
    assert any(event["timestamp"] == pytest.approx(ended[0]["timestamp"]) for event in suppressed)


class _MismatchedPendingController:
    pending_feedback_episode_id = 1

    def button_press(self, timestamp):  # pragma: no cover - mismatch must fail first
        raise AssertionError(timestamp)

    def advance_time(self, timestamp):  # pragma: no cover - mismatch must fail first
        raise AssertionError(timestamp)


def test_feedback_replay_rejects_wrong_episode_identity(tmp_path: Path) -> None:
    from foundations.events import JsonlEventLogger

    driver = TimedFeedbackDriver(
        [ScheduledFeedbackPress(1.0, 2)],
        event_logger=JsonlEventLogger(tmp_path / "events.jsonl"),
        session_id="S001",
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        driver.before_time(1.0, _MismatchedPendingController())


def test_saved_inputs_feedback_and_analysis_replay_deterministically(tmp_path: Path) -> None:
    original_config = _config(tmp_path / "original")
    original = run_integrated_experiment(original_config)

    gaze_override = tmp_path / "replay_gaze.yaml"
    gaze_override.write_text(
        f"source:\n  recording_path: {original / 'raw_glasses.h5'}\n  replay_paced: false\n",
        encoding="utf-8",
    )
    eeg_override = tmp_path / "replay_eeg.yaml"
    eeg_override.write_text(
        f"source:\n  mode: replay\n  replay_path: {original / 'raw_eeg.h5'}\n  replay_paced: false\n",
        encoding="utf-8",
    )
    replay_config = _config(tmp_path / "replay")
    replay_config["input"]["mode"] = "replay"
    replay_config["feedback"]["mode"] = "replay"
    replay_config["feedback"]["replay"]["events_path"] = str(original / "events.jsonl")
    replay_config["subsystem_config_overrides"]["gaze_interaction"] = str(gaze_override)
    replay_config["subsystem_config_overrides"]["eeg_pipeline"] = str(eeg_override)
    replay = run_integrated_experiment(replay_config)

    original_summary = json.loads((original / "analysis_summary.json").read_text(encoding="utf-8"))
    replay_summary = json.loads((replay / "analysis_summary.json").read_text(encoding="utf-8"))
    assert replay_summary == original_summary

    regenerated_dir = tmp_path / "regenerated"
    regenerated = generate_analysis(replay / "events.jsonl", regenerated_dir)
    assert regenerated == replay_summary
    assert (regenerated_dir / "learning_curve.csv").read_text(encoding="utf-8") == (
        replay / "learning_curve.csv"
    ).read_text(encoding="utf-8")


def test_incomplete_session_consumes_slot_without_learning_then_restart_is_new_session(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    broken_recording = tmp_path / "broken.h5"
    broken_recording.write_text("not an HDF5 file", encoding="utf-8")
    broken_gaze = tmp_path / "broken_gaze.yaml"
    broken_gaze.write_text(
        f"source:\n  recording_path: {broken_recording}\n  replay_paced: false\n",
        encoding="utf-8",
    )
    config["input"]["mode"] = "replay"
    config["subsystem_config_overrides"]["gaze_interaction"] = str(broken_gaze)
    with pytest.raises(OSError):
        run_integrated_experiment(config)

    incomplete_runs = sorted((tmp_path / "runs" / "integration").iterdir())
    assert len(incomplete_runs) == 1
    incomplete_events = _events(incomplete_runs[0])
    assert _named(incomplete_events, "integration_session_incomplete")
    assert not _named(incomplete_events, "experiment_episode_result")

    config["input"]["mode"] = "synthetic"
    config["subsystem_config_overrides"]["gaze_interaction"] = "configs/integration_gaze.yaml"
    resumed = run_integrated_experiment(config)
    started = _named(_events(resumed), "integration_session_started")[0]["payload"]
    assert started["session_index"] == 1
    assert started["active_condition"] == "E"
    assert started["training_counts_before"] == {"G": 0, "E": 0}
