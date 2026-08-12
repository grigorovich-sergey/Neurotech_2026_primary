from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from eeg_pipeline.contracts import EEGFeatureWindow, QualityState, WindowCompleteness
from eeg_pipeline.processing import FEATURE_NAMES as EEG_FEATURE_NAMES
from foundations.config import load_resolved_config
from gaze_interaction.contracts import BoundingBox
from gaze_interaction.episodes import CandidateEpisode, EpisodeEndReason

from experiment_learning.artifacts import artifact_digest
from experiment_learning.contracts import (
    CompletedSession,
    Condition,
    EpisodeTrainingRecord,
    GazeContextObservation,
    ModelOutcome,
    OutcomeStatus,
    TrajectoryPoint,
)
from experiment_learning.eeg_indicator import (
    ENGAGEMENT_INDEX_FORMULA,
    ENGAGEMENT_INDEX_ID,
    InvalidEEGIndicator,
    engagement_index,
)
from experiment_learning.policy import (
    FrozenSessionPolicy,
    create_cold_start_policy,
    load_frozen_policy,
    save_frozen_policy,
)
from experiment_learning.schedule import (
    ScheduleBinding,
    load_condition_schedule,
    resolve_scheduled_condition,
)
from experiment_learning.state_machine import ExperimentController, derive_common_label
from experiment_learning.synthetic import run_synthetic_experiment
from experiment_learning.trainer import TrainerConfig, train_next_session_policy


class _EEGSource:
    def __init__(
        self,
        quality: QualityState = QualityState.USABLE,
        *,
        beta: float = 6.0,
        alpha: float = 2.0,
        theta: float = 1.0,
    ) -> None:
        self.quality = quality
        self.calls: list[tuple[float, float]] = []
        self.values = np.asarray([4.0, 12.0, 0.8, 2.0, theta, alpha, beta, 0.4])

    def features(self, start: float, end: float) -> EEGFeatureWindow:
        self.calls.append((start, end))
        usable = self.quality is QualityState.USABLE
        has_samples = self.quality is not QualityState.UNAVAILABLE
        return EEGFeatureWindow(
            requested_start=start,
            requested_end=end,
            actual_start=start if has_samples else None,
            actual_end=end if has_samples else None,
            sample_count=251 if has_samples else 0,
            completeness=WindowCompleteness.COMPLETE if has_samples else WindowCompleteness.EMPTY,
            quality_state=self.quality,
            quality_reasons=() if usable else (f"test_{self.quality.value}",),
            feature_names=EEG_FEATURE_NAMES,
            values=self.values if usable else None,
        )


def _policy(*, reduction: float = 0.0, coefficient: float = 0.0) -> FrozenSessionPolicy:
    cold = create_cold_start_policy(
        participant_id="P001",
        schedule_sequence_id="g-first",
        schedule_sha256="a" * 64,
    )
    return replace(
        cold,
        maximum_eeg_reduction_fraction=reduction,
        logistic_coefficient=coefficient,
    )


def _controller(
    *,
    condition: Condition = Condition.G,
    policy: FrozenSessionPolicy | None = None,
) -> ExperimentController:
    selected = policy or _policy()
    return ExperimentController(
        policy=selected,
        policy_sha256=artifact_digest(selected.to_payload()),
        session_id="S001",
        session_number=1,
        attempt_id="attempt-001",
        active_condition=condition,
        schedule_binding=ScheduleBinding("g-first", "a" * 64),
    )


def _active(episode_id: int, start: float, dwell: float) -> CandidateEpisode:
    return CandidateEpisode(
        episode_id,
        episode_id,
        "object",
        start,
        start + dwell,
        None,
        None,
    )


def _observation(episode: CandidateEpisode, dwell: float) -> GazeContextObservation:
    return GazeContextObservation(
        episode_id=episode.episode_id,
        track_id=episode.track_id,
        timestamp=episode.last_match_timestamp,
        matched_dwell_s=dwell,
        gaze_x_normalized=0.5,
        gaze_y_normalized=0.5,
        candidate_box=BoundingBox(0.2, 0.2, 0.8, 0.8),
    )


def _evaluate(
    controller: ExperimentController,
    source: _EEGSource,
    episode_id: int = 1,
    start: float = 2.0,
    dwell: float = 0.25,
    instructed: int | None = None,
):
    episode = _active(episode_id, start, dwell)
    return controller.evaluate_update(
        episode,
        _observation(episode, dwell),
        source,
        instructed_intention=instructed,
    )


def test_engagement_index_is_exact_and_has_no_hidden_epsilon() -> None:
    features = {
        "beta_power_13_30_hz": 6.0,
        "alpha_power_8_13_hz": 2.0,
        "theta_power_4_8_hz": 1.0,
    }
    assert engagement_index(features) == 2.0
    assert ENGAGEMENT_INDEX_FORMULA == (
        "beta_power_13_30_hz / (alpha_power_8_13_hz + theta_power_4_8_hz)"
    )
    with pytest.raises(InvalidEEGIndicator, match="must be positive"):
        engagement_index({**features, "alpha_power_8_13_hz": 0.0, "theta_power_4_8_hz": 0.0})


def test_policy_artifact_is_immutable_strict_and_maps_dwell_conservatively(tmp_path: Path) -> None:
    policy = _policy(reduction=0.4)
    path = tmp_path / "policy.json"
    digest = save_frozen_policy(path, policy)
    loaded = load_frozen_policy(
        path,
        expected_participant_id="P001",
        expected_session=1,
        expected_sha256=digest,
    )
    assert loaded.e_required_dwell(0.5) == pytest.approx(0.8)
    assert loaded.dwell_parameters(Condition.G).maximum_reduction_fraction == 0.0
    assert loaded.dwell_parameters(Condition.E).maximum_reduction_fraction == 0.4
    assert save_frozen_policy(path, policy) == digest
    with pytest.raises(FileExistsError, match="different content"):
        save_frozen_policy(path, replace(policy, g_base_threshold_s=0.9))
    legacy = tmp_path / "old.pkl"
    legacy.write_bytes(b"not a JSON policy")
    with pytest.raises(ValueError, match="legacy River pickle"):
        load_frozen_policy(legacy, expected_participant_id="P001", expected_session=1)


def test_schedule_is_predetermined_and_digest_binding_rejects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "schedule.csv"
    path.write_text(
        "sequence_id,session_number,active_condition\nseq-a,1,G\nseq-a,2,E\n",
        encoding="utf-8",
    )
    first = load_condition_schedule(path)
    resolved = resolve_scheduled_condition(first, "seq-a", 2)
    assert resolved.condition is Condition.E
    path.write_text(
        "sequence_id,session_number,active_condition\nseq-a,1,E\nseq-a,2,G\n",
        encoding="utf-8",
    )
    changed = load_condition_schedule(path)
    with pytest.raises(ValueError, match="persisted"):
        resolve_scheduled_condition(changed, "seq-a", 2, resolved.binding)


def test_schedule_rejects_legacy_condition_header(tmp_path: Path) -> None:
    path = tmp_path / "schedule.csv"
    path.write_text(
        "sequence_id,session_number,condition\nseq-a,1,G\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="condition schedule header must be exactly: "
        "sequence_id,session_number,active_condition",
    ):
        load_condition_schedule(path)


def test_schedule_names_active_condition_in_value_error(tmp_path: Path) -> None:
    path = tmp_path / "schedule.csv"
    path.write_text(
        "sequence_id,session_number,active_condition\nseq-a,1,X\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="active_condition must be G or E on CSV line 2",
    ):
        load_condition_schedule(path)


def test_eeg_decision_uses_only_scalar_index_and_freezes_without_runtime_mutation() -> None:
    policy = _policy(reduction=0.4, coefficient=2.0)
    controller = _controller(condition=Condition.E, policy=policy)
    source = _EEGSource()
    before = policy.to_payload()
    first = _evaluate(controller, source)
    second = _evaluate(controller, source, dwell=0.35)

    assert first.engagement_index == 2.0
    assert first.eeg_probability == pytest.approx(1.0 / (1.0 + np.exp(-4.0)))
    assert first.intent_score == pytest.approx(max(0.0, 2.0 * first.eeg_probability - 1.0))
    assert second.engagement_index == first.engagement_index
    assert second.newly_frozen is False
    assert len(source.calls) == 1
    assert policy.to_payload() == before


@pytest.mark.parametrize("quality", [QualityState.UNAVAILABLE, QualityState.REJECTED])
def test_missing_eeg_falls_back_and_no_action_has_no_feedback_target(quality: QualityState) -> None:
    controller = _controller(condition=Condition.E)
    decision = _evaluate(controller, _EEGSource(quality))
    assert decision.intent_score is None
    assert decision.required_dwell_s == 1.0
    assert decision.training_eligible is False
    ended = CandidateEpisode(
        1, 1, "object", 2.0, 2.25, 2.5, EpisodeEndReason.CANDIDATE_CHANGE
    )
    assert controller.open_no_action_feedback(ended, 2.5) is False
    assert controller.pending_feedback_episode_id is None
    assert len(controller.records) == 1
    assert controller.records[0].training_eligible is False


def test_unusable_eeg_action_still_accepts_correction_but_never_trains() -> None:
    controller = _controller(condition=Condition.E)
    _evaluate(controller, _EEGSource(QualityState.UNAVAILABLE))
    assert controller.open_action_feedback(1, 2.5)
    resolution = controller.accept_feedback(2.6)
    assert resolution is not None
    assert resolution.record.common_label == 0
    assert resolution.record.training_eligible is False


@pytest.mark.parametrize(
    ("action", "press", "expected"),
    [(True, False, 1), (True, True, 0), (False, False, 0), (False, True, 1)],
)
def test_feedback_truth_table(action: bool, press: bool, expected: int) -> None:
    controller = _controller()
    source = _EEGSource()
    _evaluate(controller, source)
    if action:
        controller.open_action_feedback(1, 2.5)
        outcome = 2.5
    else:
        ended = CandidateEpisode(
            1, 1, "object", 2.0, 2.25, 2.5, EpisodeEndReason.CANDIDATE_CHANGE
        )
        assert controller.open_no_action_feedback(ended, 2.5)
        outcome = 2.5
    resolution = controller.accept_feedback(outcome + 0.1) if press else controller.advance_time(outcome + 1.5)
    assert derive_common_label(action_occurred=action, feedback_pressed=press) == expected
    assert resolution is not None and resolution.record.common_label == expected


def test_accepted_feedback_cancels_newer_provisional_but_timeout_does_not() -> None:
    accepted = _controller()
    source = _EEGSource()
    _evaluate(accepted, source, episode_id=1, start=2.0)
    accepted.open_action_feedback(1, 2.5)
    _evaluate(accepted, source, episode_id=2, start=2.6)
    resolution = accepted.accept_feedback(2.8)
    assert resolution is not None
    assert [item.episode_id for item in resolution.cancellation_instructions] == [2]
    assert accepted.records[-1].cancellation_reason == "feedback_accepted"

    timeout = _controller()
    _evaluate(timeout, source, episode_id=1, start=4.0)
    timeout.open_action_feedback(1, 4.5)
    _evaluate(timeout, source, episode_id=2, start=4.6)
    resolution = timeout.advance_time(6.0)
    assert resolution is not None and resolution.cancellation_instructions == ()
    assert [record.episode_id for record in timeout.records] == [1]
    assert timeout.action_gate_open is True


def test_action_prefix_marks_unobservable_shadow_outcome_censored() -> None:
    policy = _policy(reduction=0.5, coefficient=5.0)
    controller = _controller(condition=Condition.E, policy=policy)
    source = _EEGSource()
    _evaluate(controller, source, dwell=0.25)
    _evaluate(controller, source, dwell=0.6)
    controller.open_action_feedback(1, 2.6)
    resolution = controller.advance_time(4.1)
    assert resolution is not None
    record = resolution.record
    assert record.e_outcome.status is OutcomeStatus.ACTION
    assert record.g_outcome.status is OutcomeStatus.COUNTERFACTUAL_CENSORED
    assert record.eeg_feature_values is not None
    assert record.eeg_indicator_id == ENGAGEMENT_INDEX_ID


def _training_record(index: int, label: int, *, controlled: bool = False) -> EpisodeTrainingRecord:
    start = float(index * 3 + 1)
    max_dwell = 1.1 if label else 0.55
    trajectory = tuple(
        TrajectoryPoint(start + dwell, dwell)
        for dwell in np.arange(0.25, max_dwell + 0.001, 0.05)
    )
    indicator = 1.8 if label else 0.25
    feedback_open = start + max_dwell + 0.1
    exclusions = ("controlled_intention_trial",) if controlled else ()
    return EpisodeTrainingRecord(
        participant_id="P001",
        session_id="S001",
        session_number=1,
        attempt_id="attempt-001",
        episode_id=index,
        track_id=index,
        active_condition=Condition.G,
        policy_sha256="b" * 64,
        episode_start_timestamp=start,
        prediction_cutoff_timestamp=start + 0.25,
        eeg_window_start=start - 0.75,
        eeg_window_end=start + 0.25,
        eeg_quality_state="usable",
        eeg_quality_reasons=(),
        eeg_feature_names=EEG_FEATURE_NAMES,
        eeg_feature_values=(4.0, 12.0, 0.8, 2.0, 1.0, 2.0, 3.0 * indicator, 0.4),
        eeg_indicator_id=ENGAGEMENT_INDEX_ID,
        eeg_indicator_formula=ENGAGEMENT_INDEX_FORMULA,
        engagement_index=indicator,
        eeg_probability=0.5,
        eeg_evidence=0.0,
        g_required_dwell_s=1.0,
        e_required_dwell_s=1.0,
        trajectory=trajectory,
        action_occurred=False,
        action_timestamp=None,
        natural_endpoint_timestamp=feedback_open,
        feedback_window_open=feedback_open,
        feedback_deadline=feedback_open + 1.5,
        feedback_pressed=bool(label),
        feedback_resolution_timestamp=(feedback_open + 0.1 if label else feedback_open + 1.5),
        common_label=label,
        instructed_intention=label if controlled else None,
        g_outcome=ModelOutcome(OutcomeStatus.NO_ACTION, None),
        e_outcome=ModelOutcome(OutcomeStatus.NO_ACTION, None),
        training_eligible=not controlled,
        exclusion_reasons=exclusions,
        canceled=False,
        cancellation_reason=None,
    )


def _completed_session(count: int = 24, *, include_controlled: bool = True) -> CompletedSession:
    records = [_training_record(index, index % 2) for index in range(1, count + 1)]
    if include_controlled:
        records.append(_training_record(count + 1, 1, controlled=True))
    return CompletedSession(
        participant_id="P001",
        session_id="S001",
        session_number=1,
        attempt_id="attempt-001",
        active_condition=Condition.G,
        policy_sha256="b" * 64,
        schedule_sequence_id="g-first",
        schedule_sha256="a" * 64,
        completed_timestamp=100.0,
        successful=True,
        records=tuple(records),
    )


def test_trainer_is_deterministic_one_dimensional_and_writes_auditable_artifacts(tmp_path: Path) -> None:
    prior = _policy()
    session = _completed_session()
    config = TrainerConfig()
    first = train_next_session_policy([session], prior, tmp_path / "first.json", config)
    second = train_next_session_policy([session], prior, tmp_path / "second.json", config)

    assert first.policy.to_payload() == second.policy.to_payload()
    assert first.report == second.report
    assert first.policy.logistic_coefficient > 0.0
    assert first.policy.g_base_threshold_s == pytest.approx(0.6)
    assert first.policy.maximum_eeg_reduction_fraction == pytest.approx(0.5)
    assert first.report["counts"]["accepted"] == 24
    assert first.report["counts"]["excluded"] == {"controlled_intention_trial": 1}
    assert first.report["optimizer"]["initial_parameters"] == [0.0, 0.0]
    assert first.policy.eeg_indicator_formula == ENGAGEMENT_INDEX_FORMULA
    assert first.policy.policy_for_session == 2


def test_insufficient_data_writes_next_artifact_with_prior_parameters(tmp_path: Path) -> None:
    prior = _policy()
    session = _completed_session(10, include_controlled=False)
    result = train_next_session_policy([session], prior, tmp_path / "next.json", TrainerConfig())
    assert result.policy.policy_for_session == 2
    assert result.policy.g_base_threshold_s == prior.g_base_threshold_s
    assert result.policy.maximum_eeg_reduction_fraction == 0.0
    assert result.policy.cold_start_status == "carried_forward_insufficient_examples"


def test_trainer_rejects_cross_participant_and_duplicate_successful_sessions(tmp_path: Path) -> None:
    prior = _policy()
    session = _completed_session()
    other = replace(session, participant_id="P002", records=())
    with pytest.raises(ValueError, match="another participant"):
        train_next_session_policy([other], prior, tmp_path / "other.json", TrainerConfig())
    with pytest.raises(ValueError, match="duplicate successful"):
        train_next_session_policy([session, session], prior, tmp_path / "duplicate.json", TrainerConfig())


def test_small_synthetic_run_is_deterministic_and_retrains_after_every_session(tmp_path: Path) -> None:
    default = Path(__file__).resolve().parents[1] / "configs" / "experiment_learning.yaml"
    first_config = load_resolved_config(default)
    first_config["output_root"] = str(tmp_path / "first")
    first_config["schedule_path"] = str(default.parent / "experiment_condition_schedule.csv")
    first_config["episodes"] = 80
    first = run_synthetic_experiment(first_config)
    second_config = load_resolved_config(default)
    second_config["output_root"] = str(tmp_path / "second")
    second_config["schedule_path"] = str(default.parent / "experiment_condition_schedule.csv")
    second_config["episodes"] = 80
    second = run_synthetic_experiment(second_config)
    first_summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    second_summary = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    assert first_summary == second_summary
    assert first_summary["condition_sessions"] == {"E": 2, "G": 2}
    assert first_summary["completed_session_artifacts"] == 4
    assert first_summary["next_policy_session"] == 5
    assert first_summary["training_has_no_random_seed"] is True
