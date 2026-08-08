import json
from pathlib import Path

import numpy as np
import pytest

from eeg_pipeline.contracts import EEGFeatureWindow, QualityState, WindowCompleteness
from eeg_pipeline.processing import FEATURE_NAMES as EEG_FEATURE_NAMES
from foundations.config import load_resolved_config
from gaze_interaction.contracts import BoundingBox
from gaze_interaction.dwell import DwellTrigger
from gaze_interaction.episodes import CandidateEpisode, EpisodeEndReason

from experiment_learning.checkpoint import (
    ParticipantState,
    load_participant_checkpoint,
)
from experiment_learning.contracts import Condition, GazeContextObservation
from experiment_learning.features import E_FEATURE_NAMES, GAZE_FEATURE_NAMES
from experiment_learning.models import (
    FrozenPredictions,
    ModelConfig,
    ScoredPredictions,
)
from experiment_learning.schedule import SessionSchedule
from experiment_learning.state_machine import ExperimentController, derive_common_label
from experiment_learning.synthetic import run_synthetic_experiment


class _EEGSource:
    def __init__(self, quality: QualityState = QualityState.USABLE) -> None:
        self.quality = quality
        self.calls: list[tuple[float, float]] = []

    def features(self, start: float, end: float) -> EEGFeatureWindow:
        self.calls.append((start, end))
        usable = self.quality is QualityState.USABLE
        return EEGFeatureWindow(
            requested_start=start,
            requested_end=end,
            actual_start=start if usable else None,
            actual_end=end if usable else None,
            sample_count=251 if usable else 0,
            completeness=WindowCompleteness.COMPLETE if usable else WindowCompleteness.EMPTY,
            quality_state=self.quality,
            quality_reasons=() if usable else (f"test_{self.quality.value}",),
            feature_names=EEG_FEATURE_NAMES,
            values=(
                np.arange(1, len(EEG_FEATURE_NAMES) + 1, dtype=np.float64)
                if usable
                else None
            ),
        )


class _FutureEEGSource(_EEGSource):
    def features(self, start: float, end: float) -> EEGFeatureWindow:
        window = super().features(start, end + 0.01)
        return window


def _active_episode(episode_id: int = 1, start: float = 2.0, cutoff: float = 2.25) -> CandidateEpisode:
    return CandidateEpisode(
        episode_id, episode_id, "object", start, cutoff, None, None
    )


def _observation(episode: CandidateEpisode) -> GazeContextObservation:
    return GazeContextObservation(
        episode_id=episode.episode_id,
        track_id=episode.track_id,
        timestamp=episode.last_match_timestamp,
        matched_dwell_s=episode.last_match_timestamp - episode.start_timestamp,
        gaze_x_normalized=0.55,
        gaze_y_normalized=0.45,
        candidate_box=BoundingBox(0.2, 0.2, 0.8, 0.8),
    )


def _state(participant_id: str = "P001") -> ParticipantState:
    return ParticipantState.create(
        participant_id=participant_id,
        participant_sequence_index=0,
        model_config=ModelConfig(),
    )


def _controller(
    state: ParticipantState | None = None,
    *,
    condition: Condition = Condition.G,
    checkpoint_path: Path | None = None,
) -> ExperimentController:
    return ExperimentController(
        participant_state=state or _state(),
        session_id="S001",
        active_condition=condition,
        checkpoint_path=checkpoint_path,
    )


def _predict(
    controller: ExperimentController,
    episode: CandidateEpisode | None = None,
    source: _EEGSource | None = None,
    *,
    instructed_intention: int | None = None,
):
    episode = episode or _active_episode()
    source = source or _EEGSource()
    return controller.consider_prediction(
        episode,
        _observation(episode),
        source,
        instructed_intention=instructed_intention,
    )


@pytest.mark.parametrize(
    ("action_occurred", "feedback_pressed", "expected_label"),
    [
        (True, False, 1),
        (True, True, 0),
        (False, False, 0),
        (False, True, 1),
    ],
)
def test_feedback_truth_table_and_common_paired_update(
    action_occurred: bool, feedback_pressed: bool, expected_label: int
) -> None:
    state = _state()
    controller = _controller(state)
    episode = _active_episode()
    _predict(controller, episode)
    outcome = 2.5
    if action_occurred:
        controller.on_dwell_trigger(DwellTrigger(1, 1, outcome, 0.5))
    else:
        controller.on_episode_end(
            CandidateEpisode(1, 1, "object", 2.0, 2.25, outcome, EpisodeEndReason.SOURCE_END)
        )
    result = (
        controller.button_press(outcome + 0.1)
        if feedback_pressed
        else controller.advance_time(outcome + 1.5)
    )

    assert derive_common_label(
        action_occurred=action_occurred, feedback_pressed=feedback_pressed
    ) == expected_label
    assert result is not None
    assert result.common_label == expected_label
    assert state.learners.g_model.training_count == 1
    assert state.learners.e_model.training_count == 1


def test_feedback_labels_only_one_episode_and_suppresses_episode_started_while_pending() -> None:
    controller = _controller()
    first = _active_episode(1, 2.0, 2.25)
    _predict(controller, first)
    controller.on_episode_end(
        CandidateEpisode(1, 1, "object", 2.0, 2.25, 2.5, EpisodeEndReason.CANDIDATE_CHANGE)
    )

    second = _active_episode(2, 3.0, 3.25)
    suppressed = _predict(controller, second)
    assert suppressed.intent_score is None
    assert suppressed.reason == "feedback_pending_at_episode_start"

    result = controller.button_press(3.5)
    assert result is not None and result.episode_id == 1
    assert controller.button_press(3.6) is None
    assert _predict(controller, second).reason == "feedback_pending_at_episode_start"


def test_episode_end_advances_an_existing_action_feedback_timeout() -> None:
    controller = _controller()
    _predict(controller)
    controller.on_dwell_trigger(DwellTrigger(1, 1, 2.5, 0.5))

    controller.on_episode_end(
        CandidateEpisode(1, 1, "object", 2.0, 2.25, 4.5, EpisodeEndReason.SOURCE_END)
    )

    assert controller.pending_feedback_episode_id is None
    assert len(controller.results) == 1
    assert controller.results[0].action_occurred is True
    assert controller.results[0].feedback_pressed is False


class _InstrumentedLearners:
    def __init__(self, g_probability: float = 0.8, e_probability: float = 0.2) -> None:
        self.events: list[str] = []
        self.labels: list[int] = []
        self.g_probability = g_probability
        self.e_probability = e_probability
        self.config = ModelConfig()

    def predict_pair(self, g_features, e_features) -> FrozenPredictions:
        self.events.append("predict_pair")
        return FrozenPredictions(self.g_probability, self.e_probability)

    def score_pair(self, frozen: FrozenPredictions, common_label: int) -> ScoredPredictions:
        self.events.append("score_pair")
        g_label = int(frozen.g_probability >= 0.5)
        e_label = int(frozen.e_probability >= 0.5)
        return ScoredPredictions(
            frozen,
            common_label,
            g_label,
            e_label,
            g_label == common_label,
            e_label == common_label,
        )

    def learn_scored(self, scored: ScoredPredictions, g_features, e_features) -> None:
        self.events.append("learn_scored")
        self.labels.append(scored.common_label)


def _controller_with_fake(
    fake: _InstrumentedLearners, *, condition: Condition = Condition.G
) -> ExperimentController:
    state = _state()
    state.learners = fake  # explicit injection to instrument the scientific call order
    return _controller(state, condition=condition)


def test_predict_then_score_pair_then_learn_pair_order_is_mechanical() -> None:
    fake = _InstrumentedLearners()
    controller = _controller_with_fake(fake)
    _predict(controller)
    controller.on_dwell_trigger(DwellTrigger(1, 1, 2.5, 0.5))
    result = controller.advance_time(4.0)

    assert result is not None
    assert fake.events == ["predict_pair", "score_pair", "learn_scored"]
    assert fake.labels == [1]


def test_shadow_probability_cannot_change_active_control_score() -> None:
    first = _predict(_controller_with_fake(_InstrumentedLearners(0.8, 0.1)))
    second = _predict(_controller_with_fake(_InstrumentedLearners(0.8, 0.99)))
    assert first.intent_score == pytest.approx(0.8)
    assert second.intent_score == pytest.approx(0.8)

    e_active = _predict(
        _controller_with_fake(_InstrumentedLearners(0.01, 0.3), condition=Condition.E)
    )
    assert e_active.intent_score == pytest.approx(0.3)


def test_prediction_waits_for_full_history_and_eeg_request_never_exceeds_cutoff() -> None:
    controller = _controller()
    source = _EEGSource()
    early = _active_episode(1, 0.0, 0.25)
    waiting = _predict(controller, early, source)
    assert waiting.reason == "waiting_for_prediction_cutoff"
    assert source.calls == []

    eligible = _active_episode(1, 0.0, 1.0)
    decision = _predict(controller, eligible, source)
    assert decision.record is not None
    assert source.calls == [(0.0, 1.0)]
    assert decision.record.eeg_window_end == decision.record.cutoff_timestamp

    with pytest.raises(ValueError, match="requested cutoff"):
        _predict(_controller(), source=_FutureEEGSource())


@pytest.mark.parametrize("quality", [QualityState.UNAVAILABLE, QualityState.REJECTED])
def test_unusable_eeg_skips_both_models_once_without_retry(quality: QualityState) -> None:
    state = _state()
    controller = _controller(state)
    source = _EEGSource(quality)
    first = _predict(controller, source=source)
    second = _predict(controller, source=source)

    assert first.intent_score is None
    assert first.record is not None and first.record.g_probability is None
    assert first.record.e_probability is None
    assert second.record == first.record
    assert len(source.calls) == 1
    assert state.learners.g_model.training_count == 0
    assert state.learners.e_model.training_count == 0


def test_action_before_prediction_is_unscored_and_never_retried() -> None:
    state = _state()
    controller = _controller(state)
    source = _EEGSource()
    controller.on_dwell_trigger(DwellTrigger(1, 1, 2.25, 1.0))
    decision = _predict(controller, source=source)
    assert decision.intent_score is None
    assert decision.reason == "action_before_prediction"
    assert source.calls == []
    assert state.learners.g_model.training_count == 0


def test_participant_updates_are_isolated() -> None:
    first = _state("P001")
    second = _state("P002")
    controller = _controller(first)
    _predict(controller)
    controller.on_dwell_trigger(DwellTrigger(1, 1, 2.5, 0.5))
    controller.advance_time(4.0)

    assert first.learners.g_model.training_count == 1
    assert first.learners.e_model.training_count == 1
    assert second.learners.g_model.training_count == 0
    assert second.learners.e_model.training_count == 0
    neutral = {name: 0.0 for name in GAZE_FEATURE_NAMES}
    assert second.learners.g_model.predict(neutral) == pytest.approx(0.5)


def test_checkpoint_resume_preserves_models_counts_and_schedule(tmp_path: Path) -> None:
    state = _state()
    assert state.schedule.allocate_next() == (0, Condition.G)
    checkpoint = tmp_path / "P001.pkl"
    controller = _controller(state, checkpoint_path=checkpoint)
    _predict(controller)
    controller.on_dwell_trigger(DwellTrigger(1, 1, 2.5, 0.5))
    controller.advance_time(4.0)

    loaded = load_participant_checkpoint(
        checkpoint,
        expected_participant_id="P001",
        expected_participant_sequence_index=0,
        expected_model_config=ModelConfig(),
    )
    gaze = {name: 0.1 for name in GAZE_FEATURE_NAMES}
    eeg = {name: 0.1 for name in E_FEATURE_NAMES}
    assert loaded.learners.g_model.training_count == 1
    assert loaded.learners.e_model.training_count == 1
    assert loaded.schedule.next_session_index == 1
    assert loaded.learners.g_model.predict(gaze) == pytest.approx(state.learners.g_model.predict(gaze))
    assert loaded.learners.e_model.predict(eeg) == pytest.approx(state.learners.e_model.predict(eeg))
    with pytest.raises(ValueError, match="participant"):
        load_participant_checkpoint(
            checkpoint,
            expected_participant_id="P999",
            expected_participant_sequence_index=0,
            expected_model_config=ModelConfig(),
        )


def test_counterbalancing_is_deterministic_abab_baba() -> None:
    first = SessionSchedule(0)
    second = SessionSchedule(1)
    assert [first.allocate_next()[1] for _ in range(4)] == [
        Condition.G,
        Condition.E,
        Condition.G,
        Condition.E,
    ]
    assert [second.allocate_next()[1] for _ in range(4)] == [
        Condition.E,
        Condition.G,
        Condition.E,
        Condition.G,
    ]


def test_controlled_intention_is_evaluation_only_and_feedback_label_trains() -> None:
    fake = _InstrumentedLearners()
    controller = _controller_with_fake(fake)
    _predict(controller, instructed_intention=0)
    controller.on_dwell_trigger(DwellTrigger(1, 1, 2.5, 0.5))
    result = controller.advance_time(4.0)

    assert result is not None
    assert result.instructed_intention == 0
    assert result.common_label == 1
    assert fake.labels == [1]


def test_small_synthetic_run_is_deterministic_and_exercises_all_paths(tmp_path: Path) -> None:
    default = Path(__file__).resolve().parents[1] / "configs" / "experiment_learning.yaml"
    first_config = load_resolved_config(default)
    first_config["output_root"] = str(tmp_path / "first")
    first_config["episodes"] = 40
    first_config["sessions"] = 4
    first_run = run_synthetic_experiment(first_config)

    second_config = load_resolved_config(default)
    second_config["output_root"] = str(tmp_path / "second")
    second_config["episodes"] = 40
    second_config["sessions"] = 4
    second_run = run_synthetic_experiment(second_config)
    first_summary = json.loads((first_run / "summary.json").read_text(encoding="utf-8"))
    second_summary = json.loads((second_run / "summary.json").read_text(encoding="utf-8"))

    assert first_summary == second_summary
    assert first_summary["condition_sessions"] == {"E": 2, "G": 2}
    assert set(first_summary["feedback_cases"]) == set(
        first_config["synthetic"]["feedback_case_cycle"]
    )
    assert first_summary["eeg_quality"]["unavailable"] > 0
    assert first_summary["eeg_quality"]["rejected"] > 0
    assert first_summary["training_counts"]["G"] == first_summary["training_counts"]["E"]
    assert first_summary["checkpoint_resume_exercised"] is True
