"""Two independent River online intent learners and score-before-update records."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

from river import compose, linear_model, optim, preprocessing

from experiment_learning.features import E_FEATURE_NAMES, GAZE_FEATURE_NAMES


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float = 0.01
    l2: float = 0.0
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, Real)
            or not math.isfinite(float(self.learning_rate))
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if (
            isinstance(self.l2, bool)
            or not isinstance(self.l2, Real)
            or not math.isfinite(float(self.l2))
            or self.l2 < 0.0
        ):
            raise ValueError("l2 must be finite and non-negative")
        if (
            isinstance(self.decision_threshold, bool)
            or not isinstance(self.decision_threshold, Real)
            or not math.isfinite(float(self.decision_threshold))
            or not 0.0 <= self.decision_threshold <= 1.0
        ):
            raise ValueError("decision_threshold must be within [0, 1]")


@dataclass(frozen=True)
class FrozenPredictions:
    g_probability: float
    e_probability: float


@dataclass(frozen=True)
class ScoredPredictions:
    """Both model results that must exist before a learning update is accepted."""

    frozen: FrozenPredictions
    common_label: int
    g_predicted_label: int
    e_predicted_label: int
    g_correct: bool
    e_correct: bool


class OnlineIntentModel:
    """StandardScaler -> logistic regression with participant-local online state."""

    def __init__(self, feature_names: tuple[str, ...], config: ModelConfig) -> None:
        self.feature_names = feature_names
        self.config = config
        self.pipeline = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression(
                optimizer=optim.SGD(config.learning_rate), l2=config.l2
            ),
        )
        self.training_count = 0

    def _validate(self, features: Mapping[str, float]) -> dict[str, float]:
        if tuple(features) != self.feature_names:
            raise ValueError("model feature signature/order differs from checkpoint contract")
        values = {name: float(features[name]) for name in self.feature_names}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("model features must be finite")
        return values

    def predict(self, features: Mapping[str, float]) -> float:
        probabilities = self.pipeline.predict_proba_one(self._validate(features))
        probability = float(probabilities.get(True, 0.5))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("River returned an invalid intent probability")
        return probability

    def learn(self, features: Mapping[str, float], label: int) -> None:
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        self.pipeline.learn_one(self._validate(features), bool(label))
        self.training_count += 1


class ParallelIntentLearners:
    """Own independent G and E learned state for exactly one participant."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.g_model = OnlineIntentModel(GAZE_FEATURE_NAMES, config)
        self.e_model = OnlineIntentModel(E_FEATURE_NAMES, config)

    def predict_pair(
        self, g_features: Mapping[str, float], e_features: Mapping[str, float]
    ) -> FrozenPredictions:
        return FrozenPredictions(
            g_probability=self.g_model.predict(g_features),
            e_probability=self.e_model.predict(e_features),
        )

    def score_pair(self, frozen: FrozenPredictions, common_label: int) -> ScoredPredictions:
        if isinstance(common_label, bool) or common_label not in (0, 1):
            raise ValueError("common_label must be 0 or 1")
        threshold = self.config.decision_threshold
        g_label = int(frozen.g_probability >= threshold)
        e_label = int(frozen.e_probability >= threshold)
        return ScoredPredictions(
            frozen=frozen,
            common_label=common_label,
            g_predicted_label=g_label,
            e_predicted_label=e_label,
            g_correct=g_label == common_label,
            e_correct=e_label == common_label,
        )

    def learn_scored(
        self,
        scored: ScoredPredictions,
        g_features: Mapping[str, float],
        e_features: Mapping[str, float],
    ) -> None:
        """Require paired scoring to be materialized before either update."""

        self.g_model.learn(g_features, scored.common_label)
        self.e_model.learn(e_features, scored.common_label)


if __name__ == "__main__":
    config = ModelConfig()
    learners = ParallelIntentLearners(config)
    gaze = {name: 0.0 for name in GAZE_FEATURE_NAMES}
    eeg = {name: 0.0 for name in E_FEATURE_NAMES}
    frozen = learners.predict_pair(gaze, eeg)
    scored = learners.score_pair(frozen, 1)
    learners.learn_scored(scored, gaze, eeg)
    print(scored)
