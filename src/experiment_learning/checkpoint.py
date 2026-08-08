"""Trusted-local participant checkpointing with strict compatibility checks."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

import river

from experiment_learning.features import E_FEATURE_NAMES, GAZE_FEATURE_NAMES
from experiment_learning.models import ModelConfig, ParallelIntentLearners
from experiment_learning.schedule import SessionSchedule


CHECKPOINT_SCHEMA = "experiment_learning_checkpoint_v1"


@dataclass
class ParticipantState:
    """All learned and scheduling state that must remain participant-local."""

    participant_id: str
    model_config: ModelConfig
    learners: ParallelIntentLearners
    schedule: SessionSchedule

    @classmethod
    def create(
        cls,
        *,
        participant_id: str,
        participant_sequence_index: int,
        model_config: ModelConfig,
    ) -> "ParticipantState":
        if not isinstance(participant_id, str) or not participant_id:
            raise ValueError("participant_id must be a non-empty pseudonymous identifier")
        return cls(
            participant_id=participant_id,
            model_config=model_config,
            learners=ParallelIntentLearners(model_config),
            schedule=SessionSchedule(participant_sequence_index),
        )


def _payload(state: ParticipantState) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "participant_id": state.participant_id,
        "feature_signature": {
            "G": GAZE_FEATURE_NAMES,
            "E": E_FEATURE_NAMES,
        },
        "model_config": asdict(state.model_config),
        "river_version": river.__version__,
        "training_counts": {
            "G": state.learners.g_model.training_count,
            "E": state.learners.e_model.training_count,
        },
        "learners": state.learners,
        "schedule": state.schedule,
    }


def save_participant_checkpoint(path: str | Path, state: ParticipantState) -> None:
    """Atomically replace one trusted-local pickle checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(_payload(state), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_participant_checkpoint(
    path: str | Path,
    *,
    expected_participant_id: str,
    expected_participant_sequence_index: int,
    expected_model_config: ModelConfig,
) -> ParticipantState:
    """Load only an exact schema/participant/feature/model/River-compatible checkpoint."""

    # Pickle is intentionally trusted-local. Never load a checkpoint from an untrusted source.
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is incompatible")
    if payload.get("participant_id") != expected_participant_id:
        raise ValueError("checkpoint participant does not match requested participant")
    expected_signature = {"G": GAZE_FEATURE_NAMES, "E": E_FEATURE_NAMES}
    if payload.get("feature_signature") != expected_signature:
        raise ValueError("checkpoint feature signature is incompatible")
    if payload.get("model_config") != asdict(expected_model_config):
        raise ValueError("checkpoint model configuration is incompatible")
    if payload.get("river_version") != river.__version__:
        raise ValueError("checkpoint River version is incompatible")

    learners = payload.get("learners")
    schedule = payload.get("schedule")
    counts = payload.get("training_counts")
    if not isinstance(learners, ParallelIntentLearners):
        raise ValueError("checkpoint learner state is invalid")
    if not isinstance(schedule, SessionSchedule):
        raise ValueError("checkpoint schedule state is invalid")
    if schedule.participant_sequence_index != expected_participant_sequence_index:
        raise ValueError("checkpoint participant sequence index is incompatible")
    actual_counts = {
        "G": learners.g_model.training_count,
        "E": learners.e_model.training_count,
    }
    if counts != actual_counts:
        raise ValueError("checkpoint training counters are inconsistent")
    return ParticipantState(
        participant_id=expected_participant_id,
        model_config=expected_model_config,
        learners=learners,
        schedule=schedule,
    )


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        config = ModelConfig()
        state = ParticipantState.create(
            participant_id="P001", participant_sequence_index=0, model_config=config
        )
        path = Path(directory) / "participant.pkl"
        save_participant_checkpoint(path, state)
        print(
            load_participant_checkpoint(
                path,
                expected_participant_id="P001",
                expected_participant_sequence_index=0,
                expected_model_config=config,
            ).participant_id
        )
