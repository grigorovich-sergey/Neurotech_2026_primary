"""Immutable provenance for active/shadow condition selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any, Mapping

from experiment_learning.artifacts import immutable_write_json, load_json_object
from experiment_learning.contracts import Condition
from experiment_learning.schedule import ScheduleBinding


MODEL_ASSIGNMENT_SCHEMA = "experiment_model_assignment_v1"
CLI_SELECTION_ID = "cli-manual-v1"
CLI_SELECTION_SHA256 = hashlib.sha256(CLI_SELECTION_ID.encode("utf-8")).hexdigest()


class ModelSelectionSource(str, Enum):
    CLI = "cli"
    CSV = "csv"


def shadow_condition(active: Condition) -> Condition:
    if not isinstance(active, Condition):
        raise TypeError("active must be a Condition")
    return Condition.E if active is Condition.G else Condition.G


@dataclass(frozen=True)
class ModelAssignment:
    """One logical session's immutable condition-selection provenance."""

    participant_id: str
    session_number: int
    source: ModelSelectionSource
    active_condition: Condition
    shadow_condition: Condition
    binding_id: str
    binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.participant_id, str) or not self.participant_id:
            raise ValueError("participant_id must be a non-empty string")
        if (
            isinstance(self.session_number, bool)
            or not isinstance(self.session_number, int)
            or self.session_number <= 0
        ):
            raise ValueError("session_number must be a positive integer")
        if not isinstance(self.source, ModelSelectionSource):
            raise TypeError("source must be a ModelSelectionSource")
        if not isinstance(self.active_condition, Condition) or not isinstance(
            self.shadow_condition, Condition
        ):
            raise TypeError("active/shadow conditions must be Condition values")
        if self.shadow_condition is not shadow_condition(self.active_condition):
            raise ValueError("shadow condition must be the complement of active condition")
        if not isinstance(self.binding_id, str) or not self.binding_id:
            raise ValueError("binding_id must be a non-empty string")
        if (
            not isinstance(self.binding_sha256, str)
            or len(self.binding_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.binding_sha256)
        ):
            raise ValueError("binding_sha256 must be a lowercase SHA-256 digest")

    @property
    def schedule_binding(self) -> ScheduleBinding:
        """Compatibility binding consumed by the frozen-policy v1 contracts."""

        return ScheduleBinding(self.binding_id, self.binding_sha256)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_ASSIGNMENT_SCHEMA,
            "participant_id": self.participant_id,
            "session_number": self.session_number,
            "selection_source": self.source.value,
            "active_condition": self.active_condition.value,
            "shadow_condition": self.shadow_condition.value,
            "binding": {
                "id": self.binding_id,
                "sha256": self.binding_sha256,
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ModelAssignment":
        if payload.get("schema") != MODEL_ASSIGNMENT_SCHEMA:
            raise ValueError("model-assignment schema is incompatible")
        binding = payload.get("binding")
        if not isinstance(binding, Mapping):
            raise ValueError("model-assignment binding must be a mapping")
        return cls(
            participant_id=payload["participant_id"],
            session_number=payload["session_number"],
            source=ModelSelectionSource(payload["selection_source"]),
            active_condition=Condition(payload["active_condition"]),
            shadow_condition=Condition(payload["shadow_condition"]),
            binding_id=binding["id"],
            binding_sha256=binding["sha256"],
        )


def cli_model_assignment(
    participant_id: str, session_number: int, active_condition: Condition
) -> ModelAssignment:
    return ModelAssignment(
        participant_id=participant_id,
        session_number=session_number,
        source=ModelSelectionSource.CLI,
        active_condition=active_condition,
        shadow_condition=shadow_condition(active_condition),
        binding_id=CLI_SELECTION_ID,
        binding_sha256=CLI_SELECTION_SHA256,
    )


def csv_model_assignment(
    participant_id: str,
    session_number: int,
    active_condition: Condition,
    binding: ScheduleBinding,
) -> ModelAssignment:
    if not isinstance(binding, ScheduleBinding):
        raise TypeError("binding must be a ScheduleBinding")
    return ModelAssignment(
        participant_id=participant_id,
        session_number=session_number,
        source=ModelSelectionSource.CSV,
        active_condition=active_condition,
        shadow_condition=shadow_condition(active_condition),
        binding_id=binding.sequence_id,
        binding_sha256=binding.csv_sha256,
    )


def save_model_assignment(path: str | Path, assignment: ModelAssignment) -> str:
    return immutable_write_json(path, assignment.to_payload())


def load_model_assignment(path: str | Path) -> ModelAssignment:
    return ModelAssignment.from_payload(load_json_object(path))


if __name__ == "__main__":
    print(cli_model_assignment("P001", 1, Condition.G).to_payload())
