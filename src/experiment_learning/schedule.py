"""Digest-protected, predetermined G/E condition schedule resolution."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Mapping

from experiment_learning.contracts import Condition


SCHEDULE_FIELDS = ("sequence_id", "session_number", "active_condition")


@dataclass(frozen=True)
class ConditionSchedule:
    path: Path
    sha256: str
    rows: Mapping[tuple[str, int], Condition]


@dataclass(frozen=True)
class ScheduleBinding:
    sequence_id: str
    csv_sha256: str

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("sequence_id must be non-empty")
        if not self.csv_sha256:
            raise ValueError("csv_sha256 must be non-empty")


@dataclass(frozen=True)
class ScheduledCondition:
    session_number: int
    condition: Condition
    binding: ScheduleBinding


def load_condition_schedule(path: str | Path) -> ConditionSchedule:
    source = Path(path)
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("condition schedule must be UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != SCHEDULE_FIELDS:
        raise ValueError(
            "condition schedule header must be exactly: " + ",".join(SCHEDULE_FIELDS)
        )
    rows: dict[tuple[str, int], Condition] = {}
    by_sequence: dict[str, list[int]] = {}
    for line_number, row in enumerate(reader, start=2):
        sequence_id = (row.get("sequence_id") or "").strip()
        if not sequence_id:
            raise ValueError(f"empty sequence_id on CSV line {line_number}")
        try:
            session_number = int(row["session_number"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid session_number on CSV line {line_number}") from exc
        if session_number <= 0 or str(session_number) != row["session_number"].strip():
            raise ValueError(f"session_number must be a positive integer on CSV line {line_number}")
        try:
            condition = Condition((row.get("active_condition") or "").strip())
        except ValueError as exc:
            raise ValueError(
                f"active_condition must be G or E on CSV line {line_number}"
            ) from exc
        key = (sequence_id, session_number)
        if key in rows:
            raise ValueError(f"duplicate schedule row for {sequence_id} session {session_number}")
        rows[key] = condition
        by_sequence.setdefault(sequence_id, []).append(session_number)
    if not rows:
        raise ValueError("condition schedule cannot be empty")
    for sequence_id, numbers in by_sequence.items():
        ordered = sorted(numbers)
        if ordered != list(range(1, ordered[-1] + 1)):
            raise ValueError(f"sequence {sequence_id!r} must cover sessions 1..N without gaps")
    return ConditionSchedule(source, digest, rows)


def resolve_scheduled_condition(
    schedule: ConditionSchedule,
    sequence_id: str,
    session_number: int,
    persisted_binding: ScheduleBinding | None = None,
) -> ScheduledCondition:
    if not sequence_id:
        raise ValueError("sequence_id must be non-empty")
    if isinstance(session_number, bool) or not isinstance(session_number, int) or session_number <= 0:
        raise ValueError("session_number must be a positive integer")
    binding = ScheduleBinding(sequence_id, schedule.sha256)
    if persisted_binding is not None and persisted_binding != binding:
        raise ValueError("condition schedule differs from the persisted sequence/digest binding")
    try:
        condition = schedule.rows[(sequence_id, session_number)]
    except KeyError as exc:
        raise ValueError(
            f"condition schedule has no row for {sequence_id!r} session {session_number}"
        ) from exc
    return ScheduledCondition(session_number, condition, binding)


if __name__ == "__main__":
    print("Use load_condition_schedule(path) with an approved CSV schedule.")
