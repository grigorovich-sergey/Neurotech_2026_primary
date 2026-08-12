"""Persistence helpers for successful completed-session trainer inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from experiment_learning.artifacts import immutable_write_json, load_json_object
from experiment_learning.contracts import CompletedSession


def save_completed_session(path: str | Path, session: CompletedSession) -> str:
    return immutable_write_json(path, session.to_payload())


def load_completed_session(path: str | Path) -> tuple[CompletedSession, str]:
    source = Path(path)
    content = source.read_bytes()
    session = CompletedSession.from_payload(load_json_object(source))
    return session, hashlib.sha256(content).hexdigest()


if __name__ == "__main__":
    print("Completed-session artifacts are created by ExperimentController.completed_session().")
