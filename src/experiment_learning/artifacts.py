"""Canonical, immutable JSON artifact helpers for Instance 4."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize one artifact deterministically for hashing and persistence."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def artifact_digest(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def immutable_write_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Atomically create an immutable artifact and return its SHA-256 digest.

    Repeating an identical write is idempotent. Reusing a path for different
    content fails loudly instead of replacing scientific provenance.
    """

    destination = Path(path)
    content = canonical_json_bytes(payload)
    digest = sha256_bytes(content)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_bytes()
        if existing != content:
            raise FileExistsError(
                f"immutable artifact already exists with different content: {destination}"
            )
        return digest

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # The destination was checked above. os.link gives create-if-absent
        # semantics so a concurrent different writer cannot be overwritten.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise FileExistsError(
                    f"immutable artifact already exists with different content: {destination}"
                )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return digest


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() in {".pkl", ".pickle"}:
        raise ValueError("legacy River pickle checkpoints are not supported")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("artifact root must be a JSON object")
    return payload


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "artifact.json"
        print(immutable_write_json(path, {"schema": "smoke_v1", "value": 1}))
