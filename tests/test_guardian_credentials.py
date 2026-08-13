import os
from pathlib import Path

import pytest

from eeg_pipeline.credentials import load_guardian_api_token


def test_guardian_token_environment_precedes_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "not-created"

    token = load_guardian_api_token(
        environment_variable="IDUN_API_TOKEN",
        token_file=missing_file,
        environ={"IDUN_API_TOKEN": "environment-secret"},
    )

    assert token == "environment-secret"


def test_guardian_token_loads_protected_single_line_file(tmp_path: Path) -> None:
    token_file = tmp_path / "idun_api_token"
    token_file.write_text("file-secret\n", encoding="utf-8")
    token_file.chmod(0o600)

    token = load_guardian_api_token(
        environment_variable="IDUN_API_TOKEN",
        token_file="idun_api_token",
        base_directory=tmp_path,
        environ={},
    )

    assert token == "file-secret"


def test_guardian_token_rejects_multiline_without_leaking_value(
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "idun_api_token"
    token_file.write_text("first-secret\nsecond-secret", encoding="utf-8")
    token_file.chmod(0o600)

    with pytest.raises(RuntimeError) as raised:
        load_guardian_api_token(
            environment_variable="IDUN_API_TOKEN",
            token_file=token_file,
            environ={},
        )

    message = str(raised.value)
    assert "exactly one non-empty line" in message
    assert "first-secret" not in message
    assert "second-secret" not in message


def test_guardian_token_rejects_broad_posix_permissions(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX mode validation is not available on Windows")
    token_file = tmp_path / "idun_api_token"
    token_file.write_text("file-secret", encoding="utf-8")
    token_file.chmod(0o644)

    with pytest.raises(RuntimeError, match="chmod 600"):
        load_guardian_api_token(
            environment_variable="IDUN_API_TOKEN",
            token_file=token_file,
            environ={},
        )
