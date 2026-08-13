"""Secure local credential loading for live Guardian acquisition."""

from collections.abc import Mapping
import os
from pathlib import Path
import stat


def _validated_token(value: str, *, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Guardian API token from {source} is empty")
    lines = value.splitlines()
    if len(lines) != 1 or not lines[0] or lines[0] != lines[0].strip():
        raise RuntimeError(
            f"Guardian API token from {source} must be exactly one non-empty line "
            "without surrounding whitespace"
        )
    return lines[0]


def load_guardian_api_token(
    *,
    environment_variable: str,
    token_file: str | Path,
    base_directory: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Load a Guardian token from the environment, then an ignored local file.

    The token is returned only to the live adapter. It is never added to resolved
    configuration or diagnostic output. On POSIX, a file fallback must not grant
    any group or other permissions.
    """

    if not isinstance(environment_variable, str) or not environment_variable:
        raise ValueError("environment_variable must be a non-empty string")
    environment = os.environ if environ is None else environ
    environment_value = environment.get(environment_variable)
    if environment_value is not None:
        return _validated_token(
            environment_value,
            source=f"environment variable {environment_variable}",
        )

    if not isinstance(token_file, (str, Path)) or not str(token_file):
        raise ValueError("token_file must be a non-empty path")
    path = Path(token_file)
    if not path.is_absolute() and base_directory is not None:
        path = Path(base_directory) / path
    if not path.is_file():
        raise RuntimeError(
            "Guardian API token is unavailable: set environment variable "
            f"{environment_variable} or create {path}"
        )
    if os.name != "nt":
        permissions = stat.S_IMODE(path.stat().st_mode)
        if permissions & 0o077:
            raise RuntimeError(
                f"Guardian API token file {path} is accessible by group/other "
                f"(mode {permissions:04o}); run: chmod 600 {path}"
            )
    return _validated_token(
        path.read_text(encoding="utf-8"),
        source=f"file {path}",
    )
