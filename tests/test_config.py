from pathlib import Path

import pytest

from foundations.config import load_resolved_config


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_partial_config_recursively_merges_and_replaces_lists(tmp_path: Path) -> None:
    default = tmp_path / "default.yaml"
    override = tmp_path / "override.yaml"
    _write(default, "outer:\n  a: 1\n  b: 2\nitems: [1, 2]\n")
    _write(override, "outer:\n  b: 9\nitems: [3]\n")

    resolved = load_resolved_config(default, override)

    assert resolved == {"outer": {"a": 1, "b": 9}, "items": [3]}


def test_unknown_override_key_is_rejected(tmp_path: Path) -> None:
    default = tmp_path / "default.yaml"
    override = tmp_path / "override.yaml"
    _write(default, "outer:\n  known: 1\n")
    _write(override, "outer:\n  typo: 2\n")

    with pytest.raises(ValueError, match=r"unknown configuration key: outer\.typo"):
        load_resolved_config(default, override)
