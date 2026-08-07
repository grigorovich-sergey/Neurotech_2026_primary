"""YAML configuration loading and strict partial overrides."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return data


def _merge_known(
    defaults: dict[str, Any], override: dict[str, Any], prefix: str = ""
) -> dict[str, Any]:
    result = deepcopy(defaults)
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in defaults:
            raise ValueError(f"unknown configuration key: {path}")
        default_value = defaults[key]
        if isinstance(default_value, dict) and isinstance(value, dict):
            result[key] = _merge_known(default_value, value, path)
        else:
            result[key] = deepcopy(value)
    return result


def load_resolved_config(
    default_path: str | Path, override_path: str | Path | None = None
) -> dict[str, Any]:
    """Load a complete YAML config and apply an optional strict partial override."""

    defaults = _load_yaml(default_path)
    if override_path is None:
        return defaults
    return _merge_known(defaults, _load_yaml(override_path))


if __name__ == "__main__":
    print(_merge_known({"a": {"b": 1}, "items": [1]}, {"a": {"b": 2}}))
