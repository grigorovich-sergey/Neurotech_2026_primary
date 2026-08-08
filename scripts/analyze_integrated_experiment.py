"""Regenerate integrated scientific summaries from an existing events.jsonl file."""

import argparse
from pathlib import Path

from foundations.config import load_resolved_config, save_resolved_config
from integration.analysis import generate_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "integration_analysis.yaml"


def _path(value: object, *, name: str, default: Path | None = None) -> Path:
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="partial YAML configuration overriding configs/integration_analysis.yaml",
    )
    args = parser.parse_args()
    try:
        config = load_resolved_config(DEFAULT_CONFIG, args.config)
        events_path = _path(config.get("events_path"), name="events_path")
        output_directory = _path(
            config.get("output_directory"),
            name="output_directory",
            default=events_path.parent,
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, output_directory / "resolved_analysis_config.json")
        summary = generate_analysis(events_path, output_directory)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(output_directory)
    print(summary)


if __name__ == "__main__":
    main()
