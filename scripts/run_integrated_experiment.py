"""Run one complete pre-hardware gaze + EEG + paired-learning experiment session."""

import argparse
from pathlib import Path

from foundations.config import load_resolved_config
from integration.workflow import run_integrated_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "integration.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="partial YAML configuration overriding configs/integration.yaml",
    )
    args = parser.parse_args()
    try:
        config = load_resolved_config(DEFAULT_CONFIG, args.config)
        run_directory = run_integrated_experiment(config)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(run_directory)
    summary = run_directory / "analysis_summary.json"
    if summary.is_file():
        print(summary.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
