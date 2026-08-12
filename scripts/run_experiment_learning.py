"""Run frozen Instance 4 policies and deterministic between-session retraining."""

import argparse
from pathlib import Path

from foundations.config import load_resolved_config
from experiment_learning.synthetic import run_synthetic_experiment


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiment_learning.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="partial YAML configuration overriding the project default",
    )
    args = parser.parse_args()
    try:
        config = load_resolved_config(DEFAULT_CONFIG, args.config)
        run_directory = run_synthetic_experiment(config)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(run_directory)
    print((run_directory / "summary.json").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
