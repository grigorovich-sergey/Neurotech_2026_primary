"""Run the virtual-glasses recording or replay workflow."""

import argparse
from pathlib import Path

from foundations.config import load_resolved_config
from foundations.workflow import run_virtual_glasses


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "virtual_glasses.yaml"


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
        run_directory = run_virtual_glasses(config)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(run_directory)


if __name__ == "__main__":
    main()
