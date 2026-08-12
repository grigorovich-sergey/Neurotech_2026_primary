"""Run live MindLink practice with optional monitor-only Guardian EEG."""

import argparse
from pathlib import Path

from foundations.config import load_resolved_config
from practice_session import run_practice_session


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "practice_session.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="partial YAML configuration overriding the practice defaults",
    )
    eeg_group = parser.add_mutually_exclusive_group()
    eeg_group.add_argument(
        "--with-eeg",
        action="store_true",
        help="enable live Guardian acquisition and quality monitoring",
    )
    eeg_group.add_argument(
        "--without-eeg",
        action="store_true",
        help="force Guardian acquisition off",
    )
    args = parser.parse_args()
    try:
        config = load_resolved_config(DEFAULT_CONFIG, args.config)
        if args.with_eeg:
            config["eeg"]["enabled"] = True
        elif args.without_eeg:
            config["eeg"]["enabled"] = False
        run_directory = run_practice_session(config)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(run_directory)


if __name__ == "__main__":
    main()
