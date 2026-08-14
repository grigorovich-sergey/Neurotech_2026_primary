"""Run one live participant-specific NeuroTech experimental session."""

import argparse
import json
from pathlib import Path

from foundations.config import load_resolved_config
from integration.live_workflow import run_live_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="partial YAML configuration overriding configs/experiment.yaml",
    )
    parser.add_argument(
        "--subject-id",
        required=True,
        help="pseudonymous subject identifier used for recording and policy history",
    )
    parser.add_argument(
        "--active-model",
        required=True,
        choices=("G", "E"),
        help="model controlling visible actions; the other model is shadow",
    )
    args = parser.parse_args()
    try:
        config = load_resolved_config(DEFAULT_CONFIG, args.config)
        config["runtime"]["subject_id"] = args.subject_id
        config["runtime"]["active_model"] = args.active_model
        run_directory = run_live_experiment(config)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(run_directory)
    summary_path = run_directory / "attempt_summary.json"
    if summary_path.is_file():
        print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
