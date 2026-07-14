from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE_COMMANDS = {
    "data": [sys.executable, "src/data_clean_pipeline.py"],
    "model": [sys.executable, "src/train_cmm_model.py"],
    "backtest": [sys.executable, "src/model_compare_workflow.py"],
    "style": [sys.executable, "src/style_exposure_workflow.py"],
    "explain": [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "notebooks/02_explain_cmm_improvement.ipynb",
    ],
    "validate": [sys.executable, "src/validate_outputs.py"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CMM research pipeline in dependency order.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_COMMANDS,
        default=list(STAGE_COMMANDS),
        help="Stages to run. The default runs the complete reproducible workflow.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for stage in args.stages:
        command = STAGE_COMMANDS[stage]
        print(f"\n[{stage}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
