from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.evaluate import cli_main as evaluate_cli_main
from src.train import cli_main as train_cli_main


PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "src" / "app.py"
STREAMLIT_LAUNCHER_PATH = PROJECT_ROOT / "run_streamlit_app.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Potato leaf hybrid project command runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model.")
    train_parser.add_argument("--model", choices=["cnn", "hybrid", "transfer"], default="cnn")
    train_parser.add_argument("--dataset-dir", type=str, default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--learning-rate", type=float, default=None)
    train_parser.add_argument("--skip-test-eval", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    eval_parser.add_argument("--model", choices=["cnn", "hybrid", "transfer"], default="cnn")
    eval_parser.add_argument("--dataset-dir", type=str, default=None)
    eval_parser.add_argument("--batch-size", type=int, default=None)
    eval_parser.add_argument("--weights", type=str, default=None)

    subparsers.add_parser("app", help="Launch the Streamlit demo app.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        train_cli_main(args)
        return

    if args.command == "evaluate":
        evaluate_cli_main(args)
        return

    subprocess.run([sys.executable, str(STREAMLIT_LAUNCHER_PATH)], check=True)


if __name__ == "__main__":
    main()
