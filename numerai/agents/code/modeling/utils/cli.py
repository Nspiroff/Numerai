from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_CONFIG_PATH
from .pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the base Numerai model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the config file (.py or .json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory for results and predictions.",
    )
    parser.add_argument(
        "--scout-component",
        default=None,
        help="Scout component authorized for this one training run.",
    )
    parser.add_argument(
        "--scout-pre-run-receipt",
        type=Path,
        default=None,
        help="Finalized just-in-time Scout absence receipt.",
    )
    parser.add_argument(
        "--scout-pre-run-receipt-sha256",
        default=None,
        help="Exact SHA-256 of the Scout pre-run receipt.",
    )
    parser.add_argument(
        "--confirmation-component",
        default=None,
        help="Confirmation component authorized for this one training run.",
    )
    parser.add_argument(
        "--confirmation-pre-run-receipt",
        type=Path,
        default=None,
        help="Finalized just-in-time confirmation absence receipt.",
    )
    parser.add_argument(
        "--confirmation-pre-run-receipt-sha256",
        default=None,
        help="Exact SHA-256 of the confirmation pre-run receipt.",
    )
    parser.add_argument(
        "--confirmation-pretraining-receipt",
        type=Path,
        default=None,
        help="Finalized confirmation pretraining authority receipt.",
    )
    parser.add_argument(
        "--confirmation-pretraining-receipt-sha256",
        default=None,
        help="Exact SHA-256 of the confirmation pretraining receipt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        args.config,
        args.output_dir,
        scout_component=args.scout_component,
        scout_pre_run_receipt=args.scout_pre_run_receipt,
        scout_pre_run_receipt_sha256=args.scout_pre_run_receipt_sha256,
        confirmation_component=args.confirmation_component,
        confirmation_pre_run_receipt=args.confirmation_pre_run_receipt,
        confirmation_pre_run_receipt_sha256=(
            args.confirmation_pre_run_receipt_sha256
        ),
        confirmation_pretraining_receipt=args.confirmation_pretraining_receipt,
        confirmation_pretraining_receipt_sha256=(
            args.confirmation_pretraining_receipt_sha256
        ),
    )
