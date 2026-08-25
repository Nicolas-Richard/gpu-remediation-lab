"""Command-line interface for AWS distributed-training recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from . import TrainingRecoveryError
from .harness import AWSTrainingRecoveryHarness


DEFAULT_JOB_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "cluster"
    / "aws"
    / "manifests"
    / "distributed-training-job.yaml"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m tests.recovery.aws_training",
        description="Prove EFS-backed distributed CUDA training recovery on EKS",
    )
    parser.add_argument("--context", default="gpu-orch-aws")
    parser.add_argument("--job-template", type=Path, default=DEFAULT_JOB_TEMPLATE)
    parser.add_argument("--image", required=True)
    parser.add_argument("--after-step", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--run-name")
    parser.add_argument("--keep-job", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show detailed lifecycle events that are normally JSONL-only",
    )
    parser.add_argument(
        "--events-jsonl",
        type=Path,
        help="also write the structured event stream as JSON Lines",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.job_template.is_file():
        raise SystemExit(f"job template not found: {args.job_template}")
    try:
        AWSTrainingRecoveryHarness(
            args.context,
            args.job_template,
            args.image,
            after_step=args.after_step,
            timeout=args.timeout,
            run_name=args.run_name,
            keep_job=args.keep_job,
            events_jsonl=args.events_jsonl,
            verbose=args.verbose,
        ).run()
    except TrainingRecoveryError as error:
        raise SystemExit(1) from error
