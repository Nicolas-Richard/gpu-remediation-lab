"""Command-line entrypoint for the fault-recovery harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import HarnessError
from .harness import FaultRecoveryHarness


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "apps"
    / "local-training-workload"
    / "manifests"
    / "demo-job.yaml"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m tests.recovery.kind",
        description="Exercise GPU node-health and recovery behavior",
    )
    parser.add_argument("--context", default="kind-gpu-orch")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--after-step", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--keep-fault", action="store_true")
    parser.add_argument(
        "--health-source",
        choices=("annotation", "dcgm-simulator"),
        default="annotation",
        help="Fault observation source used by the lifecycle",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")

    try:
        FaultRecoveryHarness(
            args.context,
            args.manifest,
            after_step=args.after_step,
            timeout=args.timeout,
            keep_fault=args.keep_fault,
            health_source=args.health_source,
        ).run()
    except HarnessError as error:
        print(f"fault-recovery lifecycle failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
