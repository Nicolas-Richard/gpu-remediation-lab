"""Command-line interface for the AWS DCGM injection scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import DCGMInjectionError
from .dcgm_injection import DCGMInjectionScenario, SUPPORTED_XIDS, emit


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "cluster"
    / "aws"
    / "manifests"
    / "remediation-canary.yaml"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m tests.recovery.aws_dcgm",
        description="Prove real DCGM XID detection and remediation on EKS",
    )
    parser.add_argument("--context", default="gpu-orch-aws")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--xid", type=int, choices=SUPPORTED_XIDS, default=79)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--minimum-gpu-nodes", type=int, default=4)
    parser.add_argument("--keep-canary", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")
    try:
        DCGMInjectionScenario(
            args.context,
            args.manifest,
            xid=args.xid,
            timeout=args.timeout,
            minimum_gpu_nodes=args.minimum_gpu_nodes,
            keep_canary=args.keep_canary,
        ).run()
    except DCGMInjectionError as error:
        emit("FAIL", f"AWS DCGM injection failed: {error}", output=sys.stderr)
        raise SystemExit(1) from error
