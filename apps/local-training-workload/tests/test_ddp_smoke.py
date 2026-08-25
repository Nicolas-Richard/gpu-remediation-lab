#!/usr/bin/env python3
"""Process-level smoke test for the local workload's DDP checkpoints."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


TRAIN_SCRIPT = Path("/app/train.py")


class DDPCheckpointSmokeTest(unittest.TestCase):
    def run_command(self, command: list[str]) -> str:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        output = completed.stdout + completed.stderr
        if completed.returncode != 0:
            self.fail(
                f"command failed with exit code {completed.returncode}: "
                f"{' '.join(command)}\n{output}"
            )
        return output

    @staticmethod
    def training_args(checkpoint_dir: Path, max_steps: int) -> list[str]:
        return [
            str(TRAIN_SCRIPT),
            f"--checkpoint-dir={checkpoint_dir}",
            f"--max-steps={max_steps}",
            "--checkpoint-every=1",
            "--sleep-seconds=0",
        ]

    def test_single_process_and_ddp_checkpoints_are_compatible(self) -> None:
        torchrun = shutil.which("torchrun")
        self.assertIsNotNone(torchrun, "torchrun is not installed")
        self.assertTrue(TRAIN_SCRIPT.is_file(), f"missing {TRAIN_SCRIPT}")

        with tempfile.TemporaryDirectory(prefix="ddp-checkpoints-") as directory:
            checkpoint_dir = Path(directory)

            single_output = self.run_command(
                [sys.executable, *self.training_args(checkpoint_dir, 2)]
            )
            self.assertIn("checkpoint_saved=step_000002.pt", single_output)

            ddp_output = self.run_command(
                [
                    torchrun,
                    "--standalone",
                    "--nproc-per-node=2",
                    *self.training_args(checkpoint_dir, 4),
                ]
            )
            for rank in (0, 1):
                self.assertIn(f"step=2 rank={rank}", ddp_output)
            self.assertIn("checkpoint_saved=step_000004.pt", ddp_output)

            restarted_output = self.run_command(
                [
                    torchrun,
                    "--standalone",
                    "--nproc-per-node=2",
                    *self.training_args(checkpoint_dir, 5),
                ]
            )
            for rank in (0, 1):
                self.assertIn(f"step=4 rank={rank}", restarted_output)

            payload = torch.load(
                checkpoint_dir / "latest.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(5, payload["step"])
            self.assertFalse(
                any(key.startswith("module.") for key in payload["model"]),
                "DDP wrapper leaked into checkpoint model keys",
            )
            self.assertEqual([], list(checkpoint_dir.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
