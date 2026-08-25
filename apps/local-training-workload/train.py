#!/usr/bin/env python3
"""Offline synthetic DDP smoke fixture with continuous checkpointing.

This intentionally tiny model keeps local kind and process-level CI independent
of GPU access and model downloads. The AWS recovery gate uses the separate
SmolLM training workload and explicitly rejects this fixture.

When a peer disappears, the collective heartbeat fails, the process exits, the
Trainer/JobSet controllers recreate the rank group, and training resumes from
the latest shared checkpoint.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def setup_distributed(rendezvous_watchdog_seconds: int) -> tuple[int, int]:
    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)

    if world_size > 1 and not dist.is_initialized():
        rendezvous_finished = threading.Event()

        def rendezvous_watchdog() -> None:
            if not rendezvous_finished.wait(rendezvous_watchdog_seconds):
                print(
                    f"rendezvous_watchdog_timeout={rendezvous_watchdog_seconds} rank={rank}",
                    flush=True,
                )
                # A TCPStore client can remain blocked beyond the process-group
                # timeout. Exit the container so JobSet can restart the
                # distributed rank group into one rendezvous generation.
                os._exit(1)

        threading.Thread(target=rendezvous_watchdog, daemon=True).start()
        try:
            dist.init_process_group(
                backend="gloo",
                timeout=timedelta(seconds=60),
            )
        finally:
            rendezvous_finished.set()

    return rank, world_size


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(checkpoint_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def atomic_torch_save(payload: object, destination: Path) -> None:
    """Publish a torch checkpoint without exposing a partial destination."""
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_model = model.module if isinstance(model, DDP) else model
    payload = {
        "step": step,
        "model": checkpoint_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    atomic_torch_save(payload, checkpoint_dir / f"step_{step:06d}.pt")
    atomic_torch_save(payload, checkpoint_dir / "latest.pt")


def peer_heartbeat(world_size: int) -> None:
    """Fail fast if a peer disappears so pods can restart and resume."""
    if world_size <= 1:
        return
    if hasattr(dist, "monitored_barrier"):
        dist.monitored_barrier(timeout=timedelta(seconds=30))
    else:
        dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="/checkpoints")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--rendezvous-watchdog-seconds", type=int, default=75)
    args = parser.parse_args()

    rank, world_size = setup_distributed(args.rendezvous_watchdog_seconds)
    checkpoint_dir = Path(args.checkpoint_dir)

    base_model = nn.Linear(8, 1)
    start_step = 0
    payload = None

    ckpt = latest_checkpoint(checkpoint_dir)
    if ckpt is not None:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        base_model.load_state_dict(payload["model"])
        start_step = int(payload["step"])

    model: nn.Module = DDP(base_model) if world_size > 1 else base_model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        print(f"resumed_from={ckpt} step={start_step} rank={rank}", flush=True)

    try:
        peer_heartbeat(world_size)

        for step in range(start_step + 1, args.max_steps + 1):
            x = torch.randn(16, 8)
            y = torch.randn(16, 1)
            loss = ((model(x) - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"step={step} loss={loss.item():.6f}", flush=True)
                if step % args.checkpoint_every == 0:
                    save_checkpoint(checkpoint_dir, step, model, optimizer)
                    print(f"checkpoint_saved=step_{step:06d}.pt", flush=True)

            # Heartbeat every step so a killed worker fails the job quickly.
            peer_heartbeat(world_size)
            time.sleep(args.sleep_seconds)

    except Exception as exc:
        print(f"distributed_error={exc!r} rank={rank}", flush=True)
        raise SystemExit(1) from exc
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
