#!/usr/bin/env python3
"""SmolLM LoRA SFT workload for distributed GPU recovery validation.

The workload deliberately uses ordinary PyTorch DDP instead of hiding recovery
behind a training framework. JobSet recreates the complete rank group after a
worker is evicted; every rank then restores the same atomic EFS checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import tempfile
import threading
from datetime import timedelta
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_MODEL_REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"
DEFAULT_DATASET = "HuggingFaceTB/smoltalk"
DEFAULT_DATASET_CONFIG = "smol-magpie-ultra"
DEFAULT_DATASET_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def setup_distributed(watchdog_seconds: int) -> tuple[int, int, int]:
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)
    if not torch.cuda.is_available():
        raise RuntimeError("SmolLM recovery validation requires a CUDA device")

    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        rendezvous_finished = threading.Event()

        def watchdog() -> None:
            if not rendezvous_finished.wait(watchdog_seconds):
                print(
                    f"rendezvous_watchdog_timeout={watchdog_seconds} rank={rank}",
                    flush=True,
                )
                os._exit(1)

        threading.Thread(target=watchdog, daemon=True).start()
        try:
            # Cold-cache ranks can be separated by several minutes while one
            # process publishes the pinned 6 GB model into the shared HF cache.
            dist.init_process_group("nccl", timeout=timedelta(minutes=15))
        finally:
            rendezvous_finished.set()
    return rank, local_rank, world_size


def peer_heartbeat(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(checkpoint_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state().cpu(),
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def all_rank_rng_states(rank: int, world_size: int) -> dict[int, object]:
    local = capture_rng_state()
    if world_size == 1:
        return {rank: local}
    gathered: list[object | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return {found_rank: state for found_rank, state in enumerate(gathered)}


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    step: int,
    tokens_seen: int,
    examples_seen_per_rank: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    rng_by_rank: dict[int, object],
    run_config: dict[str, object],
) -> None:
    checkpoint_model = model.module if isinstance(model, DDP) else model
    adapter = {
        name: value.detach().cpu()
        for name, value in get_peft_model_state_dict(checkpoint_model).items()
    }
    payload = {
        "format_version": 1,
        "step": step,
        "tokens_seen": tokens_seen,
        "examples_seen_per_rank": examples_seen_per_rank,
        "adapter": adapter,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_by_rank": rng_by_rank,
        "run_config": run_config,
    }
    atomic_torch_save(payload, checkpoint_dir / f"step_{step:06d}.pt")
    atomic_torch_save(payload, checkpoint_dir / "latest.pt")


def render_conversation(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("SmolTalk row has no messages")
    rendered: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("SmolTalk message is not an object")
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant", "tool"} or not content:
            raise ValueError(f"invalid SmolTalk message role/content: {role!r}")
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(rendered)


def rank_examples(
    *,
    dataset_name: str,
    dataset_config: str,
    dataset_revision: str,
    seed: int,
    shuffle_buffer: int,
    rank: int,
    world_size: int,
    already_seen: int,
) -> Iterator[dict]:
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        revision=dataset_revision,
        streaming=True,
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)
    skipped = 0
    for global_index, row in enumerate(dataset):
        if global_index % world_size != rank:
            continue
        if skipped < already_seen:
            skipped += 1
            continue
        yield row


def linear_warmup_decay(
    warmup_steps: int, max_steps: int
) -> callable:
    def multiplier(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return multiplier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=1_024)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle-buffer", type=int, default=1_000)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--rendezvous-watchdog-seconds", type=int, default=150)
    parser.add_argument(
        "--worker-pid-file",
        default="/tmp/smollm-worker.pid",
        help="PID contract used by the recovery harness to pause one live rank",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed(
        args.rendezvous_watchdog_seconds
    )
    worker_pid_file = Path(args.worker_pid_file)
    worker_pid_file.parent.mkdir(parents=True, exist_ok=True)
    worker_pid_file.write_text(f"{os.getpid()}\n")
    device = torch.device("cuda", local_rank)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_bfloat16 = torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    precision = "bf16" if use_bfloat16 else "fp16"
    checkpoint_dir = Path(args.checkpoint_dir)
    run_config = {
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "world_size": world_size,
        "sequence_length": args.sequence_length,
        "gradient_accumulation": args.gradient_accumulation,
        "lora_rank": args.lora_rank,
        "seed": args.seed,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    ).to(device)

    start_step = 0
    tokens_seen = 0
    examples_seen_per_rank = 0
    payload: dict | None = None
    checkpoint = latest_checkpoint(checkpoint_dir)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("run_config") != run_config:
            raise RuntimeError(
                "checkpoint configuration differs from this run; use a new run name"
            )
        set_peft_model_state_dict(model, payload["adapter"])
        start_step = int(payload["step"])
        tokens_seen = int(payload["tokens_seen"])
        examples_seen_per_rank = int(payload["examples_seen_per_rank"])

    distributed_model: torch.nn.Module
    if world_size > 1:
        distributed_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        distributed_model = model
    optimizer = torch.optim.AdamW(
        (parameter for parameter in distributed_model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        linear_warmup_decay(args.warmup_steps, args.max_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bfloat16)
    if payload is not None:
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        rank_rng = payload["rng_by_rank"].get(rank)
        if rank_rng is None:
            raise RuntimeError(f"checkpoint has no RNG state for rank {rank}")
        restore_rng_state(rank_rng)

    print(
        "workload=smollm "
        f"model={args.model}@{args.model_revision} "
        f"dataset={args.dataset}/{args.dataset_config}@{args.dataset_revision} "
        f"device=cuda backend=nccl precision={precision} "
        f"rank={rank} world_size={world_size}",
        flush=True,
    )
    print(
        f"worker_pid={os.getpid()} worker_pid_file={worker_pid_file} rank={rank}",
        flush=True,
    )
    if payload is not None:
        print(
            f"resumed_from={checkpoint} step={start_step} rank={rank}",
            flush=True,
        )

    examples = rank_examples(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        rank=rank,
        world_size=world_size,
        already_seen=examples_seen_per_rank,
    )

    try:
        peer_heartbeat(world_size)
        for step in range(start_step + 1, args.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            local_loss = 0.0
            local_tokens = 0
            for microstep in range(args.gradient_accumulation):
                row = next(examples)
                text = render_conversation(row.get("messages"))
                encoded = tokenizer(
                    text,
                    max_length=args.sequence_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                labels = input_ids.clone()
                should_sync = microstep + 1 == args.gradient_accumulation
                sync_context = (
                    contextlib.nullcontext()
                    if should_sync or not isinstance(distributed_model, DDP)
                    else distributed_model.no_sync()
                )
                with sync_context:
                    with torch.autocast("cuda", dtype=model_dtype):
                        output = distributed_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = output.loss / args.gradient_accumulation
                    scaler.scale(loss).backward()
                local_loss += float(loss.detach())
                local_tokens += int(attention_mask.sum())
                examples_seen_per_rank += 1

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(distributed_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            metrics = torch.tensor(
                [local_loss, float(local_tokens)], device=device, dtype=torch.float64
            )
            if world_size > 1:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            tokens_seen += int(metrics[1].item())
            if rank == 0:
                print(
                    f"step={step} loss={metrics[0].item() / world_size:.6f} "
                    f"tokens={tokens_seen} lr={scheduler.get_last_lr()[0]:.8f}",
                    flush=True,
                )

            if step % args.checkpoint_every == 0:
                rng_by_rank = all_rank_rng_states(rank, world_size)
                if rank == 0:
                    save_checkpoint(
                        checkpoint_dir,
                        step=step,
                        tokens_seen=tokens_seen,
                        examples_seen_per_rank=examples_seen_per_rank,
                        model=distributed_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        rng_by_rank=rng_by_rank,
                        run_config=run_config,
                    )
                    print(f"checkpoint_saved=step_{step:06d}.pt", flush=True)
                peer_heartbeat(world_size)

    except Exception as exc:
        print(f"distributed_error={exc!r} rank={rank}", flush=True)
        raise SystemExit(1) from exc
    finally:
        try:
            if worker_pid_file.read_text().strip() == str(os.getpid()):
                worker_pid_file.unlink(missing_ok=True)
        except OSError:
            pass
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
