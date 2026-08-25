"""Parse Kubernetes state and training logs shared by recovery harnesses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence


DEGRADED_TAINT_KEY = "gpu-orch.dev/hardware-degraded"
DCGM_SELECTOR = "app=nvidia-dcgm"
EXPORTER_SELECTOR = "app=nvidia-dcgm-exporter"

STEP_PATTERN = re.compile(r"^step=(\d+)\b", re.MULTILINE)
CHECKPOINT_PATTERN = re.compile(r"^checkpoint_saved=step_(\d+)\.pt$", re.MULTILINE)
RESUME_PATTERN = re.compile(r"^resumed_from=.* step=(\d+) rank=(\d+)$", re.MULTILINE)


@dataclass(frozen=True)
class PodRecord:
    name: str
    uid: str
    node: str
    running: bool
    auto_remediate: bool


@dataclass(frozen=True)
class PodPlacement:
    name: str
    uid: str
    node: str


def pod_record(item: dict) -> PodRecord:
    metadata = item.get("metadata", {})
    spec = item.get("spec", {})
    status = item.get("status", {})
    container_statuses = status.get("containerStatuses", [])
    running = status.get("phase") == "Running" and any(
        "running" in container.get("state", {}) for container in container_statuses
    )
    return PodRecord(
        name=metadata.get("name", ""),
        uid=metadata.get("uid", ""),
        node=spec.get("nodeName", ""),
        running=running,
        auto_remediate=(
            metadata.get("labels", {}).get("gpu-orch.dev/auto-remediate") == "true"
        ),
    )


def pod_ready(item: dict) -> bool:
    return item.get("status", {}).get("phase") == "Running" and any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in item.get("status", {}).get("conditions", [])
    )


def ready_placement(
    items: Sequence[dict], *, excluded_uid: str = ""
) -> Optional[PodPlacement]:
    for item in items:
        metadata = item.get("metadata", {})
        node = item.get("spec", {}).get("nodeName", "")
        uid = metadata.get("uid", "")
        if uid and uid != excluded_uid and node and pod_ready(item):
            return PodPlacement(name=metadata.get("name", ""), uid=uid, node=node)
    return None


def daemon_pod_on_node(items: Sequence[dict], node: str) -> Optional[str]:
    for item in items:
        if item.get("spec", {}).get("nodeName") == node and pod_ready(item):
            return item.get("metadata", {}).get("name")
    return None


def ready_gpu_nodes(items: Sequence[dict]) -> list[dict]:
    ready = []
    for item in items:
        gpu_quantity = (
            item.get("status", {})
            .get("allocatable", {})
            .get("nvidia.com/gpu", "0")
        )
        try:
            has_gpu = int(gpu_quantity) > 0
        except (TypeError, ValueError):
            has_gpu = False
        if has_gpu and any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in item.get("status", {}).get("conditions", [])
        ):
            ready.append(item)
    return ready


def node_reports(node: dict, *, state: str, source: str, reason: str) -> bool:
    annotations = node.get("metadata", {}).get("annotations", {})
    return (
        annotations.get("gpu-orch.dev/health-state") == state
        and annotations.get("gpu-orch.dev/health-source") == source
        and annotations.get("gpu-orch.dev/health-reason") == reason
    )


def latest_training_step(logs: str) -> Optional[int]:
    matches = [int(value) for value in STEP_PATTERN.findall(logs)]
    return max(matches) if matches else None


def latest_checkpoint_step(logs: str) -> Optional[int]:
    matches = [int(value) for value in CHECKPOINT_PATTERN.findall(logs)]
    return max(matches) if matches else None


def resumed_checkpoint_step(logs: str, rank: int) -> Optional[int]:
    matches = [
        int(step)
        for step, found_rank in RESUME_PATTERN.findall(logs)
        if int(found_rank) == rank
    ]
    return matches[-1] if matches else None


def has_degraded_taint(taints: Sequence[dict]) -> bool:
    return any(
        taint.get("key") == DEGRADED_TAINT_KEY
        and taint.get("value") == "true"
        and taint.get("effect") == "NoSchedule"
        for taint in taints
    )
