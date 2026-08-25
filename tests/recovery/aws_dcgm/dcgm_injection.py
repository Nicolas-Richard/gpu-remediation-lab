"""Exercise the real DCGM exporter-to-remediation path on EKS GPU nodes."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TypeVar

from tests.recovery.common.kubectl import Kubectl
from tests.recovery.common.observations import (
    DCGM_SELECTOR,
    EXPORTER_SELECTOR,
    PodPlacement,
    daemon_pod_on_node,
    has_degraded_taint,
    node_reports,
    ready_gpu_nodes,
    ready_placement,
)
from tests.recovery.common.polling import wait_for
from tests.recovery.common.reporting import format_duration

from . import DCGMInjectionError


CANARY_SELECTOR = "app.kubernetes.io/name=remediation-canary"
CANARY_NAMESPACE = "gpu-remediation-test"
SUPPORTED_XIDS = (31, 43, 62, 79)

T = TypeVar("T")


def emit(status: str, message: str, *, output=None) -> None:
    """Write a timestamped human-readable scenario record."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} {status}: {message}", file=output, flush=True)


class DCGMInjectionScenario:
    def __init__(
        self,
        context: str,
        manifest: Path,
        *,
        xid: int = 79,
        timeout: int = 300,
        minimum_gpu_nodes: int = 4,
        keep_canary: bool = False,
    ) -> None:
        if xid not in SUPPORTED_XIDS:
            raise ValueError(f"xid must be one of {SUPPORTED_XIDS}")
        self.kubectl = Kubectl(context, error_type=DCGMInjectionError)
        self.manifest = manifest
        self.xid = xid
        self.timeout = timeout
        self.minimum_gpu_nodes = minimum_gpu_nodes
        self.keep_canary = keep_canary
        self.faulted_node = ""

    def wait(self, description: str, probe: Callable[[], Optional[T]]) -> T:
        result = wait_for(
            probe,
            description,
            timeout=self.timeout,
            interval=2,
            error_type=DCGMInjectionError,
        )
        emit("PASS", description)
        return result

    def run(self) -> None:
        started_at = time.monotonic()
        outcome = "FAIL"
        emit("START", f"AWS DCGM injection scenario (XID {self.xid})")
        try:
            self.wait(
                f"at least {self.minimum_gpu_nodes} Ready GPU nodes expose nvidia.com/gpu",
                self._gpu_capacity,
            )
            self.kubectl.run(["apply", "-f", str(self.manifest)])
            try:
                original = self.wait("canary is ready on a GPU node", self._canary)
                self.faulted_node = original.node
                self.wait(
                    f"DCGM exporter is ready on {original.node}",
                    lambda: self._daemon_pod(original.node, EXPORTER_SELECTOR),
                )
                self.wait(
                    f"standalone DCGM is ready on {original.node}",
                    lambda: self._daemon_pod(original.node, DCGM_SELECTOR),
                )
                self._inject(original.node, 0)
                self.wait(
                    "controller observes the seeded healthy real-DCGM value",
                    lambda: self._healthy_node(original.node),
                )
                self._inject(original.node, self.xid)
                self.wait(
                    f"node is isolated from DCGM XID {self.xid}",
                    lambda: self._degraded_node(original.node),
                )
                replacement = self.wait(
                    "evicted canary is replaced on another GPU node",
                    lambda: self._replacement(original),
                )
                emit(
                    "PASS",
                    f"canary moved {original.node} ({original.uid}) -> "
                    f"{replacement.node} ({replacement.uid})",
                )
                self._inject(original.node, 0)
                self.wait(
                    "zero injection clears controller-owned isolation",
                    lambda: self._recovered_node(original.node),
                )
            finally:
                if self.faulted_node:
                    self._best_effort_clear(self.faulted_node)
                if not self.keep_canary:
                    self.kubectl.run(
                        [
                            "delete",
                            "-f",
                            str(self.manifest),
                            "--ignore-not-found",
                            "--wait=true",
                        ],
                        check=False,
                    )
            outcome = "PASS"
        finally:
            emit(
                "END",
                (
                    f"AWS DCGM injection scenario {outcome} "
                    f"(total duration: {format_duration(time.monotonic() - started_at)})"
                )
            )

    def _gpu_capacity(self) -> Optional[int]:
        nodes = self.kubectl.json(
            ["get", "nodes", "-l", "gpu-orch.dev/node-pool=gpu"]
        ).get("items", [])
        ready = ready_gpu_nodes(nodes)
        if len(ready) < self.minimum_gpu_nodes:
            return None
        return len(ready)

    def _canary(self) -> Optional[PodPlacement]:
        pods = self.kubectl.json(
            ["get", "pods", "-l", CANARY_SELECTOR], namespace=CANARY_NAMESPACE
        )
        return ready_placement(pods.get("items", []))

    def _healthy_node(self, node_name: str) -> Optional[dict]:
        node = self.kubectl.json(["get", "node", node_name])
        if node_reports(node, state="healthy", source="dcgm", reason="xid-clear"):
            return node
        return None

    def _degraded_node(self, node_name: str) -> Optional[dict]:
        node = self.kubectl.json(["get", "node", node_name])
        annotations = node.get("metadata", {}).get("annotations", {})
        if (
            node_reports(
                node,
                state="degraded",
                source="dcgm",
                reason=f"critical-xid-{self.xid}",
            )
            and annotations.get("gpu-orch.dev/node-health-managed") == "true"
            and has_degraded_taint(node.get("spec", {}).get("taints", []))
        ):
            return node
        return None

    def _recovered_node(self, node_name: str) -> Optional[dict]:
        node = self.kubectl.json(["get", "node", node_name])
        annotations = node.get("metadata", {}).get("annotations", {})
        if (
            node_reports(node, state="healthy", source="dcgm", reason="xid-clear")
            and annotations.get("gpu-orch.dev/node-health-managed") is None
            and not has_degraded_taint(node.get("spec", {}).get("taints", []))
        ):
            return node
        return None

    def _replacement(self, original: PodPlacement) -> Optional[PodPlacement]:
        pods = self.kubectl.json(
            ["get", "pods", "-l", CANARY_SELECTOR], namespace=CANARY_NAMESPACE
        )
        replacement = ready_placement(pods.get("items", []), excluded_uid=original.uid)
        if replacement is not None and replacement.node != original.node:
            return replacement
        return None

    def _daemon_pod(self, node: str, selector: str) -> Optional[str]:
        pods = self.kubectl.json(
            ["get", "pods", "-l", selector], namespace="gpu-operator"
        )
        return daemon_pod_on_node(pods.get("items", []), node)

    def _inject(self, node: str, value: int) -> None:
        pod = self._daemon_pod(node, DCGM_SELECTOR)
        if pod is None:
            raise DCGMInjectionError(f"no Ready standalone DCGM pod found on {node}")
        output = self.kubectl.run(
            [
                "exec",
                pod,
                "-c",
                "nvidia-dcgm-ctr",
                "--",
                "dcgmi",
                "test",
                "--inject",
                "--gpuid",
                "0",
                "-f",
                "230",
                "-v",
                str(value),
            ],
            namespace="gpu-operator",
        ).strip()
        emit("PASS", f"injected DCGM field 230 value {value} on {node}: {output}")

    def _best_effort_clear(self, node: str) -> None:
        try:
            self._inject(node, 0)
        except DCGMInjectionError as error:
            emit("WARN", f"could not clear DCGM injection on {node}: {error}")
