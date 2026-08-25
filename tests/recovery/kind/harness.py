"""Orchestrate node-health and distributed-training recovery lifecycles."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, TypeVar

from . import HarnessError
from .simulator import SimulatorClient, SimulatorPortForward

from tests.recovery.common.kubectl import Kubectl
from tests.recovery.common.observations import (
    PodRecord,
    has_degraded_taint,
    latest_checkpoint_step,
    latest_training_step,
    pod_record,
    resumed_checkpoint_step,
)
from tests.recovery.common.polling import wait_for


NAMESPACE = "training"
JOB_NAME = "demo-train"
CONTAINER_NAME = "node"
CONTROLLER_NAMESPACE = "gpu-node-health-system"
INJECTED_FAULT_ANNOTATION = "gpu-orch.dev/injected-fault"
INJECTED_FAULT = "xid-79"
MANAGED_ANNOTATION = "gpu-orch.dev/node-health-managed"
RECOVERY_CONFIRMED_ANNOTATION = "gpu-orch.dev/recovery-confirmed-at"
HEALTH_STATE_ANNOTATION = "gpu-orch.dev/health-state"
HEALTH_SOURCE_ANNOTATION = "gpu-orch.dev/health-source"
SIMULATOR_LABEL = "gpu-orch.dev/health-exporter=dcgm-simulator"
SIMULATOR_SOURCE = "dcgm-simulator"
TRAINING_JOB_LABEL = "jobset.sigs.k8s.io/jobset-name"
RANK_LABEL = "batch.kubernetes.io/job-completion-index"
POLL_INTERVAL_SECONDS = 2.0

T = TypeVar("T")


class FaultRecoveryHarness:
    def __init__(
        self,
        context: str,
        manifest: Path,
        *,
        after_step: int = 100,
        timeout: int = 180,
        keep_fault: bool = False,
        health_source: str = "annotation",
    ) -> None:
        self.kube = Kubectl(
            context,
            NAMESPACE,
            command_timeout=60,
            error_type=HarnessError,
        )
        self.system_kube = Kubectl(
            context,
            CONTROLLER_NAMESPACE,
            command_timeout=60,
            error_type=HarnessError,
        )
        self.context = context
        self.manifest = manifest
        self.after_step = after_step
        self.timeout = timeout
        self.keep_fault = keep_fault
        self.health_source = health_source
        self.fault_node: Optional[str] = None
        self.simulator_client: Optional[SimulatorClient] = None
        self.simulator_forward: Optional[SimulatorPortForward] = None

    def run(self) -> None:
        try:
            self.submit()
            rank_zero = self.wait_for_running_pod(0)
            rank_one = self.wait_for_running_pod(1)
            if not rank_one.auto_remediate:
                raise HarnessError(
                    f"rank 1 pod {rank_one.name} is missing "
                    "gpu-orch.dev/auto-remediate=true"
                )

            reached_step = self.wait_for_progress(rank_zero.name, self.after_step)
            checkpoint_step = latest_checkpoint_step(self.pod_logs(rank_zero.name))
            if checkpoint_step is None:
                raise HarnessError(
                    "training passed the progress threshold without saving a checkpoint"
                )
            self.pass_message(
                f"reached step {reached_step}; latest checkpoint {checkpoint_step}"
            )

            self.inject_fault(rank_one.node)
            self.wait_for_taint(rank_one.node, expected=True)
            self.pass_message(f"node {rank_one.node} received hardware-degraded taint")

            self.wait_for_uid_to_disappear(rank_one.uid)
            self.pass_message(f"rank 1 UID {rank_one.uid} was evicted")

            replacement = self.wait_for_running_pod(1, excluded_uid=rank_one.uid)
            if replacement.node == rank_one.node:
                raise HarnessError(
                    f"replacement rank 1 pod {replacement.name} was scheduled on "
                    f"degraded node {rank_one.node}"
                )
            self.pass_message(f"replacement rank 1 scheduled on {replacement.node}")

            rank_zero_resume = self.wait_for_resume(0)
            rank_one_resume = self.wait_for_resume(1)
            if rank_zero_resume != rank_one_resume:
                raise HarnessError(
                    "ranks resumed different checkpoints: "
                    f"rank0={rank_zero_resume}, rank1={rank_one_resume}"
                )
            if rank_zero_resume < checkpoint_step:
                raise HarnessError(
                    f"checkpoint regressed from pre-fault step {checkpoint_step} "
                    f"to resumed step {rank_zero_resume}"
                )
            self.pass_message(f"both ranks resumed from checkpoint {rank_zero_resume}")

            current_rank_zero = self.wait_for_running_pod(0)
            recovery_threshold = max(reached_step, rank_zero_resume)
            recovered_step = self.wait_for_progress(
                current_rank_zero.name, recovery_threshold
            )
            self.pass_message(
                f"training advanced past pre-fault step {reached_step} and resumed "
                f"checkpoint {rank_zero_resume} to step {recovered_step}"
            )
            if self.fault_node is not None and not self.keep_fault:
                self.recover_fault(self.fault_node)
                self.fault_node = None
            print("Fault-recovery lifecycle passed.", flush=True)
        except Exception:
            self.emit_diagnostics()
            raise
        finally:
            if self.fault_node is not None and not self.keep_fault:
                self.best_effort_cleanup(self.fault_node)
            if self.simulator_forward is not None:
                self.simulator_forward.close()

    def submit(self) -> None:
        self.kube.run(["apply", "-f", str(self.manifest)], namespaced=False)
        self.pass_message(f"submitted {JOB_NAME}")

    def pods(self, rank: Optional[int] = None) -> list[PodRecord]:
        selector = f"{TRAINING_JOB_LABEL}={JOB_NAME}"
        if rank is not None:
            selector += f",{RANK_LABEL}={rank}"
        payload = self.kube.json(["get", "pods", "--selector", selector])
        return [pod_record(item) for item in payload.get("items", [])]

    def wait_for_running_pod(
        self,
        rank: int,
        *,
        excluded_uid: Optional[str] = None,
    ) -> PodRecord:
        def find() -> Optional[PodRecord]:
            return next(
                (
                    pod
                    for pod in self.pods(rank)
                    if pod.running and pod.uid != excluded_uid
                ),
                None,
            )

        return self.wait_for(
            find,
            f"running rank {rank} pod"
            + (f" replacing UID {excluded_uid}" if excluded_uid else ""),
        )

    def wait_for_progress(self, pod_name: str, threshold: int) -> int:
        def progress() -> Optional[int]:
            step = latest_training_step(self.pod_logs(pod_name, check=False))
            return step if step is not None and step > threshold else None

        return self.wait_for(progress, f"{pod_name} to advance past step {threshold}")

    def inject_fault(self, node: str) -> None:
        if self.health_source == "dcgm-simulator":
            self.inject_dcgm_fault(node)
            return
        self.annotate_node_with_fault(node)

    def annotate_node_with_fault(self, node: str) -> None:
        self.kube.run(
            [
                "annotate",
                "node",
                node,
                f"{INJECTED_FAULT_ANNOTATION}={INJECTED_FAULT}",
                f"{RECOVERY_CONFIRMED_ANNOTATION}-",
                "--overwrite",
            ],
            namespaced=False,
        )
        self.fault_node = node
        self.pass_message(f"injected synthetic fault {INJECTED_FAULT} on {node}")

    def inject_dcgm_fault(self, node: str) -> None:
        pod_name = self.wait_for_simulator_pod(node)
        self.simulator_forward = SimulatorPortForward(
            self.context,
            CONTROLLER_NAMESPACE,
            pod_name,
        )
        self.simulator_client = self.simulator_forward.start()
        self.simulator_client.clear()
        self.simulator_client.set_xid(79)
        self.fault_node = node
        self.wait_for_health_state(node, "degraded", SIMULATOR_SOURCE)
        self.pass_message(
            f"simulator pod {pod_name} emitted DCGM XID 79 for node {node}"
        )

    def wait_for_simulator_pod(self, node: str) -> str:
        def find() -> Optional[str]:
            payload = self.system_kube.json(
                ["get", "pods", "--selector", SIMULATOR_LABEL]
            )
            for pod in payload.get("items", []):
                if pod.get("spec", {}).get("nodeName") != node:
                    continue
                if pod.get("status", {}).get("phase") != "Running":
                    continue
                conditions = pod.get("status", {}).get("conditions", [])
                if any(
                    condition.get("type") == "Ready"
                    and condition.get("status") == "True"
                    for condition in conditions
                ):
                    return str(pod.get("metadata", {}).get("name", "")) or None
            return None

        return self.wait_for(find, f"ready DCGM simulator pod on node {node}")

    def wait_for_health_state(self, node: str, state: str, source: str) -> None:
        def matches() -> Optional[bool]:
            payload = self.kube.json(["get", "node", node], namespaced=False)
            annotations = payload.get("metadata", {}).get("annotations", {})
            if (
                annotations.get(HEALTH_STATE_ANNOTATION) == state
                and annotations.get(HEALTH_SOURCE_ANNOTATION) == source
            ):
                return True
            return None

        self.wait_for(matches, f"node {node} health state {state} from {source}")

    def wait_for_taint(self, node: str, *, expected: bool) -> None:
        def matches() -> Optional[bool]:
            payload = self.kube.json(["get", "node", node], namespaced=False)
            taints = payload.get("spec", {}).get("taints", [])
            annotations = payload.get("metadata", {}).get("annotations", {})
            managed = annotations.get(MANAGED_ANNOTATION) == "true"
            return (
                True
                if has_degraded_taint(taints) == expected and managed == expected
                else None
            )

        state = "on" if expected else "removed from"
        self.wait_for(matches, f"degraded taint {state} node {node}")

    def wait_for_uid_to_disappear(self, uid: str) -> None:
        self.wait_for(
            lambda: True if all(pod.uid != uid for pod in self.pods()) else None,
            f"rank pod UID {uid} to disappear",
        )

    def wait_for_resume(self, rank: int) -> int:
        def resumed() -> Optional[int]:
            for pod in self.pods(rank):
                if pod.running:
                    step = resumed_checkpoint_step(
                        self.pod_logs(pod.name, check=False), rank
                    )
                    if step is not None:
                        return step
            return None

        return self.wait_for(resumed, f"rank {rank} to resume")

    def pod_logs(self, pod_name: str, *, check: bool = True) -> str:
        return self.kube.run(
            ["logs", pod_name, "--container", CONTAINER_NAME],
            check=check,
        )

    def recover_fault(self, node: str) -> None:
        if self.health_source == "dcgm-simulator":
            self.recover_dcgm_fault(node)
            return
        self.recover_annotation_fault(node)

    def recover_annotation_fault(self, node: str) -> None:
        confirmed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.kube.run(
            [
                "annotate",
                "node",
                node,
                f"{INJECTED_FAULT_ANNOTATION}-",
                f"{RECOVERY_CONFIRMED_ANNOTATION}={confirmed_at}",
                "--overwrite",
            ],
            namespaced=False,
        )
        self.wait_for_health_state(node, "recovering", "synthetic-annotation")
        self.wait_for_taint(node, expected=False)
        self.kube.run(
            ["annotate", "node", node, f"{RECOVERY_CONFIRMED_ANNOTATION}-"],
            namespaced=False,
        )
        self.pass_message(f"confirmed synthetic recovery and cleaned node {node}")

    def recover_dcgm_fault(self, node: str) -> None:
        if self.simulator_client is None:
            raise HarnessError("DCGM simulator client is not connected")

        self.simulator_client.clear()
        self.wait_for_health_state(node, "unknown", SIMULATOR_SOURCE)
        self.wait_for_taint(node, expected=True)
        self.pass_message("metric disappearance preserved node isolation")

        self.simulator_client.set_xid(0)
        self.wait_for_health_state(node, "recovering", SIMULATOR_SOURCE)
        self.pass_message("explicit XID 0 observation started confirmed recovery")
        self.wait_for_taint(node, expected=False)
        self.wait_for_health_state(node, "healthy", SIMULATOR_SOURCE)
        self.simulator_client.clear()
        self.pass_message(f"confirmed DCGM recovery and cleaned node {node}")

    def best_effort_cleanup(self, node: str) -> None:
        try:
            if self.health_source == "dcgm-simulator":
                if self.simulator_client is None:
                    return
                self.simulator_client.set_xid(0)
                self.wait_for_taint(node, expected=False)
                self.simulator_client.clear()
            else:
                self.recover_annotation_fault(node)
        except Exception as error:
            print(
                f"warning: failed to clean node fault state: {error}",
                file=sys.stderr,
                flush=True,
            )

    def wait_for(self, operation: Callable[[], Optional[T]], description: str) -> T:
        return wait_for(
            operation,
            description,
            timeout=self.timeout,
            interval=POLL_INTERVAL_SECONDS,
            error_type=HarnessError,
        )

    def emit_diagnostics(self) -> None:
        print("\n--- fault-recovery harness diagnostics ---", file=sys.stderr)
        commands = [
            (["get", "nodes", "-o", "wide"], False),
            (["get", "pods", "-o", "wide"], True),
            (
                [
                    "--namespace",
                    CONTROLLER_NAMESPACE,
                    "get",
                    "pods",
                    "-o",
                    "wide",
                ],
                False,
            ),
            (["get", "events", "--sort-by=.lastTimestamp"], True),
            (
                [
                    "--namespace",
                    CONTROLLER_NAMESPACE,
                    "logs",
                    "deployment/gpu-node-health-controller",
                    "--tail=100",
                ],
                False,
            ),
            (
                [
                    "--namespace",
                    CONTROLLER_NAMESPACE,
                    "logs",
                    "daemonset/dcgm-metrics-simulator",
                    "--tail=100",
                ],
                False,
            ),
        ]
        for arguments, namespaced in commands:
            try:
                output = self.kube.run(arguments, namespaced=namespaced, check=False)
                print(f"$ kubectl {' '.join(arguments)}\n{output}", file=sys.stderr)
            except Exception as error:
                print(f"diagnostic command failed: {error}", file=sys.stderr)

    @staticmethod
    def pass_message(message: str) -> None:
        print(f"PASS {message}", flush=True)
