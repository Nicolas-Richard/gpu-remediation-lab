"""Exercise distributed CUDA recovery from a stalled rank and DCGM telemetry."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tests.recovery.common.kubectl import Kubectl
from tests.recovery.common.observations import (
    DCGM_SELECTOR,
    EXPORTER_SELECTOR,
    PodRecord,
    daemon_pod_on_node,
    has_degraded_taint,
    latest_checkpoint_step,
    latest_training_step,
    node_reports,
    pod_record,
    resumed_checkpoint_step,
)
from tests.recovery.common.polling import wait_for
from tests.recovery.common.reporting import ScenarioReporter, format_duration, short_uid

from . import TrainingRecoveryError


NAMESPACE = "gpu-training"
CONTAINER_NAME = "node"
TRAINING_JOB_LABEL = "jobset.sigs.k8s.io/jobset-name"
RANK_LABEL = "batch.kubernetes.io/job-completion-index"
MANAGED_ANNOTATION = "gpu-orch.dev/node-health-managed"
RUN_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
TRAINING_RANKS = 4
VICTIM_RANK = TRAINING_RANKS - 1
WORKER_PID_FILE = "/tmp/smollm-worker.pid"
QUIESCENCE_WINDOW_SECONDS = 30


def render_job_template(template: str, *, run_name: str, image: str) -> str:
    if not RUN_NAME_PATTERN.fullmatch(run_name) or len(run_name) > 63:
        raise TrainingRecoveryError(f"invalid Kubernetes run name: {run_name!r}")
    if not image.strip():
        raise TrainingRecoveryError("training image must not be empty")
    rendered = template.replace("__RUN_NAME__", run_name).replace(
        "__TRAINING_IMAGE__", image
    )
    if "__RUN_NAME__" in rendered or "__TRAINING_IMAGE__" in rendered:
        raise TrainingRecoveryError("job template contains unresolved placeholders")
    return rendered


class AWSTrainingRecoveryHarness:
    def __init__(
        self,
        context: str,
        job_template: Path,
        image: str,
        *,
        after_step: int = 10,
        timeout: int = 1800,
        run_name: Optional[str] = None,
        keep_job: bool = False,
        events_jsonl: Optional[Path] = None,
        verbose: bool = False,
        quiescence_window: float = QUIESCENCE_WINDOW_SECONDS,
        reporter: Optional[ScenarioReporter] = None,
    ) -> None:
        self.kubectl = Kubectl(
            context,
            NAMESPACE,
            command_timeout=120,
            error_type=TrainingRecoveryError,
        )
        self.job_template = job_template
        self.image = image
        self.after_step = after_step
        self.timeout = timeout
        self.run_name = run_name or datetime.now(timezone.utc).strftime(
            "aws-train-%Y%m%d%H%M%S"
        )
        self.keep_job = keep_job
        self.quiescence_window = quiescence_window
        self.faulted_node = ""
        self.paused_pod = ""
        self.reporter = reporter or ScenarioReporter(
            self.run_name,
            jsonl_path=events_jsonl,
            verbose=verbose,
        )

    def run(self) -> None:
        started_at = time.monotonic()
        marks: dict[str, float] = {}
        outcome = "FAIL"
        self.reporter.emit(
            "harness",
            "START",
            message=f"Started SmolLM recovery test with {TRAINING_RANKS} ranks",
            run=self.run_name,
            ranks=TRAINING_RANKS,
        )
        try:
            self.reporter.phase("submit")
            self._submit()
            self.reporter.emit(
                "kubernetes",
                "TRAINJOB_SUBMITTED",
                message=f"Submitted TrainJob {self.run_name}",
                trainjob=self.run_name,
            )

            self.reporter.phase("baseline")
            initial_pods = {
                rank: self._wait_for_running_pod(rank)
                for rank in range(TRAINING_RANKS)
            }
            placements = [
                f"rank-{rank}@{self.reporter.node(pod.node)}"
                for rank, pod in initial_pods.items()
            ]
            placement_summary = ", ".join(
                placement.replace("rank-", "rank ").replace("@", "→")
                for placement in placements
            )
            self.reporter.emit(
                "kubernetes",
                "RANKS_PLACED",
                message=f"Placed {placement_summary}",
                placements=placements,
            )
            self.reporter.emit(
                "kubernetes",
                "ALIASES",
                message="Recorded initial GPU node aliases",
                visible=False,
                nodes=self.reporter.nodes.mapping(),
            )
            for rank, pod in initial_pods.items():
                if not pod.auto_remediate:
                    raise TrainingRecoveryError(
                        f"rank {rank} pod {pod.name} is not opted into remediation"
                    )
                self._wait_for_smollm_cuda(pod.name, rank)
            self.reporter.emit(
                "workload",
                "RANKS_READY",
                message=(
                    f"All {TRAINING_RANKS} ranks are training SmolLM3-3B "
                    "with CUDA/NCCL"
                ),
                workload="SmolLM3-3B",
                device="CUDA",
                backend="NCCL",
                ranks=TRAINING_RANKS,
            )

            rank_zero = initial_pods[0]
            victim = initial_pods[VICTIM_RANK]
            reached_step = self._wait_for_progress(rank_zero.name, self.after_step)
            checkpoint_step = latest_checkpoint_step(self._pod_logs(rank_zero.name))
            if checkpoint_step is None:
                raise TrainingRecoveryError(
                    "training passed the progress threshold without an EFS checkpoint"
                )
            self.reporter.emit(
                "workload",
                "PROGRESS",
                message=(
                    f"Rank 0 reached step {reached_step}; checkpoint "
                    f"{checkpoint_step} is durable on EFS"
                ),
                step=reached_step,
                checkpoint=checkpoint_step,
                reporter="rank-0",
                node=self._node_alias(rank_zero.node),
            )

            self.faulted_node = victim.node
            victim_alias = self.reporter.node(victim.node)
            self.reporter.phase("fault-injection")
            self._wait_for_daemon(victim.node, EXPORTER_SELECTOR, "DCGM exporter")
            self._wait_for_daemon(victim.node, DCGM_SELECTOR, "standalone DCGM")
            self.reporter.emit(
                "harness",
                "DCGM_READY",
                message=f"DCGM metrics and injection paths are ready on {victim_alias}",
                visible=False,
                node=victim_alias,
                metrics_exporter=True,
                injection_hostengine=True,
            )
            self._inject(victim.node, 0)
            self.reporter.emit(
                "harness",
                "XID_SEEDED",
                message=f"Cleared stale injected XID state on {victim_alias}",
                visible=False,
                node=victim_alias,
                xid=0,
                device=0,
            )
            self._wait_for_healthy_node(victim.node)
            self.reporter.emit(
                "gpu-health-controller",
                "HEALTHY",
                message=f"Confirmed {victim_alias} healthy before fault injection",
                visible=False,
                node=victim_alias,
                source="dcgm",
                reason="xid-clear",
            )

            self._pause_worker(victim.name)
            self.paused_pod = victim.name
            marks["paused"] = time.monotonic()
            self.reporter.emit(
                "harness",
                "RANK_PAUSED",
                message=(
                    f"Paused rank {VICTIM_RANK} worker in pod "
                    f"{short_uid(victim.uid)} on {victim_alias}"
                ),
                rank=VICTIM_RANK,
                pod=short_uid(victim.uid),
                node=victim_alias,
                signal="SIGSTOP",
            )
            stalled_step, checkpoint_step = self._wait_for_quiescence(
                rank_zero.name,
                victim,
                minimum_checkpoint=checkpoint_step,
            )
            marks["quiescent"] = time.monotonic()
            self.reporter.emit(
                "kubernetes",
                "PAUSED_POD_STILL_RUNNING",
                message=(
                    f"Pod {short_uid(victim.uid)} still reports Running; "
                    "Kubernetes has not replaced it"
                ),
                rank=VICTIM_RANK,
                pod=short_uid(victim.uid),
                node=victim_alias,
                pod_phase="Running",
            )
            self.reporter.emit(
                "workload",
                "QUIESCENT",
                message=(
                    f"Training is stalled at step {stalled_step}; checkpoint "
                    f"{checkpoint_step} is the recovery fence"
                ),
                step=stalled_step,
                checkpoint=checkpoint_step,
                stable_seconds=self.quiescence_window,
            )

            self._inject(victim.node, 79)
            marks["fault"] = time.monotonic()
            self.reporter.emit(
                "harness",
                "XID_INJECTED",
                message=(
                    f"Injected synthetic XID 79 telemetry for GPU 0 on "
                    f"{victim_alias} using DCGM"
                ),
                node=victim_alias,
                xid=79,
                device=0,
                synthetic=True,
            )
            self._wait_for_degraded_node(victim.node)
            marks["degraded"] = time.monotonic()
            self.reporter.emit(
                "gpu-health-controller",
                "NODE_ISOLATED",
                message=(
                    f"Isolated {victim_alias} after detecting critical XID 79"
                ),
                node=victim_alias,
                reason="critical-xid-79",
                taint="hardware-degraded:NoSchedule",
            )

            self.reporter.phase("remediation")
            self._wait_for_uid_to_disappear(victim.uid)
            self.paused_pod = ""
            marks["evicted"] = time.monotonic()
            self.reporter.emit(
                "kubernetes",
                "RANK_EVICTED",
                message=(
                    f"Evicted rank {VICTIM_RANK} pod {short_uid(victim.uid)} "
                    f"from {victim_alias}"
                ),
                rank=VICTIM_RANK,
                pod=short_uid(victim.uid),
                node=victim_alias,
            )
            replacement = self._wait_for_running_pod(
                VICTIM_RANK, excluded_uid=victim.uid
            )
            if replacement.node == victim.node:
                raise TrainingRecoveryError(
                    f"replacement rank {VICTIM_RANK} remained on faulted node "
                    f"{victim.node}"
                )
            marks["replacement"] = time.monotonic()
            replacement_alias = self._node_alias(replacement.node)
            self.reporter.emit(
                "kubernetes",
                "RANK_REPLACED",
                message=(
                    f"Replaced rank {VICTIM_RANK} on {replacement_alias} "
                    f"with pod {short_uid(replacement.uid)}"
                ),
                rank=VICTIM_RANK,
                from_node=victim_alias,
                to_node=replacement_alias,
                old_pod=short_uid(victim.uid),
                new_pod=short_uid(replacement.uid),
            )

            self.reporter.phase("recovery")
            resumed_steps = {
                rank: self._wait_for_resume(rank)
                for rank in range(TRAINING_RANKS)
            }
            if len(set(resumed_steps.values())) != 1:
                raise TrainingRecoveryError(
                    f"ranks resumed different EFS checkpoints: {resumed_steps}"
                )
            resumed_step = resumed_steps[0]
            if resumed_step != checkpoint_step:
                raise TrainingRecoveryError(
                    f"ranks resumed checkpoint {resumed_step}; expected exact "
                    f"recovery fence {checkpoint_step}"
                )
            marks["resumed"] = time.monotonic()
            self.reporter.emit(
                "workload",
                "RANKS_RESUMED",
                message=(
                    f"All {TRAINING_RANKS} ranks resumed from EFS checkpoint "
                    f"{resumed_step}"
                ),
                checkpoint=resumed_step,
                ranks=list(range(TRAINING_RANKS)),
            )

            resumed_victim = self._wait_for_running_pod(VICTIM_RANK)
            if resumed_victim.node == victim.node:
                raise TrainingRecoveryError(
                    f"resumed rank {VICTIM_RANK} returned to faulted node "
                    f"{victim.node}"
                )
            self.reporter.emit(
                "kubernetes",
                "RECOVERED_RANK_PLACED",
                message=f"Confirmed recovered rank {VICTIM_RANK} remains off {victim_alias}",
                visible=False,
                rank=VICTIM_RANK,
                node=self._node_alias(resumed_victim.node),
                excluded_node=victim_alias,
                pod=short_uid(resumed_victim.uid),
            )

            current_rank_zero = self._wait_for_running_pod(0)
            recovered_step = self._wait_for_progress(
                current_rank_zero.name,
                max(reached_step, stalled_step, resumed_step),
            )
            marks["progress"] = time.monotonic()
            self.reporter.emit(
                "workload",
                "PROGRESS",
                message=f"Training advanced to step {recovered_step} after recovery",
                step=recovered_step,
                checkpoint=resumed_step,
                state="recovered",
                reporter="rank-0",
                node=self._node_alias(current_rank_zero.node),
            )

            self._inject(victim.node, 0)
            self.reporter.emit(
                "harness",
                "XID_CLEARED",
                message=f"Cleared the injected XID on {victim_alias}",
                node=victim_alias,
                xid=0,
                device=0,
            )
            self._wait_for_recovered_node(victim.node)
            self.reporter.emit(
                "gpu-health-controller",
                "NODE_RETURNED_TO_SERVICE",
                message=f"Returned {victim_alias} to service",
                node=victim_alias,
                isolation="cleared",
                reason="xid-clear",
            )
            self.faulted_node = ""
            outcome = "PASS"
        except Exception as error:
            self.reporter.emit(
                "harness",
                "ERROR",
                message=f"Recovery test failed: {error}",
                error=str(error),
            )
            self._emit_diagnostics()
            raise
        finally:
            self.reporter.phase("cleanup")
            if self.paused_pod:
                self._best_effort_resume(self.paused_pod)
                self.paused_pod = ""
            if self.faulted_node:
                self._best_effort_clear(self.faulted_node)
            if not self.keep_job:
                self._delete_job()
                self.reporter.emit(
                    "kubernetes",
                    "TRAINJOB_DELETED",
                    message=f"Deleted TrainJob {self.run_name}",
                    visible=False,
                    trainjob=self.run_name,
                )
            else:
                self.reporter.emit(
                    "kubernetes",
                    "TRAINJOB_RETAINED",
                    message=f"Retained TrainJob {self.run_name}",
                    visible=False,
                    trainjob=self.run_name,
                )
            result_fields: dict[str, object] = {
                "total": format_duration(time.monotonic() - started_at)
            }
            duration_pairs = {
                "detection": ("fault", "degraded"),
                "quiescence": ("paused", "quiescent"),
                "eviction": ("degraded", "evicted"),
                "replacement": ("degraded", "replacement"),
                "rescheduling": ("evicted", "replacement"),
                "resume": ("replacement", "resumed"),
                "workload_recovery": ("degraded", "progress"),
            }
            for name, (start, end) in duration_pairs.items():
                if start in marks and end in marks:
                    result_fields[name] = format_duration(marks[end] - marks[start])
            self.reporter.current_phase = "result"
            if outcome == "PASS":
                summary = (
                    f"PASS — detection {result_fields['detection']}, "
                    f"replacement {result_fields['replacement']}, "
                    f"workload recovery {result_fields['workload_recovery']}"
                )
            else:
                summary = f"FAIL — total runtime {result_fields['total']}"
            self.reporter.emit(
                "harness",
                outcome,
                message=summary,
                **result_fields,
            )

    def _submit(self) -> None:
        template = self.job_template.read_text()
        rendered = render_job_template(
            template,
            run_name=self.run_name,
            image=self.image,
        )
        self.kubectl.run(["apply", "-f", "-"], input_text=rendered)

    def _node_alias(self, node: str) -> str:
        known_aliases = self.reporter.nodes.mapping()
        alias = self.reporter.node(node)
        if alias not in known_aliases:
            self.reporter.emit(
                "kubernetes",
                "ALIAS",
                message=f"Mapped replacement node {node} to {alias}",
                visible=False,
                alias=alias,
                hostname=node,
            )
        return alias

    def _pods(self, rank: Optional[int] = None) -> list[PodRecord]:
        selector = f"{TRAINING_JOB_LABEL}={self.run_name}"
        if rank is not None:
            selector += f",{RANK_LABEL}={rank}"
        payload = self.kubectl.json(["get", "pods", "--selector", selector])
        return [pod_record(item) for item in payload.get("items", [])]

    def _wait_for_running_pod(
        self,
        rank: int,
        *,
        excluded_uid: Optional[str] = None,
    ) -> PodRecord:
        def find() -> Optional[PodRecord]:
            return next(
                (
                    pod
                    for pod in self._pods(rank)
                    if pod.running and pod.uid != excluded_uid
                ),
                None,
            )

        return self._wait(
            find,
            f"running rank {rank} pod"
            + (f" replacing UID {excluded_uid}" if excluded_uid else ""),
        )

    def _wait_for_smollm_cuda(self, pod_name: str, rank: int) -> None:
        def confirmed() -> Optional[bool]:
            logs = self._pod_logs(pod_name, check=False)
            return (
                True
                if "workload=smollm" in logs
                and "device=cuda backend=nccl" in logs
                and f"rank={rank} world_size={TRAINING_RANKS}" in logs
                else None
            )

        self._wait(
            confirmed,
            f"rank {rank} to confirm SmolLM CUDA/NCCL execution",
        )

    def _wait_for_progress(self, pod_name: str, threshold: int) -> int:
        def progress() -> Optional[int]:
            step = latest_training_step(self._pod_logs(pod_name, check=False))
            return step if step is not None and step > threshold else None

        return self._wait(progress, f"{pod_name} to advance past step {threshold}")

    def _pause_worker(self, pod_name: str) -> None:
        script = (
            "import os, signal; "
            f"pid = int(open({WORKER_PID_FILE!r}).read()); "
            "os.kill(pid, signal.SIGSTOP); "
            "print(f'pid={pid} signal=SIGSTOP')"
        )
        self.kubectl.run(
            [
                "exec",
                pod_name,
                "--container",
                CONTAINER_NAME,
                "--",
                "python",
                "-c",
                script,
            ]
        )
        self._wait(
            lambda: True if self._worker_state(pod_name) == "T" else None,
            f"rank {VICTIM_RANK} worker to enter stopped state",
        )

    def _worker_state(self, pod_name: str) -> str:
        script = (
            f'pid="$(cat {WORKER_PID_FILE})"; '
            "awk '/^State:/ {print $2}' \"/proc/$pid/status\""
        )
        return self.kubectl.run(
            [
                "exec",
                pod_name,
                "--container",
                CONTAINER_NAME,
                "--",
                "/bin/sh",
                "-c",
                script,
            ],
            check=False,
        ).strip()

    def _wait_for_quiescence(
        self,
        rank_zero_pod: str,
        victim: PodRecord,
        *,
        minimum_checkpoint: int,
    ) -> tuple[int, int]:
        last_observation: Optional[tuple[int, int]] = None
        stable_since = time.monotonic()

        def quiescent() -> Optional[tuple[int, int]]:
            nonlocal last_observation, stable_since
            current_victim = next(
                (pod for pod in self._pods(VICTIM_RANK) if pod.uid == victim.uid),
                None,
            )
            if current_victim is None or not current_victim.running:
                raise TrainingRecoveryError(
                    "paused rank pod changed before GPU health controller eviction"
                )
            if self._worker_state(victim.name) != "T":
                raise TrainingRecoveryError(
                    "rank worker left stopped state before GPU health remediation"
                )

            logs = self._pod_logs(rank_zero_pod, check=False)
            step = latest_training_step(logs)
            checkpoint = latest_checkpoint_step(logs)
            if step is None or checkpoint is None:
                return None
            if checkpoint < minimum_checkpoint:
                raise TrainingRecoveryError(
                    f"checkpoint regressed from {minimum_checkpoint} to {checkpoint} "
                    "while establishing the recovery fence"
                )

            observation = (step, checkpoint)
            now = time.monotonic()
            if observation != last_observation:
                last_observation = observation
                stable_since = now
                return None
            if now - stable_since >= self.quiescence_window:
                return observation
            return None

        return self._wait(
            quiescent,
            "paused rank pod to remain Running while distributed training quiesces",
        )

    def _resume_worker(self, pod_name: str) -> None:
        script = (
            "import os, signal; "
            f"pid = int(open({WORKER_PID_FILE!r}).read()); "
            "os.kill(pid, signal.SIGCONT); "
            "print(f'pid={pid} signal=SIGCONT')"
        )
        self.kubectl.run(
            [
                "exec",
                pod_name,
                "--container",
                CONTAINER_NAME,
                "--",
                "python",
                "-c",
                script,
            ]
        )

    def _best_effort_resume(self, pod_name: str) -> None:
        try:
            self._resume_worker(pod_name)
        except Exception as error:
            self.reporter.emit(
                "harness",
                "WARN",
                message=f"Could not resume paused worker in pod {pod_name}",
                pod=pod_name,
                error=str(error),
            )

    def _wait_for_resume(self, rank: int) -> int:
        def resumed() -> Optional[int]:
            for pod in self._pods(rank):
                if not pod.running:
                    continue
                step = resumed_checkpoint_step(
                    self._pod_logs(pod.name, check=False), rank
                )
                if step is not None:
                    return step
            return None

        return self._wait(resumed, f"rank {rank} to resume an EFS checkpoint")

    def _wait_for_uid_to_disappear(self, uid: str) -> None:
        self._wait(
            lambda: True if all(pod.uid != uid for pod in self._pods()) else None,
            f"rank pod UID {uid} to disappear",
        )

    def _pod_logs(self, pod_name: str, *, check: bool = True) -> str:
        return self.kubectl.run(
            ["logs", pod_name, "--container", CONTAINER_NAME],
            check=check,
        )

    def _wait_for_daemon(self, node: str, selector: str, description: str) -> str:
        return self._wait(
            lambda: self._daemon_pod(node, selector),
            f"{description} on {node}",
        )

    def _daemon_pod(self, node: str, selector: str) -> Optional[str]:
        pods = self.kubectl.json(
            ["get", "pods", "-l", selector],
            namespace="gpu-operator",
        )
        return daemon_pod_on_node(pods.get("items", []), node)

    def _inject(self, node: str, value: int) -> str:
        pod = self._daemon_pod(node, DCGM_SELECTOR)
        if pod is None:
            raise TrainingRecoveryError(f"no Ready standalone DCGM pod on {node}")
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
        return output

    def _node(self, node_name: str) -> dict:
        return self.kubectl.json(["get", "node", node_name], namespaced=False)

    def _wait_for_healthy_node(self, node_name: str) -> None:
        def healthy() -> Optional[dict]:
            node = self._node(node_name)
            if node_reports(
                node,
                state="healthy",
                source="dcgm",
                reason="xid-clear",
            ):
                return node
            return None

        self._wait(
            healthy,
            f"controller to observe healthy DCGM on {node_name}",
        )

    def _wait_for_degraded_node(self, node_name: str) -> None:
        def degraded() -> Optional[dict]:
            node = self._node(node_name)
            annotations = node.get("metadata", {}).get("annotations", {})
            if (
                node_reports(
                    node,
                    state="degraded",
                    source="dcgm",
                    reason="critical-xid-79",
                )
                and annotations.get(MANAGED_ANNOTATION) == "true"
                and has_degraded_taint(node.get("spec", {}).get("taints", []))
            ):
                return node
            return None

        self._wait(degraded, f"controller to isolate {node_name} for XID 79")

    def _wait_for_recovered_node(self, node_name: str) -> None:
        def recovered() -> Optional[dict]:
            node = self._node(node_name)
            annotations = node.get("metadata", {}).get("annotations", {})
            if (
                node_reports(
                    node,
                    state="healthy",
                    source="dcgm",
                    reason="xid-clear",
                )
                and annotations.get(MANAGED_ANNOTATION) is None
                and not has_degraded_taint(
                    node.get("spec", {}).get("taints", [])
                )
            ):
                return node
            return None

        self._wait(recovered, f"controller to recover {node_name}")

    def _best_effort_clear(self, node: str) -> None:
        try:
            self._inject(node, 0)
            self._wait_for_recovered_node(node)
        except Exception as error:
            self.reporter.emit(
                "harness",
                "WARN",
                message=f"Could not fully clear the fault on {self.reporter.node(node)}",
                node=self.reporter.node(node),
                error=str(error),
            )

    def _delete_job(self) -> None:
        self.kubectl.run(
            [
                "delete",
                "trainjob",
                self.run_name,
                "--ignore-not-found",
                "--cascade=foreground",
                "--wait=true",
            ],
            check=False,
        )

    def _wait(self, operation, description: str):
        result = wait_for(
            operation,
            description,
            timeout=self.timeout,
            interval=2,
            error_type=TrainingRecoveryError,
        )
        return result

    def _emit_diagnostics(self) -> None:
        self.reporter.emit(
            "harness",
            "DIAGNOSTICS",
            message="Collecting Kubernetes diagnostics",
        )
        commands = [
            (["get", "nodes", "-o", "wide"], False, None),
            (["get", "pods", "-o", "wide"], True, None),
            (["get", "events", "--sort-by=.lastTimestamp"], True, None),
            (["get", "pods", "-o", "wide"], True, "gpu-operator"),
            (
                [
                    "logs",
                    "deployment/gpu-node-health-controller",
                    "--tail=100",
                ],
                True,
                "gpu-node-health-system",
            ),
        ]
        for arguments, namespaced, namespace in commands:
            try:
                output = self.kubectl.run(
                    arguments,
                    namespaced=namespaced,
                    namespace=namespace,
                    check=False,
                )
                self.reporter.emit(
                    "harness",
                    "COMMAND",
                    message=f"Running kubectl {' '.join(arguments)}",
                    command=f"kubectl {' '.join(arguments)}",
                )
                self.reporter.raw("kubectl", output)
            except Exception as error:
                self.reporter.emit(
                    "harness",
                    "WARN",
                    message=f"Diagnostic command failed: {error}",
                    error=str(error),
                )
