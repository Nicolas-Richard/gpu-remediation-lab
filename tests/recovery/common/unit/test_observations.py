import unittest

from tests.recovery.common.observations import (
    daemon_pod_on_node,
    has_degraded_taint,
    latest_checkpoint_step,
    latest_training_step,
    node_reports,
    pod_record,
    ready_gpu_nodes,
    ready_placement,
    resumed_checkpoint_step,
)


def pod(name: str, uid: str, node: str, ready: bool = True) -> dict:
    return {
        "metadata": {"name": name, "uid": uid},
        "spec": {"nodeName": node},
        "status": {
            "phase": "Running",
            "conditions": [
                {"type": "Ready", "status": "True" if ready else "False"}
            ],
        },
    }


def gpu_node(name: str, quantity: str = "1", ready: bool = True) -> dict:
    return {
        "metadata": {"name": name},
        "status": {
            "allocatable": {"nvidia.com/gpu": quantity},
            "conditions": [
                {"type": "Ready", "status": "True" if ready else "False"}
            ],
        },
    }


class KubernetesObservationTests(unittest.TestCase):
    def test_ready_placement_excludes_evicted_uid(self) -> None:
        items = [pod("old", "uid-old", "node-a"), pod("new", "uid-new", "node-b")]

        self.assertEqual(
            ready_placement(items, excluded_uid="uid-old").node,
            "node-b",
        )

    def test_daemon_pod_must_be_ready_and_on_requested_node(self) -> None:
        items = [pod("wrong", "1", "node-a"), pod("right", "2", "node-b")]

        self.assertEqual(daemon_pod_on_node(items, "node-b"), "right")
        self.assertIsNone(daemon_pod_on_node(items, "node-c"))

    def test_gpu_nodes_require_ready_condition_and_positive_allocation(self) -> None:
        items = [
            gpu_node("ready"),
            gpu_node("not-ready", ready=False),
            gpu_node("zero", quantity="0"),
            gpu_node("invalid", quantity="unknown"),
        ]

        self.assertEqual(
            [item["metadata"]["name"] for item in ready_gpu_nodes(items)],
            ["ready"],
        )

    def test_degraded_contract_requires_exact_owned_taint_shape(self) -> None:
        node = {
            "metadata": {
                "annotations": {
                    "gpu-orch.dev/health-state": "degraded",
                    "gpu-orch.dev/health-source": "dcgm",
                    "gpu-orch.dev/health-reason": "critical-xid-79",
                }
            },
            "spec": {
                "taints": [
                    {
                        "key": "gpu-orch.dev/hardware-degraded",
                        "value": "true",
                        "effect": "NoSchedule",
                    }
                ]
            },
        }

        self.assertTrue(has_degraded_taint(node["spec"]["taints"]))
        self.assertTrue(
            node_reports(
                node,
                state="degraded",
                source="dcgm",
                reason="critical-xid-79",
            )
        )


class TrainingObservationTests(unittest.TestCase):
    def test_extracts_latest_progress_and_checkpoint(self) -> None:
        logs = """\
step=250 loss=0.3
checkpoint_saved=step_000250.pt
step=99 loss=0.5
checkpoint_saved=step_000100.pt
step=101 loss=0.4
"""
        self.assertEqual(250, latest_training_step(logs))
        self.assertEqual(250, latest_checkpoint_step(logs))

    def test_extracts_resume_for_requested_rank(self) -> None:
        logs = """\
resumed_from=/checkpoints/latest.pt step=100 rank=0
resumed_from=/checkpoints/latest.pt step=110 rank=1
"""
        self.assertEqual(100, resumed_checkpoint_step(logs, 0))
        self.assertEqual(110, resumed_checkpoint_step(logs, 1))
        self.assertIsNone(resumed_checkpoint_step(logs, 2))

    def test_recognizes_running_pod(self) -> None:
        pod = pod_record(
            {
                "metadata": {
                    "name": "worker",
                    "uid": "123",
                    "labels": {"gpu-orch.dev/auto-remediate": "true"},
                },
                "spec": {"nodeName": "worker-a"},
                "status": {
                    "phase": "Running",
                    "containerStatuses": [{"state": {"running": {}}}],
                },
            }
        )

        self.assertTrue(pod.running)
        self.assertEqual("worker-a", pod.node)
        self.assertTrue(pod.auto_remediate)

    def test_requires_complete_degraded_taint_contract(self) -> None:
        self.assertTrue(
            has_degraded_taint(
                [
                    {
                        "key": "gpu-orch.dev/hardware-degraded",
                        "value": "true",
                        "effect": "NoSchedule",
                    }
                ]
            )
        )
        self.assertFalse(
            has_degraded_taint(
                [
                    {
                        "key": "gpu-orch.dev/hardware-degraded",
                        "value": "true",
                        "effect": "NoExecute",
                    }
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
