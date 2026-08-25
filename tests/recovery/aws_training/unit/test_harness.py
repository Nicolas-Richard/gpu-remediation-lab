import io
import unittest
from pathlib import Path
from unittest.mock import Mock

from tests.recovery.aws_training import TrainingRecoveryError
from tests.recovery.aws_training.harness import (
    AWSTrainingRecoveryHarness,
    TRAINING_RANKS,
    VICTIM_RANK,
    render_job_template,
)
from tests.recovery.common.observations import PodRecord
from tests.recovery.common.reporting import ScenarioReporter


class TrainingJobTemplateTests(unittest.TestCase):
    def test_renders_run_name_and_training_image(self) -> None:
        rendered = render_job_template(
            "name: __RUN_NAME__\nimage: __TRAINING_IMAGE__\npath: __RUN_NAME__\n",
            run_name="aws-train-123",
            image="example.test/training:sha",
        )

        self.assertEqual(
            "name: aws-train-123\n"
            "image: example.test/training:sha\n"
            "path: aws-train-123\n",
            rendered,
        )

    def test_rejects_invalid_run_name_or_empty_image(self) -> None:
        with self.assertRaises(TrainingRecoveryError):
            render_job_template("__RUN_NAME__", run_name="INVALID", image="image")
        with self.assertRaises(TrainingRecoveryError):
            render_job_template("__TRAINING_IMAGE__", run_name="valid", image="")


class TrainingRecoveryLifecycleTests(unittest.TestCase):
    def harness(self) -> AWSTrainingRecoveryHarness:
        self.output = io.StringIO()
        return AWSTrainingRecoveryHarness(
            "test-context",
            Path("unused.yaml"),
            "example.test/training:sha",
            run_name="aws-train-test",
            reporter=ScenarioReporter("aws-train-test", output=self.output),
        )

    @staticmethod
    def pod(rank: int, uid: str, node: str) -> PodRecord:
        return PodRecord(
            name=f"rank-{rank}-{uid}",
            uid=uid,
            node=node,
            running=True,
            auto_remediate=True,
        )

    def test_runs_fault_resume_progress_and_cleanup_sequence(self) -> None:
        harness = self.harness()
        harness.reporter.emit = Mock(wraps=harness.reporter.emit)
        initial = [
            self.pod(rank, f"{rank}0000000-old-full-pod-uid", f"node-{rank}")
            for rank in range(TRAINING_RANKS)
        ]
        victim = initial[VICTIM_RANK]
        replacement = self.pod(VICTIM_RANK, "victim-new", "node-spare")
        resumed_victim = self.pod(VICTIM_RANK, "victim-final", "node-final")
        current_rank_zero = self.pod(0, "zero-new", "node-a")

        harness._submit = Mock()
        harness._wait_for_running_pod = Mock(
            side_effect=[*initial, replacement, resumed_victim, current_rank_zero]
        )
        harness._wait_for_smollm_cuda = Mock()
        harness._wait_for_progress = Mock(side_effect=[120, 130])
        harness._pod_logs = Mock(return_value="checkpoint_saved=step_000120.pt\n")
        harness._wait_for_daemon = Mock()
        harness._inject = Mock()
        harness._wait_for_healthy_node = Mock()
        harness._pause_worker = Mock()
        harness._wait_for_quiescence = Mock(return_value=(120, 120))
        harness._wait_for_degraded_node = Mock()
        harness._wait_for_uid_to_disappear = Mock()
        harness._wait_for_resume = Mock(side_effect=[120] * TRAINING_RANKS)
        harness._wait_for_recovered_node = Mock()
        harness._delete_job = Mock()
        harness._emit_diagnostics = Mock()

        harness.run()

        self.assertEqual(
            [(victim.node, 0), (victim.node, 79), (victim.node, 0)],
            [call.args for call in harness._inject.call_args_list],
        )
        harness._wait_for_uid_to_disappear.assert_called_once_with(victim.uid)
        harness._pause_worker.assert_called_once_with(victim.name)
        harness._wait_for_quiescence.assert_called_once_with(
            initial[0].name,
            victim,
            minimum_checkpoint=120,
        )
        harness._delete_job.assert_called_once_with()
        harness._emit_diagnostics.assert_not_called()
        transcript = self.output.getvalue()
        self.assertIn("[harness]", transcript)
        self.assertIn("Started SmolLM recovery test with 4 ranks", transcript)
        self.assertIn("[kubernetes]", transcript)
        self.assertIn("Placed rank 0→gpu-a", transcript)
        self.assertIn("[workload]", transcript)
        self.assertIn(
            "All 4 ranks are training SmolLM3-3B with CUDA/NCCL",
            transcript,
        )
        self.assertIn(
            "Paused rank 3 worker in pod 30000000 on gpu-d",
            transcript,
        )
        self.assertIn(
            "Pod 30000000 still reports Running; Kubernetes has not replaced it",
            transcript,
        )
        self.assertIn(
            "Training is stalled at step 120; checkpoint 120 is the recovery fence",
            transcript,
        )
        self.assertIn(
            "Injected synthetic XID 79 telemetry for GPU 0 on gpu-d using DCGM",
            transcript,
        )
        self.assertIn("[gpu-health-controller]", transcript)
        self.assertIn(
            "Isolated gpu-d after detecting critical XID 79",
            transcript,
        )
        self.assertIn("Replaced rank 3 on gpu-e with pod victim-n", transcript)
        self.assertIn(
            "All 4 ranks resumed from EFS checkpoint 120",
            transcript,
        )
        self.assertIn("Training advanced to step 130 after recovery", transcript)
        self.assertIn("PASS — detection", transcript)
        self.assertNotIn("running rank", transcript)
        self.assertNotIn(victim.uid, transcript)
        self.assertNotIn("[fault-injector", transcript)
        self.assertNotIn("PHASE", transcript)
        self.assertNotIn("ALIASES", transcript)
        self.assertNotIn("TRAINJOB_DELETED", transcript)
        running_event = next(
            call
            for call in harness.reporter.emit.call_args_list
            if call.args[1] == "PAUSED_POD_STILL_RUNNING"
        )
        self.assertEqual("Running", running_event.kwargs["pod_phase"])
        self.assertNotIn("phase", running_event.kwargs)

    def test_rejects_resume_from_a_checkpoint_other_than_the_fault_fence(self) -> None:
        harness = self.harness()
        initial = [
            self.pod(rank, f"{rank}-old", f"node-{rank}")
            for rank in range(TRAINING_RANKS)
        ]
        victim = initial[VICTIM_RANK]
        replacement = self.pod(VICTIM_RANK, "victim-new", "node-spare")

        harness._submit = Mock()
        harness._wait_for_running_pod = Mock(
            side_effect=[*initial, replacement]
        )
        harness._wait_for_smollm_cuda = Mock()
        harness._wait_for_progress = Mock(return_value=120)
        harness._pod_logs = Mock(return_value="checkpoint_saved=step_000120.pt\n")
        harness._wait_for_daemon = Mock()
        harness._inject = Mock()
        harness._wait_for_healthy_node = Mock()
        harness._pause_worker = Mock()
        harness._wait_for_quiescence = Mock(return_value=(120, 120))
        harness._wait_for_degraded_node = Mock()
        harness._wait_for_uid_to_disappear = Mock()
        harness._wait_for_resume = Mock(side_effect=[110] * TRAINING_RANKS)
        harness._emit_diagnostics = Mock()
        harness._best_effort_clear = Mock()
        harness._delete_job = Mock()

        with self.assertRaisesRegex(TrainingRecoveryError, "expected exact"):
            harness.run()

        harness._emit_diagnostics.assert_called_once_with()
        harness._best_effort_clear.assert_called_once_with(victim.node)
        harness._delete_job.assert_called_once_with()
        transcript = self.output.getvalue()
        self.assertIn(
            "Recovery test failed: ranks resumed checkpoint 110; expected exact",
            transcript,
        )
        self.assertIn("FAIL — total runtime", transcript)

    def test_rejects_final_rank_placement_on_faulted_node(self) -> None:
        harness = self.harness()
        initial = [
            self.pod(rank, f"{rank}-old", f"node-{rank}")
            for rank in range(TRAINING_RANKS)
        ]
        victim = initial[VICTIM_RANK]
        intermediate = self.pod(VICTIM_RANK, "victim-new", "node-spare")
        resumed_on_faulted = self.pod(
            VICTIM_RANK, "victim-final", victim.node
        )

        harness._submit = Mock()
        harness._wait_for_running_pod = Mock(
            side_effect=[*initial, intermediate, resumed_on_faulted]
        )
        harness._wait_for_smollm_cuda = Mock()
        harness._wait_for_progress = Mock(return_value=12)
        harness._pod_logs = Mock(return_value="checkpoint_saved=step_000010.pt\n")
        harness._wait_for_daemon = Mock()
        harness._inject = Mock()
        harness._wait_for_healthy_node = Mock()
        harness._pause_worker = Mock()
        harness._wait_for_quiescence = Mock(return_value=(12, 10))
        harness._wait_for_degraded_node = Mock()
        harness._wait_for_uid_to_disappear = Mock()
        harness._wait_for_resume = Mock(side_effect=[10] * TRAINING_RANKS)
        harness._emit_diagnostics = Mock()
        harness._best_effort_clear = Mock()
        harness._delete_job = Mock()

        with self.assertRaisesRegex(TrainingRecoveryError, "resumed rank 3 returned"):
            harness.run()

        harness._best_effort_clear.assert_called_once_with(victim.node)

    def test_quiescence_requires_same_running_stopped_pod_and_stable_fence(
        self,
    ) -> None:
        harness = self.harness()
        harness.quiescence_window = 0
        victim = self.pod(VICTIM_RANK, "victim-uid", "node-victim")
        harness._pods = Mock(return_value=[victim])
        harness._worker_state = Mock(return_value="T")
        harness._pod_logs = Mock(
            return_value="step=12 loss=1.0\ncheckpoint_saved=step_000010.pt\n"
        )

        def immediate_wait(operation, _description):
            return operation() or operation()

        harness._wait = immediate_wait

        self.assertEqual(
            (12, 10),
            harness._wait_for_quiescence(
                "rank-zero-pod",
                victim,
                minimum_checkpoint=10,
            ),
        )
        self.assertGreaterEqual(harness._pods.call_count, 2)
        self.assertGreaterEqual(harness._worker_state.call_count, 2)

    def test_pause_targets_worker_pid_and_confirms_linux_stopped_state(self) -> None:
        harness = self.harness()
        harness.kubectl.run = Mock(side_effect=["pid=42 signal=SIGSTOP\n", "T\n"])

        harness._pause_worker("rank-3-pod")

        pause_arguments = harness.kubectl.run.call_args_list[0].args[0]
        self.assertEqual("exec", pause_arguments[0])
        self.assertIn("rank-3-pod", pause_arguments)
        self.assertIn("signal.SIGSTOP", pause_arguments[-1])
        self.assertIn("/tmp/smollm-worker.pid", pause_arguments[-1])
        state_arguments = harness.kubectl.run.call_args_list[1].args[0]
        self.assertIn("/proc/$pid/status", state_arguments[-1])

    def test_quiescence_rejects_kubernetes_replacement_before_eviction(self) -> None:
        harness = self.harness()
        victim = self.pod(VICTIM_RANK, "victim-uid", "node-victim")
        replacement = self.pod(VICTIM_RANK, "replacement", "node-spare")
        harness._pods = Mock(return_value=[replacement])

        with self.assertRaisesRegex(
            TrainingRecoveryError,
            "pod changed before GPU health controller eviction",
        ):
            harness._wait_for_quiescence(
                "rank-zero-pod",
                victim,
                minimum_checkpoint=10,
            )


if __name__ == "__main__":
    unittest.main()
