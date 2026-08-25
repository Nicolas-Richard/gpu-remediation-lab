import unittest
from pathlib import Path

from tests.recovery.aws_training.harness import TRAINING_RANKS


ROOT = Path(__file__).resolve().parents[4]


class SmolLMWorkloadContractTests(unittest.TestCase):
    def test_aws_job_is_pinned_real_workload_with_one_gpu_per_rank(self) -> None:
        manifest = (
            ROOT / "cluster/aws/manifests/distributed-training-job.yaml"
        ).read_text()

        self.assertIn(f"numNodes: {TRAINING_RANKS}", manifest)
        self.assertIn("/app/train.py", manifest)
        self.assertIn(
            "--model-revision=a07cc9a04f16550a088caea529712d1d335b0ac1",
            manifest,
        )
        self.assertIn(
            "--dataset-revision=5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
            manifest,
        )
        self.assertIn("--dataset-config=smol-magpie-ultra", manifest)
        self.assertIn("--worker-pid-file=/tmp/smollm-worker.pid", manifest)
        self.assertIn('nvidia.com/gpu: "1"', manifest)

    def test_cuda_image_uses_the_smollm_workload_build_context(self) -> None:
        dockerfile = (ROOT / "apps/smollm-training-workload/Dockerfile").read_text()
        workflow = (ROOT / "infra/aws/aws-workflow.mk").read_text()

        self.assertIn("COPY train.py", dockerfile)
        self.assertIn("COPY requirements.txt", dockerfile)
        self.assertIn("apps/smollm-training-workload/", workflow)
        self.assertNotIn("apps/local-training-workload/", workflow)


if __name__ == "__main__":
    unittest.main()
