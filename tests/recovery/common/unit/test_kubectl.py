import subprocess
import unittest
from unittest.mock import patch

from tests.recovery.common import RecoveryError
from tests.recovery.common.kubectl import Kubectl


class KubectlTests(unittest.TestCase):
    @patch("tests.recovery.common.kubectl.subprocess.run")
    def test_applies_namespace_and_standard_input(self, run) -> None:
        run.return_value = subprocess.CompletedProcess([], 0, "applied\n", "")
        client = Kubectl("test-context", "training")

        output = client.run(["apply", "-f", "-"], input_text="kind: Pod\n")

        self.assertEqual("applied\n", output)
        command = run.call_args.args[0]
        self.assertEqual(
            [
                "kubectl",
                "--context",
                "test-context",
                "--namespace",
                "training",
                "apply",
                "-f",
                "-",
            ],
            command,
        )
        self.assertEqual("kind: Pod\n", run.call_args.kwargs["input"])

    @patch("tests.recovery.common.kubectl.subprocess.run")
    def test_surfaces_command_failure(self, run) -> None:
        run.return_value = subprocess.CompletedProcess([], 1, "", "denied")
        client = Kubectl("test-context")

        with self.assertRaisesRegex(RecoveryError, "denied"):
            client.run(["get", "nodes"], namespaced=False)


if __name__ == "__main__":
    unittest.main()
