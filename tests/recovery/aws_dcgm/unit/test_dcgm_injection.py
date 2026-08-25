import unittest
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from tests.recovery.aws_dcgm import DCGMInjectionError
from tests.recovery.aws_dcgm.dcgm_injection import (
    DCGMInjectionScenario,
    emit,
)
from tests.recovery.common.reporting import format_duration


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


class DCGMInjectionParsingTests(unittest.TestCase):
    def test_emit_uses_time_only(self) -> None:
        output = StringIO()
        with patch(
            "tests.recovery.aws_dcgm.dcgm_injection.datetime"
        ) as mocked_datetime:
            mocked_datetime.now.return_value = datetime(2026, 8, 24, 9, 42, 7)
            emit("PASS", "example assertion", output=output)

        self.assertEqual(output.getvalue(), "09:42:07 PASS: example assertion\n")

    def test_duration_is_reported_in_seconds(self) -> None:
        self.assertEqual(format_duration(65.24), "65.2s")

    def test_run_emits_start_and_failed_end_records(self) -> None:
        scenario = DCGMInjectionScenario("test-context", Path("unused.yaml"))
        scenario.wait = Mock(side_effect=DCGMInjectionError("no GPU capacity"))
        output = StringIO()

        with patch(
            "tests.recovery.aws_dcgm.dcgm_injection.datetime"
        ) as mocked_datetime, patch(
            "tests.recovery.aws_dcgm.dcgm_injection.time.monotonic",
            side_effect=[10.0, 12.5],
        ):
            mocked_datetime.now.return_value = datetime(2026, 8, 24, 9, 42, 7)
            with redirect_stdout(output), self.assertRaises(DCGMInjectionError):
                scenario.run()

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "09:42:07 START: AWS DCGM injection scenario (XID 79)",
                "09:42:07 END: AWS DCGM injection scenario FAIL (total duration: 2.5s)",
            ],
        )

    def test_gpu_capacity_is_pending_until_minimum_nodes_are_ready(self) -> None:
        scenario = DCGMInjectionScenario(
            "test-context",
            Path("unused.yaml"),
            minimum_gpu_nodes=2,
        )
        scenario.kubectl = Mock()
        scenario.kubectl.json.side_effect = [
            {"items": [gpu_node("one")]},
            {"items": [gpu_node("one"), gpu_node("two")]},
        ]

        self.assertIsNone(scenario._gpu_capacity())
        self.assertEqual(scenario._gpu_capacity(), 2)

if __name__ == "__main__":
    unittest.main()
