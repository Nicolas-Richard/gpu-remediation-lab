from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("dcgm_metrics_simulator", APP_PATH)
assert SPEC is not None and SPEC.loader is not None
simulator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(simulator)


class SimulatorTest(unittest.TestCase):
    def test_state_starts_absent_and_can_be_set_and_cleared(self) -> None:
        state = simulator.XIDState()

        self.assertIsNone(state.get())
        state.set(79)
        self.assertEqual(79, state.get())
        state.clear()
        self.assertIsNone(state.get())

    def test_absent_observation_omits_xid_metric(self) -> None:
        metrics = simulator.render_metrics(None, "worker-a")

        self.assertNotIn(simulator.METRIC_NAME, metrics)

    def test_xid_observation_uses_dcgm_metric_shape(self) -> None:
        metrics = simulator.render_metrics(79, "worker-a")

        self.assertIn("# TYPE DCGM_FI_DEV_XID_ERRORS gauge", metrics)
        self.assertIn('UUID="GPU-SIM-worker-a-0"', metrics)
        self.assertIn('Hostname="worker-a"', metrics)
        self.assertTrue(metrics.endswith(" 79\n"), metrics)

    def test_payload_requires_one_bounded_integer_xid(self) -> None:
        self.assertEqual(0, simulator.parse_state_payload(b'{"xid": 0}'))
        self.assertEqual(79, simulator.parse_state_payload(b'{"xid": 79}'))

        invalid = [
            b"not-json",
            b"{}",
            b'{"xid": true}',
            b'{"xid": 1.5}',
            b'{"xid": -1}',
            b'{"xid": 1000}',
            b'{"xid": 79, "state": "degraded"}',
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    simulator.parse_state_payload(payload)


if __name__ == "__main__":
    unittest.main()
