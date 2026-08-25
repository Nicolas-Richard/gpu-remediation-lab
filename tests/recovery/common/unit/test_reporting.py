import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests.recovery.common.reporting import NodeAliases, ScenarioReporter, short_uid


class NodeAliasTests(unittest.TestCase):
    def test_aliases_are_stable_in_first_observation_order(self) -> None:
        aliases = NodeAliases()

        self.assertEqual("gpu-a", aliases.get("node-z"))
        self.assertEqual("gpu-b", aliases.get("node-a"))
        self.assertEqual("gpu-a", aliases.get("node-z"))
        self.assertEqual(
            {"gpu-a": "node-z", "gpu-b": "node-a"}, aliases.mapping()
        )

    def test_short_uid_retains_human_scale_identity(self) -> None:
        self.assertEqual("3069aebf", short_uid("3069aebf-2bf5-4ed7-8128"))


class ScenarioReporterTests(unittest.TestCase):
    def test_renders_compose_style_and_writes_the_same_structured_event(self) -> None:
        output = io.StringIO()
        monotonic_values = iter((100.0, 102.25))
        now = datetime(2026, 8, 24, 15, 54, 14, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "events.jsonl"
            reporter = ScenarioReporter(
                "aws-train-test",
                jsonl_path=jsonl,
                output=output,
                wall_clock=lambda: now,
                monotonic=lambda: next(monotonic_values),
            )
            reporter.current_phase = "fault-injection"
            reporter.emit(
                "harness",
                "XID_INJECTED",
                message="Injected XID 79 into GPU 0 on gpu-d using DCGM",
                xid=79,
                device=0,
            )

            line = output.getvalue()
            self.assertIn("15:54:14 +002.2s [harness]", line)
            self.assertIn(
                "Injected XID 79 into GPU 0 on gpu-d using DCGM",
                line,
            )

            event = json.loads(jsonl.read_text())
            self.assertEqual("aws-train-test", event["run_id"])
            self.assertEqual("fault-injection", event["phase"])
            self.assertEqual("harness", event["component"])
            self.assertEqual("XID_INJECTED", event["event"])
            self.assertEqual(79, event["xid"])

    def test_hidden_events_remain_in_jsonl_and_verbose_mode(self) -> None:
        output = io.StringIO()
        ticks = iter((1.0, 1.0))

        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "events.jsonl"
            reporter = ScenarioReporter(
                "run",
                jsonl_path=jsonl,
                output=output,
                wall_clock=lambda: datetime(2026, 1, 1),
                monotonic=lambda: next(ticks),
            )
            reporter.emit(
                "kubernetes",
                "ALIASES",
                message="Recorded initial GPU node aliases",
                visible=False,
                nodes={"gpu-a": "node-a"},
            )

            self.assertEqual("", output.getvalue())
            self.assertEqual("ALIASES", json.loads(jsonl.read_text())["event"])

        verbose_output = io.StringIO()
        verbose_ticks = iter((1.0, 1.0))
        reporter = ScenarioReporter(
            "run",
            output=verbose_output,
            verbose=True,
            wall_clock=lambda: datetime(2026, 1, 1),
            monotonic=lambda: next(verbose_ticks),
        )
        reporter.emit(
            "kubernetes",
            "ALIASES",
            message="Recorded initial GPU node aliases",
            visible=False,
        )

        self.assertIn("Recorded initial GPU node aliases", verbose_output.getvalue())

    def test_raw_output_prefixes_every_source_line(self) -> None:
        output = io.StringIO()
        ticks = iter((1.0, 1.0, 1.0))
        reporter = ScenarioReporter(
            "run",
            output=output,
            wall_clock=lambda: datetime(2026, 1, 1),
            monotonic=lambda: next(ticks),
        )

        reporter.raw("train/rank-0", "first\nsecond\n")

        lines = output.getvalue().splitlines()
        self.assertEqual(2, len(lines))
        self.assertTrue(all("[train/rank-0" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
