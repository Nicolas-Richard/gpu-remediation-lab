import unittest

from tests.recovery.aws_training.cli import DEFAULT_JOB_TEMPLATE, parse_args


class TrainingRecoveryCLITests(unittest.TestCase):
    def test_defaults_point_to_checked_in_job_template(self) -> None:
        args = parse_args(["--image=example.test/training:sha"])

        self.assertEqual(DEFAULT_JOB_TEMPLATE, args.job_template)
        self.assertTrue(args.job_template.is_file(), args.job_template)
        self.assertEqual(10, args.after_step)
        self.assertEqual(1800, args.timeout)
        self.assertIsNone(args.events_jsonl)
        self.assertFalse(args.verbose)

    def test_accepts_structured_event_output_path(self) -> None:
        args = parse_args(
            [
                "--image=example.test/training:sha",
                "--events-jsonl=/tmp/aws-training-events.jsonl",
                "--verbose",
            ]
        )

        self.assertEqual(
            "/tmp/aws-training-events.jsonl",
            str(args.events_jsonl),
        )
        self.assertTrue(args.verbose)


if __name__ == "__main__":
    unittest.main()
