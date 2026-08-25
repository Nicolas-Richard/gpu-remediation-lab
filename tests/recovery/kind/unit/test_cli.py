import unittest

from tests.recovery.kind.cli import DEFAULT_MANIFEST, parse_args


class CliDefaultsTest(unittest.TestCase):
    def test_default_manifest_points_to_demo_workload(self) -> None:
        args = parse_args([])

        self.assertEqual(DEFAULT_MANIFEST, args.manifest)
        self.assertTrue(args.manifest.is_file(), args.manifest)
        self.assertEqual("demo-job.yaml", args.manifest.name)
        self.assertEqual("annotation", args.health_source)

    def test_accepts_dcgm_simulator_health_source(self) -> None:
        args = parse_args(["--health-source=dcgm-simulator"])

        self.assertEqual("dcgm-simulator", args.health_source)


if __name__ == "__main__":
    unittest.main()
