import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.golden_gate import evaluate_golden_suite  # noqa: E402


class GoldenSuiteGateTests(unittest.TestCase):
    def test_golden_suite_passes(self) -> None:
        base_dir = os.path.dirname(__file__)
        fixtures_dir = os.path.join(base_dir, "fixtures", "golden")
        baseline_path = os.path.join(fixtures_dir, "baseline_metrics.json")
        report = evaluate_golden_suite(
            baseline_path=baseline_path,
            candidate_dir=fixtures_dir,
        )
        self.assertTrue(report["overall_pass"], msg=str(report))
        self.assertGreaterEqual(len(report["cases"]), 5)

    def test_golden_suite_fails_when_candidates_missing(self) -> None:
        base_dir = os.path.dirname(__file__)
        fixtures_dir = os.path.join(base_dir, "fixtures", "golden")
        baseline_path = os.path.join(fixtures_dir, "baseline_metrics.json")
        missing_dir = os.path.join(base_dir, "fixtures", "does_not_exist")
        report = evaluate_golden_suite(
            baseline_path=baseline_path,
            candidate_dir=missing_dir,
        )
        self.assertFalse(report["overall_pass"])
        self.assertTrue(any(c["comparison"].get("error") == "missing_candidate" for c in report["cases"]))


if __name__ == "__main__":
    unittest.main()

