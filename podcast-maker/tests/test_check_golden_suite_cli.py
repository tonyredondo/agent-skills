import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import check_golden_suite as golden_cli  # noqa: E402


class CheckGoldenSuiteCLITests(unittest.TestCase):
    def test_parse_args_uses_repo_default_candidate_dir(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(sys, "argv", ["check_golden_suite.py"]):
                args = golden_cli.parse_args()
        expected = os.path.join(os.path.dirname(SCRIPTS_DIR), ".golden_candidates")
        self.assertEqual(args.candidate_dir, expected)

    def test_main_returns_zero_when_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                candidate_dir=os.path.join(tmp, "candidates"),
                fixtures_dir=os.path.join(tmp, "fixtures"),
                baseline_path=None,
                allow_fixture_fallback=False,
                json_out="",
            )
            os.makedirs(args.fixtures_dir, exist_ok=True)
            expected_baseline = os.path.join(args.fixtures_dir, "baseline_metrics.json")
            with mock.patch.object(golden_cli, "parse_args", return_value=args):
                with mock.patch.object(
                    golden_cli,
                    "evaluate_golden_suite",
                    return_value={"overall_pass": True, "cases": []},
                ) as eval_mock:
                    rc = golden_cli.main()
            self.assertEqual(rc, 0)
            called = eval_mock.call_args.kwargs
            self.assertEqual(called["baseline_path"], expected_baseline)
            self.assertEqual(called["candidate_dir"], os.path.abspath(args.candidate_dir))
            self.assertIsNone(called["fixtures_dir"])

    def test_main_returns_one_when_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                candidate_dir=os.path.join(tmp, "candidates"),
                fixtures_dir=os.path.join(tmp, "fixtures"),
                baseline_path=None,
                allow_fixture_fallback=True,
                json_out="",
            )
            os.makedirs(args.fixtures_dir, exist_ok=True)
            with mock.patch.object(golden_cli, "parse_args", return_value=args):
                with mock.patch.object(
                    golden_cli,
                    "evaluate_golden_suite",
                    return_value={"overall_pass": False, "cases": [{"name": "x"}]},
                ):
                    rc = golden_cli.main()
            self.assertEqual(rc, 1)

    def test_main_writes_json_output_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "report.json")
            args = argparse.Namespace(
                candidate_dir=os.path.join(tmp, "candidates"),
                fixtures_dir=os.path.join(tmp, "fixtures"),
                baseline_path=os.path.join(tmp, "custom_baseline.json"),
                allow_fixture_fallback=True,
                json_out=out_path,
            )
            os.makedirs(args.fixtures_dir, exist_ok=True)
            report = {"overall_pass": True, "cases": [{"name": "ok"}]}
            with mock.patch.object(golden_cli, "parse_args", return_value=args):
                with mock.patch.object(golden_cli, "evaluate_golden_suite", return_value=report):
                    rc = golden_cli.main()
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["cases"][0]["name"], "ok")

    def test_main_real_fixture_fallback_passes_when_candidates_missing(self) -> None:
        base_dir = os.path.dirname(__file__)
        fixtures_dir = os.path.join(base_dir, "fixtures", "golden")
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                candidate_dir=os.path.join(tmp, "missing_candidates"),
                fixtures_dir=fixtures_dir,
                baseline_path=None,
                allow_fixture_fallback=True,
                json_out="",
            )
            with mock.patch.object(golden_cli, "parse_args", return_value=args):
                rc = golden_cli.main()
            self.assertEqual(rc, 0)

    def test_main_handles_evaluation_error_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "report.json")
            args = argparse.Namespace(
                candidate_dir=os.path.join(tmp, "candidates"),
                fixtures_dir=os.path.join(tmp, "fixtures"),
                baseline_path=os.path.join(tmp, "missing_baseline.json"),
                allow_fixture_fallback=False,
                json_out=report_path,
            )
            os.makedirs(args.fixtures_dir, exist_ok=True)
            with mock.patch.object(golden_cli, "parse_args", return_value=args):
                with mock.patch.object(
                    golden_cli,
                    "evaluate_golden_suite",
                    side_effect=FileNotFoundError("missing baseline"),
                ):
                    with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        rc = golden_cli.main()
            self.assertEqual(rc, 1)
            summary = json.loads(stdout.getvalue().strip())
            self.assertFalse(summary["overall_pass"])
            self.assertIn("error", summary)
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertIn("FileNotFoundError", str(report.get("error", "")))


if __name__ == "__main__":
    unittest.main()

