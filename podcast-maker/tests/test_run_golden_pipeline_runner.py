import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import run_golden_pipeline as runner  # noqa: E402


class RunGoldenPipelineRunnerTests(unittest.TestCase):
    def test_load_cases_rejects_non_list_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cases.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"bad": 1}, f)
            with self.assertRaises(RuntimeError):
                runner._load_cases(path)

    def test_load_cases_rejects_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cases.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)
            with self.assertRaises(RuntimeError):
                runner._load_cases(path)

    def test_main_success_creates_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "src.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido")
            cases_path = os.path.join(tmp, "cases.json")
            with open(cases_path, "w", encoding="utf-8") as f:
                json.dump([{"case_name": "c1", "profile": "short", "source_path": source}], f)

            candidate_dir = os.path.join(tmp, "candidates")
            args = argparse.Namespace(
                cases_path=cases_path,
                candidate_dir=candidate_dir,
                debug=False,
                stop_on_error=False,
                report_json="",
            )

            def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
                with open(cmd[-1], "w", encoding="utf-8") as out:
                    json.dump(
                        {
                            "lines": [
                                {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "hola"}
                            ]
                        },
                        out,
                    )
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

            with mock.patch.object(runner, "parse_args", return_value=args):
                with mock.patch("run_golden_pipeline.subprocess.run", side_effect=fake_run):
                    rc = runner.main()

            self.assertEqual(rc, 0)
            report_path = os.path.join(candidate_dir, "golden_pipeline_run_report.json")
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertTrue(report["overall_pass"])
            self.assertEqual(report["failures"], 0)

    def test_main_stop_on_error_halts_without_running_next_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "src.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido")
            cases_path = os.path.join(tmp, "cases.json")
            with open(cases_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"case_name": "bad", "profile": "short"},
                        {"case_name": "good", "profile": "short", "source_path": source},
                    ],
                    f,
                )
            args = argparse.Namespace(
                cases_path=cases_path,
                candidate_dir=os.path.join(tmp, "candidates"),
                debug=False,
                stop_on_error=True,
                report_json="",
            )
            with mock.patch.object(runner, "parse_args", return_value=args):
                with mock.patch("run_golden_pipeline.subprocess.run") as run_mock:
                    rc = runner.main()
            self.assertEqual(rc, 1)
            run_mock.assert_not_called()

    def test_main_debug_adds_debug_flag_to_subprocess_cmd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "src.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido")
            cases_path = os.path.join(tmp, "cases.json")
            with open(cases_path, "w", encoding="utf-8") as f:
                json.dump([{"case_name": "c1", "profile": "short", "source_path": source}], f)
            args = argparse.Namespace(
                cases_path=cases_path,
                candidate_dir=os.path.join(tmp, "candidates"),
                debug=True,
                stop_on_error=False,
                report_json="",
            )

            seen = {"cmd": None}

            def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
                seen["cmd"] = cmd
                with open(cmd[-1], "w", encoding="utf-8") as out:
                    json.dump(
                        {"lines": [{"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "hola"}]},
                        out,
                    )
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

            with mock.patch.object(runner, "parse_args", return_value=args):
                with mock.patch("run_golden_pipeline.subprocess.run", side_effect=fake_run):
                    rc = runner.main()

            self.assertEqual(rc, 0)
            self.assertIsNotNone(seen["cmd"])
            self.assertIn("--debug", seen["cmd"])
            self.assertNotIn("--engine", seen["cmd"])

    def test_main_subprocess_failure_marks_case_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "src.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido")
            cases_path = os.path.join(tmp, "cases.json")
            with open(cases_path, "w", encoding="utf-8") as f:
                json.dump([{"case_name": "c1", "profile": "short", "source_path": source}], f)
            args = argparse.Namespace(
                cases_path=cases_path,
                candidate_dir=os.path.join(tmp, "candidates"),
                debug=False,
                stop_on_error=False,
                report_json="",
            )
            fail_proc = subprocess.CompletedProcess(args=["cmd"], returncode=1, stdout="", stderr="bad")
            with mock.patch.object(runner, "parse_args", return_value=args):
                with mock.patch("run_golden_pipeline.subprocess.run", return_value=fail_proc):
                    rc = runner.main()
            self.assertEqual(rc, 1)

    def test_main_passes_optional_length_flags_and_custom_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "src.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("contenido")
            cases_path = os.path.join(tmp, "cases.json")
            with open(cases_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "case_name": "c1",
                            "profile": "long",
                            "source_path": source,
                            "target_minutes": 25,
                            "words_per_min": 140,
                            "min_words": 250,
                            "max_words": 320,
                        }
                    ],
                    f,
                )
            report_json = os.path.join(tmp, "reports", "custom_report.json")
            args = argparse.Namespace(
                cases_path=cases_path,
                candidate_dir=os.path.join(tmp, "candidates"),
                debug=True,
                stop_on_error=False,
                report_json=report_json,
            )
            seen = {"cmd": None}

            def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
                seen["cmd"] = cmd
                with open(cmd[-1], "w", encoding="utf-8") as out:
                    json.dump(
                        {"lines": [{"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "hola"}]},
                        out,
                    )
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

            with mock.patch.object(runner, "parse_args", return_value=args):
                with mock.patch("run_golden_pipeline.subprocess.run", side_effect=fake_run):
                    rc = runner.main()

            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(report_json))
            self.assertIsNotNone(seen["cmd"])
            cmd = seen["cmd"]
            self.assertIn("--debug", cmd)
            self.assertIn("--target-minutes", cmd)
            self.assertIn("25", cmd)
            self.assertIn("--words-per-min", cmd)
            self.assertIn("140", cmd)
            self.assertIn("--min-words", cmd)
            self.assertIn("250", cmd)
            self.assertIn("--max-words", cmd)
            self.assertIn("320", cmd)


if __name__ == "__main__":
    unittest.main()

