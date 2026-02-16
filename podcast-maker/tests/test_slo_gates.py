import os
import json
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.slo_gates import append_slo_event, evaluate_slo_windows  # noqa: E402


class SLOGatesTests(unittest.TestCase):
    def test_trigger_rollback_after_two_failed_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = os.path.join(tmp, "history.jsonl")
            # Two windows of 4 events each, all failed and slow.
            for _ in range(8):
                append_slo_event(
                    profile="short",
                    component="script",
                    status="failed",
                    elapsed_seconds=999,
                    history_path=history,
                )
            report = evaluate_slo_windows(
                profile="short",
                component="script",
                history_path=history,
                window_size=4,
                required_failed_windows=2,
            )
            self.assertTrue(report["has_data"])
            self.assertTrue(report["should_rollback"])

    def test_no_rollback_when_successful(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = os.path.join(tmp, "history.jsonl")
            for _ in range(10):
                append_slo_event(
                    profile="standard",
                    component="audio",
                    status="completed",
                    elapsed_seconds=120,
                    history_path=history,
                )
            report = evaluate_slo_windows(
                profile="standard",
                component="audio",
                history_path=history,
                window_size=5,
                required_failed_windows=2,
            )
            self.assertFalse(report["should_rollback"])

    def test_retry_rate_kpi_can_trigger_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = os.path.join(tmp, "history.jsonl")
            # All completed and fast, but retry rate too high.
            for _ in range(8):
                append_slo_event(
                    profile="standard",
                    component="audio",
                    status="completed",
                    elapsed_seconds=60,
                    retry_rate=0.5,
                    history_path=history,
                )
            report = evaluate_slo_windows(
                profile="standard",
                component="audio",
                history_path=history,
                window_size=4,
                required_failed_windows=2,
            )
            self.assertTrue(report["should_rollback"])
            self.assertFalse(report["window_reports"][0]["retry_rate_ok"])

    def test_resume_runtime_slo_triggers_when_resume_runs_are_slow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = os.path.join(tmp, "history.jsonl")
            # Resume runs are slower than short-profile resume SLO (4 minutes).
            for _ in range(8):
                append_slo_event(
                    profile="short",
                    component="script",
                    status="completed",
                    elapsed_seconds=400,  # below total SLO (8 min), above resume SLO (4 min)
                    is_resume=True,
                    history_path=history,
                )
            report = evaluate_slo_windows(
                profile="short",
                component="script",
                history_path=history,
                window_size=4,
                required_failed_windows=2,
            )
            self.assertTrue(report["should_rollback"])
            self.assertFalse(report["window_reports"][0]["resume_runtime_ok"])

    def test_cost_error_uses_p90_not_p95(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = os.path.join(tmp, "history.jsonl")
            # One outlier should not break p90 in a 10-item window.
            for idx in range(10):
                append_slo_event(
                    profile="standard",
                    component="audio",
                    status="completed",
                    elapsed_seconds=100,
                    cost_estimation_error_pct=(100.0 if idx == 9 else 0.0),
                    history_path=history,
                )
            report = evaluate_slo_windows(
                profile="standard",
                component="audio",
                history_path=history,
                window_size=10,
                required_failed_windows=1,
            )
            self.assertFalse(report["should_rollback"])
            self.assertTrue(report["window_reports"][0]["cost_error_ok"])

    def test_append_slo_event_writes_failure_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = os.path.join(tmp, "history.jsonl")
            append_slo_event(
                profile="standard",
                component="audio",
                status="failed",
                elapsed_seconds=10,
                failure_kind="TIMEOUT",
                history_path=history,
            )
            with open(history, "r", encoding="utf-8") as f:
                event = json.loads(f.readline())
            self.assertEqual(event.get("failure_kind"), "timeout")

    def test_script_quality_rejected_rate_kpi_can_trigger_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = os.path.join(tmp, "history.jsonl")
            for idx in range(10):
                append_slo_event(
                    profile="standard",
                    component="audio",
                    status="failed" if idx < 7 else "completed",
                    elapsed_seconds=120,
                    failure_kind=("script_quality_rejected" if idx < 7 else None),
                    history_path=history,
                )
            report = evaluate_slo_windows(
                profile="standard",
                component="audio",
                history_path=history,
                window_size=10,
                required_failed_windows=1,
            )
            self.assertTrue(report["should_rollback"])
            self.assertFalse(report["window_reports"][0]["script_quality_rejected_ok"])


if __name__ == "__main__":
    unittest.main()

