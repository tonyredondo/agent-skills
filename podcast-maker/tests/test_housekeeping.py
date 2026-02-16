import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.housekeeping import cleanup_dir  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.config import LoggingConfig  # noqa: E402


class HousekeepingTests(unittest.TestCase):
    def test_failed_recent_artifacts_protected_unless_force_clean(self) -> None:
        logger = Logger.create(LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False))
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "ep1")
            os.makedirs(run_dir, exist_ok=True)
            summary_path = os.path.join(run_dir, "run_summary.json")
            artifact_path = os.path.join(run_dir, "artifact.tmp")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"status": "failed"}, f)
            with open(artifact_path, "w", encoding="utf-8") as f:
                f.write("x")
            # Make artifact old enough to be deletion candidate.
            old = time.time() - (3 * 86400)
            os.utime(artifact_path, (old, old))

            cleanup_dir(
                base_dir=tmp,
                retention_days=1,
                retention_log_days=1,
                retention_intermediate_audio_days=1,
                max_storage_mb=10,
                max_log_storage_mb=10,
                logger=logger,
                dry_run=False,
                force_clean=False,
            )
            self.assertTrue(os.path.exists(artifact_path))

            cleanup_dir(
                base_dir=tmp,
                retention_days=1,
                retention_log_days=1,
                retention_intermediate_audio_days=1,
                max_storage_mb=10,
                max_log_storage_mb=10,
                logger=logger,
                dry_run=False,
                force_clean=True,
            )
            self.assertFalse(os.path.exists(artifact_path))

    def test_cleanup_ignores_file_disappearing_during_sort(self) -> None:
        logger = Logger.create(LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False))
        with tempfile.TemporaryDirectory() as tmp:
            keep_path = os.path.join(tmp, "keep.txt")
            race_path = os.path.join(tmp, "race.txt")
            with open(keep_path, "w", encoding="utf-8") as f:
                f.write("a")
            with open(race_path, "w", encoding="utf-8") as f:
                f.write("b")

            original_getmtime = os.path.getmtime
            first_race = {"raised": False}

            def flaky_getmtime(path):  # noqa: ANN001
                if path == race_path and not first_race["raised"]:
                    first_race["raised"] = True
                    raise FileNotFoundError("simulated concurrent deletion")
                return original_getmtime(path)

            with mock.patch("pipeline.housekeeping.os.path.getmtime", side_effect=flaky_getmtime):
                report = cleanup_dir(
                    base_dir=tmp,
                    retention_days=1,
                    retention_log_days=1,
                    retention_intermediate_audio_days=1,
                    max_storage_mb=10,
                    max_log_storage_mb=10,
                    logger=logger,
                    dry_run=False,
                    force_clean=False,
                )
            self.assertIsNotNone(report)


if __name__ == "__main__":
    unittest.main()

