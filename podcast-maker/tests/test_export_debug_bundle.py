import json
import os
import sys
import tempfile
import unittest
import zipfile
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from export_debug_bundle import create_debug_bundle, parse_args  # noqa: E402


class ExportDebugBundleTests(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def test_create_debug_bundle_includes_expected_files_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_abc"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            audio_run = os.path.join(audio_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)
            os.makedirs(audio_run, exist_ok=True)

            self._write_json(os.path.join(script_run, "script_checkpoint.json"), {"status": "completed"})
            self._write_json(os.path.join(script_run, "run_summary.json"), {"word_count": 1234})
            self._write_json(os.path.join(audio_run, "audio_manifest.json"), {"status": "completed"})
            self._write_json(os.path.join(audio_run, "podcast_run_summary.json"), {"status": "completed"})

            script_path = os.path.join(tmp, "script.json")
            source_path = os.path.join(tmp, "source.txt")
            log_path = os.path.join(tmp, "podcast.log")
            self._write_json(script_path, {"lines": []})
            with open(source_path, "w", encoding="utf-8") as f:
                f.write("fuente base de prueba\n")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("log line\n")

            output_zip = os.path.join(tmp, "bundle.zip")
            argv = [
                episode,
                "--script-checkpoint-dir",
                script_ckpt,
                "--audio-checkpoint-dir",
                audio_ckpt,
                "--script-path",
                script_path,
                "--source-path",
                source_path,
                "--log-path",
                log_path,
                "--output",
                output_zip,
            ]
            args = parse_args(argv)
            with mock.patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "secret-key",
                    "OPENAI_AUTH_TOKEN": "also-secret",
                    "USER_PASSWORD": "super-secret",
                    "PODCAST_MAKER_VERSION": "9.9.9-test",
                    "SCRIPT_TIMEOUT_SECONDS": "180",
                    "TTS_TIMEOUT_SECONDS": "90",
                },
                clear=False,
            ):
                out = create_debug_bundle(args)

            self.assertEqual(out, os.path.abspath(output_zip))
            self.assertTrue(os.path.exists(out))
            with zipfile.ZipFile(out, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("debug_bundle_metadata.json", names)
                self.assertIn("debug_bundle_tree.txt", names)
                self.assertTrue(any(name.endswith("script_checkpoint.json") for name in names))
                self.assertTrue(any(name.endswith("audio_manifest.json") for name in names))
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            self.assertIn("env", metadata)
            self.assertNotIn("OPENAI_API_KEY", metadata["env"])
            self.assertNotIn("OPENAI_AUTH_TOKEN", metadata["env"])
            self.assertNotIn("USER_PASSWORD", metadata["env"])
            self.assertEqual(metadata["env"].get("SCRIPT_TIMEOUT_SECONDS"), "180")
            self.assertEqual(metadata.get("skill_version"), "9.9.9-test")
            self.assertIn("git_commit", metadata)
            self.assertIn("effective_params", metadata)
            self.assertIn("collection_complete", metadata)
            self.assertIn("collection_status_counts", metadata)
            self.assertEqual(
                metadata.get("effective_params", {}).get("episode_id"),
                episode,
            )
            log_paths = metadata.get("effective_params", {}).get("log_paths", [])
            self.assertIn(os.path.abspath(log_path), log_paths)
            self.assertTrue(str(metadata.get("invocation", "")).strip())
            self.assertTrue(len(metadata.get("included_files", [])) >= 4)

    def test_create_debug_bundle_records_missing_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_missing"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            os.makedirs(script_ckpt, exist_ok=True)
            os.makedirs(audio_ckpt, exist_ok=True)
            output_zip = os.path.join(tmp, "bundle_missing.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            self.assertGreater(len(metadata.get("missing_candidates", [])), 0)
            self.assertFalse(bool(metadata.get("collection_complete", True)))
            status_counts = metadata.get("collection_status_counts", {})
            self.assertIsInstance(status_counts, dict)
            self.assertGreater(int(status_counts.get("missing", 0)), 0)

    def test_create_debug_bundle_records_resolved_episode_id_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            requested_episode = "episode_requested"
            resolved_episode = "episode_real"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, requested_episode), exist_ok=True)
            os.makedirs(os.path.join(script_ckpt, resolved_episode), exist_ok=True)
            os.makedirs(os.path.join(audio_ckpt, resolved_episode), exist_ok=True)
            self._write_json(
                os.path.join(script_ckpt, requested_episode, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": resolved_episode,
                    "script_checkpoint_dir": script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                },
            )
            self._write_json(
                os.path.join(script_ckpt, resolved_episode, "run_summary.json"),
                {"episode_id": resolved_episode, "status": "completed"},
            )
            output_zip = os.path.join(tmp, "bundle_resolved_episode.zip")
            args = parse_args(
                [
                    requested_episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            self.assertEqual(metadata.get("episode_id"), requested_episode)
            self.assertEqual(metadata.get("resolved_episode_id"), resolved_episode)

    def test_create_debug_bundle_resolves_relative_manifest_pointer_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_relative_pointer"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            audio_run = os.path.join(audio_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)
            os.makedirs(audio_run, exist_ok=True)
            reports_dir = os.path.join(script_run, "reports")
            os.makedirs(reports_dir, exist_ok=True)

            self._write_json(
                os.path.join(reports_dir, "quality_report.json"),
                {"status": "completed", "source": "relative_manifest_pointer"},
            )
            self._write_json(
                os.path.join(script_run, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": episode,
                    "script_checkpoint_dir": script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                    "script": {"quality_report_path": "reports/quality_report.json"},
                },
            )
            output_zip = os.path.join(tmp, "bundle_relative_pointer.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            included = list(metadata.get("included_files", []))
            self.assertTrue(any("reports__quality_report.json" in name for name in included))

    def test_create_debug_bundle_resolves_relative_audio_manifest_pointer_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_relative_audio_pointer"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            audio_run = os.path.join(audio_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)
            os.makedirs(audio_run, exist_ok=True)
            reports_dir = os.path.join(audio_run, "reports")
            os.makedirs(reports_dir, exist_ok=True)

            self._write_json(
                os.path.join(reports_dir, "podcast_run_summary.json"),
                {
                    "status": "failed",
                    "audio_executed": True,
                    "failure_kind": "timeout",
                },
            )
            self._write_json(
                os.path.join(script_run, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": episode,
                    "script_checkpoint_dir": script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                    "status_by_stage": {"script": "completed", "audio": "failed"},
                    "audio": {"podcast_run_summary_path": "reports/podcast_run_summary.json"},
                },
            )
            output_zip = os.path.join(tmp, "bundle_relative_audio_pointer.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            included = list(metadata.get("included_files", []))
            self.assertTrue(any("reports__podcast_run_summary.json" in name for name in included))

    def test_create_debug_bundle_records_read_error_for_directory_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_read_error_log"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, episode), exist_ok=True)
            os.makedirs(os.path.join(audio_ckpt, episode), exist_ok=True)
            log_dir = os.path.join(tmp, "logs_dir_only")
            os.makedirs(log_dir, exist_ok=True)

            output_zip = os.path.join(tmp, "bundle_read_error_log.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--log-path",
                    log_dir,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            status_counts = dict(metadata.get("collection_status_counts", {}))
            self.assertGreater(int(status_counts.get("read_error", 0)), 0)

    def test_create_debug_bundle_deduplicates_overlapping_log_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_log_dedup"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, episode), exist_ok=True)
            os.makedirs(os.path.join(audio_ckpt, episode), exist_ok=True)
            run_log = os.path.join(tmp, "run.log")
            with open(run_log, "w", encoding="utf-8") as f:
                f.write("line\n")

            output_zip = os.path.join(tmp, "bundle_log_dedup.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--log-path",
                    "./run.log",
                    "--output",
                    output_zip,
                ]
            )
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                out = create_debug_bundle(args)
            finally:
                os.chdir(cwd)
            with zipfile.ZipFile(out, "r") as zf:
                collection_report = json.loads(zf.read("collection_report.json").decode("utf-8"))
            run_log_entries = [
                item
                for item in collection_report
                if str(item.get("category", "")) == "logs" and str(item.get("archive_name", "")) == "run.log"
            ]
            self.assertEqual(len(run_log_entries), 1)
            self.assertEqual(str(run_log_entries[0].get("status", "")), "found")

    def test_collection_complete_is_false_when_read_error_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_collection_read_error"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            audio_run = os.path.join(audio_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)
            os.makedirs(audio_run, exist_ok=True)

            # Provide required artefacts so read_error is the decisive negative state.
            self._write_json(os.path.join(script_run, "script_checkpoint.json"), {"status": "completed", "lines": []})
            self._write_json(
                os.path.join(script_run, "run_summary.json"),
                {"status": "completed", "quality_gate_executed": True},
            )
            self._write_json(os.path.join(script_run, "quality_report.json"), {"pass": True})
            self._write_json(
                os.path.join(script_run, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": episode,
                    "script_checkpoint_dir": script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                    "status_by_stage": {"script": "completed", "audio": "completed"},
                },
            )
            self._write_json(os.path.join(script_run, "pipeline_summary.json"), {"overall_status": "completed"})
            self._write_json(os.path.join(audio_run, "audio_manifest.json"), {"status": "completed"})
            self._write_json(os.path.join(audio_run, "run_summary.json"), {"status": "completed"})
            self._write_json(
                os.path.join(audio_run, "podcast_run_summary.json"),
                {"status": "completed", "audio_executed": True},
            )
            self._write_json(os.path.join(audio_run, "quality_report.json"), {"pass": True})
            self._write_json(os.path.join(audio_run, "normalized_script.json"), {"lines": []})

            bad_log_path = os.path.join(tmp, "logs_dir_only")
            os.makedirs(bad_log_path, exist_ok=True)
            output_zip = os.path.join(tmp, "bundle_collection_read_error.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--log-path",
                    bad_log_path,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            self.assertFalse(bool(metadata.get("collection_complete", True)))
            status_counts = dict(metadata.get("collection_status_counts", {}))
            self.assertGreater(int(status_counts.get("read_error", 0)), 0)

    def test_create_debug_bundle_uses_manifest_run_summary_pointer_before_quality_applicability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_manifest_run_summary_pointer"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            reports_dir = os.path.join(script_run, "reports")
            os.makedirs(reports_dir, exist_ok=True)

            self._write_json(
                os.path.join(script_run, "script_checkpoint.json"),
                {"lines": [{"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "hola"}]},
            )
            self._write_json(
                os.path.join(reports_dir, "run_summary.json"),
                {
                    "status": "failed",
                    "quality_gate_executed": False,
                },
            )
            self._write_json(
                os.path.join(script_run, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": episode,
                    "script_checkpoint_dir": script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                    "status_by_stage": {"script": "failed", "audio": "not_started"},
                    "script": {
                        "run_summary_path": "reports/run_summary.json",
                        "quality_report_path": "reports/quality_report.json",
                    },
                },
            )
            self._write_json(
                os.path.join(script_run, "pipeline_summary.json"),
                {"overall_status": "failed"},
            )

            output_zip = os.path.join(tmp, "bundle_manifest_run_summary_pointer.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
                collection_report = json.loads(zf.read("collection_report.json").decode("utf-8"))
            self.assertTrue(bool(metadata.get("collection_complete", False)))
            self.assertEqual(int(metadata.get("collection_status_counts", {}).get("missing", 0)), 0)
            self.assertTrue(
                any(
                    item.get("status") == "not_applicable"
                    and str(item.get("reason", "")) == "manifest_pointer_override"
                    and str(item.get("path", "")).endswith("run_summary.json")
                    and str(item.get("category", "")) == "script_checkpoint"
                    for item in collection_report
                )
            )
            self.assertTrue(
                any(
                    item.get("status") == "not_applicable"
                    and str(item.get("reason", "")) == "script_quality_gate_not_executed"
                    and str(item.get("archive_name", "")).endswith("quality_report.json")
                    for item in collection_report
                )
            )

    def test_create_debug_bundle_marks_script_quality_report_not_applicable_when_gate_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_gate_off"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)

            self._write_json(
                os.path.join(script_run, "script_checkpoint.json"),
                {"lines": [{"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "hola"}]},
            )
            self._write_json(
                os.path.join(script_run, "run_summary.json"),
                {
                    "status": "failed",
                    "quality_gate_executed": False,
                },
            )
            self._write_json(
                os.path.join(script_run, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": episode,
                    "script_checkpoint_dir": script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                    "status_by_stage": {"script": "failed", "audio": "not_started"},
                },
            )
            self._write_json(
                os.path.join(script_run, "pipeline_summary.json"),
                {"overall_status": "failed"},
            )

            output_zip = os.path.join(tmp, "bundle_gate_off.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
                collection_report = json.loads(zf.read("collection_report.json").decode("utf-8"))
            self.assertTrue(bool(metadata.get("collection_complete", False)))
            self.assertTrue(
                any(
                    item.get("status") == "not_applicable"
                    and str(item.get("reason", "")) == "script_quality_gate_not_executed"
                    and str(item.get("path", "")).endswith("quality_report.json")
                    for item in collection_report
                )
            )
            self.assertTrue(
                all(
                    str(item.get("status", "")).strip().lower() in {"found", "not_applicable"}
                    for item in collection_report
                )
            )
            self.assertTrue(
                any(
                    item.get("status") == "not_applicable"
                    and str(item.get("reason", "")) == "log_candidate_not_found"
                    and str(item.get("category", "")) == "logs"
                    for item in collection_report
                )
            )

    def test_create_debug_bundle_keeps_initial_quality_report_when_stage_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_quality_interrupted"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)
            self._write_json(
                os.path.join(script_run, "script_checkpoint.json"),
                {"lines": [{"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "contenido"}]},
            )
            self._write_json(
                os.path.join(script_run, "run_summary.json"),
                {
                    "status": "interrupted",
                    "quality_gate_executed": True,
                    "quality_stage_started": True,
                    "quality_stage_finished": False,
                    "quality_stage_interrupted": True,
                },
            )
            self._write_json(
                os.path.join(script_run, "quality_report_initial.json"),
                {"status": "failed", "quality_report_phase": "initial"},
            )
            self._write_json(
                os.path.join(script_run, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": episode,
                    "script_checkpoint_dir": script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                    "status_by_stage": {"script": "interrupted", "audio": "not_started"},
                    "script": {
                        "quality_stage_started": True,
                        "quality_stage_finished": False,
                        "quality_stage_interrupted": True,
                    },
                },
            )
            self._write_json(os.path.join(script_run, "pipeline_summary.json"), {"overall_status": "interrupted"})
            output_zip = os.path.join(tmp, "bundle_quality_interrupted.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
                collection_report = json.loads(zf.read("collection_report.json").decode("utf-8"))
            self.assertTrue(
                any(
                    item.get("status") == "found"
                    and str(item.get("archive_name", "")).endswith("quality_report_initial.json")
                    for item in collection_report
                )
            )
            self.assertTrue(
                any(
                    item.get("status") == "found"
                    and str(item.get("archive_name", "")).endswith("quality_report_initial.json")
                    and str(item.get("reason", "")) == "initial_only_due_to_interruption"
                    for item in collection_report
                )
            )
            self.assertFalse(
                any(
                    item.get("status") == "missing"
                    and str(item.get("archive_name", "")).endswith("quality_report.json")
                    for item in collection_report
                )
            )
            self.assertTrue(bool(metadata.get("collection_complete", False)))
            diagnostics = dict(metadata.get("collection_diagnostics", {}))
            self.assertTrue(bool(diagnostics))
            self.assertEqual(bool(diagnostics.get("script_quality_stage_interrupted")), True)

    def test_create_debug_bundle_autodiscovers_checkpoint_root_from_neighbor_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_root_discovery"
            real_script_ckpt = os.path.join(tmp, "script_ckpt")
            wrong_script_ckpt = os.path.join(tmp, "wrong_script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(real_script_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)
            self._write_json(os.path.join(script_run, "script_checkpoint.json"), {"status": "completed"})
            self._write_json(
                os.path.join(script_run, "run_summary.json"),
                {"status": "completed", "quality_gate_executed": False},
            )
            self._write_json(
                os.path.join(script_run, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": episode,
                    "script_checkpoint_dir": real_script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                    "status_by_stage": {"script": "completed", "audio": "not_started"},
                },
            )
            self._write_json(os.path.join(script_run, "pipeline_summary.json"), {"overall_status": "partial"})
            output_zip = os.path.join(tmp, "bundle_root_discovery.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    wrong_script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            diagnostics = dict(metadata.get("collection_diagnostics", {}))
            script_root = dict(diagnostics.get("script_root", {}))
            self.assertEqual(str(script_root.get("chosen_root", "")), os.path.abspath(real_script_ckpt))

    def test_create_debug_bundle_reconstructs_script_when_external_script_missing_on_failed_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_reconstruct_script"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)
            script_lines = [
                {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Linea uno completa."},
                {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "Linea dos completa."},
            ]
            self._write_json(os.path.join(script_run, "script_checkpoint.json"), {"lines": script_lines})
            self._write_json(
                os.path.join(script_run, "run_summary.json"),
                {"status": "failed", "quality_gate_executed": False},
            )
            self._write_json(
                os.path.join(script_run, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": episode,
                    "script_checkpoint_dir": script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                    "status_by_stage": {"script": "failed", "audio": "not_started"},
                },
            )
            self._write_json(os.path.join(script_run, "pipeline_summary.json"), {"overall_status": "failed"})
            missing_script_path = os.path.join(tmp, "script_missing.json")

            output_zip = os.path.join(tmp, "bundle_reconstructed_script.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--script-path",
                    missing_script_path,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
                collection_report = json.loads(zf.read("collection_report.json").decode("utf-8"))
                reconstructed = json.loads(zf.read("reconstructed_script_from_checkpoint.json").decode("utf-8"))
            self.assertTrue(bool(metadata.get("collection_complete", False)))
            self.assertIn("reconstructed_script_from_checkpoint.json", list(metadata.get("included_files", [])))
            self.assertIn("reconstructed_script_from_checkpoint.json", list(metadata.get("derived_files", [])))
            self.assertEqual(reconstructed.get("lines"), script_lines)
            self.assertTrue(
                any(
                    item.get("status") == "not_applicable"
                    and str(item.get("reason", "")) == "script_not_generated"
                    and str(item.get("category", "")) == "external_input"
                    and str(item.get("path", "")) == os.path.abspath(missing_script_path)
                    for item in collection_report
                )
            )
            self.assertTrue(
                any(
                    item.get("status") == "found"
                    and str(item.get("archive_name", "")) == "reconstructed_script_from_checkpoint.json"
                    and str(item.get("category", "")) == "derived"
                    for item in collection_report
                )
            )

    def test_create_debug_bundle_marks_audio_quality_report_not_applicable_when_audio_gate_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_audio_gate_off"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            audio_run = os.path.join(audio_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)
            os.makedirs(audio_run, exist_ok=True)

            self._write_json(os.path.join(script_run, "script_checkpoint.json"), {"status": "completed", "lines": []})
            self._write_json(
                os.path.join(script_run, "run_summary.json"),
                {"status": "completed", "quality_gate_executed": True},
            )
            self._write_json(os.path.join(script_run, "quality_report.json"), {"pass": True})
            self._write_json(
                os.path.join(script_run, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": episode,
                    "script_checkpoint_dir": script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                    "status_by_stage": {"script": "completed", "audio": "completed"},
                },
            )
            self._write_json(os.path.join(script_run, "pipeline_summary.json"), {"overall_status": "completed"})

            self._write_json(os.path.join(audio_run, "audio_manifest.json"), {"status": "completed"})
            self._write_json(os.path.join(audio_run, "run_summary.json"), {"status": "completed"})
            self._write_json(
                os.path.join(audio_run, "podcast_run_summary.json"),
                {
                    "status": "completed",
                    "audio_executed": True,
                    "quality_gate_executed": False,
                },
            )
            self._write_json(os.path.join(audio_run, "normalized_script.json"), {"lines": []})
            script_path = os.path.join(tmp, "script_output.json")
            self._write_json(script_path, {"lines": []})

            output_zip = os.path.join(tmp, "bundle_audio_gate_off.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--script-path",
                    script_path,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
                collection_report = json.loads(zf.read("collection_report.json").decode("utf-8"))
            self.assertTrue(bool(metadata.get("collection_complete", False)))
            self.assertEqual(int(metadata.get("collection_status_counts", {}).get("missing", 0)), 0)
            self.assertTrue(
                any(
                    item.get("status") == "not_applicable"
                    and str(item.get("reason", "")) == "audio_quality_gate_not_executed"
                    and str(item.get("path", "")).endswith("quality_report.json")
                    and str(item.get("category", "")) == "audio_checkpoint"
                    for item in collection_report
                )
            )

    def test_create_debug_bundle_marks_missing_when_script_completed_but_script_path_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_script_path_required"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)

            self._write_json(
                os.path.join(script_run, "script_checkpoint.json"),
                {"lines": [{"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "contenido"}]},
            )
            self._write_json(
                os.path.join(script_run, "run_summary.json"),
                {"status": "completed", "quality_gate_executed": True},
            )
            self._write_json(os.path.join(script_run, "quality_report.json"), {"pass": True})
            self._write_json(
                os.path.join(script_run, "run_manifest.json"),
                {
                    "manifest_version": 2,
                    "episode_id": episode,
                    "script_checkpoint_dir": script_ckpt,
                    "audio_checkpoint_dir": audio_ckpt,
                    "status_by_stage": {"script": "completed", "audio": "not_started"},
                },
            )
            self._write_json(os.path.join(script_run, "pipeline_summary.json"), {"overall_status": "partial"})

            output_zip = os.path.join(tmp, "bundle_script_path_required.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
                collection_report = json.loads(zf.read("collection_report.json").decode("utf-8"))
            self.assertFalse(bool(metadata.get("collection_complete", True)))
            self.assertEqual(int(metadata.get("collection_status_counts", {}).get("missing", 0)), 1)
            self.assertTrue(
                any(
                    item.get("status") == "missing"
                    and str(item.get("category", "")) == "external_input"
                    and str(item.get("reason", "")) == "script_path_required_when_completed"
                    for item in collection_report
                )
            )

    def test_create_debug_bundle_collects_script_checkpoint_from_script_basename_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "podcast_15min"
            script_episode = "podcast_script"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, script_episode), exist_ok=True)
            os.makedirs(os.path.join(audio_ckpt, episode), exist_ok=True)

            self._write_json(
                os.path.join(script_ckpt, script_episode, "run_summary.json"),
                {"status": "completed", "component": "make_script"},
            )
            self._write_json(
                os.path.join(audio_ckpt, episode, "podcast_run_summary.json"),
                {"status": "failed", "component": "make_podcast"},
            )
            script_path = os.path.join(tmp, "podcast_script.json")
            self._write_json(script_path, {"lines": []})
            output_zip = os.path.join(tmp, "bundle_alias.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--script-path",
                    script_path,
                    "--output",
                    output_zip,
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            included = list(metadata.get("included_files", []))
            self.assertTrue(
                any("podcast_script" in name and name.endswith("run_summary.json") for name in included)
            )

    def test_create_debug_bundle_collects_all_script_run_files_from_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "podcast_15min"
            script_episode = "podcast_script"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, script_episode), exist_ok=True)
            os.makedirs(os.path.join(audio_ckpt, episode), exist_ok=True)

            self._write_json(os.path.join(script_ckpt, script_episode, "script_checkpoint.json"), {"status": "ok"})
            self._write_json(os.path.join(script_ckpt, script_episode, "run_summary.json"), {"status": "ok"})
            self._write_json(os.path.join(script_ckpt, script_episode, "quality_report.json"), {"status": "ok"})
            self._write_json(os.path.join(audio_ckpt, episode, "podcast_run_summary.json"), {"status": "failed"})
            self._write_json(os.path.join(audio_ckpt, episode, "normalized_script.json"), {"lines": []})
            script_path = os.path.join(tmp, "podcast_script.json")
            self._write_json(script_path, {"lines": []})

            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--script-path",
                    script_path,
                    "--output",
                    os.path.join(tmp, "bundle_alias_full.zip"),
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            included = list(metadata.get("included_files", []))
            self.assertTrue(any("podcast_script" in name and name.endswith("script_checkpoint.json") for name in included))
            self.assertTrue(any("podcast_script" in name and name.endswith("run_summary.json") for name in included))
            self.assertTrue(any("podcast_script" in name and name.endswith("quality_report.json") for name in included))
            self.assertTrue(any(name.endswith("normalized_script.json") for name in included))
            missing = list(metadata.get("missing_candidates", []))
            self.assertFalse(
                any(
                    "script_ckpt" in name and "podcast_15min" in name and name.endswith("script_checkpoint.json")
                    for name in missing
                )
            )
            self.assertFalse(
                any(
                    "script_ckpt" in name and "podcast_15min" in name and name.endswith("run_summary.json")
                    for name in missing
                )
            )
            self.assertFalse(
                any(
                    "script_ckpt" in name and "podcast_15min" in name and name.endswith("quality_report.json")
                    for name in missing
                )
            )

    def test_create_debug_bundle_without_script_path_does_not_resolve_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "podcast_15min"
            script_episode = "podcast_script"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, script_episode), exist_ok=True)
            os.makedirs(os.path.join(audio_ckpt, episode), exist_ok=True)
            self._write_json(
                os.path.join(script_ckpt, script_episode, "run_summary.json"),
                {"status": "completed", "component": "make_script"},
            )
            self._write_json(
                os.path.join(audio_ckpt, episode, "podcast_run_summary.json"),
                {"status": "failed", "component": "make_podcast"},
            )
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    os.path.join(tmp, "bundle_no_alias.zip"),
                ]
            )
            out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            included = list(metadata.get("included_files", []))
            self.assertFalse(any("podcast_script" in name and name.endswith("run_summary.json") for name in included))

    def test_create_debug_bundle_reads_git_commit_from_gitfile_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = os.path.join(tmp, "project")
            os.makedirs(project, exist_ok=True)
            gitdir = os.path.join(tmp, "gitdir")
            os.makedirs(os.path.join(gitdir, "refs", "heads"), exist_ok=True)
            commit = "1234567890abcdef1234567890abcdef12345678"
            with open(os.path.join(gitdir, "HEAD"), "w", encoding="utf-8") as f:
                f.write("ref: refs/heads/main\n")
            with open(os.path.join(gitdir, "refs", "heads", "main"), "w", encoding="utf-8") as f:
                f.write(commit + "\n")
            with open(os.path.join(project, ".git"), "w", encoding="utf-8") as f:
                f.write(f"gitdir: {gitdir}\n")

            episode = "ep_git"
            script_ckpt = os.path.join(project, "script_ckpt")
            audio_ckpt = os.path.join(project, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, episode), exist_ok=True)
            output_zip = os.path.join(project, "bundle_git.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            with mock.patch("export_debug_bundle.os.getcwd", return_value=project):
                out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            self.assertEqual(metadata.get("git_commit"), commit)

    def test_create_debug_bundle_reads_git_commit_from_git_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = os.path.join(tmp, "project")
            os.makedirs(project, exist_ok=True)
            gitdir = os.path.join(project, ".git")
            os.makedirs(os.path.join(gitdir, "refs", "heads"), exist_ok=True)
            commit = "abcdefabcdefabcdefabcdefabcdefabcdefabcd"
            with open(os.path.join(gitdir, "HEAD"), "w", encoding="utf-8") as f:
                f.write("ref: refs/heads/main\n")
            with open(os.path.join(gitdir, "refs", "heads", "main"), "w", encoding="utf-8") as f:
                f.write(commit + "\n")

            episode = "ep_git_dir"
            script_ckpt = os.path.join(project, "script_ckpt")
            audio_ckpt = os.path.join(project, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, episode), exist_ok=True)
            output_zip = os.path.join(project, "bundle_git_dir.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            with mock.patch("export_debug_bundle.os.getcwd", return_value=project):
                out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            self.assertEqual(metadata.get("git_commit"), commit)

    def test_create_debug_bundle_reads_git_commit_from_relative_gitdir_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = os.path.join(tmp, "project")
            os.makedirs(project, exist_ok=True)
            shared = os.path.join(tmp, "shared_git")
            os.makedirs(os.path.join(shared, "refs", "heads"), exist_ok=True)
            commit = "fedcba9876543210fedcba9876543210fedcba98"
            with open(os.path.join(shared, "HEAD"), "w", encoding="utf-8") as f:
                f.write("ref: refs/heads/main\n")
            with open(os.path.join(shared, "refs", "heads", "main"), "w", encoding="utf-8") as f:
                f.write(commit + "\n")
            with open(os.path.join(project, ".git"), "w", encoding="utf-8") as f:
                f.write("gitdir: ../shared_git\n")

            episode = "ep_git_relative"
            script_ckpt = os.path.join(project, "script_ckpt")
            audio_ckpt = os.path.join(project, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, episode), exist_ok=True)
            output_zip = os.path.join(project, "bundle_git_relative.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            with mock.patch("export_debug_bundle.os.getcwd", return_value=project):
                out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            self.assertEqual(metadata.get("git_commit"), commit)

    def test_create_debug_bundle_reads_git_commit_from_packed_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = os.path.join(tmp, "project")
            os.makedirs(project, exist_ok=True)
            gitdir = os.path.join(project, ".git")
            os.makedirs(gitdir, exist_ok=True)
            commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            ref = "refs/heads/main"
            with open(os.path.join(gitdir, "HEAD"), "w", encoding="utf-8") as f:
                f.write(f"ref: {ref}\n")
            with open(os.path.join(gitdir, "packed-refs"), "w", encoding="utf-8") as f:
                f.write("# pack-refs with: peeled fully-peeled\n")
                f.write(f"{commit} {ref}\n")

            episode = "ep_git_packed"
            script_ckpt = os.path.join(project, "script_ckpt")
            audio_ckpt = os.path.join(project, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, episode), exist_ok=True)
            output_zip = os.path.join(project, "bundle_git_packed.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            with mock.patch("export_debug_bundle.os.getcwd", return_value=project):
                out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            self.assertEqual(metadata.get("git_commit"), commit)

    def test_create_debug_bundle_reads_git_commit_from_detached_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = os.path.join(tmp, "project")
            os.makedirs(project, exist_ok=True)
            gitdir = os.path.join(project, ".git")
            os.makedirs(gitdir, exist_ok=True)
            commit = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            with open(os.path.join(gitdir, "HEAD"), "w", encoding="utf-8") as f:
                f.write(commit + "\n")

            episode = "ep_git_detached"
            script_ckpt = os.path.join(project, "script_ckpt")
            audio_ckpt = os.path.join(project, "audio_ckpt")
            os.makedirs(os.path.join(script_ckpt, episode), exist_ok=True)
            output_zip = os.path.join(project, "bundle_git_detached.zip")
            args = parse_args(
                [
                    episode,
                    "--script-checkpoint-dir",
                    script_ckpt,
                    "--audio-checkpoint-dir",
                    audio_ckpt,
                    "--output",
                    output_zip,
                ]
            )
            with mock.patch("export_debug_bundle.os.getcwd", return_value=project):
                out = create_debug_bundle(args)
            with zipfile.ZipFile(out, "r") as zf:
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
            self.assertEqual(metadata.get("git_commit"), commit)


if __name__ == "__main__":
    unittest.main()

