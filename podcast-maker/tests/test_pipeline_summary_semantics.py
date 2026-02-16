import json
import os
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.run_manifest import (  # noqa: E402
    init_manifest,
    pipeline_summary_path,
    run_manifest_path,
    update_manifest,
)


class PipelineSummarySemanticsTests(unittest.TestCase):
    def _read_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertIsInstance(payload, dict)
        return payload

    def test_pipeline_summary_tracks_not_started_completed_and_failed_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_id = "ep_summary"
            init_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                run_token="run_abc",
                script_output_path="/tmp/script.json",
                script_checkpoint_dir=tmp,
                audio_checkpoint_dir=os.path.join(tmp, "audio"),
            )
            summary_path = pipeline_summary_path(checkpoint_dir=tmp, episode_id=episode_id)
            summary_initial = self._read_json(summary_path)
            self.assertEqual(summary_initial.get("overall_status"), "running")

            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"script": "completed", "audio": "not_started"},
                    "script": {"status": "completed"},
                },
            )
            summary_partial = self._read_json(summary_path)
            self.assertEqual(summary_partial.get("overall_status"), "partial")

            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"audio": "completed"},
                    "audio": {"status": "completed"},
                },
            )
            summary_completed = self._read_json(summary_path)
            self.assertEqual(summary_completed.get("overall_status"), "completed")

            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"audio": "failed"},
                    "audio": {"status": "failed", "failure_kind": "timeout"},
                },
            )
            summary_failed = self._read_json(summary_path)
            self.assertEqual(summary_failed.get("overall_status"), "failed")

    def test_run_manifest_file_is_updated_with_stage_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_id = "ep_manifest"
            init_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                run_token="run_xyz",
                script_output_path="/tmp/script.json",
                script_checkpoint_dir=tmp,
                audio_checkpoint_dir=os.path.join(tmp, "audio"),
            )
            manifest_path = run_manifest_path(checkpoint_dir=tmp, episode_id=episode_id)
            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"script": "completed"},
                    "script": {"status": "completed", "run_summary_path": "/tmp/run_summary.json"},
                },
            )
            manifest = self._read_json(manifest_path)
            self.assertEqual(manifest.get("episode_id"), episode_id)
            self.assertEqual(manifest.get("status_by_stage", {}).get("script"), "completed")
            self.assertEqual(manifest.get("script", {}).get("run_summary_path"), "/tmp/run_summary.json")

    def test_pipeline_summary_running_for_started_script_before_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_id = "ep_running"
            init_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                run_token="run_started",
                script_output_path="/tmp/script.json",
                script_checkpoint_dir=tmp,
                audio_checkpoint_dir=os.path.join(tmp, "audio"),
            )
            summary_path = pipeline_summary_path(checkpoint_dir=tmp, episode_id=episode_id)
            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"script": "started", "audio": "not_started"},
                    "script": {"status": "started"},
                },
            )
            summary = self._read_json(summary_path)
            self.assertEqual(summary.get("overall_status"), "running")

    def test_pipeline_summary_partial_when_audio_running_after_script_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_id = "ep_partial_running_audio"
            init_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                run_token="run_partial",
                script_output_path="/tmp/script.json",
                script_checkpoint_dir=tmp,
                audio_checkpoint_dir=os.path.join(tmp, "audio"),
            )
            summary_path = pipeline_summary_path(checkpoint_dir=tmp, episode_id=episode_id)
            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"script": "completed", "audio": "running"},
                    "script": {"status": "completed"},
                    "audio": {"status": "running"},
                },
            )
            summary = self._read_json(summary_path)
            self.assertEqual(summary.get("overall_status"), "partial")

    def test_pipeline_summary_interrupted_when_script_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_id = "ep_interrupted"
            init_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                run_token="run_interrupt",
                script_output_path="/tmp/script.json",
                script_checkpoint_dir=tmp,
                audio_checkpoint_dir=os.path.join(tmp, "audio"),
            )
            summary_path = pipeline_summary_path(checkpoint_dir=tmp, episode_id=episode_id)
            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"script": "interrupted", "audio": "not_started"},
                    "script": {"status": "interrupted", "failure_kind": "interrupted"},
                },
            )
            summary = self._read_json(summary_path)
            self.assertEqual(summary.get("overall_status"), "interrupted")

    def test_pipeline_summary_unknown_audio_status_defaults_to_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_id = "ep_audio_unknown"
            init_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                run_token="run_unknown",
                script_output_path="/tmp/script.json",
                script_checkpoint_dir=tmp,
                audio_checkpoint_dir=os.path.join(tmp, "audio"),
            )
            summary_path = pipeline_summary_path(checkpoint_dir=tmp, episode_id=episode_id)
            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"script": "completed", "audio": "weird_status"},
                    "script": {"status": "completed"},
                    "audio": {"status": "weird_status"},
                },
            )
            summary = self._read_json(summary_path)
            self.assertEqual(summary.get("overall_status"), "partial")

    def test_pipeline_summary_audio_failure_wins_when_script_status_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_id = "ep_audio_failed_unknown_script"
            init_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                run_token="run_audio_failed",
                script_output_path="/tmp/script.json",
                script_checkpoint_dir=tmp,
                audio_checkpoint_dir=os.path.join(tmp, "audio"),
            )
            summary_path = pipeline_summary_path(checkpoint_dir=tmp, episode_id=episode_id)
            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"script": "weird_status", "audio": "failed"},
                    "script": {"status": "weird_status"},
                    "audio": {"status": "failed", "failure_kind": "tts_timeout"},
                },
            )
            summary = self._read_json(summary_path)
            self.assertEqual(summary.get("overall_status"), "failed")

    def test_pipeline_summary_audio_completed_with_unknown_script_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_id = "ep_audio_completed_unknown_script"
            init_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                run_token="run_audio_completed",
                script_output_path="/tmp/script.json",
                script_checkpoint_dir=tmp,
                audio_checkpoint_dir=os.path.join(tmp, "audio"),
            )
            summary_path = pipeline_summary_path(checkpoint_dir=tmp, episode_id=episode_id)
            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"script": "mystery_state", "audio": "completed"},
                    "script": {"status": "mystery_state"},
                    "audio": {"status": "completed"},
                },
            )
            summary = self._read_json(summary_path)
            self.assertEqual(summary.get("overall_status"), "partial")

    def test_pipeline_summary_audio_interrupted_after_script_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode_id = "ep_audio_interrupted_after_script"
            init_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                run_token="run_audio_interrupted",
                script_output_path="/tmp/script.json",
                script_checkpoint_dir=tmp,
                audio_checkpoint_dir=os.path.join(tmp, "audio"),
            )
            summary_path = pipeline_summary_path(checkpoint_dir=tmp, episode_id=episode_id)
            update_manifest(
                checkpoint_dir=tmp,
                episode_id=episode_id,
                updates={
                    "status_by_stage": {"script": "completed", "audio": "interrupted"},
                    "script": {"status": "completed"},
                    "audio": {"status": "interrupted", "failure_kind": "interrupted"},
                },
            )
            summary = self._read_json(summary_path)
            self.assertEqual(summary.get("overall_status"), "interrupted")


if __name__ == "__main__":
    unittest.main()

