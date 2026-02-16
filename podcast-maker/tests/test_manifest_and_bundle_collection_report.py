import json
import os
import sys
import tempfile
import unittest
import zipfile


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from export_debug_bundle import create_debug_bundle, parse_args  # noqa: E402


class ManifestAndBundleCollectionReportTests(unittest.TestCase):
    def _write_json(self, path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def test_bundle_v2_includes_collection_report_and_not_applicable_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_bundle_v2"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            audio_run = os.path.join(audio_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)
            os.makedirs(audio_run, exist_ok=True)

            self._write_json(os.path.join(script_run, "script_checkpoint.json"), {"status": "completed"})
            self._write_json(os.path.join(script_run, "run_summary.json"), {"status": "completed"})
            self._write_json(os.path.join(script_run, "run_manifest.json"), {"manifest_version": 2})
            self._write_json(os.path.join(script_run, "pipeline_summary.json"), {"overall_status": "partial"})
            self._write_json(
                os.path.join(audio_run, "podcast_run_summary.json"),
                {"status": "failed", "audio_executed": False},
            )

            output_zip = os.path.join(tmp, "bundle_v2.zip")
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
                names = set(zf.namelist())
                self.assertIn("collection_report.json", names)
                self.assertTrue(any(name.endswith("run_manifest.json") for name in names))
                self.assertTrue(any(name.endswith("pipeline_summary.json") for name in names))
                metadata = json.loads(zf.read("debug_bundle_metadata.json").decode("utf-8"))
                collection_report = json.loads(zf.read("collection_report.json").decode("utf-8"))

            self.assertEqual(metadata.get("bundle_version"), 2)
            self.assertEqual(metadata.get("collection_report_path"), "collection_report.json")
            self.assertTrue(
                any(
                    item.get("status") == "not_applicable"
                    and str(item.get("reason", "")) == "audio_not_executed"
                    and str(item.get("path", "")).endswith("audio_manifest.json")
                    for item in collection_report
                )
            )

    def test_bundle_v2_manifest_not_started_audio_marks_audio_files_not_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = "episode_script_only"
            script_ckpt = os.path.join(tmp, "script_ckpt")
            audio_ckpt = os.path.join(tmp, "audio_ckpt")
            script_run = os.path.join(script_ckpt, episode)
            os.makedirs(script_run, exist_ok=True)

            self._write_json(os.path.join(script_run, "script_checkpoint.json"), {"status": "completed"})
            self._write_json(os.path.join(script_run, "run_summary.json"), {"status": "completed"})
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
            script_path = os.path.join(tmp, "episode_script_only.json")
            self._write_json(script_path, {"lines": [{"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "ok"}]})

            output_zip = os.path.join(tmp, "bundle_script_only.zip")
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
            status_counts = dict(metadata.get("collection_status_counts", {}))
            self.assertEqual(int(status_counts.get("missing", 0)), 0)
            self.assertGreater(int(status_counts.get("not_applicable", 0)), 0)
            self.assertTrue(
                any(
                    item.get("status") == "not_applicable"
                    and str(item.get("reason", "")) == "audio_not_executed"
                    and str(item.get("path", "")).endswith("podcast_run_summary.json")
                    for item in collection_report
                )
            )


if __name__ == "__main__":
    unittest.main()

