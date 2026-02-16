import dataclasses
import os
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.audio_checkpoint import AudioCheckpointStore  # noqa: E402
from pipeline.config import ReliabilityConfig  # noqa: E402
from pipeline.script_checkpoint import ScriptCheckpointStore  # noqa: E402


class CheckpointMigrationTests(unittest.TestCase):
    def test_script_checkpoint_minor_version_migration(self) -> None:
        reliability = dataclasses.replace(ReliabilityConfig.from_env(), checkpoint_version="2.1")
        with tempfile.TemporaryDirectory() as tmp:
            store = ScriptCheckpointStore(base_dir=tmp, episode_id="ep_minor", reliability=reliability)
            state = store.create_initial_state(source_hash="h1", config_fingerprint="f1")
            state["checkpoint_version"] = "2.0"
            migrated = store.validate_resume(
                state,
                source_hash="h1",
                config_fingerprint="f1",
                resume_force=False,
            )
            self.assertTrue(migrated)
            self.assertEqual(state["checkpoint_version"], "2.1")
            self.assertEqual(state["migrated_from_version"], "2.0")

    def test_script_checkpoint_major_version_mismatch_raises(self) -> None:
        reliability = dataclasses.replace(ReliabilityConfig.from_env(), checkpoint_version="3.0")
        with tempfile.TemporaryDirectory() as tmp:
            store = ScriptCheckpointStore(base_dir=tmp, episode_id="ep_major", reliability=reliability)
            state = store.create_initial_state(source_hash="h1", config_fingerprint="f1")
            state["checkpoint_version"] = "2.9"
            with self.assertRaises(RuntimeError):
                store.validate_resume(
                    state,
                    source_hash="h1",
                    config_fingerprint="f1",
                    resume_force=False,
                )

    def test_audio_manifest_minor_version_migration(self) -> None:
        reliability = dataclasses.replace(ReliabilityConfig.from_env(), checkpoint_version="2.2")
        with tempfile.TemporaryDirectory() as tmp:
            store = AudioCheckpointStore(base_dir=tmp, episode_id="ep_audio_minor", reliability=reliability)
            manifest = store.init_manifest(
                config_fingerprint="cfg",
                script_hash="script",
                segments=[],
            )
            manifest["checkpoint_version"] = "2.0"
            migrated = store.validate_resume(
                manifest,
                config_fingerprint="cfg",
                script_hash="script",
                resume_force=False,
            )
            self.assertTrue(migrated)
            self.assertEqual(manifest["checkpoint_version"], "2.2")
            self.assertEqual(manifest["migrated_from_version"], "2.0")

    def test_audio_manifest_major_version_mismatch_raises(self) -> None:
        reliability = dataclasses.replace(ReliabilityConfig.from_env(), checkpoint_version="4.0")
        with tempfile.TemporaryDirectory() as tmp:
            store = AudioCheckpointStore(base_dir=tmp, episode_id="ep_audio_major", reliability=reliability)
            manifest = store.init_manifest(
                config_fingerprint="cfg",
                script_hash="script",
                segments=[],
            )
            manifest["checkpoint_version"] = "2.0"
            with self.assertRaises(RuntimeError):
                store.validate_resume(
                    manifest,
                    config_fingerprint="cfg",
                    script_hash="script",
                    resume_force=False,
                )


if __name__ == "__main__":
    unittest.main()

