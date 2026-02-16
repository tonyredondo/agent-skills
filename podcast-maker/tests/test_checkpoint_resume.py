import os
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import ReliabilityConfig  # noqa: E402
from pipeline.audio_checkpoint import AudioCheckpointStore  # noqa: E402
from pipeline.script_checkpoint import ScriptCheckpointStore  # noqa: E402


class CheckpointResumeTests(unittest.TestCase):
    def test_resume_validation_blocks_mismatch(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            store = ScriptCheckpointStore(
                base_dir=tmp,
                episode_id="ep1",
                reliability=reliability,
            )
            state = store.create_initial_state(source_hash="aaa", config_fingerprint="bbb")
            store.save(state)
            loaded = store.load()
            self.assertIsNotNone(loaded)
            with self.assertRaises(RuntimeError):
                store.validate_resume(
                    loaded,  # type: ignore[arg-type]
                    source_hash="changed",
                    config_fingerprint="bbb",
                    resume_force=False,
                )

    def test_resume_force_allows_mismatch(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            store = ScriptCheckpointStore(
                base_dir=tmp,
                episode_id="ep2",
                reliability=reliability,
            )
            state = store.create_initial_state(source_hash="aaa", config_fingerprint="bbb")
            store.save(state)
            loaded = store.load()
            self.assertIsNotNone(loaded)
            store.validate_resume(
                loaded,  # type: ignore[arg-type]
                source_hash="changed",
                config_fingerprint="changed",
                resume_force=True,
            )

    def test_script_validate_resume_rejects_empty_state(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            store = ScriptCheckpointStore(base_dir=tmp, episode_id="ep_empty_script", reliability=reliability)
            with self.assertRaises(RuntimeError):
                store.validate_resume(
                    {},
                    source_hash="h",
                    config_fingerprint="f",
                    resume_force=False,
                )

    def test_audio_validate_resume_rejects_empty_manifest(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            store = AudioCheckpointStore(base_dir=tmp, episode_id="ep_empty_audio", reliability=reliability)
            with self.assertRaises(RuntimeError):
                store.validate_resume(
                    {},
                    config_fingerprint="f",
                    script_hash="h",
                    resume_force=False,
                )

    def test_corrupt_script_checkpoint_is_quarantined(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            store = ScriptCheckpointStore(base_dir=tmp, episode_id="ep3", reliability=reliability)
            os.makedirs(os.path.dirname(store.checkpoint_path), exist_ok=True)
            with open(store.checkpoint_path, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            loaded = store.load()
            self.assertIsNone(loaded)
            self.assertTrue(store.last_corrupt_backup_path)
            self.assertFalse(os.path.exists(store.checkpoint_path))
            self.assertTrue(os.path.exists(store.last_corrupt_backup_path))

    def test_corrupt_audio_manifest_is_quarantined(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            store = AudioCheckpointStore(base_dir=tmp, episode_id="ep4", reliability=reliability)
            os.makedirs(os.path.dirname(store.manifest_path), exist_ok=True)
            with open(store.manifest_path, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            loaded = store.load()
            self.assertIsNone(loaded)
            self.assertTrue(store.last_corrupt_backup_path)
            self.assertFalse(os.path.exists(store.manifest_path))
            self.assertTrue(os.path.exists(store.last_corrupt_backup_path))


if __name__ == "__main__":
    unittest.main()

