import dataclasses
import os
import sys
import tempfile
import threading
import time
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.audio_checkpoint import AudioCheckpointStore  # noqa: E402
from pipeline.config import ReliabilityConfig  # noqa: E402
from pipeline.script_checkpoint import LockError, ScriptCheckpointStore  # noqa: E402


class ChaosLockingTests(unittest.TestCase):
    def test_script_lock_conflict(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            s1 = ScriptCheckpointStore(base_dir=tmp, episode_id="ep", reliability=reliability)
            s2 = ScriptCheckpointStore(base_dir=tmp, episode_id="ep", reliability=reliability)
            s1.acquire_lock()
            try:
                with self.assertRaises(LockError):
                    s2.acquire_lock()
            finally:
                s1.release_lock()

    def test_audio_lock_force_unlock(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            a1 = AudioCheckpointStore(base_dir=tmp, episode_id="ep", reliability=reliability)
            a2 = AudioCheckpointStore(base_dir=tmp, episode_id="ep", reliability=reliability)
            a1.acquire_lock()
            try:
                with self.assertRaises(RuntimeError):
                    a2.acquire_lock()
                a2.acquire_lock(force_unlock=True)
            finally:
                a2.release_lock()

    def test_script_lock_heartbeat_prevents_ttl_takeover(self) -> None:
        reliability = dataclasses.replace(ReliabilityConfig.from_env(), lock_ttl_seconds=1)
        with tempfile.TemporaryDirectory() as tmp:
            s1 = ScriptCheckpointStore(base_dir=tmp, episode_id="hb_script", reliability=reliability)
            s2 = ScriptCheckpointStore(base_dir=tmp, episode_id="hb_script", reliability=reliability)
            s1.acquire_lock()
            try:
                time.sleep(2.1)
                with self.assertRaises(LockError):
                    s2.acquire_lock()
            finally:
                s1.release_lock()

    def test_audio_lock_heartbeat_prevents_ttl_takeover(self) -> None:
        reliability = dataclasses.replace(ReliabilityConfig.from_env(), lock_ttl_seconds=1)
        with tempfile.TemporaryDirectory() as tmp:
            a1 = AudioCheckpointStore(base_dir=tmp, episode_id="hb_audio", reliability=reliability)
            a2 = AudioCheckpointStore(base_dir=tmp, episode_id="hb_audio", reliability=reliability)
            a1.acquire_lock()
            try:
                time.sleep(2.1)
                with self.assertRaises(RuntimeError):
                    a2.acquire_lock()
            finally:
                a1.release_lock()

    def test_script_release_does_not_remove_foreign_lock(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            s1 = ScriptCheckpointStore(base_dir=tmp, episode_id="foreign_script", reliability=reliability)
            s2 = ScriptCheckpointStore(base_dir=tmp, episode_id="foreign_script", reliability=reliability)
            s3 = ScriptCheckpointStore(base_dir=tmp, episode_id="foreign_script", reliability=reliability)
            s1.acquire_lock()
            s2.acquire_lock(force_unlock=True)
            try:
                s1.release_lock()
                with self.assertRaises(LockError):
                    s3.acquire_lock()
            finally:
                s2.release_lock()

    def test_script_unreadable_lock_requires_force_unlock(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            store = ScriptCheckpointStore(base_dir=tmp, episode_id="corrupt_script", reliability=reliability)
            with open(store.lock_path, "w", encoding="utf-8") as f:
                f.write("{broken")
            with self.assertRaises(LockError):
                store.acquire_lock()
            store.acquire_lock(force_unlock=True)
            store.release_lock()

    def test_script_lock_is_atomic_under_race(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            release_holder = threading.Event()
            holder_acquired = threading.Event()
            results: list[str] = []

            def holder() -> None:
                store = ScriptCheckpointStore(base_dir=tmp, episode_id="race", reliability=reliability)
                try:
                    store.acquire_lock()
                    results.append("acquired")
                    holder_acquired.set()
                    release_holder.wait(timeout=0.2)
                except LockError:
                    results.append("blocked")
                finally:
                    store.release_lock()

            def contender() -> None:
                holder_acquired.wait(timeout=0.2)
                store = ScriptCheckpointStore(base_dir=tmp, episode_id="race", reliability=reliability)
                try:
                    store.acquire_lock()
                    results.append("acquired")
                except LockError:
                    results.append("blocked")
                finally:
                    store.release_lock()

            t1 = threading.Thread(target=holder)
            t2 = threading.Thread(target=contender)
            t1.start()
            t2.start()
            holder_acquired.wait(timeout=0.2)
            time.sleep(0.05)
            release_holder.set()
            t1.join()
            t2.join()

            self.assertEqual(results.count("acquired"), 1)
            self.assertEqual(results.count("blocked"), 1)

    def test_audio_lock_is_atomic_under_race(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            release_holder = threading.Event()
            holder_acquired = threading.Event()
            results: list[str] = []

            def holder() -> None:
                store = AudioCheckpointStore(base_dir=tmp, episode_id="race_audio", reliability=reliability)
                try:
                    store.acquire_lock()
                    results.append("acquired")
                    holder_acquired.set()
                    release_holder.wait(timeout=0.2)
                except RuntimeError:
                    results.append("blocked")
                finally:
                    store.release_lock()

            def contender() -> None:
                holder_acquired.wait(timeout=0.2)
                store = AudioCheckpointStore(base_dir=tmp, episode_id="race_audio", reliability=reliability)
                try:
                    store.acquire_lock()
                    results.append("acquired")
                except RuntimeError:
                    results.append("blocked")
                finally:
                    store.release_lock()

            t1 = threading.Thread(target=holder)
            t2 = threading.Thread(target=contender)
            t1.start()
            t2.start()
            holder_acquired.wait(timeout=0.2)
            time.sleep(0.05)
            release_holder.set()
            t1.join()
            t2.join()

            self.assertEqual(results.count("acquired"), 1)
            self.assertEqual(results.count("blocked"), 1)

    def test_audio_unreadable_lock_requires_force_unlock(self) -> None:
        reliability = ReliabilityConfig.from_env()
        with tempfile.TemporaryDirectory() as tmp:
            store = AudioCheckpointStore(base_dir=tmp, episode_id="corrupt_audio", reliability=reliability)
            with open(store.lock_path, "w", encoding="utf-8") as f:
                f.write("{broken")
            with self.assertRaises(RuntimeError):
                store.acquire_lock()
            store.acquire_lock(force_unlock=True)
            store.release_lock()


if __name__ == "__main__":
    unittest.main()

