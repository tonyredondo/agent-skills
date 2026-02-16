#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .config import ReliabilityConfig


class LockError(RuntimeError):
    pass


def _parse_version(value: Any) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, 0
    if isinstance(value, str):
        m = re.match(r"^\s*(\d+)(?:\.(\d+))?\s*$", value)
        if m:
            major = int(m.group(1))
            minor = int(m.group(2) or 0)
            return major, minor
    return -1, -1


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _atomic_create_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


@dataclass
class ScriptCheckpointStore:
    base_dir: str
    episode_id: str
    reliability: ReliabilityConfig
    last_corrupt_backup_path: Optional[str] = None
    last_corrupt_error: str = ""
    _lock_token: Optional[str] = field(default=None, init=False, repr=False)
    _lock_heartbeat_stop: Optional[threading.Event] = field(default=None, init=False, repr=False)
    _lock_heartbeat_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        self.run_dir = os.path.join(self.base_dir, self.episode_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.run_dir, "script_checkpoint.json")
        self.lock_path = os.path.join(self.run_dir, ".lock")

    def _heartbeat_interval_seconds(self) -> float:
        return float(max(1, min(30, int(self.reliability.lock_ttl_seconds / 3))))

    def _stop_lock_heartbeat(self) -> None:
        stop = self._lock_heartbeat_stop
        thread = self._lock_heartbeat_thread
        self._lock_heartbeat_stop = None
        self._lock_heartbeat_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _refresh_lock_timestamp(self) -> None:
        token = self._lock_token
        if not token:
            return
        try:
            with open(self.lock_path, "r", encoding="utf-8") as f:
                try:
                    lock_data = json.load(f)
                except Exception:
                    return
                if str(lock_data.get("token", "")) != token:
                    return
            now = time.time()
            os.utime(self.lock_path, (now, now))
        except OSError:
            return

    def _lock_heartbeat_loop(self, stop: threading.Event, interval_s: float) -> None:
        while not stop.wait(interval_s):
            self._refresh_lock_timestamp()

    def _start_lock_heartbeat(self) -> None:
        self._stop_lock_heartbeat()
        stop = threading.Event()
        thread = threading.Thread(
            target=self._lock_heartbeat_loop,
            args=(stop, self._heartbeat_interval_seconds()),
            daemon=True,
            name=f"script-lock-heartbeat-{self.episode_id}",
        )
        self._lock_heartbeat_stop = stop
        self._lock_heartbeat_thread = thread
        thread.start()

    def acquire_lock(self, force_unlock: bool = False) -> None:
        self._stop_lock_heartbeat()
        self._lock_token = None
        while True:
            now = int(time.time())
            token = uuid.uuid4().hex
            payload = {"pid": os.getpid(), "ts": now, "token": token}
            try:
                _atomic_create_json(self.lock_path, payload)
                self._lock_token = token
                self._start_lock_heartbeat()
                return
            except FileExistsError:
                try:
                    with open(self.lock_path, "r", encoding="utf-8") as f:
                        lock_data = json.load(f)
                except Exception as exc:
                    if force_unlock:
                        try:
                            os.remove(self.lock_path)
                        except OSError:
                            pass
                        continue
                    raise LockError(
                        "Checkpoint lock exists but is unreadable/invalid. "
                        f"Use --force-unlock if this lock is stale. ({exc})"
                    )
                if not isinstance(lock_data, dict) or "ts" not in lock_data:
                    if force_unlock:
                        try:
                            os.remove(self.lock_path)
                        except OSError:
                            pass
                        continue
                    raise LockError(
                        "Checkpoint lock exists but has invalid metadata. "
                        "Use --force-unlock if this lock is stale."
                    )
                try:
                    ts = int(lock_data.get("ts", 0))
                except (TypeError, ValueError):
                    if force_unlock:
                        try:
                            os.remove(self.lock_path)
                        except OSError:
                            pass
                        continue
                    raise LockError(
                        "Checkpoint lock exists but has invalid timestamp metadata. "
                        "Use --force-unlock if this lock is stale."
                    )
                try:
                    lock_mtime = int(os.path.getmtime(self.lock_path))
                except OSError:
                    lock_mtime = ts
                age = now - max(ts, lock_mtime)
                if force_unlock or age > self.reliability.lock_ttl_seconds:
                    try:
                        os.remove(self.lock_path)
                    except OSError:
                        # Another process may have removed/replaced it; retry atomically.
                        pass
                    continue
                owner = lock_data.get("pid", "unknown")
                raise LockError(f"Checkpoint lock active (pid={owner}, age={age}s)")

    def release_lock(self) -> None:
        token = self._lock_token
        self._lock_token = None
        self._stop_lock_heartbeat()
        if not token:
            return
        try:
            with open(self.lock_path, "r", encoding="utf-8") as f:
                lock_data = json.load(f)
        except Exception:
            return
        if str(lock_data.get("token", "")) != token:
            return
        try:
            os.remove(self.lock_path)
        except OSError:
            pass

    def load(self) -> Optional[Dict[str, Any]]:
        self.last_corrupt_backup_path = None
        self.last_corrupt_error = ""
        if not os.path.exists(self.checkpoint_path):
            return None
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            ts = int(time.time())
            backup = f"{self.checkpoint_path}.corrupt.{ts}.json"
            try:
                os.replace(self.checkpoint_path, backup)
                self.last_corrupt_backup_path = backup
            except OSError:
                self.last_corrupt_backup_path = ""
            self.last_corrupt_error = str(exc)
            return None

    def save(self, state: Dict[str, Any]) -> None:
        _atomic_write_json(self.checkpoint_path, state)

    def create_initial_state(
        self,
        *,
        source_hash: str,
        config_fingerprint: str,
    ) -> Dict[str, Any]:
        return {
            "checkpoint_version": self.reliability.checkpoint_version,
            "episode_id": self.episode_id,
            "source_hash": source_hash,
            "config_fingerprint": config_fingerprint,
            "chunks_done": 0,
            "lines": [],
            "current_word_count": 0,
            "last_success_at": int(time.time()),
            "status": "running",
        }

    def validate_resume(
        self,
        state: Dict[str, Any],
        *,
        source_hash: str,
        config_fingerprint: str,
        resume_force: bool,
    ) -> bool:
        if not state or not isinstance(state, dict):
            raise RuntimeError("Invalid checkpoint state for resume")
        current_version = self.reliability.checkpoint_version
        existing_version = state.get("checkpoint_version")
        current_major, _ = _parse_version(current_version)
        existing_major, _ = _parse_version(existing_version)
        if existing_major != current_major:
            raise RuntimeError(
                f"Checkpoint version mismatch: {existing_version} != {current_version}. "
                "Use a clean run for major version changes."
            )
        migrated = False
        if existing_version != current_version:
            state["migrated_from_version"] = existing_version
            state["checkpoint_version"] = current_version
            migrated = True
        if not self.reliability.resume_require_matching_fingerprint:
            return migrated
        if resume_force:
            return migrated
        if state.get("source_hash") != source_hash:
            raise RuntimeError(
                "Resume blocked because source changed. Use --resume-force to override."
            )
        if state.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError(
                "Resume blocked because configuration changed. Use --resume-force to override."
            )
        return migrated

