#!/usr/bin/env python3
from __future__ import annotations

"""Audio checkpoint persistence with lock ownership and resume validation."""

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import ReliabilityConfig


def _content_type_for_extension(extension: str) -> str:
    normalized = str(extension or "").strip().lower().lstrip(".")
    if normalized == "wav":
        return "audio/wav"
    if normalized == "ogg":
        return "audio/ogg"
    if normalized == "flac":
        return "audio/flac"
    if normalized == "webm":
        return "audio/webm"
    return "audio/mpeg"


def _extension_from_file_name(file_name: str) -> str:
    value = str(file_name or "").strip()
    if "." not in value:
        return ""
    return str(value.rsplit(".", 1)[-1]).strip().lower()


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid partial manifest files."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _atomic_create_json(path: str, payload: Dict[str, Any]) -> None:
    """Create JSON atomically and fail when path already exists."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _parse_version(value: Any) -> Tuple[int, int]:
    """Parse `major.minor` version values into integer tuple."""
    if isinstance(value, int):
        return value, 0
    if isinstance(value, str):
        m = re.match(r"^\s*(\d+)(?:\.(\d+))?\s*$", value)
        if m:
            return int(m.group(1)), int(m.group(2) or 0)
    return -1, -1


@dataclass
class AudioCheckpointStore:
    """Owns audio manifest files, lock lifecycle, and resume checks."""

    base_dir: str
    episode_id: str
    reliability: ReliabilityConfig
    last_corrupt_backup_path: Optional[str] = None
    last_corrupt_error: str = ""
    _lock_token: Optional[str] = field(default=None, init=False, repr=False)
    _lock_heartbeat_stop: Optional[threading.Event] = field(default=None, init=False, repr=False)
    _lock_heartbeat_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize run-scoped directories and file paths."""
        os.makedirs(self.base_dir, exist_ok=True)
        self.run_dir = os.path.join(self.base_dir, self.episode_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.run_dir, "audio_manifest.json")
        self.summary_path = os.path.join(self.run_dir, "run_summary.json")
        self.lock_path = os.path.join(self.run_dir, ".lock")
        self.segments_dir = os.path.join(self.run_dir, "segments")
        os.makedirs(self.segments_dir, exist_ok=True)

    def _heartbeat_interval_seconds(self) -> float:
        """Return lock heartbeat interval derived from lock TTL."""
        return float(max(1, min(30, int(self.reliability.lock_ttl_seconds / 3))))

    def _stop_lock_heartbeat(self) -> None:
        """Stop and join lock heartbeat thread if running."""
        stop = self._lock_heartbeat_stop
        thread = self._lock_heartbeat_thread
        self._lock_heartbeat_stop = None
        self._lock_heartbeat_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _refresh_lock_timestamp(self) -> None:
        """Refresh lock mtime when current token still owns lock."""
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
        """Periodic lock timestamp refresher loop."""
        while not stop.wait(interval_s):
            self._refresh_lock_timestamp()

    def _start_lock_heartbeat(self) -> None:
        """Start daemon thread that refreshes lock timestamp."""
        self._stop_lock_heartbeat()
        stop = threading.Event()
        thread = threading.Thread(
            target=self._lock_heartbeat_loop,
            args=(stop, self._heartbeat_interval_seconds()),
            daemon=True,
            name=f"audio-lock-heartbeat-{self.episode_id}",
        )
        self._lock_heartbeat_stop = stop
        self._lock_heartbeat_thread = thread
        thread.start()

    def acquire_lock(self, force_unlock: bool = False) -> None:
        """Acquire exclusive audio checkpoint lock, honoring TTL rules."""
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
                # Lock already exists: inspect metadata and reclaim only if
                # forced or stale by TTL.
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
                    raise RuntimeError(
                        "Audio checkpoint lock exists but is unreadable/invalid. "
                        f"Use --force-unlock if this lock is stale. ({exc})"
                    )
                if not isinstance(lock_data, dict) or "ts" not in lock_data:
                    if force_unlock:
                        try:
                            os.remove(self.lock_path)
                        except OSError:
                            pass
                        continue
                    raise RuntimeError(
                        "Audio checkpoint lock exists but has invalid metadata. "
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
                    raise RuntimeError(
                        "Audio checkpoint lock exists but has invalid timestamp metadata. "
                        "Use --force-unlock if this lock is stale."
                    )
                try:
                    lock_mtime = int(os.path.getmtime(self.lock_path))
                except OSError:
                    lock_mtime = ts
                age = now - max(ts, lock_mtime)
                if force_unlock or age > self.reliability.lock_ttl_seconds:
                    # Reclaim stale lock and retry atomic creation.
                    try:
                        os.remove(self.lock_path)
                    except OSError:
                        pass
                    continue
                owner = lock_data.get("pid", "unknown")
                raise RuntimeError(f"Audio checkpoint lock active (pid={owner}, age={age}s)")

    def release_lock(self) -> None:
        """Release lock only if token matches current owner."""
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
        """Load audio manifest; quarantine corrupt payloads when unreadable."""
        self.last_corrupt_backup_path = None
        self.last_corrupt_error = ""
        if not os.path.exists(self.manifest_path):
            return None
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            ts = int(time.time())
            backup = f"{self.manifest_path}.corrupt.{ts}.json"
            try:
                os.replace(self.manifest_path, backup)
                self.last_corrupt_backup_path = backup
            except OSError:
                self.last_corrupt_backup_path = ""
            self.last_corrupt_error = str(exc)
            return None

    def save(self, manifest: Dict[str, Any]) -> None:
        """Persist current audio manifest payload."""
        _atomic_write_json(self.manifest_path, manifest)

    def init_manifest(
        self,
        *,
        config_fingerprint: str,
        script_hash: str,
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create initial manifest for a new TTS synthesis run."""
        return {
            "checkpoint_version": self.reliability.checkpoint_version,
            "episode_id": self.episode_id,
            "config_fingerprint": config_fingerprint,
            "script_hash": script_hash,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "segments": segments,
            "status": "running",
        }

    def validate_resume(
        self,
        manifest: Dict[str, Any],
        *,
        config_fingerprint: str,
        script_hash: str,
        resume_force: bool,
    ) -> bool:
        """Validate manifest compatibility for resume behavior."""
        if not manifest or not isinstance(manifest, dict):
            raise RuntimeError("Invalid audio manifest state for resume")
        current_version = self.reliability.checkpoint_version
        existing_version = manifest.get("checkpoint_version")
        current_major, _ = _parse_version(current_version)
        existing_major, _ = _parse_version(existing_version)
        if existing_major != current_major:
            raise RuntimeError(
                f"Manifest version mismatch: {existing_version} != {current_version}. "
                "Use a clean run for major version changes."
            )
        migrated = False
        if existing_version != current_version:
            manifest["migrated_from_version"] = existing_version
            manifest["checkpoint_version"] = current_version
            migrated = True
        segments = manifest.get("segments", [])
        if isinstance(segments, list):
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                ext = str(seg.get("audio_format", "")).strip().lower()
                if not ext:
                    ext = _extension_from_file_name(str(seg.get("file_name", "")))
                    if ext:
                        seg["audio_format"] = ext
                        migrated = True
                content_type = str(seg.get("content_type", "")).split(";", 1)[0].strip().lower()
                if not content_type and ext:
                    seg["content_type"] = _content_type_for_extension(ext)
                    migrated = True
        if not self.reliability.resume_require_matching_fingerprint or resume_force:
            return migrated
        if manifest.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError("Resume blocked: audio config fingerprint changed")
        if manifest.get("script_hash") != script_hash:
            raise RuntimeError("Resume blocked: script content changed")
        return migrated

