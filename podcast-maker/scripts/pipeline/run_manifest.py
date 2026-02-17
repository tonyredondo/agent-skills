#!/usr/bin/env python3
from __future__ import annotations

"""Run manifest helpers for cross-stage handoff and observability.

The manifest tracks script/audio stage status and feeds a lightweight pipeline
summary used by tooling and incident triage.
"""

import json
import os
import time
from typing import Any, Dict


RUN_MANIFEST_FILENAME = "run_manifest.json"
PIPELINE_SUMMARY_FILENAME = "pipeline_summary.json"


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write manifest/summary JSON atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def resolve_episode_id(*, output_path: str, override: str | None = None) -> str:
    """Resolve stable episode id from override or output filename."""
    candidate = str(override or "").strip()
    if candidate:
        if os.path.basename(candidate) != candidate:
            raise ValueError("episode_id must be a plain name without path separators")
        return candidate
    return os.path.splitext(os.path.basename(output_path))[0] or "episode"


def run_manifest_path(*, checkpoint_dir: str, episode_id: str) -> str:
    """Return manifest path for an episode."""
    return os.path.join(checkpoint_dir, episode_id, RUN_MANIFEST_FILENAME)


def pipeline_summary_path(*, checkpoint_dir: str, episode_id: str) -> str:
    """Return pipeline summary path for an episode."""
    return os.path.join(checkpoint_dir, episode_id, PIPELINE_SUMMARY_FILENAME)


def load_manifest(path: str) -> Dict[str, Any] | None:
    """Load manifest from disk when valid dict payload exists."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def normalize_path_for_compare(path: str) -> str:
    """Normalize path for robust equality checks."""
    value = str(path or "").strip()
    if not value:
        return ""
    return os.path.abspath(os.path.realpath(value))


def manifest_script_path_matches(*, manifest: Dict[str, Any] | None, script_path: str) -> bool:
    """Check whether provided script path matches manifest script pointer."""
    if not isinstance(manifest, dict):
        return True
    manifest_script_path = normalize_path_for_compare(str(manifest.get("script_output_path", "")))
    if not manifest_script_path:
        return True
    return manifest_script_path == normalize_path_for_compare(script_path)


def init_manifest(
    *,
    checkpoint_dir: str,
    episode_id: str,
    run_token: str,
    script_output_path: str,
    script_checkpoint_dir: str,
    audio_checkpoint_dir: str,
) -> Dict[str, Any]:
    """Create initial v2 manifest and matching pipeline summary."""
    now = int(time.time())
    manifest = {
        "manifest_version": 2,
        "episode_id": episode_id,
        "run_token": run_token,
        "script_output_path": script_output_path,
        "script_checkpoint_dir": script_checkpoint_dir,
        "audio_checkpoint_dir": audio_checkpoint_dir,
        "status_by_stage": {
            "script": "running",
            "audio": "not_started",
            "bundle": "not_started",
        },
        "script": {
            "started_at": now,
            "completed_at": None,
            "status": "running",
            "failure_kind": None,
            "run_summary_path": "",
            "quality_report_path": "",
        },
        "audio": {
            "started_at": None,
            "completed_at": None,
            "status": "not_started",
            "failure_kind": None,
            "podcast_run_summary_path": "",
        },
        "updated_at": now,
    }
    path = run_manifest_path(checkpoint_dir=checkpoint_dir, episode_id=episode_id)
    _atomic_write_json(path, manifest)
    _write_pipeline_summary(
        checkpoint_dir=checkpoint_dir,
        episode_id=episode_id,
        manifest=manifest,
    )
    return manifest


def update_manifest(
    *,
    checkpoint_dir: str,
    episode_id: str,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge partial updates into manifest and refresh pipeline summary."""
    path = run_manifest_path(checkpoint_dir=checkpoint_dir, episode_id=episode_id)
    current = load_manifest(path) or {}
    merged = dict(current)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**dict(merged.get(key, {})), **value}
        else:
            merged[key] = value
    merged["updated_at"] = int(time.time())
    _atomic_write_json(path, merged)
    _write_pipeline_summary(
        checkpoint_dir=checkpoint_dir,
        episode_id=episode_id,
        manifest=merged,
    )
    return merged


def _write_pipeline_summary(
    *,
    checkpoint_dir: str,
    episode_id: str,
    manifest: Dict[str, Any],
) -> None:
    """Derive compact pipeline status summary from full manifest."""
    status_by_stage = manifest.get("status_by_stage", {}) if isinstance(manifest, dict) else {}
    script_status = ""
    audio_status = ""
    if isinstance(status_by_stage, dict):
        script_status = str(status_by_stage.get("script", "")).strip().lower()
        audio_status = str(status_by_stage.get("audio", "")).strip().lower()
    overall_status = "running"
    if script_status in {"failed", "interrupted"}:
        overall_status = script_status
    elif audio_status in {"failed", "interrupted"}:
        # If audio reached a terminal failure/interruption state, prefer that
        # even when script status is missing or non-standard.
        overall_status = audio_status
    elif script_status == "completed":
        if audio_status in {"completed"}:
            overall_status = "completed"
        else:
            # Once script is completed, any non-terminal/unknown audio state is
            # treated as partial handoff rather than "running".
            overall_status = "partial"
    elif audio_status == "completed":
        # Audio can be terminal even if script stage is missing/corrupted in
        # status_by_stage. Treat as partial, not running.
        overall_status = "partial"
    summary = {
        "manifest_version": 2,
        "episode_id": str(manifest.get("episode_id", episode_id)),
        "run_token": str(manifest.get("run_token", "")),
        "overall_status": overall_status,
        "status_by_stage": status_by_stage if isinstance(status_by_stage, dict) else {},
        "script": manifest.get("script", {}),
        "audio": manifest.get("audio", {}),
        "updated_at": int(manifest.get("updated_at", int(time.time()))),
    }
    path = pipeline_summary_path(checkpoint_dir=checkpoint_dir, episode_id=episode_id)
    _atomic_write_json(path, summary)

