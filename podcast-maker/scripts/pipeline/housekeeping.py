#!/usr/bin/env python3
from __future__ import annotations

"""Disk-space checks and retention cleanup for checkpoint/log artifacts."""

import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Iterable, List

from .logging_utils import Logger


@dataclass
class CleanupReport:
    """Summary stats returned by cleanup operations."""

    deleted_files: int
    deleted_bytes: int
    kept_files: int


def ensure_min_free_disk(path: str, min_free_mb: int) -> None:
    """Raise when free disk space is below required threshold."""
    usage = shutil.disk_usage(path)
    free_mb = usage.free // (1024 * 1024)
    if free_mb < min_free_mb:
        raise RuntimeError(
            f"Not enough free disk space: {free_mb}MB available, {min_free_mb}MB required"
        )


def _iter_files(base_dir: str) -> Iterable[str]:
    """Yield files recursively under base dir."""
    if not os.path.exists(base_dir):
        return
    for root, _, files in os.walk(base_dir):
        for name in files:
            yield os.path.join(root, name)


def _safe_mtime(path: str) -> float:
    """Return file mtime or inf when unavailable."""
    try:
        return float(os.path.getmtime(path))
    except OSError:
        # File may vanish between os.walk and sort under concurrent cleanup/writes.
        return float("inf")


def _is_under(path: str, root: str) -> bool:
    """Return True when path is located under root directory."""
    try:
        common = os.path.commonpath([os.path.abspath(path), os.path.abspath(root)])
    except ValueError:
        return False
    return common == os.path.abspath(root)


def _detect_failed_recent_roots(base_dir: str, *, recent_seconds: int) -> List[str]:
    """Detect recent failed run roots that should be protected from cleanup."""
    now = time.time()
    protected: List[str] = []
    for root, _, files in os.walk(base_dir):
        for name in ("run_summary.json", "podcast_run_summary.json"):
            if name not in files:
                continue
            summary_path = os.path.join(root, name)
            try:
                age = now - os.path.getmtime(summary_path)
                if age > recent_seconds:
                    continue
                with open(summary_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if str(payload.get("status", "")).lower() == "failed":
                    protected.append(root)
            except Exception:
                continue
    return protected


def _file_category(path: str) -> str:
    """Classify file into cleanup category."""
    lower = path.lower()
    if lower.endswith(".log"):
        return "log"
    if "/segments/" in lower or "\\segments\\" in lower:
        return "intermediate_audio"
    return "checkpoint"


def cleanup_dir(
    *,
    base_dir: str,
    retention_days: int,
    max_storage_mb: int,
    retention_log_days: int = 7,
    retention_intermediate_audio_days: int = 3,
    max_log_storage_mb: int = 0,
    protect_failed_recent_hours: int = 24,
    force_clean: bool = False,
    logger: Logger,
    dry_run: bool = False,
) -> CleanupReport:
    """Cleanup directory by retention age and storage budget caps."""
    if not os.path.exists(base_dir):
        return CleanupReport(deleted_files=0, deleted_bytes=0, kept_files=0)

    now = time.time()
    checkpoint_age_s = retention_days * 86400
    log_age_s = max(1, retention_log_days) * 86400
    intermediate_age_s = max(1, retention_intermediate_audio_days) * 86400
    protected_roots: List[str] = []
    if not force_clean:
        protected_roots = _detect_failed_recent_roots(
            base_dir,
            recent_seconds=max(1, protect_failed_recent_hours) * 3600,
        )
        if protected_roots:
            logger.info("cleanup_protected_failed_roots", count=len(protected_roots))

    files = sorted(_iter_files(base_dir), key=_safe_mtime)
    deleted_files = 0
    deleted_bytes = 0

    # First pass: retention by age.
    for path in files:
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not force_clean and any(_is_under(path, root) for root in protected_roots):
            continue
        age = now - st.st_mtime
        category = _file_category(path)
        if category == "log":
            max_age_s = log_age_s
        elif category == "intermediate_audio":
            max_age_s = intermediate_age_s
        else:
            max_age_s = checkpoint_age_s
        if age <= max_age_s:
            continue
        if dry_run:
            logger.info("cleanup_candidate_old_file", path=path, age_s=int(age))
            continue
        try:
            os.remove(path)
            deleted_files += 1
            deleted_bytes += st.st_size
        except OSError:
            pass

    # Second pass: trim to storage budgets by deleting oldest.
    files_after = sorted(_iter_files(base_dir), key=_safe_mtime)

    def _trim_group(group_files: List[str], cap_mb: int, tag: str) -> None:
        """Trim oldest files in one category until cap is met."""
        nonlocal deleted_files, deleted_bytes
        cap_bytes = cap_mb * 1024 * 1024
        if cap_bytes <= 0:
            return
        total = 0
        sizes = {}
        for p in group_files:
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            sizes[p] = sz
            total += sz
        if total <= cap_bytes:
            return
        for p in group_files:
            if total <= cap_bytes:
                break
            if not force_clean and any(_is_under(p, root) for root in protected_roots):
                continue
            size = sizes.get(p)
            if size is None:
                continue
            if dry_run:
                logger.info("cleanup_candidate_size_file", category=tag, path=p, size=size)
                total -= size
                continue
            try:
                os.remove(p)
                deleted_files += 1
                deleted_bytes += size
                total -= size
            except OSError:
                pass

    log_files = [p for p in files_after if _file_category(p) == "log"]
    non_log_files = [p for p in files_after if _file_category(p) != "log"]
    _trim_group(non_log_files, max_storage_mb, "checkpoint")
    _trim_group(log_files, max_log_storage_mb, "log")

    kept_files = sum(1 for _ in _iter_files(base_dir))
    return CleanupReport(
        deleted_files=deleted_files,
        deleted_bytes=deleted_bytes,
        kept_files=kept_files,
    )

