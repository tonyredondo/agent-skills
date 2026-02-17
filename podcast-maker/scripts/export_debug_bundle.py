#!/usr/bin/env python3
from __future__ import annotations

"""Build portable debug bundles for script/audio incident triage."""

import argparse
import json
import os
import sys
import time
import zipfile
from typing import Dict, List, Tuple


SCRIPT_RUN_FILES = (
    "script_checkpoint.json",
    "run_summary.json",
    "quality_report_initial.json",
    "quality_report.json",
    "run_manifest.json",
    "pipeline_summary.json",
)

AUDIO_RUN_FILES = (
    "audio_manifest.json",
    "run_summary.json",
    "podcast_run_summary.json",
    "quality_report.json",
    "normalized_script.json",
)

DEFAULT_LOG_CANDIDATES = (
    "./podcast_run_logs.txt",
    "./run.log",
)

SENSITIVE_ENV_TOKENS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH")

SAFE_ENV_KEYS = {
    "PODCAST_DURATION_PROFILE",
    "TARGET_MINUTES",
    "WORDS_PER_MIN",
    "MIN_WORDS",
    "MAX_WORDS",
    "ALLOW_RAW_ONLY",
    "CHECKPOINT_VERSION",
    "OPENAI_CIRCUIT_BREAKER_FAILURES",
}

SAFE_ENV_PREFIXES = (
    "SCRIPT_",
    "TTS_",
    "SLO_",
    "RETENTION_",
    "MAX_",
    "MIN_FREE_DISK_",
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for debug bundle export."""
    parser = argparse.ArgumentParser(description="Create a debug bundle ZIP for podcast runs.")
    parser.add_argument("episode_id", help="Episode id used by checkpoint directories")
    parser.add_argument(
        "--script-checkpoint-dir",
        default=os.environ.get("SCRIPT_CHECKPOINT_DIR", "./.script_checkpoints"),
        help="Script checkpoint base directory",
    )
    parser.add_argument(
        "--audio-checkpoint-dir",
        default=os.environ.get("AUDIO_CHECKPOINT_DIR", "./.audio_checkpoints"),
        help="Audio checkpoint base directory",
    )
    parser.add_argument("--script-path", default="", help="Optional script.json path")
    parser.add_argument("--source-path", default="", help="Optional source text path")
    parser.add_argument(
        "--log-path",
        action="append",
        default=[],
        help="Optional log file path (can be repeated)",
    )
    parser.add_argument("--output", default="", help="Output .zip path")
    return parser.parse_args(argv)


def _safe_env_snapshot() -> Dict[str, str]:
    """Capture a sanitized environment snapshot excluding sensitive keys."""
    out: Dict[str, str] = {}
    for key in sorted(os.environ.keys()):
        upper = key.upper()
        if any(token in upper for token in SENSITIVE_ENV_TOKENS):
            continue
        if key in SAFE_ENV_KEYS or any(key.startswith(prefix) for prefix in SAFE_ENV_PREFIXES):
            out[key] = str(os.environ.get(key, ""))
    return out


def _archive_name(path: str) -> str:
    """Map absolute path to stable archive-relative name."""
    abs_path = os.path.abspath(path)
    cwd = os.path.abspath(os.getcwd())
    if abs_path == cwd:
        return "."
    if abs_path.startswith(cwd + os.sep):
        return os.path.relpath(abs_path, cwd)
    sanitized = abs_path.strip(os.sep).replace(os.sep, "__")
    if not sanitized:
        sanitized = os.path.basename(abs_path) or "root"
    return os.path.join("external", sanitized)


def _read_json_dict(path: str) -> Dict[str, object] | None:
    """Load JSON dict payload from disk when valid."""
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


def _report_entry(*, status: str, path: str, reason: str = "", category: str = "") -> Dict[str, str]:
    """Build one path status entry for bundle report."""
    return {
        "status": str(status),
        "path": os.path.abspath(path) if str(path).strip() else "",
        "archive_name": _archive_name(path) if str(path).strip() else "",
        "category": str(category or ""),
        "reason": str(reason or ""),
    }


def _virtual_found_entry(*, archive_name: str, reason: str, category: str) -> Dict[str, str]:
    """Build status entry for virtual/generated bundle artifacts."""
    return {
        "status": "found",
        "path": "",
        "archive_name": str(archive_name),
        "category": str(category or ""),
        "reason": str(reason or ""),
    }


def _resolve_manifest_pointer_path(*, raw_path: str, run_dir: str, checkpoint_dir: str) -> str:
    """Resolve manifest-relative pointer path to concrete absolute path."""
    candidate = str(raw_path or "").strip()
    if not candidate:
        return ""
    if os.path.isabs(candidate):
        return candidate
    run_candidate = os.path.abspath(os.path.join(run_dir, candidate))
    if os.path.exists(run_candidate):
        return run_candidate
    checkpoint_candidate = os.path.abspath(os.path.join(checkpoint_dir, candidate))
    if os.path.exists(checkpoint_candidate):
        return checkpoint_candidate
    cwd_candidate = os.path.abspath(candidate)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return run_candidate


def _discover_checkpoint_root(
    *,
    base_dir: str,
    episode_id: str,
    prefer_manifest: bool,
    episode_aliases: List[str] | None = None,
) -> tuple[str, Dict[str, object]]:
    """Discover most likely checkpoint root for an episode id."""
    base = os.path.abspath(str(base_dir or "."))
    episode = str(episode_id or "").strip()
    aliases: List[str] = []
    for item in list(episode_aliases or []):
        name = str(item or "").strip()
        if name and name not in aliases:
            aliases.append(name)
    episodes_to_probe = [episode] + [name for name in aliases if name != episode]
    roots: List[str] = []

    def _push(path: str) -> None:
        candidate = os.path.abspath(path)
        if candidate not in roots:
            roots.append(candidate)

    _push(base)
    parent = os.path.dirname(base)
    cwd = os.path.abspath(os.getcwd())
    for root in (
        parent,
        cwd,
        os.path.join(parent, ".script_checkpoints"),
        os.path.join(parent, ".audio_checkpoints"),
        os.path.join(cwd, ".script_checkpoints"),
        os.path.join(cwd, ".audio_checkpoints"),
        os.path.join(parent, "script_ckpt"),
        os.path.join(parent, "audio_ckpt"),
        os.path.join(cwd, "script_ckpt"),
        os.path.join(cwd, "audio_ckpt"),
    ):
        _push(root)

    best = base
    reason = "base_dir_default"
    manifest_rel = os.path.join(episode, "run_manifest.json")
    episode_hit_rels = [os.path.join(name) for name in episodes_to_probe]
    for candidate in roots:
        if prefer_manifest and os.path.exists(os.path.join(candidate, manifest_rel)):
            best = candidate
            reason = "manifest_found"
            break
        if any(os.path.isdir(os.path.join(candidate, rel)) for rel in episode_hit_rels):
            best = candidate
            reason = "episode_dir_found"
            break
    diagnostics: Dict[str, object] = {
        "base_dir": base,
        "episode_id": episode,
        "episode_aliases": aliases,
        "candidate_roots": roots,
        "chosen_root": best,
        "reason": reason,
    }
    return best, diagnostics


def _read_git_commit() -> str:
    """Resolve current git commit hash without invoking git CLI."""
    root = os.path.abspath(os.getcwd())
    current = root
    while True:
        git_dir = os.path.join(current, ".git")
        if os.path.isfile(git_dir):
            try:
                with open(git_dir, "r", encoding="utf-8") as f:
                    marker = f.read().strip()
                if marker.lower().startswith("gitdir:"):
                    candidate = marker.split(":", 1)[1].strip()
                    git_dir = (
                        os.path.abspath(os.path.join(current, candidate))
                        if not os.path.isabs(candidate)
                        else candidate
                    )
            except Exception:
                git_dir = os.path.join(current, ".git")
        head_path = os.path.join(git_dir, "HEAD")
        if os.path.exists(head_path):
            try:
                with open(head_path, "r", encoding="utf-8") as f:
                    head = f.read().strip()
                if head.startswith("ref:"):
                    ref = head.split(" ", 1)[-1].strip()
                    ref_path = os.path.join(git_dir, ref.replace("/", os.sep))
                    if os.path.exists(ref_path):
                        with open(ref_path, "r", encoding="utf-8") as rf:
                            return rf.read().strip()
                    packed_refs = os.path.join(git_dir, "packed-refs")
                    if os.path.exists(packed_refs):
                        try:
                            with open(packed_refs, "r", encoding="utf-8") as pf:
                                for row in pf:
                                    line = row.strip()
                                    if not line or line.startswith("#") or line.startswith("^"):
                                        continue
                                    parts = line.split(" ", 1)
                                    if len(parts) == 2 and parts[1].strip() == ref:
                                        return parts[0].strip()
                        except Exception:
                            pass
                    return head
                return head
            except Exception:
                return ""
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return ""


def _skill_version() -> str:
    """Resolve skill version marker from environment."""
    version = str(os.environ.get("PODCAST_MAKER_VERSION", "")).strip()
    if version:
        return version
    return "unknown"


def _append_path_status(
    *,
    requested: List[str],
    missing: List[str],
    collection_report: List[Dict[str, str]],
    path: str,
    category: str,
    missing_reason: str = "",
    not_applicable_reason: str = "",
) -> None:
    """Append path status and include existing paths in archive request list."""
    candidate = str(path or "").strip()
    if not candidate:
        return
    if not_applicable_reason:
        collection_report.append(
            _report_entry(
                status="not_applicable",
                path=candidate,
                category=category,
                reason=not_applicable_reason,
            )
        )
        return
    if not os.path.exists(candidate):
        reason = str(missing_reason or "not_found")
        missing.append(os.path.abspath(candidate))
        collection_report.append(
            _report_entry(
                status="missing",
                path=candidate,
                category=category,
                reason=reason,
            )
        )
        return
    try:
        with open(candidate, "rb") as f:
            _ = f.read(1)
    except Exception as exc:  # noqa: BLE001
        missing.append(os.path.abspath(candidate))
        collection_report.append(
            _report_entry(
                status="read_error",
                path=candidate,
                category=category,
                reason=str(exc),
            )
        )
        return
    requested.append(os.path.abspath(candidate))
    collection_report.append(
        _report_entry(
            status="found",
            path=candidate,
            category=category,
        )
    )


def _build_consistency_warnings(
    *,
    expected_episode_id: str,
    manifest_payload: Dict[str, object] | None,
    include_paths: List[str],
) -> List[str]:
    """Generate cross-artifact consistency warnings for troubleshooting."""
    warnings: List[str] = []
    expected = str(expected_episode_id or "").strip()
    if isinstance(manifest_payload, dict):
        manifest_episode = str(manifest_payload.get("episode_id", "")).strip()
        if expected and manifest_episode and manifest_episode != expected:
            warnings.append(
                f"episode_id_mismatch: metadata={expected} manifest={manifest_episode}"
            )
    for path in include_paths:
        filename = os.path.basename(path)
        if filename not in {"run_summary.json", "podcast_run_summary.json", "pipeline_summary.json", "run_manifest.json"}:
            continue
        payload = _read_json_dict(path)
        if not isinstance(payload, dict):
            continue
        value = str(payload.get("episode_id", "")).strip()
        if expected and value and value != expected:
            warnings.append(
                f"episode_id_mismatch: metadata={expected} file={filename} value={value}"
            )
    return sorted(set(warnings))


def _collect_paths(
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[Dict[str, str]], List[str], str, Dict[str, object]]:
    """Collect files and metadata candidates for bundle export."""
    episode = str(args.episode_id).strip()
    script_ckpt_input = os.path.abspath(str(args.script_checkpoint_dir))
    audio_ckpt_input = os.path.abspath(str(args.audio_checkpoint_dir))
    script_aliases: List[str] = []
    script_path_arg = str(args.script_path or "").strip()
    if script_path_arg:
        script_basename = os.path.splitext(os.path.basename(script_path_arg))[0].strip()
        if script_basename:
            script_aliases.append(script_basename)
    script_ckpt_dir, script_root_diag = _discover_checkpoint_root(
        base_dir=script_ckpt_input,
        episode_id=episode,
        prefer_manifest=True,
        episode_aliases=script_aliases,
    )
    audio_ckpt_dir, audio_root_diag = _discover_checkpoint_root(
        base_dir=audio_ckpt_input,
        episode_id=episode,
        prefer_manifest=False,
    )
    collection_diagnostics: Dict[str, object] = {
        "script_root": script_root_diag,
        "audio_root": audio_root_diag,
    }

    manifest_path = os.path.join(script_ckpt_dir, episode, "run_manifest.json")
    manifest_payload = _read_json_dict(manifest_path)
    use_manifest_layout = isinstance(manifest_payload, dict)

    run_episode = episode
    if use_manifest_layout:
        run_episode = str(manifest_payload.get("episode_id", "")).strip() or episode
        script_ckpt_dir = os.path.abspath(str(manifest_payload.get("script_checkpoint_dir") or script_ckpt_dir))
        audio_ckpt_dir = os.path.abspath(str(manifest_payload.get("audio_checkpoint_dir") or audio_ckpt_dir))
    collection_diagnostics["manifest_path"] = manifest_path
    collection_diagnostics["manifest_found"] = bool(use_manifest_layout)
    collection_diagnostics["resolved_script_checkpoint_dir"] = script_ckpt_dir
    collection_diagnostics["resolved_audio_checkpoint_dir"] = audio_ckpt_dir

    requested: List[str] = []
    missing: List[str] = []
    collection_report: List[Dict[str, str]] = []
    script_stage_status = ""
    script_quality_gate_executed: bool | None = None
    script_quality_stage_started: bool | None = None
    script_quality_stage_finished: bool | None = None
    script_quality_stage_interrupted: bool | None = None

    def _consume_script_run_summary(path: str) -> None:
        nonlocal script_stage_status
        nonlocal script_quality_gate_executed
        nonlocal script_quality_stage_started
        nonlocal script_quality_stage_finished
        nonlocal script_quality_stage_interrupted
        payload = _read_json_dict(path)
        if not isinstance(payload, dict):
            return
        status = str(payload.get("status", "")).strip().lower()
        if status:
            script_stage_status = status
        if "quality_gate_executed" in payload:
            script_quality_gate_executed = bool(payload.get("quality_gate_executed"))
        elif str(payload.get("script_gate_action_effective", "")).strip().lower() == "off":
            script_quality_gate_executed = False
        if "quality_stage_started" in payload:
            script_quality_stage_started = bool(payload.get("quality_stage_started"))
        if "quality_stage_finished" in payload:
            script_quality_stage_finished = bool(payload.get("quality_stage_finished"))
        if "quality_stage_interrupted" in payload:
            script_quality_stage_interrupted = bool(payload.get("quality_stage_interrupted"))

    def _script_quality_not_applicable_reason(file_name: str) -> str:
        if file_name not in {"quality_report.json", "quality_report_initial.json"}:
            return ""
        if script_quality_gate_executed is False:
            return "script_quality_gate_not_executed"
        if file_name == "quality_report_initial.json" and script_quality_stage_started is not True:
            return "quality_stage_not_started"
        if script_quality_stage_started is False:
            return "quality_stage_not_started"
        return ""

    def _script_quality_missing_reason(file_name: str, *, fallback: str) -> str:
        if file_name == "quality_report_initial.json":
            if script_quality_stage_started is True:
                return "quality_stage_started_initial_report_missing"
            return fallback
        if file_name == "quality_report.json":
            if script_quality_stage_interrupted is True and script_quality_stage_finished is not True:
                return "quality_stage_interrupted_final_report_missing"
            if script_quality_stage_started is True:
                return "quality_report_expected_missing"
            if script_quality_gate_executed is True:
                return "quality_report_expected_missing"
            return fallback
        return fallback

    def _append_interrupted_quality_final_fallback(
        *,
        initial_candidates: List[str],
        category: str,
    ) -> bool:
        if script_quality_stage_interrupted is not True or script_quality_stage_finished is True:
            return False
        for candidate in initial_candidates:
            initial_path = str(candidate or "").strip()
            if not initial_path or not os.path.exists(initial_path):
                continue
            requested.append(os.path.abspath(initial_path))
            collection_report.append(
                _report_entry(
                    status="found",
                    path=initial_path,
                    category=category,
                    reason="initial_only_due_to_interruption",
                )
            )
            return True
        return False

    if use_manifest_layout:
        script_run_dir = os.path.join(script_ckpt_dir, run_episode)
        script_block = manifest_payload.get("script", {}) if isinstance(manifest_payload, dict) else {}
        script_manifest_paths: Dict[str, str] = {}
        script_manifest_run_summary_path = ""
        script_manifest_quality_report_path = ""
        script_manifest_quality_report_initial_path = ""
        if isinstance(script_block, dict):
            for key in ("run_summary_path", "quality_report_path", "quality_report_initial_path"):
                raw_path = str(script_block.get(key, "")).strip()
                if not raw_path:
                    continue
                resolved_path = _resolve_manifest_pointer_path(
                    raw_path=raw_path,
                    run_dir=script_run_dir,
                    checkpoint_dir=script_ckpt_dir,
                )
                script_manifest_paths[key] = resolved_path
                if key == "run_summary_path":
                    script_manifest_run_summary_path = resolved_path
                elif key == "quality_report_path":
                    script_manifest_quality_report_path = resolved_path
                elif key == "quality_report_initial_path":
                    script_manifest_quality_report_initial_path = resolved_path
            if script_quality_stage_started is None and "quality_stage_started" in script_block:
                script_quality_stage_started = bool(script_block.get("quality_stage_started"))
            if script_quality_stage_finished is None and "quality_stage_finished" in script_block:
                script_quality_stage_finished = bool(script_block.get("quality_stage_finished"))
            if script_quality_stage_interrupted is None and "quality_stage_interrupted" in script_block:
                script_quality_stage_interrupted = bool(script_block.get("quality_stage_interrupted"))

        default_script_run_summary_path = os.path.join(script_run_dir, "run_summary.json")
        _consume_script_run_summary(default_script_run_summary_path)
        if (
            script_manifest_run_summary_path
            and os.path.abspath(script_manifest_run_summary_path)
            != os.path.abspath(default_script_run_summary_path)
        ):
            _consume_script_run_summary(script_manifest_run_summary_path)
        for name in SCRIPT_RUN_FILES:
            path = os.path.join(script_run_dir, name)
            manifest_pointer_override = False
            if name == "run_summary.json" and script_manifest_run_summary_path:
                manifest_pointer_override = (
                    os.path.abspath(script_manifest_run_summary_path) != os.path.abspath(path)
                    and not os.path.exists(path)
                )
            elif name == "quality_report.json" and script_manifest_quality_report_path:
                manifest_pointer_override = (
                    os.path.abspath(script_manifest_quality_report_path) != os.path.abspath(path)
                    and not os.path.exists(path)
                )
            elif name == "quality_report_initial.json" and script_manifest_quality_report_initial_path:
                manifest_pointer_override = (
                    os.path.abspath(script_manifest_quality_report_initial_path) != os.path.abspath(path)
                    and not os.path.exists(path)
                )
            if name == "quality_report.json" and not os.path.exists(path):
                if _append_interrupted_quality_final_fallback(
                    initial_candidates=[
                        script_manifest_quality_report_initial_path,
                        os.path.join(script_run_dir, "quality_report_initial.json"),
                    ],
                    category="script_checkpoint",
                ):
                    continue
            not_applicable_reason = ""
            if not os.path.exists(path):
                not_applicable_reason = (
                    "manifest_pointer_override"
                    if manifest_pointer_override
                    else _script_quality_not_applicable_reason(name)
                )
            _append_path_status(
                requested=requested,
                missing=missing,
                collection_report=collection_report,
                path=path,
                category="script_checkpoint",
                missing_reason=_script_quality_missing_reason(name, fallback="not_found"),
                not_applicable_reason=not_applicable_reason,
            )
        for key, resolved_path in script_manifest_paths.items():
            if not str(resolved_path).strip():
                continue
            if key == "run_summary_path":
                default_path = os.path.join(script_run_dir, "run_summary.json")
                pointer_name = "run_summary.json"
            elif key == "quality_report_initial_path":
                default_path = os.path.join(script_run_dir, "quality_report_initial.json")
                pointer_name = "quality_report_initial.json"
            else:
                default_path = os.path.join(script_run_dir, "quality_report.json")
                pointer_name = "quality_report.json"
            if os.path.abspath(resolved_path) == os.path.abspath(default_path):
                # Already covered via SCRIPT_RUN_FILES.
                continue
            if pointer_name == "quality_report.json" and not os.path.exists(resolved_path):
                if _append_interrupted_quality_final_fallback(
                    initial_candidates=[
                        script_manifest_quality_report_initial_path,
                        os.path.join(script_run_dir, "quality_report_initial.json"),
                    ],
                    category="manifest_pointer",
                ):
                    continue
            not_applicable_reason = (
                _script_quality_not_applicable_reason(pointer_name)
                if not os.path.exists(resolved_path)
                else ""
            )
            _append_path_status(
                requested=requested,
                missing=missing,
                collection_report=collection_report,
                path=resolved_path,
                category="manifest_pointer",
                missing_reason=_script_quality_missing_reason(pointer_name, fallback="manifest_pointer_missing"),
                not_applicable_reason=not_applicable_reason,
            )
    else:
        script_episode_candidates = [episode]
        script_path = str(args.script_path or "").strip()
        if script_path:
            script_basename = os.path.splitext(os.path.basename(script_path))[0].strip()
            if script_basename and script_basename not in script_episode_candidates:
                script_episode_candidates.append(script_basename)
        for name in SCRIPT_RUN_FILES:
            primary_path = os.path.join(script_ckpt_dir, episode, name)
            found_any = False
            for script_episode in script_episode_candidates:
                path = os.path.join(script_ckpt_dir, script_episode, name)
                if os.path.exists(path):
                    if name == "run_summary.json":
                        _consume_script_run_summary(path)
                    _append_path_status(
                        requested=requested,
                        missing=missing,
                        collection_report=collection_report,
                        path=path,
                        category="script_checkpoint",
                        missing_reason=_script_quality_missing_reason(name, fallback="not_found_in_episode_or_alias"),
                        not_applicable_reason="",
                    )
                    found_any = True
            if not found_any:
                if name == "quality_report.json":
                    initial_candidates = [
                        os.path.join(script_ckpt_dir, script_episode, "quality_report_initial.json")
                        for script_episode in script_episode_candidates
                    ]
                    if _append_interrupted_quality_final_fallback(
                        initial_candidates=initial_candidates,
                        category="script_checkpoint",
                    ):
                        continue
                _append_path_status(
                    requested=requested,
                    missing=missing,
                    collection_report=collection_report,
                    path=primary_path,
                    category="script_checkpoint",
                    missing_reason=_script_quality_missing_reason(name, fallback="not_found_in_episode_or_alias"),
                    not_applicable_reason=_script_quality_not_applicable_reason(name),
                )

    if use_manifest_layout and isinstance(manifest_payload, dict) and not script_stage_status:
        status_by_stage = manifest_payload.get("status_by_stage", {})
        if isinstance(status_by_stage, dict):
            script_stage_status = str(status_by_stage.get("script", "")).strip().lower()

    audio_run_dir = os.path.join(audio_ckpt_dir, run_episode)
    audio_summary = _read_json_dict(os.path.join(audio_run_dir, "podcast_run_summary.json")) or {}
    has_audio_executed_flag = "audio_executed" in audio_summary
    audio_executed = bool(audio_summary.get("audio_executed", True))
    audio_quality_gate_executed: bool | None = None
    if "quality_gate_executed" in audio_summary:
        audio_quality_gate_executed = bool(audio_summary.get("quality_gate_executed"))
    elif str(audio_summary.get("script_gate_action_effective", "")).strip().lower() == "off":
        audio_quality_gate_executed = False
    if use_manifest_layout and isinstance(manifest_payload, dict) and not has_audio_executed_flag:
        status_by_stage = manifest_payload.get("status_by_stage", {})
        if isinstance(status_by_stage, dict):
            manifest_audio_status = str(status_by_stage.get("audio", "")).strip().lower()
            # For script-only/early-failure runs, audio may never execute and there
            # may be no podcast_run_summary yet. In that case treat missing audio
            # artefacts as not applicable.
            if manifest_audio_status in {"not_started"}:
                audio_executed = False
    if use_manifest_layout and isinstance(manifest_payload, dict):
        audio_block = manifest_payload.get("audio", {})
        if isinstance(audio_block, dict):
            summary_pointer = str(audio_block.get("podcast_run_summary_path", "")).strip()
            if summary_pointer:
                resolved_pointer = _resolve_manifest_pointer_path(
                    raw_path=summary_pointer,
                    run_dir=audio_run_dir,
                    checkpoint_dir=audio_ckpt_dir,
                )
                _append_path_status(
                    requested=requested,
                    missing=missing,
                    collection_report=collection_report,
                    path=resolved_pointer,
                    category="manifest_pointer",
                )
    for name in AUDIO_RUN_FILES:
        path = os.path.join(audio_run_dir, name)
        audio_quality_not_applicable = (
            name == "quality_report.json"
            and audio_quality_gate_executed is False
            and not os.path.exists(path)
        )
        if not audio_executed and not os.path.exists(path):
            _append_path_status(
                requested=requested,
                missing=missing,
                collection_report=collection_report,
                path=path,
                category="audio_checkpoint",
                not_applicable_reason="audio_not_executed",
            )
            continue
        _append_path_status(
            requested=requested,
            missing=missing,
            collection_report=collection_report,
            path=path,
            category="audio_checkpoint",
            not_applicable_reason="audio_quality_gate_not_executed" if audio_quality_not_applicable else "",
        )

    script_path_arg = str(args.script_path or "").strip()
    if script_path_arg:
        script_not_generated = (
            script_stage_status in {"failed", "interrupted", "not_started"}
            and not os.path.exists(script_path_arg)
        )
        _append_path_status(
            requested=requested,
            missing=missing,
            collection_report=collection_report,
            path=script_path_arg,
            category="external_input",
            not_applicable_reason="script_not_generated" if script_not_generated else "",
        )
    else:
        if script_stage_status == "completed":
            collection_report.append(
                _report_entry(
                    status="missing",
                    path="",
                    category="external_input",
                    reason="script_path_required_when_completed",
                )
            )
        else:
            collection_report.append(
                _report_entry(
                    status="not_applicable",
                    path="",
                    category="external_input",
                    reason="script_path_not_provided",
                )
            )

    source_path_arg = str(args.source_path or "").strip()
    if source_path_arg:
        _append_path_status(
            requested=requested,
            missing=missing,
            collection_report=collection_report,
            path=source_path_arg,
            category="external_input",
        )
    else:
        collection_report.append(
            _report_entry(
                status="not_applicable",
                path="",
                category="external_input",
                reason="source_path_not_provided",
            )
        )

    explicit_logs = [str(path) for path in list(args.log_path or []) if str(path).strip()]
    explicit_log_keys = {os.path.abspath(path) for path in explicit_logs}
    seen_log_paths: set[str] = set()
    for raw_path in explicit_logs + list(DEFAULT_LOG_CANDIDATES):
        path = str(raw_path or "").strip()
        if not path:
            continue
        normalized_path = os.path.abspath(path)
        if normalized_path in seen_log_paths:
            continue
        seen_log_paths.add(normalized_path)
        if normalized_path in explicit_log_keys:
            _append_path_status(
                requested=requested,
                missing=missing,
                collection_report=collection_report,
                path=path,
                category="logs",
            )
        elif os.path.exists(path):
            _append_path_status(
                requested=requested,
                missing=missing,
                collection_report=collection_report,
                path=path,
                category="logs",
            )
        else:
            _append_path_status(
                requested=requested,
                missing=missing,
                collection_report=collection_report,
                path=path,
                category="logs",
                not_applicable_reason="log_candidate_not_found",
            )

    seen = set()
    unique_requested: List[str] = []
    for path in requested:
        if path in seen:
            continue
        seen.add(path)
        unique_requested.append(path)
    collection_diagnostics["script_stage_status"] = script_stage_status
    collection_diagnostics["script_quality_gate_executed"] = script_quality_gate_executed
    collection_diagnostics["script_quality_stage_started"] = script_quality_stage_started
    collection_diagnostics["script_quality_stage_finished"] = script_quality_stage_finished
    collection_diagnostics["script_quality_stage_interrupted"] = script_quality_stage_interrupted
    consistency_warnings = _build_consistency_warnings(
        expected_episode_id=episode,
        manifest_payload=manifest_payload,
        include_paths=unique_requested,
    )
    return (
        unique_requested,
        sorted(set(missing)),
        collection_report,
        consistency_warnings,
        run_episode,
        collection_diagnostics,
    )


def create_debug_bundle(args: argparse.Namespace) -> str:
    """Create debug bundle ZIP and return output file path."""
    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_path = str(args.output or "").strip()
    if not output_path:
        output_path = f"./debug_bundle_{args.episode_id}_{now}.zip"
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    (
        include_paths,
        missing_paths,
        collection_report,
        consistency_warnings,
        resolved_episode_id,
        collection_diagnostics,
    ) = _collect_paths(args)
    virtual_files: Dict[str, str] = {}
    should_reconstruct_script = any(
        str(item.get("category", "")).strip() == "external_input"
        and str(item.get("status", "")).strip().lower() == "not_applicable"
        and str(item.get("reason", "")).strip() == "script_not_generated"
        for item in collection_report
    )
    if should_reconstruct_script:
        # If script output is missing but checkpoint lines exist, include a
        # reconstructed script artifact to improve replay/debug workflows.
        checkpoint_candidates = [
            str(item.get("path", "")).strip()
            for item in collection_report
            if str(item.get("category", "")).strip() == "script_checkpoint"
            and str(item.get("status", "")).strip().lower() == "found"
            and str(item.get("path", "")).strip().endswith("script_checkpoint.json")
        ]
        for checkpoint_path in checkpoint_candidates:
            checkpoint_payload = _read_json_dict(checkpoint_path)
            lines = checkpoint_payload.get("lines") if isinstance(checkpoint_payload, dict) else None
            if not isinstance(lines, list) or not lines:
                continue
            virtual_name = "reconstructed_script_from_checkpoint.json"
            virtual_files[virtual_name] = (
                json.dumps({"lines": lines}, indent=2, ensure_ascii=False) + "\n"
            )
            collection_report.append(
                _virtual_found_entry(
                    archive_name=virtual_name,
                    reason="reconstructed_from_script_checkpoint",
                    category="derived",
                )
            )
            break

    archive_paths = sorted(set([_archive_name(path) for path in include_paths] + list(virtual_files.keys())))
    file_tree = "\n".join(archive_paths) if archive_paths else "(no collected files)"
    collection_status_counts: Dict[str, int] = {}
    for item in collection_report:
        status = str(item.get("status", "")).strip().lower() or "unknown"
        collection_status_counts[status] = int(collection_status_counts.get(status, 0)) + 1
    collection_complete = all(
        str(item.get("status", "")).strip().lower() in {"found", "not_applicable"}
        for item in collection_report
    )
    metadata = {
        "bundle_version": 2,
        "generated_at": int(time.time()),
        "episode_id": str(args.episode_id),
        "resolved_episode_id": str(resolved_episode_id),
        "invocation": " ".join(sys.argv),
        "effective_params": {
            "episode_id": str(args.episode_id),
            "script_checkpoint_dir": os.path.abspath(args.script_checkpoint_dir),
            "audio_checkpoint_dir": os.path.abspath(args.audio_checkpoint_dir),
            "script_path": os.path.abspath(str(args.script_path)) if str(args.script_path).strip() else "",
            "source_path": os.path.abspath(str(args.source_path)) if str(args.source_path).strip() else "",
            "log_paths": [
                os.path.abspath(str(p))
                for p in list(args.log_path or [])
                if str(p).strip()
            ],
        },
        "skill_version": _skill_version(),
        "git_commit": _read_git_commit(),
        "script_checkpoint_dir": os.path.abspath(args.script_checkpoint_dir),
        "audio_checkpoint_dir": os.path.abspath(args.audio_checkpoint_dir),
        "included_files": archive_paths,
        "missing_candidates": sorted(_archive_name(path) for path in missing_paths),
        "collection_report_path": "collection_report.json",
        "collection_complete": bool(collection_complete),
        "collection_status_counts": collection_status_counts,
        "collection_diagnostics": collection_diagnostics,
        "derived_files": sorted(list(virtual_files.keys())),
        "consistency_warnings": consistency_warnings,
        "env": _safe_env_snapshot(),
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file_path in include_paths:
            bundle.write(file_path, arcname=_archive_name(file_path))
        for archive_name, content in virtual_files.items():
            bundle.writestr(archive_name, content)
        bundle.writestr("debug_bundle_metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        bundle.writestr("debug_bundle_tree.txt", file_tree + "\n")
        bundle.writestr("collection_report.json", json.dumps(collection_report, indent=2, ensure_ascii=False) + "\n")

    return output_path


def main(argv: List[str] | None = None) -> int:
    """CLI entrypoint for debug bundle creation."""
    args = parse_args(argv)
    out = create_debug_bundle(args)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

