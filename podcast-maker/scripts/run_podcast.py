#!/usr/bin/env python3
from __future__ import annotations

"""Orchestrate the full podcast pipeline (script -> audio).

This entrypoint keeps a shared `episode_id` and `run_token` across both
stages so manifests, checkpoints, and summaries remain correlated even when
retries or resumes happen.
"""

import argparse
import os
import sys
import uuid

import make_podcast
import make_script
from pipeline.run_manifest import resolve_episode_id


def _basename_arg(value: str) -> str:
    """Validate CLI basename/episode arguments.

    The value must be a plain filename token, never a path.
    """
    name = str(value).strip()
    if not name:
        raise argparse.ArgumentTypeError("name must not be empty")
    if os.path.basename(name) != name:
        raise argparse.ArgumentTypeError("name must be a file name, not a path")
    return name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run full podcast pipeline (script + audio) with shared run identity.",
    )
    parser.add_argument("input_path", help="Source text file for script generation")
    parser.add_argument("outdir", help="Output directory for audio artifacts")
    parser.add_argument(
        "basename",
        nargs="?",
        default="episode",
        type=_basename_arg,
        help="Output base filename for final audio files",
    )
    parser.add_argument("--script-path", default=None, help="Optional script JSON path")
    parser.add_argument("--episode-id", type=_basename_arg, default=None)
    parser.add_argument("--profile", choices=["short", "standard", "long"], default=None)
    parser.add_argument("--target-minutes", type=float, default=None)
    parser.add_argument("--words-per-min", type=float, default=None)
    parser.add_argument("--min-words", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-force", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    parser.add_argument("--allow-raw-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run-cleanup", action="store_true")
    parser.add_argument("--force-clean", action="store_true")
    return parser.parse_args(argv)


def _append_optional(argv: list[str], flag: str, value: object | None) -> None:
    """Append a `--flag value` pair only when value is present."""
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    argv.extend([flag, text])


def _env_int(name: str, default: int) -> int:
    """Read an integer env var with a safe fallback."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _latest_script_failure_kind(*, episode_id: str, expected_run_token: str) -> str:
    """Read script failure kind from run summary for retry decisions.

    The run token check prevents mixing signals from a previous run of the
    same episode id.
    """
    script_checkpoint_dir = str(os.environ.get("SCRIPT_CHECKPOINT_DIR", "./.script_checkpoints") or "").strip()
    if not script_checkpoint_dir:
        script_checkpoint_dir = "./.script_checkpoints"
    run_summary_path = os.path.join(script_checkpoint_dir, episode_id, "run_summary.json")
    if not os.path.exists(run_summary_path):
        return ""
    try:
        import json

        with open(run_summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    run_token = str(payload.get("run_token", "")).strip()
    if run_token and run_token != str(expected_run_token or "").strip():
        return ""
    return str(payload.get("failure_kind", "")).strip().lower()


def _is_nonretryable_script_failure_kind(kind: str) -> bool:
    """Return True when retrying script stage would not help."""
    normalized = str(kind or "").strip().lower()
    return normalized in {"source_too_short", "resume_blocked"}


def main(argv: list[str] | None = None) -> int:
    """Run script stage first, then hand off to audio stage.

    Script generation can be retried in-process based on classified failure
    kinds. Audio runs only after script stage succeeds.
    """
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    audio_checkpoint_was_configured = "AUDIO_CHECKPOINT_DIR" in os.environ
    if not audio_checkpoint_was_configured:
        # Keep script/audio manifest handoff consistent across both stages.
        os.environ["AUDIO_CHECKPOINT_DIR"] = os.path.join(args.outdir, ".audio_checkpoints")

    episode_id = resolve_episode_id(
        output_path=args.script_path or f"{args.basename}.json",
        override=args.episode_id or args.basename,
    )
    run_token = uuid.uuid4().hex
    script_path = args.script_path or os.path.join(args.outdir, f"{episode_id}_script.json")

    try:
        script_attempts = max(1, _env_int("RUN_PODCAST_SCRIPT_ATTEMPTS", 1))
        script_resume = bool(args.resume)
        script_resume_force = bool(args.resume_force)
        script_force_unlock = bool(args.force_unlock)
        script_rc = 1
        for attempt in range(1, script_attempts + 1):
            # Build child argv explicitly so retries can force resume semantics
            # without mutating the original CLI.
            script_argv: list[str] = [
                args.input_path,
                script_path,
                "--episode-id",
                episode_id,
                "--run-token",
                run_token,
            ]
            _append_optional(script_argv, "--profile", args.profile)
            _append_optional(script_argv, "--target-minutes", args.target_minutes)
            _append_optional(script_argv, "--words-per-min", args.words_per_min)
            _append_optional(script_argv, "--min-words", args.min_words)
            _append_optional(script_argv, "--max-words", args.max_words)
            if script_resume:
                script_argv.append("--resume")
            if script_resume_force:
                script_argv.append("--resume-force")
            if script_force_unlock:
                script_argv.append("--force-unlock")
            if args.verbose:
                script_argv.append("--verbose")
            if args.debug:
                script_argv.append("--debug")
            if args.dry_run_cleanup:
                script_argv.append("--dry-run-cleanup")
            if args.force_clean:
                script_argv.append("--force-clean")

            script_rc = make_script.main(script_argv)
            if script_rc == 0:
                break
            if script_rc == 130:
                return script_rc
            # Retry only recoverable script failures. Non-recoverable failures
            # should fail fast to avoid useless API usage.
            failure_kind = _latest_script_failure_kind(
                episode_id=episode_id,
                expected_run_token=run_token,
            )
            if _is_nonretryable_script_failure_kind(failure_kind):
                return script_rc
            if attempt < script_attempts:
                # After the first failed attempt, retries always run in
                # force-resume mode so work already produced can be reused.
                script_resume = True
                script_resume_force = True
                script_force_unlock = True
        if script_rc != 0:
            return script_rc

        podcast_argv: list[str] = [
            script_path,
            args.outdir,
            args.basename,
            "--episode-id",
            episode_id,
            "--run-token",
            run_token,
        ]
        _append_optional(podcast_argv, "--profile", args.profile)
        if args.resume:
            podcast_argv.append("--resume")
        if args.resume_force:
            podcast_argv.append("--resume-force")
        if args.force_unlock:
            podcast_argv.append("--force-unlock")
        if args.allow_raw_only:
            podcast_argv.append("--allow-raw-only")
        if args.verbose:
            podcast_argv.append("--verbose")
        if args.debug:
            podcast_argv.append("--debug")
        if args.dry_run_cleanup:
            podcast_argv.append("--dry-run-cleanup")
        if args.force_clean:
            podcast_argv.append("--force-clean")

        # Audio stage receives the same episode/run identifiers for strict
        # manifest consistency and easier post-mortem correlation.
        return make_podcast.main(podcast_argv)
    finally:
        if not audio_checkpoint_was_configured:
            os.environ.pop("AUDIO_CHECKPOINT_DIR", None)


if __name__ == "__main__":
    sys.exit(main())

