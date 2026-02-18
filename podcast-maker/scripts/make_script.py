#!/usr/bin/env python3
from __future__ import annotations

"""CLI entrypoint for script generation and script-stage quality gates.

This command is responsible for:
- loading and validating source input,
- running `ScriptGenerator` with checkpoint-aware retries,
- enforcing or warning on script quality outcomes,
- writing run summaries/manifests used by downstream audio stage.
"""

import argparse
import dataclasses
import json
import os
import signal
import sys
import time
import uuid

from pipeline.config import AudioConfig, LoggingConfig, ReliabilityConfig, ScriptConfig
from pipeline.errors import (
    ERROR_KIND_INTERRUPTED,
    ERROR_KIND_INVALID_SCHEMA,
    ERROR_KIND_OPENAI_EMPTY_OUTPUT,
    ERROR_KIND_SCRIPT_COMPLETENESS,
    ERROR_KIND_SCRIPT_QUALITY,
    ERROR_KIND_STUCK,
    ERROR_KIND_UNKNOWN,
    ScriptOperationError,
)
from pipeline.gate_action import default_script_gate_action, resolve_script_gate_action
from pipeline.housekeeping import cleanup_dir, ensure_min_free_disk
from pipeline.io_utils import read_text_file_with_fallback
from pipeline.logging_utils import Logger
from pipeline.openai_client import OpenAIClient
from pipeline.run_manifest import (
    init_manifest,
    resolve_episode_id,
    run_manifest_path,
    update_manifest,
)
from pipeline.schema import count_words_from_lines, validate_script_payload
from pipeline.script_generator import ScriptGenerator
from pipeline.script_postprocess import (
    evaluate_script_completeness,
    harden_script_structure,
    repair_script_completeness,
)
from pipeline.script_quality_gate import (
    ScriptQualityGateConfig,
    attempt_script_quality_repair,
    evaluate_script_quality,
    write_quality_report,
)
from pipeline.slo_gates import append_slo_event, evaluate_slo_windows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for script generation."""
    parser = argparse.ArgumentParser(
        description="Generate podcast script JSON with chunking, checkpoints and resume."
    )
    parser.add_argument("input_path", help="Source text file")
    parser.add_argument("output_path", help="Output script JSON path")
    parser.add_argument("--profile", choices=["short", "standard", "long"], default=None)
    parser.add_argument("--target-minutes", type=float, default=None)
    parser.add_argument("--words-per-min", type=float, default=None)
    parser.add_argument("--min-words", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--episode-id", type=_episode_id_arg, default=None)
    parser.add_argument("--run-token", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-force", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run-cleanup", action="store_true")
    parser.add_argument("--force-clean", action="store_true")
    return parser.parse_args(argv)


def _env_int(name: str, default: int) -> int:
    """Read integer env var with fallback on missing/invalid values."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    """Read boolean env var using common truthy literals."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized == "":
        return default
    return normalized in {"1", "true", "yes", "on"}


def _episode_id_arg(value: str) -> str:
    """Validate `--episode-id` as a plain token (no path parts)."""
    name = str(value or "").strip()
    if not name:
        raise argparse.ArgumentTypeError("episode_id must not be empty")
    if os.path.basename(name) != name:
        raise argparse.ArgumentTypeError("episode_id must be a plain name, not a path")
    return name


def _default_script_gate_action(*, script_profile_name: str) -> str:
    """Compatibility shim around centralized gate action defaults."""
    # Backward-compatible alias kept for tests and external callers.
    return default_script_gate_action(script_profile_name=script_profile_name)


def _load_script_failure_signals(
    checkpoint_dir: str,
    output_path: str,
    *,
    expected_run_token: str | None = None,
    episode_id: str | None = None,
) -> dict[str, object]:
    """Load structured failure hints from script run summary.

    Signals are filtered by run token when available to avoid leaking data from
    a previous execution of the same episode.
    """
    run_episode_id = resolve_episode_id(output_path=output_path, override=episode_id)
    run_summary = os.path.join(checkpoint_dir, run_episode_id, "run_summary.json")
    if not os.path.exists(run_summary):
        return {}
    try:
        with open(run_summary, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if expected_run_token is not None:
        summary_token = str(payload.get("run_token", "")).strip()
        if summary_token != expected_run_token:
            return {}

    stuck_abort = bool(payload.get("stuck_abort", False))
    invalid_schema = False
    try:
        invalid_schema = float(payload.get("invalid_schema_rate", 0.0)) > 0.0
    except (TypeError, ValueError):
        invalid_schema = False
    if not invalid_schema:
        try:
            invalid_schema = int(payload.get("schema_validation_failures", 0) or 0) > 0
        except (TypeError, ValueError):
            invalid_schema = False
    if not invalid_schema:
        try:
            invalid_schema = int(payload.get("script_json_parse_failures", 0) or 0) > 0
        except (TypeError, ValueError):
            invalid_schema = False
    retry_rate = None
    try:
        raw_retry = payload.get("script_retry_rate")
        if raw_retry is not None:
            retry_rate = float(raw_retry)
            if retry_rate < 0.0:
                retry_rate = 0.0
    except (TypeError, ValueError):
        retry_rate = None
    failure_kind = str(payload.get("failure_kind", "")).strip().lower() or None
    return {
        "stuck_abort": stuck_abort,
        "invalid_schema": invalid_schema,
        "retry_rate": retry_rate,
        "failure_kind": failure_kind,
    }


def _run_summary_has_current_token(path: str, *, expected_run_token: str) -> bool:
    """Return True when run summary exists and belongs to this run token."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    token = str(payload.get("run_token", "")).strip()
    return token == expected_run_token


def _atomic_write_json(path: str, payload: dict[str, object]) -> None:
    """Write JSON atomically to avoid partial files on interruption."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _sync_script_artifacts_after_repair(
    *,
    checkpoint_path: str,
    run_summary_path: str,
    repaired_lines: list[dict[str, str]] | None,
    status: str,
    failure_kind: str | None,
    logger: Logger,
) -> None:
    """Keep checkpoint and run summary aligned after quality-stage repairs."""
    now = int(time.time())
    repaired_wc = count_words_from_lines(repaired_lines) if repaired_lines is not None else None
    repaired_lc = len(repaired_lines) if repaired_lines is not None else None

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_payload = json.load(f)
            if isinstance(checkpoint_payload, dict):
                if repaired_lines is not None and repaired_wc is not None:
                    checkpoint_payload["lines"] = repaired_lines
                    checkpoint_payload["current_word_count"] = repaired_wc
                checkpoint_payload["status"] = status
                if status == "completed":
                    checkpoint_payload["last_success_at"] = now
                    checkpoint_payload["completed_at"] = now
                    checkpoint_payload.pop("failed_at", None)
                    checkpoint_payload.pop("failure_kind", None)
                else:
                    checkpoint_payload["failed_at"] = now
                    if failure_kind:
                        checkpoint_payload["failure_kind"] = failure_kind
                _atomic_write_json(checkpoint_path, checkpoint_payload)
        except Exception as exc:  # noqa: BLE001
            logger.warn("script_quality_repair_checkpoint_sync_failed", error=str(exc), path=checkpoint_path)

    if os.path.exists(run_summary_path):
        try:
            with open(run_summary_path, "r", encoding="utf-8") as f:
                summary_payload = json.load(f)
            if isinstance(summary_payload, dict):
                if repaired_lc is not None and repaired_wc is not None:
                    summary_payload["line_count"] = repaired_lc
                    summary_payload["word_count"] = repaired_wc
                summary_payload["status"] = status
                if status == "completed":
                    summary_payload.pop("failure_kind", None)
                elif failure_kind:
                    summary_payload["failure_kind"] = failure_kind
                _atomic_write_json(run_summary_path, summary_payload)
        except Exception as exc:  # noqa: BLE001
            logger.warn("script_quality_repair_summary_sync_failed", error=str(exc), path=run_summary_path)


def _sync_script_phase_metrics(
    *,
    run_summary_path: str,
    phase_metrics: dict[str, float],
    logger: Logger,
) -> None:
    """Patch phase timing metrics into existing run summary."""
    if not os.path.exists(run_summary_path):
        return
    try:
        with open(run_summary_path, "r", encoding="utf-8") as f:
            summary_payload = json.load(f)
        if not isinstance(summary_payload, dict):
            return
        phase_seconds = summary_payload.get("phase_seconds")
        if not isinstance(phase_seconds, dict):
            phase_seconds = {}
        for key, value in phase_metrics.items():
            try:
                phase_seconds[str(key)] = round(float(value), 3)
            except (TypeError, ValueError):
                continue
        summary_payload["phase_seconds"] = phase_seconds
        _atomic_write_json(run_summary_path, summary_payload)
    except Exception as exc:  # noqa: BLE001
        logger.warn("script_phase_metrics_sync_failed", error=str(exc), path=run_summary_path)


def _sync_script_summary_fields(
    *,
    run_summary_path: str,
    updates: dict[str, object],
    logger: Logger,
) -> None:
    """Apply partial field updates to run summary when it already exists."""
    if not os.path.exists(run_summary_path):
        return
    try:
        with open(run_summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return
        payload.update(dict(updates))
        _atomic_write_json(run_summary_path, payload)
    except Exception as exc:  # noqa: BLE001
        logger.warn("script_summary_sync_failed", error=str(exc), path=run_summary_path)


def _recoverable_script_failure_kinds() -> set[str]:
    """Resolve failure kinds eligible for orchestrated script retries."""
    raw = str(
        os.environ.get(
            "SCRIPT_ORCHESTRATED_RETRY_FAILURE_KINDS",
            ",".join(
                [
                    ERROR_KIND_OPENAI_EMPTY_OUTPUT,
                    ERROR_KIND_INVALID_SCHEMA,
                    ERROR_KIND_SCRIPT_QUALITY,
                    ERROR_KIND_SCRIPT_COMPLETENESS,
                ]
            ),
        )
    ).strip()
    values: set[str] = set()
    for part in raw.split(","):
        normalized = str(part or "").strip().lower()
        if normalized:
            values.add(normalized)
    if not values:
        values = {
            ERROR_KIND_OPENAI_EMPTY_OUTPUT,
            ERROR_KIND_INVALID_SCHEMA,
            ERROR_KIND_SCRIPT_QUALITY,
            ERROR_KIND_SCRIPT_COMPLETENESS,
        }
    values.discard(ERROR_KIND_STUCK)
    values.discard(ERROR_KIND_INTERRUPTED)
    if not values:
        values = {
            ERROR_KIND_OPENAI_EMPTY_OUTPUT,
            ERROR_KIND_INVALID_SCHEMA,
            ERROR_KIND_SCRIPT_QUALITY,
            ERROR_KIND_SCRIPT_COMPLETENESS,
        }
    return values


def _classify_retry_failure_kind(
    *,
    exc: BaseException,
    signals: dict[str, object],
    client: OpenAIClient | None,
    client_failure_deltas: dict[str, int] | None = None,
) -> str | None:
    """Classify an exception into a retry policy failure kind."""
    if isinstance(exc, ScriptOperationError):
        exc_kind = str(exc.error_kind or "").strip().lower()
        if exc_kind:
            return exc_kind
    signal_kind = str(signals.get("failure_kind", "")).strip().lower()
    if signal_kind:
        return signal_kind
    message = str(exc or "").strip().lower()
    if "parse_failure_kind=empty_output" in message or "openai returned empty text" in message:
        return ERROR_KIND_OPENAI_EMPTY_OUTPUT
    if (
        "schema" in message
        or "failed to parse json output" in message
        or "parse_failure_kind=truncation" in message
        or "parse_failure_kind=wrapper" in message
        or "parse_failure_kind=malformed" in message
    ):
        return ERROR_KIND_INVALID_SCHEMA
    if "quality gate rejected" in message:
        return ERROR_KIND_SCRIPT_QUALITY
    if bool(signals.get("invalid_schema", False)):
        return ERROR_KIND_INVALID_SCHEMA
    if bool(signals.get("stuck_abort", False)):
        return ERROR_KIND_STUCK
    if client is not None:
        empty_output_failures = int(getattr(client, "script_empty_output_failures", 0))
        json_parse_failures = int(getattr(client, "script_json_parse_failures", 0))
        if isinstance(client_failure_deltas, dict):
            empty_output_failures = int(client_failure_deltas.get("script_empty_output_failures", 0))
            json_parse_failures = int(client_failure_deltas.get("script_json_parse_failures", 0))
        if empty_output_failures > 0:
            return ERROR_KIND_OPENAI_EMPTY_OUTPUT
        if json_parse_failures > 0:
            return ERROR_KIND_INVALID_SCHEMA
    return None


def _run_generation_with_orchestrated_retry(
    *,
    generator: ScriptGenerator,
    source_text: str,
    output_path: str,
    episode_id: str,
    run_token: str,
    resume: bool,
    resume_force: bool,
    force_unlock: bool,
    cancel_check,
    checkpoint_dir: str,
    logger: Logger,
    client: OpenAIClient | None,
    max_attempts: int,
    backoff_seconds: float,
    retry_enabled: bool,
) -> tuple[object, list[dict[str, object]], int]:
    """Execute script generation with bounded recoverable retries.

    Returns `(result, retry_events, attempts_used)`.
    """
    attempts = max(1, int(max_attempts))
    if not retry_enabled:
        attempts = 1
    recoverable_kinds = _recoverable_script_failure_kinds()
    use_resume = bool(resume)
    use_resume_force = bool(resume_force)
    retry_events: list[dict[str, object]] = []
    for attempt in range(1, attempts + 1):
        before_empty_output_failures = 0
        before_json_parse_failures = 0
        if client is not None:
            before_empty_output_failures = int(getattr(client, "script_empty_output_failures", 0))
            before_json_parse_failures = int(getattr(client, "script_json_parse_failures", 0))
        try:
            result = generator.generate(
                source_text=source_text,
                output_path=output_path,
                episode_id=episode_id,
                resume=use_resume,
                resume_force=use_resume_force,
                force_unlock=force_unlock,
                cancel_check=cancel_check,
                run_token=run_token,
            )
            return result, retry_events, attempt
        except (InterruptedError, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001
            # Failure signals from checkpoint artifacts often carry richer
            # classifier hints than the top-level exception message alone.
            signals = _load_script_failure_signals(
                checkpoint_dir,
                output_path,
                expected_run_token=run_token,
                episode_id=episode_id,
            )
            deltas: dict[str, int] | None = None
            if client is not None:
                current_empty_output_failures = int(getattr(client, "script_empty_output_failures", 0))
                current_json_parse_failures = int(getattr(client, "script_json_parse_failures", 0))
                deltas = {
                    "script_empty_output_failures": max(
                        0,
                        current_empty_output_failures - before_empty_output_failures,
                    ),
                    "script_json_parse_failures": max(
                        0,
                        current_json_parse_failures - before_json_parse_failures,
                    ),
                }
            failure_kind = _classify_retry_failure_kind(
                exc=exc,
                signals=signals,
                client=client,
                client_failure_deltas=deltas,
            )
            # Keep retries narrow: only known recoverable kinds are retried.
            should_retry = bool(
                attempt < attempts
                and failure_kind is not None
                and failure_kind in recoverable_kinds
            )
            if not should_retry:
                raise
            retry_events.append(
                {
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "failure_kind": failure_kind,
                    "resume": True,
                    "resume_force": True,
                }
            )
            logger.warn(
                "script_orchestrated_retry",
                attempt=attempt,
                next_attempt=attempt + 1,
                failure_kind=failure_kind,
                recoverable_kinds=sorted(recoverable_kinds),
            )
            # Force checkpoint-aware resume flags so follow-up attempts reuse
            # existing progress instead of recomputing completed chunks.
            use_resume = True
            use_resume_force = True
            if cancel_check and cancel_check():
                raise InterruptedError("Interrupted before orchestrated retry attempt")
            if backoff_seconds > 0.0:
                time.sleep(backoff_seconds)
    raise RuntimeError("script_orchestrated_retry_unexpected_exhaustion")


def main(argv: list[str] | None = None) -> int:
    """Run script generation, quality validation/repair, and reporting.

    The function keeps operational guarantees for maintainers:
    - resilient retries for recoverable generation failures,
    - deterministic + optional LLM quality checks before handoff,
    - consistent summaries/manifests even when interrupted.
    """
    args = parse_args(argv)
    started = time.time()
    run_token = str(getattr(args, "run_token", "") or "").strip() or uuid.uuid4().hex

    log_cfg = LoggingConfig.from_env()
    if args.debug:
        log_cfg = dataclasses.replace(log_cfg, level="DEBUG", debug_events=True)
    elif args.verbose:
        log_cfg = dataclasses.replace(log_cfg, level="INFO")

    logger = Logger.create(log_cfg)
    shutdown = {"requested": False}

    def _signal_handler(signum, _frame):  # type: ignore[no-untyped-def]
        shutdown["requested"] = True
        logger.warn("signal_received", signal=signum)

    signal.signal(signal.SIGINT, _signal_handler)
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is not None:
        signal.signal(sigterm, _signal_handler)

    reliability = ReliabilityConfig.from_env()
    script_cfg = ScriptConfig.from_env(
        target_minutes=args.target_minutes,
        words_per_min=args.words_per_min,
        min_words=args.min_words,
        max_words=args.max_words,
        profile_name=args.profile,
    )
    # OpenAI client needs both script and tts config values.
    audio_cfg = AudioConfig.from_env(profile_name=script_cfg.profile_name)
    episode_id = resolve_episode_id(
        output_path=args.output_path,
        override=getattr(args, "episode_id", None),
    )
    manifest_path = run_manifest_path(checkpoint_dir=script_cfg.checkpoint_dir, episode_id=episode_id)
    manifest_initialized = False
    manifest_v2_enabled = _env_bool("RUN_MANIFEST_V2", True)

    status = "failed"
    output_path = ""
    exit_code = 1
    source_text = ""
    retry_rate = None
    invalid_schema = False
    stuck_abort = False
    estimated_cost_usd = 0.0
    failure_kind = None
    client = None
    quality_eval_seconds = 0.0
    quality_repair_seconds = 0.0
    quality_report_path = ""
    quality_report_initial_path = ""
    quality_stage_started = False
    quality_stage_finished = False
    quality_stage_interrupted = False
    completeness_v2_enabled = _env_bool("SCRIPT_COMPLETENESS_CHECK_V2", True)
    script_started_at = int(started)
    script_completed_at: int | None = None
    script_gate_action_effective = "off"
    quality_gate_executed = False
    handoff_to_audio_started = False
    handoff_to_audio_completed = False
    script_orchestrated_retry_enabled = _env_bool("SCRIPT_ORCHESTRATED_RETRY_ENABLED", True)
    script_orchestrated_retry_max_attempts = max(1, _env_int("SCRIPT_ORCHESTRATED_MAX_ATTEMPTS", 2))
    script_orchestrated_retry_backoff_seconds = max(
        0.0,
        float(_env_int("SCRIPT_ORCHESTRATED_RETRY_BACKOFF_MS", 400)) / 1000.0,
    )
    script_orchestrated_retry_attempts_used = 0
    script_orchestrated_retry_events: list[dict[str, object]] = []

    if manifest_v2_enabled:
        # Best-effort manifest bootstrap. Generation can still proceed even when
        # manifest initialization fails; finalizer will attempt to reconcile.
        try:
            init_manifest(
                checkpoint_dir=script_cfg.checkpoint_dir,
                episode_id=episode_id,
                run_token=run_token,
                script_output_path=args.output_path,
                script_checkpoint_dir=script_cfg.checkpoint_dir,
                audio_checkpoint_dir=audio_cfg.checkpoint_dir,
            )
            manifest_initialized = True
        except Exception as exc:  # noqa: BLE001
            logger.warn(
                "run_manifest_init_failed",
                episode_id=episode_id,
                path=manifest_path,
                error=str(exc),
            )

    try:
        source_text, used_encoding = read_text_file_with_fallback(
            args.input_path,
            on_fallback=lambda enc: logger.warn("input_encoding_fallback", encoding=enc),
        )
        logger.info("input_loaded", input_path=args.input_path, encoding=used_encoding, chars=len(source_text))
    except Exception as exc:  # noqa: BLE001
        logger.error("source_read_failed", input_path=args.input_path, error=str(exc))
        status = "failed"
        exit_code = 1

    try:
        if not source_text:
            raise RuntimeError("Input source unavailable")
        # Pre-flight housekeeping protects disk usage and keeps checkpoint dirs
        # bounded before new artifacts are produced.
        ensure_min_free_disk(".", reliability.min_free_disk_mb)
        report = cleanup_dir(
            base_dir=script_cfg.checkpoint_dir,
            retention_days=reliability.retention_checkpoint_days,
            max_storage_mb=reliability.max_checkpoint_storage_mb,
            retention_log_days=reliability.retention_log_days,
            retention_intermediate_audio_days=reliability.retention_intermediate_audio_days,
            max_log_storage_mb=reliability.max_log_storage_mb,
            force_clean=args.force_clean,
            logger=logger,
            dry_run=args.dry_run_cleanup,
        )
        logger.info(
            "checkpoint_cleanup",
            deleted_files=report.deleted_files,
            deleted_bytes=report.deleted_bytes,
            kept_files=report.kept_files,
        )

        client = OpenAIClient.from_configs(
            logger=logger,
            reliability=reliability,
            script_model=script_cfg.model,
            tts_model=audio_cfg.model,
            script_timeout_seconds=script_cfg.timeout_seconds,
            script_retries=script_cfg.retries,
            tts_timeout_seconds=audio_cfg.timeout_seconds,
            tts_retries=audio_cfg.retries,
            tts_backoff_base_ms=audio_cfg.retry_backoff_base_ms,
            tts_backoff_max_ms=audio_cfg.retry_backoff_max_ms,
        )
        generator = ScriptGenerator(
            config=script_cfg,
            reliability=reliability,
            logger=logger,
            client=client,
        )
        result, script_orchestrated_retry_events, script_orchestrated_retry_attempts_used = (
            _run_generation_with_orchestrated_retry(
                generator=generator,
                source_text=source_text,
                output_path=args.output_path,
                episode_id=episode_id,
                run_token=run_token,
                resume=args.resume,
                resume_force=args.resume_force,
                force_unlock=args.force_unlock,
                cancel_check=lambda: shutdown["requested"],
                checkpoint_dir=script_cfg.checkpoint_dir,
                logger=logger,
                client=client,
                max_attempts=script_orchestrated_retry_max_attempts,
                backoff_seconds=script_orchestrated_retry_backoff_seconds,
                retry_enabled=script_orchestrated_retry_enabled,
            )
        )
        gate_cfg = ScriptQualityGateConfig.from_env(profile_name=script_cfg.profile_name)
        script_gate_action = resolve_script_gate_action(
            script_profile_name=script_cfg.profile_name,
            fallback_action=gate_cfg.action,
        )
        script_gate_action_effective = script_gate_action
        quality_gate_executed = script_gate_action in {"warn", "enforce"}
        if script_gate_action in {"warn", "enforce"}:
            # Script-stage quality gate can hard-fail (`enforce`) or emit
            # warnings (`warn`) while still allowing downstream audio.
            quality_stage_started = True
            quality_report_path = os.path.join(
                script_cfg.checkpoint_dir,
                episode_id,
                "quality_report.json",
            )
            quality_report_initial_path = os.path.join(
                script_cfg.checkpoint_dir,
                episode_id,
                "quality_report_initial.json",
            )
            if not os.path.exists(result.output_path):
                raise ScriptOperationError(
                    "Script quality gate could not read generated output",
                    error_kind=ERROR_KIND_SCRIPT_QUALITY,
                )
            else:
                with open(result.output_path, "r", encoding="utf-8") as f:
                    generated_payload = json.load(f)
                gate_cfg = dataclasses.replace(gate_cfg, action=script_gate_action)
                validated_payload = validate_script_payload(generated_payload)
                hardened_lines = harden_script_structure(
                    list(validated_payload.get("lines", [])),
                    max_consecutive_same_speaker=gate_cfg.max_consecutive_same_speaker,
                )
                if completeness_v2_enabled:
                    hardened_lines = repair_script_completeness(
                        hardened_lines,
                        max_consecutive_same_speaker=gate_cfg.max_consecutive_same_speaker,
                    )
                    completeness_report = evaluate_script_completeness(hardened_lines)
                    if not bool(completeness_report.get("pass", False)):
                        raise ScriptOperationError(
                            "Script completeness check failed before quality gate",
                            error_kind=ERROR_KIND_SCRIPT_QUALITY,
                        )
                if hardened_lines != list(validated_payload.get("lines", [])):
                    validated_payload = {"lines": hardened_lines}
                    _atomic_write_json(result.output_path, validated_payload)
                    logger.info(
                        "script_structural_hardening_applied",
                        output_path=result.output_path,
                        lines=len(hardened_lines),
                        words=count_words_from_lines(hardened_lines),
                    )
                initial_lines_for_gate = list(validated_payload.get("lines", []))
                quality_eval_started = time.time()
                quality_report_initial = evaluate_script_quality(
                    validated_payload=validated_payload,
                    script_cfg=script_cfg,
                    quality_cfg=gate_cfg,
                    script_path=result.output_path,
                    client=client,
                    logger=logger,
                    source_context=source_text,
                )
                quality_eval_seconds = time.time() - quality_eval_started
                quality_report_initial = dict(quality_report_initial)
                quality_report_initial.update(
                    {
                        "quality_stage_started": True,
                        "quality_stage_finished": False,
                        "quality_stage_interrupted": False,
                        "quality_report_phase": "initial",
                        "quality_report_path": quality_report_path,
                        "quality_report_initial_path": quality_report_initial_path,
                    }
                )
                write_quality_report(quality_report_initial_path, quality_report_initial)

                repair_total_timeout_seconds = max(
                    1,
                    _env_int("SCRIPT_QUALITY_GATE_REPAIR_TOTAL_TIMEOUT_SECONDS", 300),
                )
                repair_attempt_timeout_seconds = max(
                    1,
                    _env_int(
                        "SCRIPT_QUALITY_GATE_REPAIR_ATTEMPT_TIMEOUT_SECONDS",
                        90,
                    ),
                )
                quality_repair_started = time.time()
                try:
                    # Attempt deterministic and optional LLM-backed repair while
                    # respecting cancellation and bounded repair deadlines.
                    repair_result = attempt_script_quality_repair(
                        validated_payload=validated_payload,
                        initial_report=quality_report_initial,
                        script_cfg=script_cfg,
                        quality_cfg=gate_cfg,
                        script_path=result.output_path,
                        client=client,
                        logger=logger,
                        stage_prefix="script_gate_repair",
                        cancel_check=lambda: shutdown["requested"],
                        total_timeout_seconds=float(repair_total_timeout_seconds),
                        attempt_timeout_seconds=repair_attempt_timeout_seconds,
                        source_context=source_text,
                    )
                except InterruptedError:
                    quality_repair_seconds = time.time() - quality_repair_started
                    quality_stage_interrupted = True
                    interrupted_report = dict(quality_report_initial)
                    interrupted_reasons = list(interrupted_report.get("reasons", []))
                    if "quality_stage_interrupted" not in interrupted_reasons:
                        interrupted_reasons.append("quality_stage_interrupted")
                    interrupted_report.update(
                        {
                            "status": "interrupted",
                            "pass": False,
                            "failure_kind": ERROR_KIND_INTERRUPTED,
                            "reasons": interrupted_reasons,
                            "quality_stage_started": True,
                            "quality_stage_finished": False,
                            "quality_stage_interrupted": True,
                            "quality_report_phase": "interrupted",
                            "quality_report_path": quality_report_path,
                            "quality_report_initial_path": quality_report_initial_path,
                        }
                    )
                    write_quality_report(quality_report_path, interrupted_report)
                    _sync_script_artifacts_after_repair(
                        checkpoint_path=result.checkpoint_path,
                        run_summary_path=result.run_summary_path,
                        repaired_lines=None,
                        status="interrupted",
                        failure_kind=ERROR_KIND_INTERRUPTED,
                        logger=logger,
                    )
                    _sync_script_phase_metrics(
                        run_summary_path=result.run_summary_path,
                        phase_metrics={
                            "quality_eval": quality_eval_seconds,
                            "repair": quality_repair_seconds,
                            "quality_repair": quality_repair_seconds,
                        },
                        logger=logger,
                    )
                    raise
                quality_repair_seconds = time.time() - quality_repair_started
                quality_report = dict(repair_result.get("report", quality_report_initial))
                final_payload = repair_result.get("payload", validated_payload)
                if not isinstance(final_payload, dict):
                    final_payload = validated_payload
                final_lines = list(final_payload.get("lines", [])) if isinstance(final_payload, dict) else []
                final_completeness_pass = True
                if final_lines:
                    final_lines = harden_script_structure(
                        final_lines,
                        max_consecutive_same_speaker=gate_cfg.max_consecutive_same_speaker,
                    )
                    if completeness_v2_enabled:
                        final_lines = repair_script_completeness(
                            final_lines,
                            max_consecutive_same_speaker=gate_cfg.max_consecutive_same_speaker,
                        )
                        final_completeness = evaluate_script_completeness(final_lines)
                        final_completeness_pass = bool(final_completeness.get("pass", False))
                        if not final_completeness_pass:
                            reasons = list(quality_report.get("reasons", []))
                            for item in list(final_completeness.get("reasons", [])):
                                if item not in reasons:
                                    reasons.append(str(item))
                            quality_report["reasons"] = reasons
                            quality_report["pass"] = False
                applied_repair = bool(final_lines) and (
                    bool(repair_result.get("repaired", False)) or final_lines != initial_lines_for_gate
                )
                persist_repaired_lines = applied_repair and final_completeness_pass
                gate_passed = bool(quality_report.get("pass", False)) and final_completeness_pass
                gate_enforce_failed = script_gate_action == "enforce" and not gate_passed
                quality_stage_finished = True
                if persist_repaired_lines:
                    # Persist repaired script and synchronize checkpoint/summary
                    # so resume keeps using the corrected content.
                    _atomic_write_json(result.output_path, {"lines": final_lines})
                    logger.info(
                        "script_quality_repair_applied",
                        output_path=result.output_path,
                        attempts=quality_report.get("repair_attempts_used", 0),
                    )
                    artifact_status = "failed" if gate_enforce_failed else "completed"
                    _sync_script_artifacts_after_repair(
                        checkpoint_path=result.checkpoint_path,
                        run_summary_path=result.run_summary_path,
                        repaired_lines=final_lines,
                        status=artifact_status,
                        failure_kind=ERROR_KIND_SCRIPT_QUALITY if gate_enforce_failed else None,
                        logger=logger,
                    )
                    result.line_count = len(final_lines)
                    result.word_count = count_words_from_lines(final_lines)
                elif applied_repair and not final_completeness_pass:
                    logger.warn(
                        "script_quality_repair_discarded_incomplete_candidate",
                        output_path=result.output_path,
                        reasons=list(quality_report.get("reasons", [])),
                    )
                if gate_enforce_failed and not persist_repaired_lines:
                    # Keep status consistent when quality gate fails after generation.
                    _sync_script_artifacts_after_repair(
                        checkpoint_path=result.checkpoint_path,
                        run_summary_path=result.run_summary_path,
                        repaired_lines=None,
                        status="failed",
                        failure_kind=ERROR_KIND_SCRIPT_QUALITY,
                        logger=logger,
                    )
                quality_report.update(
                    {
                        "quality_stage_started": True,
                        "quality_stage_finished": True,
                        "quality_stage_interrupted": False,
                        "quality_report_phase": "final",
                        "quality_report_path": quality_report_path,
                        "quality_report_initial_path": quality_report_initial_path,
                    }
                )
                write_quality_report(quality_report_path, quality_report)
                _sync_script_phase_metrics(
                    run_summary_path=result.run_summary_path,
                    phase_metrics={
                        "quality_eval": quality_eval_seconds,
                        "repair": quality_repair_seconds,
                        "quality_repair": quality_repair_seconds,
                    },
                    logger=logger,
                )
                if not gate_passed:
                    if script_gate_action == "enforce":
                        # In enforce mode, quality rejection becomes a hard
                        # operational failure for the script stage.
                        raise ScriptOperationError(
                            "Script quality gate rejected generated script",
                            error_kind=ERROR_KIND_SCRIPT_QUALITY,
                        )
                    logger.warn(
                        "script_quality_gate_warn_continue",
                        report_path=quality_report_path,
                        reasons=quality_report.get("reasons", []),
                    )
        logger.info(
            "script_success",
            output_path=result.output_path,
            lines=result.line_count,
            words=result.word_count,
            checkpoint=result.checkpoint_path,
            run_summary=result.run_summary_path,
        )
        print(result.output_path)
        status = "completed"
        script_completed_at = int(time.time())
        output_path = result.output_path
        exit_code = 0
        script_requests = float(max(1, int(getattr(client, "script_requests_made", client.requests_made))))
        retry_rate = (
            round(float(getattr(client, "script_retries_total", 0)) / script_requests, 4)
            if client is not None
            else result.script_retry_rate
        )
        invalid_schema = result.invalid_schema_rate > 0.0
        stuck_abort = False
        failure_kind = None
        estimated_cost_usd = client.estimated_cost_usd
    except (InterruptedError, KeyboardInterrupt) as exc:
        logger.warn("script_interrupted", error=str(exc))
        status = "interrupted"
        exit_code = 130
        # Signals persisted by lower layers preserve the most recent classified
        # failure context even when interruption surfaces as a generic exception.
        signals = _load_script_failure_signals(
            script_cfg.checkpoint_dir,
            args.output_path,
            expected_run_token=run_token,
            episode_id=episode_id,
        )
        stuck_abort = bool(signals.get("stuck_abort", False))
        invalid_schema = bool(signals.get("invalid_schema", False))
        signal_failure_kind = str(signals.get("failure_kind", "")).strip().lower()
        failure_kind = signal_failure_kind or ERROR_KIND_INTERRUPTED
        signal_retry_rate = signals.get("retry_rate")
        if client is not None:
            script_requests = float(max(1, int(getattr(client, "script_requests_made", client.requests_made))))
            retry_rate = round(
                float(getattr(client, "script_retries_total", 0)) / script_requests,
                4,
            )
        elif isinstance(signal_retry_rate, (int, float)):
            retry_rate = round(float(signal_retry_rate), 4)
        estimated_cost_usd = client.estimated_cost_usd if client is not None else 0.0
    except ScriptOperationError as exc:
        logger.error("script_failed_operation", error=str(exc), error_kind=exc.error_kind)
        status = "failed"
        exit_code = 1
        # Prefer checkpoint-stored failure kind when available because it can be
        # more specific than the coarse operation-level exception taxonomy.
        signals = _load_script_failure_signals(
            script_cfg.checkpoint_dir,
            args.output_path,
            expected_run_token=run_token,
            episode_id=episode_id,
        )
        stuck_abort = bool(signals.get("stuck_abort", False))
        invalid_schema = bool(signals.get("invalid_schema", False))
        failure_kind = str(signals.get("failure_kind", "")).strip().lower() or exc.error_kind
        estimated_cost_usd = client.estimated_cost_usd if client is not None else 0.0
        signal_retry_rate = signals.get("retry_rate")
        if client is not None:
            script_requests = float(max(1, int(getattr(client, "script_requests_made", client.requests_made))))
            retry_rate = round(
                float(getattr(client, "script_retries_total", 0)) / script_requests,
                4,
            )
            if not invalid_schema:
                invalid_schema = int(getattr(client, "script_json_parse_failures", 0)) > 0
        elif isinstance(signal_retry_rate, (int, float)):
            retry_rate = round(float(signal_retry_rate), 4)
    except Exception as exc:  # noqa: BLE001
        logger.error("script_failed", error=str(exc))
        status = "failed"
        exit_code = 1
        # Unknown failure path still pulls checkpoint signals so summaries remain
        # usable by orchestration and triage tooling.
        signals = _load_script_failure_signals(
            script_cfg.checkpoint_dir,
            args.output_path,
            expected_run_token=run_token,
            episode_id=episode_id,
        )
        stuck_abort = bool(signals.get("stuck_abort", False))
        invalid_schema = bool(signals.get("invalid_schema", False))
        failure_kind = (
            str(signals.get("failure_kind", "")).strip().lower() or None
        )
        estimated_cost_usd = client.estimated_cost_usd if client is not None else 0.0
        signal_retry_rate = signals.get("retry_rate")
        if client is not None:
            script_requests = float(max(1, int(getattr(client, "script_requests_made", client.requests_made))))
            retry_rate = round(
                float(getattr(client, "script_retries_total", 0)) / script_requests,
                4,
            )
            if not invalid_schema:
                invalid_schema = int(getattr(client, "script_json_parse_failures", 0)) > 0
        elif isinstance(signal_retry_rate, (int, float)):
            retry_rate = round(float(signal_retry_rate), 4)
        if failure_kind is None:
            # Last-resort taxonomy keeps dashboards and policy checks stable even
            # when the exact new error class was not mapped yet.
            if stuck_abort:
                failure_kind = ERROR_KIND_STUCK
            elif invalid_schema:
                failure_kind = ERROR_KIND_INVALID_SCHEMA
            else:
                failure_kind = ERROR_KIND_UNKNOWN
    finally:
        elapsed = time.time() - started
        run_summary_path = os.path.join(script_cfg.checkpoint_dir, episode_id, "run_summary.json")
        if status != "completed" and not _run_summary_has_current_token(
            run_summary_path,
            expected_run_token=run_token,
        ):
            # If generator did not write a current-token summary (for example
            # early failure), synthesize a fallback summary for observability.
            fallback_summary: dict[str, object] = {
                "component": "make_script",
                "episode_id": episode_id,
                "run_token": run_token,
                "status": status,
                "input_path": args.input_path,
                "output_path": args.output_path,
                "profile": script_cfg.profile_name,
                "script_started_at": script_started_at,
                "script_completed_at": script_completed_at,
                "script_gate_action_effective": script_gate_action_effective,
                "quality_gate_executed": quality_gate_executed,
                "quality_stage_started": quality_stage_started,
                "quality_stage_finished": quality_stage_finished,
                "quality_stage_interrupted": quality_stage_interrupted,
                "handoff_to_audio_started": handoff_to_audio_started,
                "handoff_to_audio_completed": handoff_to_audio_completed,
                "quality_report_path": quality_report_path,
                "quality_report_initial_path": quality_report_initial_path,
                "script_orchestrated_retry_enabled": bool(script_orchestrated_retry_enabled),
                "script_orchestrated_retry_max_attempts": int(script_orchestrated_retry_max_attempts),
                "script_orchestrated_retry_attempts_used": int(script_orchestrated_retry_attempts_used),
                "script_orchestrated_retry_recoveries": int(len(script_orchestrated_retry_events)),
                "script_orchestrated_retry_events": list(script_orchestrated_retry_events),
                "elapsed_seconds": round(elapsed, 2),
                "requests_made": int(getattr(client, "requests_made", 0) if client is not None else 0),
                "estimated_cost_usd": round(float(estimated_cost_usd), 4),
                "script_retry_rate": float(retry_rate) if retry_rate is not None else 0.0,
                "invalid_schema_rate": 1.0 if invalid_schema else 0.0,
                "invalid_schema": bool(invalid_schema),
                "stuck_abort": bool(stuck_abort),
                "failure_kind": failure_kind,
                "failed_stage": "make_script",
                "source_word_count": int(len(source_text.split())) if source_text else 0,
                "run_manifest_path": manifest_path,
                "phase_seconds": {
                    "quality_eval": round(float(quality_eval_seconds), 3),
                    "repair": round(float(quality_repair_seconds), 3),
                    "quality_repair": round(float(quality_repair_seconds), 3),
                },
            }
            if os.path.exists(run_summary_path):
                try:
                    with open(run_summary_path, "r", encoding="utf-8") as f:
                        existing_summary = json.load(f)
                    if isinstance(existing_summary, dict):
                        existing_token = str(existing_summary.get("run_token", "")).strip()
                        if not existing_token or existing_token == run_token:
                            # Keep any extra fields written earlier in the run
                            # while overriding core fields with current fallback values.
                            fallback_summary = {**existing_summary, **fallback_summary}
                except Exception:
                    pass
            try:
                _atomic_write_json(run_summary_path, fallback_summary)
            except Exception as exc:  # noqa: BLE001
                logger.warn("script_run_summary_write_failed", error=str(exc), path=run_summary_path)
        _sync_script_summary_fields(
            run_summary_path=run_summary_path,
            updates={
                "script_started_at": script_started_at,
                "script_completed_at": script_completed_at if status == "completed" else None,
                "script_gate_action_effective": script_gate_action_effective,
                "quality_gate_executed": quality_gate_executed,
                "quality_stage_started": quality_stage_started,
                "quality_stage_finished": quality_stage_finished,
                "quality_stage_interrupted": quality_stage_interrupted,
                "handoff_to_audio_started": handoff_to_audio_started,
                "handoff_to_audio_completed": handoff_to_audio_completed,
                "quality_report_path": quality_report_path,
                "quality_report_initial_path": quality_report_initial_path,
                "script_orchestrated_retry_enabled": bool(script_orchestrated_retry_enabled),
                "script_orchestrated_retry_max_attempts": int(script_orchestrated_retry_max_attempts),
                "script_orchestrated_retry_attempts_used": int(script_orchestrated_retry_attempts_used),
                "script_orchestrated_retry_recoveries": int(len(script_orchestrated_retry_events)),
                "script_orchestrated_retry_events": list(script_orchestrated_retry_events),
            },
            logger=logger,
        )
        if manifest_v2_enabled:
            if not manifest_initialized:
                logger.warn(
                    "run_manifest_final_update_after_init_failure",
                    episode_id=episode_id,
                    path=manifest_path,
                )
            try:
                stage_status = status
                # Manifest expects terminal enum values for script stage.
                # Defensive normalization avoids leaking unknown status strings.
                if stage_status not in {"completed", "failed", "interrupted"}:
                    stage_status = "failed"
                update_manifest(
                    checkpoint_dir=script_cfg.checkpoint_dir,
                    episode_id=episode_id,
                    updates={
                        "run_token": run_token,
                        "script_output_path": output_path or args.output_path,
                        "script_checkpoint_dir": script_cfg.checkpoint_dir,
                        "audio_checkpoint_dir": audio_cfg.checkpoint_dir,
                        "status_by_stage": {
                            "script": stage_status,
                            "audio": "not_started",
                            "bundle": "not_started",
                        },
                        "script": {
                            "started_at": script_started_at,
                            "status": stage_status,
                            "completed_at": int(time.time()) if stage_status == "completed" else None,
                            "failure_kind": None if stage_status == "completed" else failure_kind,
                            "run_summary_path": run_summary_path,
                            "quality_report_path": quality_report_path,
                            "quality_report_initial_path": quality_report_initial_path,
                            "quality_stage_started": quality_stage_started,
                            "quality_stage_finished": quality_stage_finished,
                            "quality_stage_interrupted": quality_stage_interrupted,
                            "script_orchestrated_retry_attempts_used": int(
                                script_orchestrated_retry_attempts_used
                            ),
                            "script_orchestrated_retry_recoveries": int(
                                len(script_orchestrated_retry_events)
                            ),
                        },
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warn(
                    "run_manifest_script_update_failed",
                    episode_id=episode_id,
                    error=str(exc),
                    path=manifest_path,
                )
        try:
            # SLO telemetry and window gates are intentionally best-effort: they
            # should not hide the primary stage result when telemetry fails.
            cost_error_pct = 0.0
            retry_rate_value = float(retry_rate) if isinstance(retry_rate, (int, float)) else 0.0
            actual_cost_raw = os.environ.get("ACTUAL_COST_USD", "").strip()
            if actual_cost_raw:
                try:
                    actual_cost = float(actual_cost_raw)
                    if actual_cost > 0:
                        cost_error_pct = abs(estimated_cost_usd - actual_cost) / actual_cost * 100.0
                except ValueError:
                    cost_error_pct = 0.0
            try:
                append_slo_event(
                    profile=script_cfg.profile_name,
                    component="script",
                    status=status,
                    elapsed_seconds=elapsed,
                    output_path=output_path,
                    is_resume=bool(args.resume or script_orchestrated_retry_attempts_used > 1),
                    retry_rate=retry_rate_value,
                    stuck_abort=stuck_abort,
                    invalid_schema=invalid_schema,
                    failure_kind=failure_kind,
                    cost_estimation_error_pct=cost_error_pct,
                )
            except Exception as append_exc:  # noqa: BLE001
                logger.warn("slo_event_append_failed", error=str(append_exc))
            mode = os.environ.get("SLO_GATE_MODE", "warn").strip().lower()
            if mode in {"warn", "enforce"}:
                report = evaluate_slo_windows(
                    profile=script_cfg.profile_name,
                    component="script",
                    window_size=max(2, _env_int("SLO_WINDOW_SIZE", 20)),
                    required_failed_windows=max(
                        1, _env_int("SLO_REQUIRED_FAILED_WINDOWS", 2)
                    ),
                )
                if report.get("should_rollback"):
                    logger.warn("slo_gate_triggered", profile=script_cfg.profile_name, report=report)
                    if mode == "enforce" and exit_code == 0:
                        exit_code = 3
        except Exception as exc:  # noqa: BLE001
            logger.warn("slo_gate_eval_failed", error=str(exc))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
