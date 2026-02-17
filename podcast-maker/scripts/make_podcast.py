#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import sys
import time
import uuid

from pipeline.audio_mixer import AudioMixer
from pipeline.config import AudioConfig, LoggingConfig, ReliabilityConfig, ScriptConfig
from pipeline.errors import (
    ERROR_KIND_INTERRUPTED,
    ERROR_KIND_NETWORK,
    ERROR_KIND_RATE_LIMIT,
    ERROR_KIND_RUN_MISMATCH,
    ERROR_KIND_SCRIPT_COMPLETENESS,
    ERROR_KIND_SCRIPT_QUALITY,
    ERROR_KIND_TIMEOUT,
    ERROR_KIND_UNKNOWN,
    ScriptOperationError,
    TTSBatchError,
    TTSOperationError,
    is_stuck_error_kind,
)
from pipeline.gate_action import resolve_script_gate_action
from pipeline.housekeeping import cleanup_dir, ensure_min_free_disk
from pipeline.logging_utils import Logger
from pipeline.openai_client import OpenAIClient
from pipeline.run_manifest import (
    init_manifest,
    load_manifest,
    normalize_path_for_compare,
    resolve_episode_id,
    run_manifest_path,
    update_manifest,
)
from pipeline.schema import validate_script_payload
from pipeline.script_postprocess import (
    evaluate_script_completeness,
    harden_script_structure,
    repair_script_completeness,
)
from pipeline.script_quality_gate import (
    ScriptQualityGateConfig,
    ScriptQualityGateError,
    evaluate_script_quality,
    write_quality_report,
)
from pipeline.slo_gates import append_slo_event, evaluate_slo_windows
from pipeline.tts_synthesizer import TTSSynthesizer


def _basename_arg(value: str) -> str:
    name = str(value).strip()
    if not name:
        raise argparse.ArgumentTypeError("basename must not be empty")
    if name in {".", ".."}:
        raise argparse.ArgumentTypeError("basename cannot be '.' or '..'")
    if os.path.basename(name) != name:
        raise argparse.ArgumentTypeError("basename must be a file name, not a path")
    if os.path.sep in name:
        raise argparse.ArgumentTypeError("basename cannot contain path separators")
    if os.path.altsep and os.path.altsep in name:
        raise argparse.ArgumentTypeError("basename cannot contain path separators")
    return name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate podcast audio from script JSON with resume and checkpoints."
    )
    parser.add_argument("script_path", help="Input script JSON path")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument(
        "basename",
        nargs="?",
        default="episode",
        type=_basename_arg,
        help="Output base filename (must not contain path separators)",
    )
    parser.add_argument("--profile", choices=["short", "standard", "long"], default=None)
    parser.add_argument("--episode-id", type=_basename_arg, default=None)
    parser.add_argument("--run-token", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-force", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    parser.add_argument("--allow-raw-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run-cleanup", action="store_true")
    parser.add_argument("--force-clean", action="store_true")
    return parser.parse_args(argv)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _recoverable_audio_failure_kinds() -> set[str]:
    raw = str(
        os.environ.get(
            "AUDIO_ORCHESTRATED_RETRY_FAILURE_KINDS",
            ",".join(
                [
                    ERROR_KIND_TIMEOUT,
                    ERROR_KIND_NETWORK,
                    ERROR_KIND_RATE_LIMIT,
                ]
            ),
        )
        or ""
    ).strip()
    values: set[str] = set()
    for part in raw.split(","):
        normalized = str(part or "").strip().lower()
        if normalized:
            values.add(normalized)
    if not values:
        values = {
            ERROR_KIND_TIMEOUT,
            ERROR_KIND_NETWORK,
            ERROR_KIND_RATE_LIMIT,
        }
    values.discard(ERROR_KIND_INTERRUPTED)
    if not values:
        values = {
            ERROR_KIND_TIMEOUT,
            ERROR_KIND_NETWORK,
            ERROR_KIND_RATE_LIMIT,
        }
    return values


def _classify_audio_failure_kind(exc: BaseException) -> str:
    if isinstance(exc, TTSBatchError):
        return str(exc.primary_kind or ERROR_KIND_UNKNOWN).strip().lower() or ERROR_KIND_UNKNOWN
    if isinstance(exc, TTSOperationError):
        return str(exc.error_kind or ERROR_KIND_UNKNOWN).strip().lower() or ERROR_KIND_UNKNOWN
    if isinstance(exc, ScriptOperationError):
        return str(exc.error_kind or ERROR_KIND_UNKNOWN).strip().lower() or ERROR_KIND_UNKNOWN
    if isinstance(exc, InterruptedError):
        return ERROR_KIND_INTERRUPTED
    message = str(exc or "").strip().lower()
    if "rate limit" in message or "429" in message:
        return ERROR_KIND_RATE_LIMIT
    if "timeout" in message or "timed out" in message:
        return ERROR_KIND_TIMEOUT
    if (
        "network" in message
        or "connection" in message
        or "urlopen error" in message
        or "name or service not known" in message
        or "temporary failure in name resolution" in message
    ):
        return ERROR_KIND_NETWORK
    return ERROR_KIND_UNKNOWN


def _write_raw_only_mp3(segment_files: list[str], output_path: str) -> str:
    tmp = f"{output_path}.tmp"
    try:
        with open(tmp, "wb") as out:
            for seg in segment_files:
                with open(seg, "rb") as src:
                    out.write(src.read())
        os.replace(tmp, output_path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return output_path


def _write_podcast_run_summary(path: str, payload: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    run_token = str(getattr(args, "run_token", "") or "").strip()

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
    script_cfg = ScriptConfig.from_env(profile_name=args.profile)
    audio_cfg = AudioConfig.from_env(profile_name=script_cfg.profile_name)
    quality_cfg = ScriptQualityGateConfig.from_env(profile_name=script_cfg.profile_name)
    quality_cfg = dataclasses.replace(
        quality_cfg,
        action=resolve_script_gate_action(
            script_profile_name=script_cfg.profile_name,
            fallback_action=quality_cfg.action,
        ),
    )
    if "AUDIO_CHECKPOINT_DIR" not in os.environ:
        audio_cfg = dataclasses.replace(audio_cfg, checkpoint_dir=os.path.join(args.outdir, ".audio_checkpoints"))
    episode_id = resolve_episode_id(
        output_path=f"{args.basename}.json",
        override=getattr(args, "episode_id", None) or args.basename,
    )
    normalized_script_input_path = normalize_path_for_compare(args.script_path)
    audio_run_dir = os.path.join(audio_cfg.checkpoint_dir, episode_id)
    run_summary_path = os.path.join(audio_run_dir, "podcast_run_summary.json")
    run_manifest_file = run_manifest_path(checkpoint_dir=script_cfg.checkpoint_dir, episode_id=episode_id)
    run_manifest_initialized = False
    manifest_v2_enabled = _env_bool("RUN_MANIFEST_V2", True)
    prior_manifest = load_manifest(run_manifest_file) if manifest_v2_enabled else None
    if not run_token and isinstance(prior_manifest, dict):
        run_token = str(prior_manifest.get("run_token", "")).strip()
    if not run_token:
        run_token = uuid.uuid4().hex
    script_started_at_from_manifest: int | None = None
    script_completed_at_from_manifest: int | None = None
    manifest_mismatch_error: ScriptOperationError | None = None
    if manifest_v2_enabled:
        if prior_manifest is None:
            try:
                init_manifest(
                    checkpoint_dir=script_cfg.checkpoint_dir,
                    episode_id=episode_id,
                    run_token=run_token,
                    script_output_path=args.script_path,
                    script_checkpoint_dir=script_cfg.checkpoint_dir,
                    audio_checkpoint_dir=audio_cfg.checkpoint_dir,
                )
                run_manifest_initialized = True
            except Exception as exc:  # noqa: BLE001
                logger.warn(
                    "run_manifest_init_failed",
                    episode_id=episode_id,
                    path=run_manifest_file,
                    error=str(exc),
                )
        else:
            run_manifest_initialized = True
            manifest_episode = str(prior_manifest.get("episode_id", "")).strip()
            if manifest_episode and manifest_episode != episode_id:
                manifest_mismatch_error = ScriptOperationError(
                    "Run manifest episode_id mismatch for audio stage",
                    error_kind=ERROR_KIND_RUN_MISMATCH,
                )
            manifest_script_path = normalize_path_for_compare(str(prior_manifest.get("script_output_path", "")))
            if (
                manifest_mismatch_error is None
                and manifest_script_path
                and manifest_script_path != normalized_script_input_path
            ):
                if os.path.exists(manifest_script_path):
                    manifest_mismatch_error = ScriptOperationError(
                        "Run manifest script_output_path does not match provided script_path",
                        error_kind=ERROR_KIND_RUN_MISMATCH,
                    )
                else:
                    logger.warn(
                        "run_manifest_stale_script_path_ignored",
                        episode_id=episode_id,
                        manifest_script_output_path=manifest_script_path,
                        provided_script_path=normalized_script_input_path,
                    )
            script_block = prior_manifest.get("script", {})
            if isinstance(script_block, dict):
                try:
                    started_raw = script_block.get("started_at")
                    if started_raw is not None:
                        script_started_at_from_manifest = int(started_raw)
                except (TypeError, ValueError):
                    script_started_at_from_manifest = None
                try:
                    completed_raw = script_block.get("completed_at")
                    if completed_raw is not None:
                        script_completed_at_from_manifest = int(completed_raw)
                except (TypeError, ValueError):
                    script_completed_at_from_manifest = None
    if run_manifest_initialized and manifest_v2_enabled:
        try:
            update_manifest(
                checkpoint_dir=script_cfg.checkpoint_dir,
                episode_id=episode_id,
                updates={
                    "run_token": run_token,
                    "script_output_path": args.script_path,
                    "audio_checkpoint_dir": audio_cfg.checkpoint_dir,
                    "status_by_stage": {"audio": "not_started"},
                    "audio": {
                        "started_at": None,
                        "status": "not_started",
                        "audio_stage": "not_started",
                        "failure_kind": None,
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warn(
                "run_manifest_audio_start_update_failed",
                episode_id=episode_id,
                path=run_manifest_file,
                error=str(exc),
            )
    status = "failed"
    output_path = ""
    exit_code = 1
    validated = None
    retry_rate = None
    stuck_abort = False
    estimated_cost_usd = 0.0
    failure_kind = None
    client = None
    quality_report_path = ""
    quality_report = None
    quality_eval_seconds = 0.0
    tts_seconds = 0.0
    mix_seconds = 0.0
    manifest_path = ""
    tts_summary_path = ""
    output_mode = ""
    raw_path = ""
    norm_path = ""
    final_path = ""
    segment_count = 0
    normalized_script_path = ""
    run_summary_written = False
    completeness_v2_enabled = _env_bool("SCRIPT_COMPLETENESS_CHECK_V2", True)
    quality_gate_executed = False
    script_gate_action_effective = ""
    handoff_to_audio_started = False
    handoff_to_audio_completed = False
    audio_stage = "not_started"
    audio_started_at: int | None = None
    audio_completed_at: int | None = None
    audio_orchestrated_retry_enabled = _env_bool("AUDIO_ORCHESTRATED_RETRY_ENABLED", True)
    audio_orchestrated_retry_max_attempts = max(1, _env_int("AUDIO_ORCHESTRATED_MAX_ATTEMPTS", 2))
    audio_orchestrated_retry_backoff_seconds = max(
        0.0,
        float(_env_int("AUDIO_ORCHESTRATED_RETRY_BACKOFF_MS", 1200)) / 1000.0,
    )
    audio_orchestrated_retry_attempts_used = 0
    audio_orchestrated_retry_events: list[dict[str, object]] = []

    try:
        if manifest_mismatch_error is not None:
            raise manifest_mismatch_error
        with open(args.script_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        validated = validate_script_payload(payload)
        hardened_lines = harden_script_structure(
            list(validated.get("lines", [])),
            max_consecutive_same_speaker=quality_cfg.max_consecutive_same_speaker,
        )
        if completeness_v2_enabled:
            hardened_lines = repair_script_completeness(
                hardened_lines,
                max_consecutive_same_speaker=quality_cfg.max_consecutive_same_speaker,
            )
            completeness_report = evaluate_script_completeness(hardened_lines)
            if not bool(completeness_report.get("pass", False)):
                raise ScriptOperationError(
                    "Script completeness check failed before audio: "
                    + ", ".join(str(r) for r in list(completeness_report.get("reasons", []))),
                    error_kind=ERROR_KIND_SCRIPT_COMPLETENESS,
                )
        if hardened_lines != list(validated.get("lines", [])):
            validated = {"lines": hardened_lines}
            logger.info(
                "script_structural_hardening_applied",
                script_path=args.script_path,
                lines=len(hardened_lines),
            )
        normalized_script_path = os.path.join(audio_run_dir, "normalized_script.json")
        try:
            _write_podcast_run_summary(
                normalized_script_path,
                {"lines": list(validated.get("lines", []))},
            )
            logger.info(
                "normalized_script_snapshot_written",
                path=normalized_script_path,
                lines=len(list(validated.get("lines", []))),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warn(
                "normalized_script_snapshot_write_failed",
                path=normalized_script_path,
                error=str(exc),
            )
    except ScriptOperationError as exc:
        logger.error("podcast_precheck_failed_operation", error=str(exc), error_kind=exc.error_kind)
        status = "failed"
        exit_code = 1
        failure_kind = exc.error_kind
        validated = None
    except Exception as exc:  # noqa: BLE001
        logger.error("script_json_invalid", script_path=args.script_path, error=str(exc))
        status = "failed"
        exit_code = 1
        validated = None

    try:
        if validated is None:
            raise RuntimeError("Input script validation failed")
        os.makedirs(args.outdir, exist_ok=True)
        ensure_min_free_disk(args.outdir, reliability.min_free_disk_mb)
        report = cleanup_dir(
            base_dir=audio_cfg.checkpoint_dir,
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
            "audio_checkpoint_cleanup",
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
        quality_gate_executed = bool(quality_cfg.enabled)
        script_gate_action_effective = str(quality_cfg.action)
        if quality_cfg.enabled:
            os.makedirs(audio_run_dir, exist_ok=True)
            quality_report_path = os.path.join(audio_run_dir, "quality_report.json")
            quality_eval_started = time.time()
            quality_report = evaluate_script_quality(
                validated_payload=validated,
                script_cfg=script_cfg,
                quality_cfg=quality_cfg,
                script_path=args.script_path,
                client=client,
                logger=logger,
            )
            quality_eval_seconds = time.time() - quality_eval_started
            write_quality_report(quality_report_path, quality_report)
            if not bool(quality_report.get("pass", False)):
                failure_kind = ERROR_KIND_SCRIPT_QUALITY
                if quality_cfg.action == "enforce":
                    raise ScriptQualityGateError(
                        "Script quality gate rejected script before audio synthesis",
                        report=quality_report,
                    )
                logger.warn(
                    "script_quality_gate_warn_continue",
                    report_path=quality_report_path,
                    reasons=quality_report.get("reasons", []),
                )
        handoff_to_audio_started = True
        audio_stage = "started"
        audio_started_at = int(time.time())
        if run_manifest_initialized and manifest_v2_enabled:
            try:
                update_manifest(
                    checkpoint_dir=script_cfg.checkpoint_dir,
                    episode_id=episode_id,
                    updates={
                        "run_token": run_token,
                        "status_by_stage": {"audio": "running"},
                        "audio": {
                            "started_at": audio_started_at,
                            "status": "running",
                            "audio_stage": "started",
                            "failure_kind": None,
                        },
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warn(
                    "run_manifest_audio_handoff_start_update_failed",
                    episode_id=episode_id,
                    path=run_manifest_file,
                    error=str(exc),
                )
        allow_raw_only = args.allow_raw_only or _env_bool("ALLOW_RAW_ONLY", False)
        mixer = AudioMixer(config=audio_cfg, logger=logger)
        mixer_available = True
        try:
            mixer.check_dependencies()
        except Exception as exc:  # noqa: BLE001
            if not allow_raw_only:
                raise
            mixer_available = False
            logger.warn("ffmpeg_missing_raw_only_mode", error=str(exc))

        synth = TTSSynthesizer(
            config=audio_cfg,
            reliability=reliability,
            logger=logger,
            client=client,
        )
        max_audio_attempts = int(audio_orchestrated_retry_max_attempts)
        if not audio_orchestrated_retry_enabled:
            max_audio_attempts = 1
        recoverable_audio_kinds = _recoverable_audio_failure_kinds()
        use_resume = bool(args.resume)
        use_resume_force = bool(args.resume_force)
        use_force_unlock = bool(args.force_unlock)
        tts_result = None
        for audio_attempt in range(1, max_audio_attempts + 1):
            try:
                tts_started = time.time()
                tts_result = synth.synthesize(
                    lines=validated["lines"],
                    episode_id=episode_id,
                    resume=use_resume,
                    resume_force=use_resume_force,
                    force_unlock=use_force_unlock,
                    cancel_check=lambda: shutdown["requested"],
                )
                tts_seconds += time.time() - tts_started
                handoff_to_audio_completed = True
                manifest_path = tts_result.manifest_path
                tts_summary_path = tts_result.summary_path
                segment_count = len(tts_result.segment_files)
                audio_run_dir = str(getattr(tts_result, "checkpoint_dir", "") or "").strip() or audio_run_dir
                run_summary_path = os.path.join(audio_run_dir, "podcast_run_summary.json")

                mix_started = time.time()
                if mixer_available:
                    mix_result = mixer.mix(
                        segment_files=tts_result.segment_files,
                        outdir=args.outdir,
                        basename=args.basename,
                    )
                    final_path = mix_result.final_path
                    raw_path = mix_result.raw_path
                    norm_path = mix_result.norm_path
                    output_mode = "full_pipeline"
                else:
                    raw_only = os.path.join(args.outdir, f"{args.basename}_raw_only.mp3")
                    final_path = _write_raw_only_mp3(tts_result.segment_files, raw_only)
                    raw_path = final_path
                    norm_path = ""
                    output_mode = "raw_only"
                    logger.warn("raw_only_output_created", final_path=final_path)
                mix_seconds += time.time() - mix_started
                audio_orchestrated_retry_attempts_used = audio_attempt
                break
            except (InterruptedError, KeyboardInterrupt):
                audio_orchestrated_retry_attempts_used = audio_attempt
                raise
            except Exception as audio_exc:  # noqa: BLE001
                audio_orchestrated_retry_attempts_used = audio_attempt
                failure_kind_candidate = _classify_audio_failure_kind(audio_exc)
                should_retry = bool(
                    audio_attempt < max_audio_attempts
                    and failure_kind_candidate in recoverable_audio_kinds
                )
                if not should_retry:
                    raise
                audio_orchestrated_retry_events.append(
                    {
                        "attempt": audio_attempt,
                        "next_attempt": audio_attempt + 1,
                        "failure_kind": failure_kind_candidate,
                        "resume": True,
                        "resume_force": True,
                        "force_unlock": True,
                    }
                )
                logger.warn(
                    "audio_orchestrated_retry",
                    attempt=audio_attempt,
                    next_attempt=audio_attempt + 1,
                    failure_kind=failure_kind_candidate,
                    recoverable_kinds=sorted(recoverable_audio_kinds),
                )
                use_resume = True
                use_resume_force = True
                use_force_unlock = True
                if shutdown["requested"]:
                    raise InterruptedError("Interrupted before audio retry attempt")
                if audio_orchestrated_retry_backoff_seconds > 0.0:
                    time.sleep(audio_orchestrated_retry_backoff_seconds)
                continue
        if tts_result is None:
            raise RuntimeError("audio_orchestrated_retry_exhausted_without_result")
        audio_stage = "completed"
        audio_completed_at = int(time.time())

        run_summary = {
            "component": "make_podcast",
            "episode_id": episode_id,
            "run_token": run_token,
            "status": "completed",
            "audio_stage": audio_stage,
            "output_mode": output_mode,
            "script_path": args.script_path,
            "script_path_normalized": normalized_script_input_path,
            "run_manifest_path": run_manifest_file,
            "audio_executed": True,
            "quality_gate_executed": quality_gate_executed,
            "script_gate_action_effective": script_gate_action_effective,
            "handoff_to_audio_started": handoff_to_audio_started,
            "handoff_to_audio_completed": handoff_to_audio_completed,
            "audio_started_at": audio_started_at,
            "audio_completed_at": audio_completed_at,
            "script_started_at": script_started_at_from_manifest,
            "script_completed_at": script_completed_at_from_manifest,
            "segment_count": len(tts_result.segment_files),
            "requests_made": client.requests_made,
            "estimated_cost_usd": round(client.estimated_cost_usd, 4),
            "elapsed_seconds": round(time.time() - started, 2),
            "manifest_path": tts_result.manifest_path,
            "tts_summary_path": tts_result.summary_path,
            "raw_path": raw_path,
            "norm_path": norm_path,
            "final_path": final_path,
            "phase_seconds": {
                "quality_eval": round(quality_eval_seconds, 3),
                "tts": round(tts_seconds, 3),
                "mix": round(mix_seconds, 3),
            },
            "audio_orchestrated_retry_enabled": bool(audio_orchestrated_retry_enabled),
            "audio_orchestrated_retry_max_attempts": int(audio_orchestrated_retry_max_attempts),
            "audio_orchestrated_retry_attempts_used": int(audio_orchestrated_retry_attempts_used),
            "audio_orchestrated_retry_recoveries": int(len(audio_orchestrated_retry_events)),
            "audio_orchestrated_retry_events": list(audio_orchestrated_retry_events),
        }
        if normalized_script_path and os.path.exists(normalized_script_path):
            run_summary["normalized_script_path"] = normalized_script_path
        if quality_report is not None:
            run_summary["quality_gate_pass"] = bool(quality_report.get("pass", False))
            run_summary["quality_report_path"] = quality_report_path
        run_summary["status_by_stage"] = {"script": "completed", "audio": "completed"}
        _write_podcast_run_summary(run_summary_path, run_summary)
        run_summary_written = True

        logger.info("podcast_success", final_path=final_path, run_summary=run_summary_path)
        print(final_path)
        status = "completed"
        output_path = final_path
        exit_code = 0
        tts_requests = float(max(1, int(getattr(client, "tts_requests_made", client.requests_made))))
        retry_rate = round(
            float(getattr(client, "tts_retries_total", 0)) / tts_requests,
            4,
        )
        stuck_abort = False
        failure_kind = None
        estimated_cost_usd = client.estimated_cost_usd
    except (InterruptedError, KeyboardInterrupt) as exc:
        logger.warn("podcast_interrupted", error=str(exc))
        status = "interrupted"
        exit_code = 130
        failure_kind = ERROR_KIND_INTERRUPTED
        audio_stage = "failed_during_tts" if handoff_to_audio_started else "failed_before_tts"
        estimated_cost_usd = client.estimated_cost_usd if client is not None else 0.0
    except ScriptOperationError as exc:
        logger.error("podcast_failed_operation", error=str(exc), error_kind=exc.error_kind)
        status = "failed"
        exit_code = 1
        stuck_abort = False
        failure_kind = exc.error_kind
        audio_stage = "failed_before_tts"
        estimated_cost_usd = client.estimated_cost_usd if client is not None else 0.0
    except ScriptQualityGateError as exc:
        logger.error(
            "podcast_failed_script_quality_gate",
            error=str(exc),
            report=getattr(exc, "report", {}),
        )
        status = "failed"
        exit_code = 4
        stuck_abort = False
        failure_kind = ERROR_KIND_SCRIPT_QUALITY
        audio_stage = "failed_before_tts"
        estimated_cost_usd = client.estimated_cost_usd if client is not None else 0.0
        if client is not None:
            tts_requests = float(max(1, int(getattr(client, "tts_requests_made", client.requests_made))))
            retry_rate = round(
                float(getattr(client, "tts_retries_total", 0)) / tts_requests,
                4,
            )
    except TTSBatchError as exc:
        logger.error(
            "podcast_failed_tts_batch",
            error=str(exc),
            failed_kinds=exc.failed_kinds,
            manifest_path=exc.manifest_path,
        )
        status = "failed"
        exit_code = 1
        stuck_abort = bool(exc.stuck_abort)
        failure_kind = exc.primary_kind
        audio_stage = "failed_during_tts" if handoff_to_audio_started else "failed_before_tts"
        manifest_path = str(exc.manifest_path or "").strip()
        estimated_cost_usd = client.estimated_cost_usd if client is not None else 0.0
        if client is not None:
            tts_requests = float(max(1, int(getattr(client, "tts_requests_made", client.requests_made))))
            retry_rate = round(
                float(getattr(client, "tts_retries_total", 0)) / tts_requests,
                4,
            )
    except TTSOperationError as exc:
        logger.error("podcast_failed_tts_operation", error=str(exc), error_kind=exc.error_kind)
        status = "failed"
        exit_code = 1
        stuck_abort = is_stuck_error_kind(exc.error_kind)
        failure_kind = exc.error_kind
        audio_stage = "failed_during_tts" if handoff_to_audio_started else "failed_before_tts"
        estimated_cost_usd = client.estimated_cost_usd if client is not None else 0.0
        if client is not None:
            tts_requests = float(max(1, int(getattr(client, "tts_requests_made", client.requests_made))))
            retry_rate = round(
                float(getattr(client, "tts_retries_total", 0)) / tts_requests,
                4,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("podcast_failed", error=str(exc))
        status = "failed"
        exit_code = 1
        stuck_abort = False
        failure_kind = str(failure_kind or "").strip().lower() or ERROR_KIND_UNKNOWN
        audio_stage = "failed_during_tts" if handoff_to_audio_started else "failed_before_tts"
        estimated_cost_usd = client.estimated_cost_usd if client is not None else 0.0
        if client is not None:
            tts_requests = float(max(1, int(getattr(client, "tts_requests_made", client.requests_made))))
            retry_rate = round(
                float(getattr(client, "tts_retries_total", 0)) / tts_requests,
                4,
            )
    finally:
        elapsed = time.time() - started
        if not run_summary_written:
            fallback_summary: dict[str, object] = {
                "component": "make_podcast",
                "episode_id": episode_id,
                "run_token": run_token,
                "status": status,
                "audio_stage": audio_stage,
                "output_mode": output_mode or "unknown",
                "script_path": args.script_path,
                "script_path_normalized": normalized_script_input_path,
                "run_manifest_path": run_manifest_file,
                "audio_executed": bool(segment_count > 0),
                "quality_gate_executed": quality_gate_executed,
                "script_gate_action_effective": script_gate_action_effective,
                "handoff_to_audio_started": handoff_to_audio_started,
                "handoff_to_audio_completed": handoff_to_audio_completed,
                "audio_started_at": audio_started_at,
                "audio_completed_at": audio_completed_at,
                "script_started_at": script_started_at_from_manifest,
                "script_completed_at": script_completed_at_from_manifest,
                "segment_count": int(max(0, segment_count)),
                "requests_made": int(getattr(client, "requests_made", 0) if client is not None else 0),
                "estimated_cost_usd": round(float(estimated_cost_usd), 4),
                "elapsed_seconds": round(elapsed, 2),
                "phase_seconds": {
                    "quality_eval": round(float(quality_eval_seconds), 3),
                    "tts": round(float(tts_seconds), 3),
                    "mix": round(float(mix_seconds), 3),
                },
                "audio_orchestrated_retry_enabled": bool(audio_orchestrated_retry_enabled),
                "audio_orchestrated_retry_max_attempts": int(audio_orchestrated_retry_max_attempts),
                "audio_orchestrated_retry_attempts_used": int(audio_orchestrated_retry_attempts_used),
                "audio_orchestrated_retry_recoveries": int(len(audio_orchestrated_retry_events)),
                "audio_orchestrated_retry_events": list(audio_orchestrated_retry_events),
                "failure_kind": failure_kind,
            }
            fallback_summary["status_by_stage"] = {
                "script": "completed" if validated is not None else "unknown",
                "audio": (
                    "completed"
                    if status == "completed"
                    else ("interrupted" if status == "interrupted" else ("failed" if status == "failed" else "unknown"))
                ),
            }
            if manifest_path:
                fallback_summary["manifest_path"] = manifest_path
            if tts_summary_path:
                fallback_summary["tts_summary_path"] = tts_summary_path
            if raw_path:
                fallback_summary["raw_path"] = raw_path
            if norm_path:
                fallback_summary["norm_path"] = norm_path
            if final_path:
                fallback_summary["final_path"] = final_path
            if normalized_script_path and os.path.exists(normalized_script_path):
                fallback_summary["normalized_script_path"] = normalized_script_path
            if quality_report is not None:
                fallback_summary["quality_gate_pass"] = bool(quality_report.get("pass", False))
                fallback_summary["quality_report_path"] = quality_report_path
            try:
                _write_podcast_run_summary(run_summary_path, fallback_summary)
            except Exception as exc:  # noqa: BLE001
                logger.warn("podcast_run_summary_write_failed", error=str(exc), path=run_summary_path)
        if run_manifest_initialized and manifest_v2_enabled:
            try:
                audio_stage_status = status if status in {"completed", "failed", "interrupted"} else "failed"
                update_manifest(
                    checkpoint_dir=script_cfg.checkpoint_dir,
                    episode_id=episode_id,
                    updates={
                        "run_token": run_token,
                        "audio_checkpoint_dir": audio_cfg.checkpoint_dir,
                        "status_by_stage": {"audio": audio_stage_status},
                        "audio": {
                            "status": audio_stage_status,
                            "audio_stage": audio_stage,
                            "started_at": audio_started_at,
                            "completed_at": audio_completed_at if audio_stage_status == "completed" else None,
                            "failure_kind": None if audio_stage_status == "completed" else failure_kind,
                            "podcast_run_summary_path": run_summary_path,
                        },
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warn(
                    "run_manifest_audio_update_failed",
                    episode_id=episode_id,
                    path=run_manifest_file,
                    error=str(exc),
                )
        try:
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
                    component="audio",
                    status=status,
                    elapsed_seconds=elapsed,
                    output_path=output_path,
                    is_resume=bool(args.resume or audio_orchestrated_retry_attempts_used > 1),
                    retry_rate=retry_rate_value,
                    stuck_abort=stuck_abort,
                    invalid_schema=False,
                    failure_kind=failure_kind,
                    cost_estimation_error_pct=cost_error_pct,
                )
            except Exception as append_exc:  # noqa: BLE001
                logger.warn("slo_event_append_failed", error=str(append_exc))
            mode = os.environ.get("SLO_GATE_MODE", "warn").strip().lower()
            if mode in {"warn", "enforce"}:
                report = evaluate_slo_windows(
                    profile=script_cfg.profile_name,
                    component="audio",
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
