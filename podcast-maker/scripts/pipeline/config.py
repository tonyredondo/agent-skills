#!/usr/bin/env python3
from __future__ import annotations

"""Centralized runtime configuration for podcast-maker pipeline.

This module maps environment variables and optional CLI overrides into typed
dataclasses used by script/audio/orchestration components.
"""

import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _env_str(name: str, default: str) -> str:
    """Read string env var with trim + default fallback."""
    v = os.environ.get(name)
    return default if v is None else str(v).strip()


def _env_int(name: str, default: int) -> int:
    """Read integer env var with defensive fallback."""
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read finite float env var with defensive fallback."""
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        value = float(str(v).strip())
        if not math.isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read boolean env var from common truthy literals."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _coalesce(value: Any, fallback: Any) -> Any:
    """Return fallback when value is None."""
    return fallback if value is None else value


def _clamp_float(value: float, low: float, high: float) -> float:
    """Clamp float to inclusive range."""
    return max(low, min(high, value))


def _clamp_int(value: int, low: int, high: int) -> int:
    """Clamp integer to inclusive range."""
    return max(low, min(high, value))


def _default_source_validation_policy(*, profile_name: str, target_minutes: float) -> tuple[str, float, float]:
    """Return default source-validation policy by profile/duration."""
    normalized = str(profile_name or "standard").strip().lower()
    minutes = max(1.0, float(target_minutes))
    if normalized == "long" or minutes >= 25.0:
        return ("enforce", 0.60, 0.45)
    if normalized == "standard" or minutes >= 10.0:
        return ("enforce", 0.50, 0.35)
    return ("warn", 0.35, 0.22)


@dataclass(frozen=True)
class DurationProfile:
    """Profile defaults used to derive script/audio runtime tuning."""

    name: str
    default_target_minutes: float
    chunk_target_minutes: float
    max_context_lines: int
    max_continuations_per_chunk: int
    no_progress_rounds: int
    min_word_delta: int
    tts_max_concurrent: int


PROFILE_DEFAULTS: Dict[str, DurationProfile] = {
    "short": DurationProfile(
        name="short",
        default_target_minutes=5.0,
        chunk_target_minutes=1.8,
        max_context_lines=12,
        max_continuations_per_chunk=3,
        no_progress_rounds=2,
        min_word_delta=30,
        tts_max_concurrent=1,
    ),
    "standard": DurationProfile(
        name="standard",
        default_target_minutes=15.0,
        chunk_target_minutes=2.5,
        max_context_lines=20,
        max_continuations_per_chunk=4,
        no_progress_rounds=3,
        min_word_delta=40,
        tts_max_concurrent=1,
    ),
    "long": DurationProfile(
        name="long",
        default_target_minutes=30.0,
        chunk_target_minutes=3.0,
        max_context_lines=28,
        max_continuations_per_chunk=5,
        no_progress_rounds=4,
        min_word_delta=50,
        tts_max_concurrent=2,
    ),
}


def resolve_profile(profile_name: str) -> DurationProfile:
    """Resolve known duration profile, defaulting to standard."""
    normalized = (profile_name or "standard").strip().lower()
    return PROFILE_DEFAULTS.get(normalized, PROFILE_DEFAULTS["standard"])


@dataclass(frozen=True)
class LoggingConfig:
    """Logging behavior used by `Logger`."""

    level: str
    heartbeat_seconds: int
    debug_events: bool
    include_event_ids: bool

    @staticmethod
    def from_env() -> "LoggingConfig":
        """Build logging config from environment."""
        return LoggingConfig(
            level=_env_str("LOG_LEVEL", "INFO").upper(),
            heartbeat_seconds=max(1, _env_int("LOG_HEARTBEAT_SECONDS", 15)),
            debug_events=_env_bool("LOG_DEBUG_EVENTS", False),
            include_event_ids=_env_bool("LOG_INCLUDE_EVENT_IDS", True),
        )


@dataclass(frozen=True)
class ReliabilityConfig:
    """Cross-cutting reliability and retention configuration."""

    checkpoint_version: int
    lock_ttl_seconds: int
    max_requests_per_run: int
    max_estimated_cost_usd: float
    min_free_disk_mb: int
    resume_require_matching_fingerprint: bool
    max_checkpoint_storage_mb: int
    max_log_storage_mb: int
    retention_checkpoint_days: int
    retention_log_days: int
    retention_intermediate_audio_days: int

    @staticmethod
    def from_env() -> "ReliabilityConfig":
        """Build reliability config from environment."""
        return ReliabilityConfig(
            checkpoint_version=_env_int("CHECKPOINT_VERSION", 3),
            lock_ttl_seconds=max(60, _env_int("LOCK_TTL_SECONDS", 1800)),
            max_requests_per_run=max(0, _env_int("MAX_REQUESTS_PER_RUN", 0)),
            max_estimated_cost_usd=max(0.0, _env_float("MAX_ESTIMATED_COST_USD", 0.0)),
            min_free_disk_mb=max(32, _env_int("MIN_FREE_DISK_MB", 512)),
            resume_require_matching_fingerprint=_env_bool("RESUME_REQUIRE_MATCHING_FINGERPRINT", True),
            max_checkpoint_storage_mb=max(256, _env_int("MAX_CHECKPOINT_STORAGE_MB", 4096)),
            max_log_storage_mb=max(128, _env_int("MAX_LOG_STORAGE_MB", 1024)),
            retention_checkpoint_days=max(1, _env_int("RETENTION_CHECKPOINT_DAYS", 14)),
            retention_log_days=max(1, _env_int("RETENTION_LOG_DAYS", 7)),
            retention_intermediate_audio_days=max(1, _env_int("RETENTION_INTERMEDIATE_AUDIO_DAYS", 3)),
        )


@dataclass(frozen=True)
class ScriptConfig:
    """Script-generation specific runtime configuration."""

    model: str
    profile_name: str
    target_minutes: float
    words_per_min: float
    min_words: int
    max_words: int
    chunk_target_minutes: float
    max_context_lines: int
    max_continuations_per_chunk: int
    no_progress_rounds: int
    min_word_delta: int
    timeout_seconds: int
    retries: int
    checkpoint_dir: str
    pre_summary_trigger_words: int
    pre_summary_chunk_target_minutes: float
    pre_summary_target_words: int
    pre_summary_max_rounds: int
    repair_max_attempts: int
    max_output_tokens_initial: int
    max_output_tokens_chunk: int
    max_output_tokens_continuation: int
    source_validation_mode: str
    source_validation_warn_ratio: float
    source_validation_enforce_ratio: float
    adaptive_defaults_enabled: bool
    pre_summary_parallel: bool
    pre_summary_parallel_workers: int
    expected_words_per_chunk: int
    expected_tokens_per_chunk: int

    @staticmethod
    def from_env(
        *,
        target_minutes: Optional[float] = None,
        words_per_min: Optional[float] = None,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
        profile_name: Optional[str] = None,
    ) -> "ScriptConfig":
        """Build script config from env and optional CLI overrides."""
        requested_profile = _coalesce(profile_name, _env_str("PODCAST_DURATION_PROFILE", "standard"))
        profile = resolve_profile(requested_profile)

        default_target = profile.default_target_minutes if profile.name != "standard" else 15.0
        resolved_target = max(1.0, float(_coalesce(target_minutes, _env_float("TARGET_MINUTES", default_target))))
        resolved_wpm = max(80.0, float(_coalesce(words_per_min, _env_float("WORDS_PER_MIN", 130.0))))
        resolved_min = int(_coalesce(min_words, _env_int("MIN_WORDS", int(resolved_target * resolved_wpm))))
        resolved_max = int(_coalesce(max_words, _env_int("MAX_WORDS", int(resolved_min * 1.15))))
        if resolved_max < resolved_min:
            resolved_max = resolved_min
        resolved_min = max(60, resolved_min)
        resolved_max = max(80, resolved_max)
        target_words = max(resolved_min, int((resolved_min + resolved_max) / 2))

        adaptive_defaults_enabled = _env_bool("SCRIPT_ADAPTIVE_DEFAULTS", True)
        if adaptive_defaults_enabled:
            # Scale generation defaults with target duration to keep behavior
            # stable across short/standard/long episodes.
            chunk_target_default = _clamp_float(1.4 + (resolved_target * 0.05), 1.4, 3.8)
            max_context_default = _clamp_int(int(round(10 + (resolved_target * 0.7))), 10, 44)
            max_continuations_default = _clamp_int(int(round(2 + (resolved_target / 20.0))), 2, 7)
            no_progress_rounds_default = _clamp_int(int(round(2 + (resolved_target / 25.0))), 2, 6)
            min_word_delta_default = _clamp_int(int(round(target_words * 0.015)), 20, 90)
        else:
            chunk_target_default = profile.chunk_target_minutes
            max_context_default = profile.max_context_lines
            max_continuations_default = profile.max_continuations_per_chunk
            no_progress_rounds_default = profile.no_progress_rounds
            min_word_delta_default = profile.min_word_delta

        resolved_chunk_target_minutes = max(
            0.8,
            _env_float("SCRIPT_CHUNK_TARGET_MINUTES", chunk_target_default),
        )
        chunk_count_estimate = max(
            1,
            int(math.ceil(max(1.0, resolved_target) / max(0.8, resolved_chunk_target_minutes))),
        )
        expected_words_per_chunk = max(80, int(math.ceil(target_words / float(chunk_count_estimate))))
        expected_tokens_per_chunk = _clamp_int(int(round(expected_words_per_chunk * 2.0)), 900, 7000)
        adaptive_timeout_default = _clamp_int(
            int(round(60 + (expected_tokens_per_chunk / 50.0))),
            90,
            240,
        )
        if adaptive_defaults_enabled:
            max_output_tokens_initial_default = min(16000, int(round(expected_tokens_per_chunk * 2.5)))
            max_output_tokens_chunk_default = int(round(expected_tokens_per_chunk * 2.0))
            max_output_tokens_continuation_default = _clamp_int(
                int(round(expected_tokens_per_chunk * 1.6)),
                1400,
                10000,
            )
            openai_timeout_default = adaptive_timeout_default
        else:
            max_output_tokens_initial_default = 14000
            max_output_tokens_chunk_default = 8000
            max_output_tokens_continuation_default = 6000
            openai_timeout_default = 120

        default_source_validation_mode, default_warn_ratio, default_enforce_ratio = _default_source_validation_policy(
            profile_name=profile.name,
            target_minutes=resolved_target,
        )
        source_validation_mode = _env_str("SCRIPT_SOURCE_VALIDATION_MODE", default_source_validation_mode).lower()
        if source_validation_mode not in {"off", "warn", "enforce"}:
            source_validation_mode = default_source_validation_mode
        source_validation_warn_ratio = _clamp_float(
            _env_float("SCRIPT_SOURCE_VALIDATION_WARN_RATIO", default_warn_ratio),
            0.0,
            1.0,
        )
        source_validation_enforce_ratio = _clamp_float(
            _env_float("SCRIPT_SOURCE_VALIDATION_ENFORCE_RATIO", default_enforce_ratio),
            0.0,
            1.0,
        )
        if source_validation_enforce_ratio > source_validation_warn_ratio:
            source_validation_enforce_ratio = source_validation_warn_ratio

        return ScriptConfig(
            model=_env_str("SCRIPT_MODEL", _env_str("MODEL", "gpt-5.2")),
            profile_name=profile.name,
            target_minutes=resolved_target,
            words_per_min=resolved_wpm,
            min_words=resolved_min,
            max_words=resolved_max,
            chunk_target_minutes=resolved_chunk_target_minutes,
            max_context_lines=max(6, _env_int("SCRIPT_MAX_CONTEXT_LINES", max_context_default)),
            max_continuations_per_chunk=max(
                1,
                _env_int("SCRIPT_MAX_CONTINUATIONS_PER_CHUNK", max_continuations_default),
            ),
            no_progress_rounds=max(1, _env_int("SCRIPT_NO_PROGRESS_ROUNDS", no_progress_rounds_default)),
            min_word_delta=max(1, _env_int("SCRIPT_MIN_WORD_DELTA", min_word_delta_default)),
            timeout_seconds=max(
                10,
                _env_int(
                    "SCRIPT_TIMEOUT_SECONDS",
                    _env_int("OPENAI_TIMEOUT", openai_timeout_default),
                ),
            ),
            retries=max(1, _env_int("SCRIPT_RETRIES", _env_int("OPENAI_RETRIES", 3))),
            checkpoint_dir=_env_str("SCRIPT_CHECKPOINT_DIR", "./.script_checkpoints"),
            pre_summary_trigger_words=max(500, _env_int("SCRIPT_PRE_SUMMARY_TRIGGER_WORDS", 6000)),
            pre_summary_chunk_target_minutes=max(
                1.0, _env_float("SCRIPT_PRE_SUMMARY_CHUNK_TARGET_MINUTES", 6.0)
            ),
            pre_summary_target_words=max(200, _env_int("SCRIPT_PRE_SUMMARY_TARGET_WORDS", 1800)),
            pre_summary_max_rounds=max(1, _env_int("SCRIPT_PRE_SUMMARY_MAX_ROUNDS", 2)),
            repair_max_attempts=max(0, _env_int("SCRIPT_REPAIR_MAX_ATTEMPTS", 1)),
            max_output_tokens_initial=max(
                512,
                _env_int("SCRIPT_MAX_OUTPUT_TOKENS_INITIAL", max_output_tokens_initial_default),
            ),
            max_output_tokens_chunk=max(
                512,
                _env_int("SCRIPT_MAX_OUTPUT_TOKENS_CHUNK", max_output_tokens_chunk_default),
            ),
            max_output_tokens_continuation=max(
                512,
                _env_int(
                    "SCRIPT_MAX_OUTPUT_TOKENS_CONTINUATION",
                    max_output_tokens_continuation_default,
                ),
            ),
            source_validation_mode=source_validation_mode,
            source_validation_warn_ratio=source_validation_warn_ratio,
            source_validation_enforce_ratio=source_validation_enforce_ratio,
            adaptive_defaults_enabled=adaptive_defaults_enabled,
            pre_summary_parallel=_env_bool("SCRIPT_PRESUMMARY_PARALLEL", False),
            pre_summary_parallel_workers=_clamp_int(
                _env_int("SCRIPT_PRESUMMARY_PARALLEL_WORKERS", 2),
                1,
                4,
            ),
            expected_words_per_chunk=expected_words_per_chunk,
            expected_tokens_per_chunk=expected_tokens_per_chunk,
        )


@dataclass(frozen=True)
class AudioConfig:
    """Audio synthesis/mixing runtime configuration."""

    tts_provider: str
    model: str
    tts_openai_model: str
    tts_alibaba_model: str
    tts_alibaba_base_url: str
    tts_alibaba_language_type: str
    tts_alibaba_optimize_instructions: bool
    tts_alibaba_female_voice: str
    tts_alibaba_male_voice: str
    tts_alibaba_default_voice: str
    timeout_seconds: int
    retries: int
    max_concurrent: int
    checkpoint_dir: str
    tts_max_chars_per_segment: int
    retry_backoff_base_ms: int
    retry_backoff_max_ms: int
    global_timeout_seconds: int
    pause_between_segments_ms: int
    loudnorm_i: float
    loudnorm_tp: float
    loudnorm_lra: float
    bass_eq_freq: int
    bass_eq_gain: float
    ffmpeg_loglevel: str
    chunk_lines: int
    cross_chunk_parallel: bool
    tts_speed_default: float
    tts_speed_intro: float
    tts_speed_body: float
    tts_speed_closing: float
    tts_phase_intro_ratio: float
    tts_phase_closing_ratio: float

    @staticmethod
    def from_env(*, profile_name: Optional[str] = None) -> "AudioConfig":
        """Build audio config from env and optional profile override."""
        profile = resolve_profile(_coalesce(profile_name, _env_str("PODCAST_DURATION_PROFILE", "standard")))
        provider = _env_str("TTS_PROVIDER", "openai").strip().lower()
        if provider not in {"openai", "alibaba"}:
            provider = "openai"
        legacy_model_env = os.environ.get("TTS_MODEL")
        legacy_model = _env_str("TTS_MODEL", "gpt-4o-mini-tts")
        tts_openai_model = _env_str("TTS_OPENAI_MODEL", legacy_model)
        if os.environ.get("TTS_ALIBABA_MODEL", "").strip():
            tts_alibaba_model = _env_str("TTS_ALIBABA_MODEL", "qwen3-tts-instruct-flash")
        elif legacy_model_env is not None and str(legacy_model_env).strip():
            tts_alibaba_model = legacy_model
        else:
            tts_alibaba_model = "qwen3-tts-instruct-flash"
        # Keep this alias stable during migration for call sites that still read audio_cfg.model.
        resolved_model = tts_alibaba_model if provider == "alibaba" else tts_openai_model
        tts_speed_default = _clamp_float(_env_float("TTS_SPEED_DEFAULT", 1.0), 0.25, 4.0)

        def _read_tts_speed(name: str, missing_default: float) -> float:
            """Read bounded TTS speed value with sensible fallback."""
            raw = os.environ.get(name)
            if raw is None or str(raw).strip() == "":
                return _clamp_float(float(missing_default), 0.25, 4.0)
            try:
                parsed = float(str(raw).strip())
                if not math.isfinite(parsed):
                    return tts_speed_default
            except (TypeError, ValueError):
                return tts_speed_default
            return _clamp_float(parsed, 0.25, 4.0)

        # Keep neutral speed by default to avoid robotic artifacts.
        tts_speed_intro = _read_tts_speed("TTS_SPEED_INTRO", tts_speed_default)
        tts_speed_body = _read_tts_speed("TTS_SPEED_BODY", tts_speed_default)
        tts_speed_closing = _read_tts_speed("TTS_SPEED_CLOSING", tts_speed_default)
        tts_phase_intro_ratio = _clamp_float(_env_float("TTS_PHASE_INTRO_RATIO", 0.15), 0.0, 0.45)
        tts_phase_closing_ratio = _clamp_float(_env_float("TTS_PHASE_CLOSING_RATIO", 0.15), 0.0, 0.45)
        return AudioConfig(
            tts_provider=provider,
            model=resolved_model,
            tts_openai_model=tts_openai_model,
            tts_alibaba_model=tts_alibaba_model,
            tts_alibaba_base_url=_env_str(
                "TTS_ALIBABA_BASE_URL",
                "https://dashscope-intl.aliyuncs.com/api/v1",
            ).rstrip("/"),
            tts_alibaba_language_type=_env_str("TTS_ALIBABA_LANGUAGE_TYPE", "Spanish"),
            tts_alibaba_optimize_instructions=_env_bool("TTS_ALIBABA_OPTIMIZE_INSTRUCTIONS", True),
            tts_alibaba_female_voice=_env_str("TTS_ALIBABA_FEMALE_VOICE", "Cherry"),
            tts_alibaba_male_voice=_env_str("TTS_ALIBABA_MALE_VOICE", "Ethan"),
            tts_alibaba_default_voice=_env_str("TTS_ALIBABA_DEFAULT_VOICE", "Cherry"),
            timeout_seconds=max(5, _env_int("TTS_TIMEOUT_SECONDS", _env_int("TTS_TIMEOUT", 60))),
            retries=max(1, _env_int("TTS_RETRIES", 3)),
            max_concurrent=max(1, _env_int("TTS_MAX_CONCURRENT", profile.tts_max_concurrent)),
            checkpoint_dir=_env_str("AUDIO_CHECKPOINT_DIR", "./.audio_checkpoints"),
            tts_max_chars_per_segment=max(80, _env_int("TTS_MAX_CHARS_PER_SEGMENT", 450)),
            retry_backoff_base_ms=max(100, _env_int("TTS_RETRY_BACKOFF_BASE_MS", 800)),
            retry_backoff_max_ms=max(500, _env_int("TTS_RETRY_BACKOFF_MAX_MS", 8000)),
            global_timeout_seconds=max(0, _env_int("TTS_GLOBAL_TIMEOUT_SECONDS", 0)),
            pause_between_segments_ms=max(0, _env_int("PAUSE_BETWEEN_SEGMENTS_MS", 0)),
            loudnorm_i=_env_float("LOUDNORM_LUFS", -16.0),
            loudnorm_tp=_env_float("LOUDNORM_TP", -1.5),
            loudnorm_lra=_env_float("LOUDNORM_LRA", 11.0),
            bass_eq_freq=max(20, _env_int("BASS_EQ_FREQ", 100)),
            bass_eq_gain=_env_float("BASS_EQ_GAIN", 3.0),
            ffmpeg_loglevel=_env_str("FFMPEG_LOGLEVEL", "warning"),
            chunk_lines=max(0, _env_int("CHUNK_LINES", 0)),
            cross_chunk_parallel=_env_bool("TTS_CROSS_CHUNK_PARALLEL", False),
            tts_speed_default=tts_speed_default,
            tts_speed_intro=tts_speed_intro,
            tts_speed_body=tts_speed_body,
            tts_speed_closing=tts_speed_closing,
            tts_phase_intro_ratio=tts_phase_intro_ratio,
            tts_phase_closing_ratio=tts_phase_closing_ratio,
        )


def fingerprint_dict(value: Dict[str, Any]) -> str:
    """Return stable SHA-256 hash for a dictionary payload."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_fingerprint(
    script_cfg: Optional[ScriptConfig] = None,
    audio_cfg: Optional[AudioConfig] = None,
    reliability_cfg: Optional[ReliabilityConfig] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a composite fingerprint across config sections."""
    payload: Dict[str, Any] = {}
    if script_cfg is not None:
        payload["script"] = dataclasses.asdict(script_cfg)
    if audio_cfg is not None:
        payload["audio"] = dataclasses.asdict(audio_cfg)
    if reliability_cfg is not None:
        payload["reliability"] = dataclasses.asdict(reliability_cfg)
    if extra:
        payload["extra"] = extra
    return fingerprint_dict(payload)

