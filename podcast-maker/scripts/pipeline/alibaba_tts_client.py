#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from .config import AudioConfig, ReliabilityConfig
from .logging_utils import Logger
from .tts_provider import TTSAudioResult, normalize_file_extension


def _redact_sensitive_text(text: str, *, api_key: str) -> str:
    rendered = str(text or "")
    secret = str(api_key or "").strip()
    if secret:
        rendered = rendered.replace(secret, "***")
    # Defensive redaction for accidental bearer echoes from upstream errors.
    rendered = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1***", rendered)
    return rendered


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        parsed = float(str(raw).strip())
        if not math.isfinite(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def _resolve_dashscope_api_key() -> str:
    env_key = str(os.environ.get("DASHSCOPE_API_KEY", "") or "").strip()
    if env_key:
        return env_key
    auth_path = os.path.expanduser("~/.codex/auth.json")
    try:
        with open(auth_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    flat_keys = (
        "DASHSCOPE_API_KEY",
        "dashscope_api_key",
        "dashscopeApiKey",
        "api_key",
    )
    for key in flat_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = payload.get("dashscope")
    if isinstance(nested, dict):
        for key in ("api_key", "DASHSCOPE_API_KEY"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _is_retriable_http(code: int) -> bool:
    return int(code) in {408, 409, 429, 500, 502, 503, 504}


def _extract_url_extension(url: str) -> str:
    path = urllib.parse.urlparse(str(url or "")).path
    if "." not in path:
        return ""
    return str(path.rsplit(".", 1)[-1]).strip().lower()


def _search_audio_url(node: Any) -> str:
    if isinstance(node, dict):
        audio = node.get("audio")
        if isinstance(audio, dict):
            candidate = str(audio.get("url", "")).strip()
            if candidate:
                return candidate
        for key in ("audio_url", "url"):
            candidate = str(node.get(key, "")).strip()
            if key == "audio_url" and candidate:
                return candidate
        for value in node.values():
            candidate = _search_audio_url(value)
            if candidate:
                return candidate
    elif isinstance(node, list):
        for item in node:
            candidate = _search_audio_url(item)
            if candidate:
                return candidate
    return ""


@dataclass
class AlibabaInstructTTSProvider:
    api_key: str
    model_name: str
    base_url: str
    language_type: str
    optimize_instructions: bool
    timeout_seconds: int
    retries: int
    backoff_base_ms: int
    backoff_max_ms: int
    reliability: ReliabilityConfig
    logger: Logger
    provider_name: str = "alibaba"
    default_file_extension: str = "wav"
    _requests_made: int = 0
    _estimated_cost_usd: float = 0.0
    _retries_total: int = 0
    _state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @staticmethod
    def from_audio_config(
        *,
        audio_cfg: AudioConfig,
        reliability: ReliabilityConfig,
        logger: Logger,
    ) -> "AlibabaInstructTTSProvider":
        key = _resolve_dashscope_api_key()
        if not key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is required when TTS_PROVIDER=alibaba "
                "(env or ~/.codex/auth.json)"
            )
        base_url = str(audio_cfg.tts_alibaba_base_url or "").strip().rstrip("/")
        if not base_url.startswith("https://"):
            raise RuntimeError(
                "TTS_ALIBABA_BASE_URL must be a valid https URL "
                "(example: https://dashscope-intl.aliyuncs.com/api/v1)"
            )
        return AlibabaInstructTTSProvider(
            api_key=key,
            model_name=str(audio_cfg.tts_alibaba_model or "qwen3-tts-instruct-flash").strip(),
            base_url=base_url,
            language_type=str(audio_cfg.tts_alibaba_language_type or "Spanish").strip(),
            optimize_instructions=bool(audio_cfg.tts_alibaba_optimize_instructions),
            timeout_seconds=max(5, int(audio_cfg.timeout_seconds)),
            retries=max(1, int(audio_cfg.retries)),
            backoff_base_ms=max(100, int(audio_cfg.retry_backoff_base_ms)),
            backoff_max_ms=max(500, int(audio_cfg.retry_backoff_max_ms)),
            reliability=reliability,
            logger=logger,
        )

    @property
    def requests_made(self) -> int:
        with self._state_lock:
            return int(self._requests_made)

    @property
    def estimated_cost_usd(self) -> float:
        with self._state_lock:
            return float(self._estimated_cost_usd)

    @property
    def retries_total(self) -> int:
        with self._state_lock:
            return int(self._retries_total)

    def _check_budget_locked(self) -> None:
        if (
            self.reliability.max_requests_per_run > 0
            and self._requests_made >= self.reliability.max_requests_per_run
        ):
            raise RuntimeError("Request budget reached (MAX_REQUESTS_PER_RUN)")
        if (
            self.reliability.max_estimated_cost_usd > 0
            and self._estimated_cost_usd >= self.reliability.max_estimated_cost_usd
        ):
            raise RuntimeError("Estimated cost budget reached (MAX_ESTIMATED_COST_USD)")

    def _estimate_cost_per_request(self) -> float:
        direct = _env_float("ESTIMATED_COST_PER_ALIBABA_TTS_REQUEST_USD", float("nan"))
        if math.isfinite(direct):
            return max(0.0, direct)
        fallback = _env_float("ESTIMATED_COST_PER_TTS_REQUEST_USD", 0.01)
        return max(0.0, fallback)

    def _track_usage_locked(self) -> None:
        self._requests_made += 1
        self._estimated_cost_usd += self._estimate_cost_per_request()

    def check_budget(self) -> None:
        with self._state_lock:
            self._check_budget_locked()

    def reserve_request_slot(self) -> None:
        with self._state_lock:
            self._check_budget_locked()
            self._track_usage_locked()

    def _record_retry(self) -> None:
        with self._state_lock:
            self._retries_total += 1

    def _sleep_backoff(
        self,
        *,
        attempt: int,
        retry_is_5xx: bool,
        cancel_check: Optional[Callable[[], bool]],
    ) -> None:
        backoff_s = min(
            self.backoff_max_ms / 1000.0,
            (self.backoff_base_ms / 1000.0) * (2 ** max(0, int(attempt) - 1)),
        )
        if retry_is_5xx:
            backoff_s = min(self.backoff_max_ms / 1000.0, backoff_s * 1.6)
        backoff_s += random.uniform(0.0, 0.2)
        if cancel_check is None:
            time.sleep(backoff_s)
            return
        wake_interval_s = max(0.05, min(0.25, backoff_s))
        deadline = time.time() + backoff_s
        while True:
            if cancel_check():
                raise InterruptedError("Interrupted during Alibaba TTS retry backoff")
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(wake_interval_s, remaining))

    def _post_generation(
        self,
        *,
        payload: Dict[str, Any],
        timeout_seconds: int,
        stage: str,
        cancel_check: Optional[Callable[[], bool]],
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/services/aigc/multimodal-generation/generation"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Interrupted before Alibaba TTS request")
            self.reserve_request_slot()
            retry_is_5xx = False
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                self.logger.info(
                    "alibaba_tts_request_ok",
                    stage=stage,
                    attempt=attempt,
                    elapsed_ms=int((time.time() - started) * 1000),
                    requests_made=self.requests_made,
                    estimated_cost_usd=round(self.estimated_cost_usd, 4),
                )
                return parsed
            except urllib.error.HTTPError as exc:
                code = int(getattr(exc, "code", 0) or 0)
                retry_is_5xx = 500 <= code <= 599
                retriable = _is_retriable_http(code)
                body_preview = exc.read().decode("utf-8", errors="ignore")[:500]
                self.logger.warn(
                    "alibaba_tts_http_error",
                    stage=stage,
                    attempt=attempt,
                    code=code,
                    retriable=retriable,
                    detail=_redact_sensitive_text(body_preview, api_key=self.api_key),
                )
                last_exc = RuntimeError(f"HTTP {code}")
                if not retriable or attempt >= self.retries:
                    break
            except Exception as exc:  # noqa: BLE001
                self.logger.warn(
                    "alibaba_tts_request_error",
                    stage=stage,
                    attempt=attempt,
                    error=_redact_sensitive_text(str(exc), api_key=self.api_key),
                )
                last_exc = exc
                if attempt >= self.retries:
                    break
            self._record_retry()
            self._sleep_backoff(
                attempt=attempt,
                retry_is_5xx=retry_is_5xx,
                cancel_check=cancel_check,
            )
        if last_exc is not None:
            safe_error = _redact_sensitive_text(str(last_exc), api_key=self.api_key)
            raise RuntimeError(f"Alibaba TTS generation failed for stage={stage}: {safe_error}") from last_exc
        raise RuntimeError(f"Alibaba TTS generation failed for stage={stage}")

    def _download_audio(
        self,
        *,
        url: str,
        timeout_seconds: int,
        stage: str,
        cancel_check: Optional[Callable[[], bool]],
    ) -> Tuple[bytes, str, str]:
        if not str(url or "").strip():
            raise RuntimeError("Alibaba response did not include output.audio.url")
        request = urllib.request.Request(str(url).strip())
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Interrupted before Alibaba audio download")
            retry_is_5xx = False
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
                    audio_bytes = resp.read()
                    content_type = str(resp.headers.get("Content-Type", "") or "").split(";", 1)[0].strip()
                if not audio_bytes:
                    raise RuntimeError("Alibaba audio download returned empty content")
                extension = normalize_file_extension(
                    file_extension=_extract_url_extension(url),
                    content_type=content_type,
                    fallback=self.default_file_extension,
                )
                if not content_type:
                    content_type = "audio/wav" if extension == "wav" else "audio/mpeg"
                self.logger.info(
                    "alibaba_tts_audio_download_ok",
                    stage=stage,
                    attempt=attempt,
                    elapsed_ms=int((time.time() - started) * 1000),
                    bytes=len(audio_bytes),
                    content_type=content_type,
                )
                return audio_bytes, content_type, extension
            except urllib.error.HTTPError as exc:
                code = int(getattr(exc, "code", 0) or 0)
                retry_is_5xx = 500 <= code <= 599
                retriable = _is_retriable_http(code)
                self.logger.warn(
                    "alibaba_tts_audio_download_http_error",
                    stage=stage,
                    attempt=attempt,
                    code=code,
                    retriable=retriable,
                )
                last_exc = RuntimeError(f"HTTP {code}")
                if not retriable or attempt >= self.retries:
                    break
            except Exception as exc:  # noqa: BLE001
                self.logger.warn(
                    "alibaba_tts_audio_download_error",
                    stage=stage,
                    attempt=attempt,
                    error=_redact_sensitive_text(str(exc), api_key=self.api_key),
                )
                last_exc = exc
                if attempt >= self.retries:
                    break
            self._record_retry()
            self._sleep_backoff(
                attempt=attempt,
                retry_is_5xx=retry_is_5xx,
                cancel_check=cancel_check,
            )
        if last_exc is not None:
            safe_error = _redact_sensitive_text(str(last_exc), api_key=self.api_key)
            raise RuntimeError(f"Alibaba audio download failed for stage={stage}: {safe_error}") from last_exc
        raise RuntimeError(f"Alibaba audio download failed for stage={stage}")

    def synthesize_speech(
        self,
        *,
        text: str,
        instructions: str,
        voice: str,
        speed: Optional[float] = None,
        stage: str,
        timeout_seconds_override: Optional[int] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> TTSAudioResult:
        timeout_seconds = (
            max(1, int(timeout_seconds_override))
            if timeout_seconds_override is not None
            else self.timeout_seconds
        )
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": {
                "text": text,
                "voice": voice,
                "language_type": self.language_type,
            },
            "parameters": {
                "instructions": instructions,
                "optimize_instructions": bool(self.optimize_instructions),
            },
        }
        if speed is not None:
            # API does not document a speed parameter for this endpoint; keep it as hint.
            payload["parameters"]["speed"] = speed
        response = self._post_generation(
            payload=payload,
            timeout_seconds=timeout_seconds,
            stage=stage,
            cancel_check=cancel_check,
        )
        audio_url = _search_audio_url(response.get("output", response))
        if not audio_url:
            raise RuntimeError(
                "Alibaba response did not include audio URL; check region/base URL, model, and permissions."
            )
        audio_bytes, content_type, extension = self._download_audio(
            url=audio_url,
            timeout_seconds=timeout_seconds,
            stage=stage,
            cancel_check=cancel_check,
        )
        return TTSAudioResult(
            audio_bytes=audio_bytes,
            content_type=content_type,
            file_extension=extension,
            provider=self.provider_name,
            model=self.model_name,
        )
