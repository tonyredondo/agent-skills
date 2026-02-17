#!/usr/bin/env python3
from __future__ import annotations

"""OpenAI transport and resilience layer for script/TTS requests.

This module centralizes:
- request budgeting and coarse cost tracking,
- retry/backoff/circuit-breaker behavior,
- script JSON parse classification + repair attempts.
"""

import json
import math
import os
import random
import threading
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .config import ReliabilityConfig
from .logging_utils import Logger


def _env_int(name: str, default: int) -> int:
    """Read integer env var with defensive fallback."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read finite float env var with defensive fallback."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
        if not math.isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _clamp_tts_speed(value: Any, *, fallback: float = 1.0) -> float:
    """Clamp TTS speed to API-supported range."""
    try:
        parsed = float(value)
        if not math.isfinite(parsed):
            parsed = float(fallback)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if not math.isfinite(parsed):
        parsed = 1.0
    return round(max(0.25, min(4.0, parsed)), 3)


def _resolve_openai_api_key() -> str:
    """Resolve API key from env first, then local auth file."""
    env_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_api_key:
        return env_api_key
    auth_path = os.path.expanduser("~/.codex/auth.json")
    try:
        with open(auth_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    for key_name in ("OPENAI_API_KEY", "openai_api_key", "api_key"):
        candidate = payload.get(key_name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _resolve_script_reasoning_effort() -> str:
    """Resolve reasoning effort level for script requests."""
    effort = str(os.environ.get("SCRIPT_REASONING_EFFORT", "medium") or "").strip().lower()
    if effort in {"low", "medium", "high"}:
        return effort
    return "medium"


def _resolve_quality_eval_reasoning_effort(default_effort: str) -> str:
    """Resolve reasoning effort override for script quality evaluation stage."""
    override = str(os.environ.get("SCRIPT_QUALITY_EVAL_REASONING_EFFORT", "high") or "").strip().lower()
    if override in {"low", "medium", "high"}:
        return override
    normalized_default = str(default_effort or "").strip().lower()
    if normalized_default in {"low", "medium", "high"}:
        return normalized_default
    return "low"


def _extract_text_from_responses_payload(payload: Dict[str, Any]) -> str:
    """Extract output text from Responses API payload variants."""
    text = ""
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text += str(content.get("text", ""))
    if not text and payload.get("output_text"):
        text = str(payload["output_text"])
    return text.strip()


PARSE_FAILURE_TRUNCATION = "truncation"
PARSE_FAILURE_WRAPPER = "wrapper"
PARSE_FAILURE_MALFORMED = "malformed"
PARSE_FAILURE_EMPTY_OUTPUT = "empty_output"


def _extract_json_object_candidate(raw_text: str) -> tuple[str, bool, bool]:
    """Extract best JSON-object candidate and wrapper/truncation hints."""
    text = str(raw_text or "").strip()
    if not text:
        return "", False, False
    start = text.find("{")
    if start < 0:
        return "", False, False
    end = text.rfind("}")
    if end > start:
        candidate = text[start : end + 1]
        wrapped = start > 0 or end < (len(text) - 1)
        return candidate, wrapped, False
    candidate = text[start:]
    wrapped = start > 0
    return candidate, wrapped, True


def _looks_like_truncated_json(*, raw_text: str, parse_error: str) -> bool:
    """Heuristic detection for truncated JSON responses."""
    text = str(raw_text or "").strip()
    if not text:
        return False
    error = str(parse_error or "").strip().lower()
    if "unterminated string" in error:
        return True
    if "unexpected end" in error or "end of data" in error:
        return True
    if "expecting value" in error and text.endswith(("{", "[", ":", ",", "\"")):
        return True
    if text.count("{") > text.count("}") or text.count("[") > text.count("]"):
        return True
    if text.endswith(("...", "…")):
        return True
    return False


def _classify_json_parse_failure(
    *,
    raw_text: str,
    parse_error: str,
    wrapper_hint: bool,
    incomplete_object_hint: bool,
) -> str:
    """Classify parse failures to drive targeted repair policy."""
    if not str(raw_text or "").strip():
        return PARSE_FAILURE_EMPTY_OUTPUT
    if incomplete_object_hint or _looks_like_truncated_json(raw_text=raw_text, parse_error=parse_error):
        return PARSE_FAILURE_TRUNCATION
    if wrapper_hint or "```" in str(raw_text or ""):
        return PARSE_FAILURE_WRAPPER
    return PARSE_FAILURE_MALFORMED


def _parse_repair_budget(*, base_tokens: int, attempt: int, parse_failure_kind: str) -> int:
    """Compute adaptive token budget for parse repair attempts."""
    base = max(64, int(base_tokens))
    kind = str(parse_failure_kind or "").strip().lower()
    if kind == PARSE_FAILURE_EMPTY_OUTPUT:
        growth = max(1.0, _env_float("SCRIPT_PARSE_REPAIR_EMPTY_OUTPUT_OUTPUT_TOKENS_GROWTH", 1.15))
        cap = max(
            base,
            _env_int(
                "SCRIPT_PARSE_REPAIR_EMPTY_OUTPUT_MAX_OUTPUT_TOKENS",
                _env_int("SCRIPT_PARSE_REPAIR_MAX_OUTPUT_TOKENS", 10000),
            ),
        )
    elif kind == PARSE_FAILURE_TRUNCATION:
        growth = max(1.0, _env_float("SCRIPT_PARSE_REPAIR_TRUNCATION_OUTPUT_TOKENS_GROWTH", 1.55))
        cap = max(
            base,
            _env_int(
                "SCRIPT_PARSE_REPAIR_TRUNCATION_MAX_OUTPUT_TOKENS",
                _env_int("SCRIPT_PARSE_REPAIR_MAX_OUTPUT_TOKENS", 10000),
            ),
        )
    elif kind == PARSE_FAILURE_WRAPPER:
        growth = max(1.0, _env_float("SCRIPT_PARSE_REPAIR_WRAPPER_OUTPUT_TOKENS_GROWTH", 1.2))
        cap = max(
            base,
            _env_int(
                "SCRIPT_PARSE_REPAIR_WRAPPER_MAX_OUTPUT_TOKENS",
                _env_int("SCRIPT_PARSE_REPAIR_MAX_OUTPUT_TOKENS", 10000),
            ),
        )
    else:
        growth = max(1.0, _env_float("SCRIPT_PARSE_REPAIR_OUTPUT_TOKENS_GROWTH", 1.35))
        cap = max(base, _env_int("SCRIPT_PARSE_REPAIR_MAX_OUTPUT_TOKENS", 10000))
    candidate = int(round(float(base) * (growth ** max(0, int(attempt) - 1))))
    return max(base, min(cap, candidate))


def _build_parse_repair_input(
    *,
    raw_text: str,
    extracted_candidate: str,
    parse_failure_kind: str,
    max_input_chars: int,
) -> str:
    """Prepare bounded invalid content input for repair prompts."""
    kind = str(parse_failure_kind or "").strip().lower()
    candidate = str(extracted_candidate or raw_text or "")
    if kind == PARSE_FAILURE_TRUNCATION:
        tail_chars = max(2000, _env_int("SCRIPT_PARSE_REPAIR_TRUNCATION_TAIL_CHARS", 32000))
        if len(candidate) > tail_chars:
            candidate = candidate[-tail_chars:]
            start = candidate.find("{")
            if start > 0:
                candidate = candidate[start:]
    elif len(candidate) > max_input_chars:
        candidate = candidate[:max_input_chars]
    if len(candidate) > max_input_chars:
        candidate = candidate[:max_input_chars]
    return candidate


@dataclass
class OpenAIClient:
    """Thin client wrapper with retry/budget/telemetry semantics."""

    api_key: str
    logger: Logger
    reliability: ReliabilityConfig
    script_model: str
    script_reasoning_effort: str
    tts_model: str
    script_timeout_seconds: int
    script_retries: int
    tts_timeout_seconds: int
    tts_retries: int
    tts_backoff_base_ms: int
    tts_backoff_max_ms: int
    circuit_breaker_failures: int
    consecutive_script_failures: int = 0
    consecutive_tts_failures: int = 0
    requests_made: int = 0
    script_requests_made: int = 0
    tts_requests_made: int = 0
    estimated_cost_usd: float = 0.0
    script_retries_total: int = 0
    tts_retries_total: int = 0
    script_json_parse_failures: int = 0
    script_json_parse_repair_successes: int = 0
    script_json_parse_repair_failures: int = 0
    script_json_parse_failures_by_stage: Dict[str, int] = field(default_factory=dict)
    script_json_parse_failures_by_kind: Dict[str, int] = field(default_factory=dict)
    script_json_parse_repair_successes_by_stage: Dict[str, int] = field(default_factory=dict)
    script_json_parse_repair_successes_by_kind: Dict[str, int] = field(default_factory=dict)
    script_json_parse_repair_failures_by_stage: Dict[str, int] = field(default_factory=dict)
    script_json_parse_repair_failures_by_kind: Dict[str, int] = field(default_factory=dict)
    script_empty_output_events: int = 0
    script_empty_output_retries: int = 0
    script_empty_output_failures: int = 0
    script_empty_output_by_stage: Dict[str, int] = field(default_factory=dict)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @staticmethod
    def from_configs(
        *,
        logger: Logger,
        reliability: ReliabilityConfig,
        script_model: str,
        tts_model: str,
        script_timeout_seconds: int,
        script_retries: int,
        tts_timeout_seconds: int,
        tts_retries: int,
        tts_backoff_base_ms: int,
        tts_backoff_max_ms: int,
    ) -> "OpenAIClient":
        """Factory that resolves credentials and initializes runtime config."""
        api_key = _resolve_openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required (env or ~/.codex/auth.json)")
        return OpenAIClient(
            api_key=api_key,
            logger=logger,
            reliability=reliability,
            script_model=script_model,
            script_reasoning_effort=_resolve_script_reasoning_effort(),
            tts_model=tts_model,
            script_timeout_seconds=script_timeout_seconds,
            script_retries=script_retries,
            tts_timeout_seconds=tts_timeout_seconds,
            tts_retries=tts_retries,
            tts_backoff_base_ms=tts_backoff_base_ms,
            tts_backoff_max_ms=tts_backoff_max_ms,
            circuit_breaker_failures=max(0, _env_int("OPENAI_CIRCUIT_BREAKER_FAILURES", 0)),
        )

    def _check_budget(self) -> None:
        """Enforce run-level request/cost limits."""
        if self.reliability.max_requests_per_run > 0 and self.requests_made >= self.reliability.max_requests_per_run:
            raise RuntimeError("Request budget reached (MAX_REQUESTS_PER_RUN)")
        if self.reliability.max_estimated_cost_usd > 0 and self.estimated_cost_usd >= self.reliability.max_estimated_cost_usd:
            raise RuntimeError("Estimated cost budget reached (MAX_ESTIMATED_COST_USD)")

    def _track_usage(self, *, request_kind: str) -> None:
        """Track request counters and coarse estimated cost."""
        self.requests_made += 1
        if request_kind == "script":
            self.script_requests_made += 1
        elif request_kind == "tts":
            self.tts_requests_made += 1
        # Coarse estimate, disabled by default unless budget is configured.
        if request_kind == "script":
            self.estimated_cost_usd += max(0.0, _env_float("ESTIMATED_COST_PER_SCRIPT_REQUEST_USD", 0.02))
        elif request_kind == "tts":
            self.estimated_cost_usd += max(0.0, _env_float("ESTIMATED_COST_PER_TTS_REQUEST_USD", 0.01))

    def _reserve_request_slot(self, *, request_kind: str) -> None:
        """Atomically enforce budget then reserve one request slot."""
        with self._state_lock:
            self._check_budget()
            self._track_usage(request_kind=request_kind)

    def _post_json(
        self,
        *,
        endpoint: str,
        payload: Dict[str, Any],
        timeout_seconds: int,
        retries: int,
        request_kind: str,
        stage: str,
    ) -> Dict[str, Any]:
        """POST JSON with retry/backoff and structured logging."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        last_exc: Optional[BaseException] = None
        if (
            request_kind == "script"
            and self.circuit_breaker_failures > 0
            and self.consecutive_script_failures >= self.circuit_breaker_failures
        ):
            # Circuit breaker avoids burning budget when upstream service is
            # likely unavailable and recent failures are consecutive.
            raise RuntimeError("Circuit breaker open for script requests")
        for attempt in range(1, retries + 1):
            # Reserve budget before request dispatch so accounting reflects real
            # attempted workload, even if transport fails immediately.
            self._reserve_request_slot(request_kind=request_kind)
            retry_is_5xx = False
            started = time.time()
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                elapsed_ms = int((time.time() - started) * 1000)
                self.logger.info(
                    "openai_request_ok",
                    stage=stage,
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                    request_kind=request_kind,
                    requests_made=self.requests_made,
                    estimated_cost_usd=round(self.estimated_cost_usd, 4),
                )
                if request_kind == "script":
                    with self._state_lock:
                        self.consecutive_script_failures = 0
                return json.loads(raw)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                retriable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                self.logger.warn(
                    "openai_http_error",
                    stage=stage,
                    attempt=attempt,
                    code=exc.code,
                    retriable=retriable,
                    detail=body[:500],
                )
                last_exc = exc
                retry_is_5xx = 500 <= int(exc.code) <= 599
                if request_kind == "script":
                    with self._state_lock:
                        self.consecutive_script_failures += 1
                if not retriable or attempt >= retries:
                    break
            except Exception as exc:  # noqa: BLE001
                self.logger.warn(
                    "openai_request_error",
                    stage=stage,
                    attempt=attempt,
                    error=str(exc),
                )
                last_exc = exc
                if request_kind == "script":
                    with self._state_lock:
                        self.consecutive_script_failures += 1
                if attempt >= retries:
                    break

            backoff_s = min(
                self.tts_backoff_max_ms / 1000.0,
                (self.tts_backoff_base_ms / 1000.0) * (2 ** max(0, attempt - 1)),
            )
            if retry_is_5xx:
                # 5xx bursts tend to indicate service pressure; increase delay
                # to reduce retry storms against a degraded upstream.
                backoff_s = min(self.tts_backoff_max_ms / 1000.0, backoff_s * 1.6)
            if request_kind == "script":
                with self._state_lock:
                    self.script_retries_total += 1
            elif request_kind == "tts":
                with self._state_lock:
                    self.tts_retries_total += 1
            backoff_s += random.uniform(0.0, 0.2)
            self.logger.info(
                "openai_retry_wait",
                stage=stage,
                attempt=attempt,
                backoff_s=round(backoff_s, 2),
            )
            time.sleep(backoff_s)
        if last_exc is not None:
            raise RuntimeError(f"OpenAI request failed for stage={stage}: {last_exc}") from last_exc
        raise RuntimeError(f"OpenAI request failed for stage={stage}")

    def _record_script_parse_failure(self, *, stage: str, parse_failure_kind: str) -> None:
        """Track script JSON parse failures by stage and kind."""
        with self._state_lock:
            self.script_json_parse_failures += 1
            self.script_json_parse_failures_by_stage[stage] = int(
                self.script_json_parse_failures_by_stage.get(stage, 0)
            ) + 1
            self.script_json_parse_failures_by_kind[parse_failure_kind] = int(
                self.script_json_parse_failures_by_kind.get(parse_failure_kind, 0)
            ) + 1

    def _recover_empty_script_output(
        self,
        *,
        prompt: str,
        schema: Dict[str, Any],
        stage: str,
        max_output_tokens: int,
        timeout_seconds: int,
        retries: int,
    ) -> str:
        """Retry script call with reduced budget when output is empty."""
        with self._state_lock:
            self.script_empty_output_events += 1
            self.script_empty_output_by_stage[stage] = int(self.script_empty_output_by_stage.get(stage, 0)) + 1
        self._record_script_parse_failure(stage=stage, parse_failure_kind=PARSE_FAILURE_EMPTY_OUTPUT)
        empty_output_retries = max(0, _env_int("SCRIPT_EMPTY_OUTPUT_RETRIES", 2))
        retry_budget_scale = max(
            0.25,
            min(1.25, _env_float("SCRIPT_EMPTY_OUTPUT_RETRY_OUTPUT_BUDGET_SCALE", 0.8)),
        )
        for attempt in range(1, empty_output_retries + 1):
            with self._state_lock:
                self.script_empty_output_retries += 1
            retry_tokens = max(128, int(round(float(max_output_tokens) * retry_budget_scale)))
            retry_payload = self._script_json_payload(
                prompt=prompt,
                schema=schema,
                max_output_tokens=retry_tokens,
            )
            raw_retry = self._post_json(
                endpoint="https://api.openai.com/v1/responses",
                payload=retry_payload,
                timeout_seconds=timeout_seconds,
                retries=retries,
                request_kind="script",
                stage=f"{stage}_empty_output_retry_{attempt}",
            )
            retry_text = _extract_text_from_responses_payload(raw_retry)
            if retry_text:
                self.logger.warn(
                    "script_empty_output_recovered",
                    stage=stage,
                    attempt=attempt,
                )
                return retry_text
        with self._state_lock:
            self.script_empty_output_failures += 1
        return ""

    def _script_json_payload(
        self,
        *,
        prompt: str,
        schema: Dict[str, Any],
        max_output_tokens: int,
    ) -> Dict[str, Any]:
        """Build Responses API payload for structured script generation."""
        return {
            "model": self.script_model,
            "input": [
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a senior podcast scriptwriter for Spanish spoken audio. Write natural, engaging dialogue with smooth transitions and no explicit section labels. Keep strict alternation between Host1 and Host2 turns. Avoid repetitive line openers across consecutive turns (especially repeated 'Y ...'). Prefer natural Spanish technical phrasing and avoid unnecessary anglicisms (for example, use 'donante adicional' instead of 'donor extra'). Never include internal workflow/tooling notes in spoken output (for example script paths, shell commands, DailyRead pipeline notes, Tavily, Serper). If source metadata includes author names, reference them naturally without inventing names. If source covers multiple topics or includes an index, start with a brief spoken roadmap and then pivot to the first topic. Make the conversation genuinely interactive: hosts should ask each other direct questions, challenge assumptions respectfully, and respond to each other's points instead of delivering isolated monologues. Keep question cadence natural (avoid constant back-to-back questions) and mix in concise assertions and reactions. Never leave an explicit question unresolved before recap or farewell: if one host asks near the end, the counterpart must answer before the closing lines. In the final 2-3 turns before recap/farewell, avoid introducing new questions and use declarative closure turns. Keep recap lines concise and easy to follow; split dense closing summaries across two short turns when needed. Use occasional everyday analogies to clarify technical ideas, and allow only brief, respectful humor when it feels natural.",
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            "reasoning": {"effort": self.script_reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "podcast_script",
                    "schema": schema,
                }
            },
            "max_output_tokens": max_output_tokens,
        }

    def _script_freeform_payload(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
        reasoning_effort: str | None = None,
    ) -> Dict[str, Any]:
        """Build Responses API payload for free-form text requests."""
        effort = str(reasoning_effort or self.script_reasoning_effort or "low").strip().lower()
        if effort not in {"low", "medium", "high"}:
            effort = str(self.script_reasoning_effort or "low").strip().lower()
        if effort not in {"low", "medium", "high"}:
            effort = "low"
        return {
            "model": self.script_model,
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are a precise writing assistant."}],
                },
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            "reasoning": {"effort": effort},
            "max_output_tokens": max_output_tokens,
        }

    def generate_freeform_text(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
        stage: str,
        timeout_seconds_override: Optional[int] = None,
    ) -> str:
        """Generate plain text output for evaluators and helper prompts."""
        reasoning_effort = self.script_reasoning_effort
        if stage == "script_quality_eval":
            # Keep expensive reasoning scoped to quality evaluation only so
            # generation stages preserve throughput and parse stability.
            reasoning_effort = _resolve_quality_eval_reasoning_effort(self.script_reasoning_effort)
        payload = self._script_freeform_payload(
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        request_timeout = (
            max(1, int(timeout_seconds_override))
            if timeout_seconds_override is not None
            else self.script_timeout_seconds
        )
        raw = self._post_json(
            endpoint="https://api.openai.com/v1/responses",
            payload=payload,
            timeout_seconds=request_timeout,
            retries=self.script_retries,
            request_kind="script",
            stage=stage,
        )
        text = _extract_text_from_responses_payload(raw)
        if not text:
            raise RuntimeError(f"OpenAI returned empty text for stage={stage}")
        return text

    def generate_script_json(
        self,
        *,
        prompt: str,
        schema: Dict[str, Any],
        max_output_tokens: int,
        stage: str,
        timeout_seconds_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate schema-aligned script JSON with parse repair fallback."""
        payload = self._script_json_payload(prompt=prompt, schema=schema, max_output_tokens=max_output_tokens)
        request_timeout = (
            max(1, int(timeout_seconds_override))
            if timeout_seconds_override is not None
            else self.script_timeout_seconds
        )
        raw = self._post_json(
            endpoint="https://api.openai.com/v1/responses",
            payload=payload,
            timeout_seconds=request_timeout,
            retries=self.script_retries,
            request_kind="script",
            stage=stage,
        )
        text = _extract_text_from_responses_payload(raw)
        if not text:
            # Empty payloads may still be recoverable via lower output budget
            # retries that nudge the model to return non-empty JSON.
            text = self._recover_empty_script_output(
                prompt=prompt,
                schema=schema,
                stage=stage,
                max_output_tokens=max_output_tokens,
                timeout_seconds=request_timeout,
                retries=self.script_retries,
            )
        if not text:
            raise RuntimeError(
                f"OpenAI returned empty text for stage={stage}; parse_failure_kind={PARSE_FAILURE_EMPTY_OUTPUT}"
            )
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("root_json_value_is_not_object")
        except Exception as exc:  # noqa: BLE001
            parse_exc: BaseException = exc
            # First recovery path: extract object when model wraps JSON with
            # prose/markdown but object itself is still parseable.
            extracted_candidate, wrapper_hint, incomplete_object_hint = _extract_json_object_candidate(text)
            if extracted_candidate and (wrapper_hint or extracted_candidate != text):
                try:
                    extracted_parsed = json.loads(extracted_candidate)
                    if isinstance(extracted_parsed, dict):
                        self.logger.info(
                            "script_json_parse_wrapper_recovered",
                            stage=stage,
                            extracted_chars=len(extracted_candidate),
                        )
                        return extracted_parsed
                    parse_exc = ValueError("extracted_root_json_value_is_not_object")
                except Exception as extracted_exc:  # noqa: BLE001
                    parse_exc = extracted_exc
            parse_failure_kind = _classify_json_parse_failure(
                raw_text=text,
                parse_error=str(parse_exc),
                wrapper_hint=wrapper_hint,
                incomplete_object_hint=incomplete_object_hint,
            )
            self._record_script_parse_failure(
                stage=stage,
                parse_failure_kind=parse_failure_kind,
            )
            self.logger.warn(
                "script_json_parse_failed",
                stage=stage,
                error=str(parse_exc),
                parse_failure_kind=parse_failure_kind,
                preview=text[:500],
            )
            repair_attempts_base = max(0, _env_int("SCRIPT_PARSE_REPAIR_ATTEMPTS", 2))
            truncation_bonus_attempts = max(0, _env_int("SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS", 2))
            wrapper_bonus_attempts = max(0, _env_int("SCRIPT_PARSE_REPAIR_WRAPPER_BONUS_ATTEMPTS", 1))
            # Parse-kind-specific attempt budgets reduce over-retrying benign
            # wrappers while giving truncated payloads more room to recover.
            if parse_failure_kind == PARSE_FAILURE_TRUNCATION:
                repair_attempts = repair_attempts_base + truncation_bonus_attempts
            elif parse_failure_kind == PARSE_FAILURE_WRAPPER:
                repair_attempts = repair_attempts_base + wrapper_bonus_attempts
            else:
                repair_attempts = repair_attempts_base
            last_exc: BaseException = parse_exc
            max_input_chars = max(2000, _env_int("SCRIPT_PARSE_REPAIR_MAX_INPUT_CHARS", 120000))
            invalid_content = _build_parse_repair_input(
                raw_text=text,
                extracted_candidate=extracted_candidate,
                parse_failure_kind=parse_failure_kind,
                max_input_chars=max_input_chars,
            )
            for attempt in range(1, repair_attempts + 1):
                # LLM repair path asks for a strict JSON-only response that
                # preserves content while fixing syntax/shape defects.
                repair_max_tokens = _parse_repair_budget(
                    base_tokens=max_output_tokens,
                    attempt=attempt,
                    parse_failure_kind=parse_failure_kind,
                )
                if repair_max_tokens > max_output_tokens:
                    self.logger.info(
                        "script_json_parse_repair_budget_increased",
                        stage=stage,
                        attempt=attempt,
                        max_output_tokens=repair_max_tokens,
                        parse_failure_kind=parse_failure_kind,
                    )
                repair_prompt = textwrap.dedent(
                    f"""
                    Repair the content below into valid JSON matching the requested schema.
                    Return ONLY JSON object with key "lines".
                    Keep as much original meaning as possible.
                    Do not add markdown or explanations.
                    Parse failure kind: {parse_failure_kind}.
                    If content is truncated, complete quotes/brackets/braces and close all JSON structures.
                    Preserve the existing "lines" content unless syntax repair requires minimal edits.

                    INVALID CONTENT:
                    {invalid_content}
                    """
                ).strip()
                repair_payload = self._script_json_payload(
                    prompt=repair_prompt,
                    schema=schema,
                    max_output_tokens=repair_max_tokens,
                )
                raw_repair = self._post_json(
                    endpoint="https://api.openai.com/v1/responses",
                    payload=repair_payload,
                    timeout_seconds=request_timeout,
                    retries=self.script_retries,
                    request_kind="script",
                    stage=f"{stage}_parse_repair_{attempt}",
                )
                repaired_text = _extract_text_from_responses_payload(raw_repair)
                if not repaired_text:
                    # Missing repair payload counts as inconclusive attempt and
                    # falls through to the next retry budget.
                    continue
                repaired_candidate, _, _ = _extract_json_object_candidate(repaired_text)
                try:
                    repaired_payload = json.loads(repaired_candidate or repaired_text)
                    if not isinstance(repaired_payload, dict):
                        raise ValueError("repaired_root_json_value_is_not_object")
                    self.logger.info(
                        "script_json_parse_repair_ok",
                        stage=stage,
                        attempt=attempt,
                        parse_failure_kind=parse_failure_kind,
                    )
                    with self._state_lock:
                        self.script_json_parse_repair_successes += 1
                        self.script_json_parse_repair_successes_by_stage[stage] = int(
                            self.script_json_parse_repair_successes_by_stage.get(stage, 0)
                        ) + 1
                        self.script_json_parse_repair_successes_by_kind[parse_failure_kind] = int(
                            self.script_json_parse_repair_successes_by_kind.get(parse_failure_kind, 0)
                        ) + 1
                    return repaired_payload
                except Exception as repair_exc:  # noqa: BLE001
                    last_exc = repair_exc
                    with self._state_lock:
                        self.script_json_parse_repair_failures += 1
                        self.script_json_parse_repair_failures_by_stage[stage] = int(
                            self.script_json_parse_repair_failures_by_stage.get(stage, 0)
                        ) + 1
                        self.script_json_parse_repair_failures_by_kind[parse_failure_kind] = int(
                            self.script_json_parse_repair_failures_by_kind.get(parse_failure_kind, 0)
                        ) + 1
                    self.logger.warn(
                        "script_json_parse_repair_failed",
                        stage=stage,
                        attempt=attempt,
                        error=str(repair_exc),
                        parse_failure_kind=parse_failure_kind,
                    )
            raise RuntimeError(f"Failed to parse JSON output for stage={stage}: {last_exc}")

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
    ) -> bytes:
        """Synthesize one TTS segment with retry/backoff/cancel support."""
        payload = {
            "model": self.tts_model,
            "voice": voice,
            "format": "mp3",
            "input": text,
            "instructions": instructions,
        }
        if speed is not None:
            payload["speed"] = _clamp_tts_speed(speed, fallback=1.0)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        last_exc: Optional[BaseException] = None
        if (
            self.circuit_breaker_failures > 0
            and self.consecutive_tts_failures >= self.circuit_breaker_failures
        ):
            raise RuntimeError("Circuit breaker open for TTS requests")
        for attempt in range(1, self.tts_retries + 1):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Interrupted before TTS request")
            # Reserve request slot before transport call for consistent budget
            # accounting across successful and failed attempts.
            self._reserve_request_slot(request_kind="tts")
            retry_is_5xx = False
            started = time.time()
            request_timeout = (
                max(1, int(timeout_seconds_override))
                if timeout_seconds_override is not None
                else self.tts_timeout_seconds
            )
            try:
                with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                    audio = resp.read()
                elapsed_ms = int((time.time() - started) * 1000)
                self.logger.info(
                    "tts_ok",
                    stage=stage,
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                    bytes=len(audio),
                )
                with self._state_lock:
                    self.consecutive_tts_failures = 0
                return audio
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                retriable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                self.logger.warn(
                    "tts_http_error",
                    stage=stage,
                    attempt=attempt,
                    code=exc.code,
                    retriable=retriable,
                    detail=body[:500],
                )
                last_exc = exc
                retry_is_5xx = 500 <= int(exc.code) <= 599
                with self._state_lock:
                    self.consecutive_tts_failures += 1
                if not retriable or attempt >= self.tts_retries:
                    break
            except Exception as exc:  # noqa: BLE001
                self.logger.warn("tts_error", stage=stage, attempt=attempt, error=str(exc))
                last_exc = exc
                with self._state_lock:
                    self.consecutive_tts_failures += 1
                if attempt >= self.tts_retries:
                    break

            backoff_s = min(
                self.tts_backoff_max_ms / 1000.0,
                (self.tts_backoff_base_ms / 1000.0) * (2 ** max(0, attempt - 1)),
            )
            if retry_is_5xx:
                # 5xx responses typically recover slower than rate-limit jitter,
                # so apply a stronger backoff multiplier.
                backoff_s = min(self.tts_backoff_max_ms / 1000.0, backoff_s * 1.6)
            with self._state_lock:
                self.tts_retries_total += 1
            backoff_s += random.uniform(0.0, 0.2)
            self.logger.info("tts_retry_wait", stage=stage, attempt=attempt, backoff_s=round(backoff_s, 2))
            if cancel_check is None:
                time.sleep(backoff_s)
            else:
                # Sleep in short intervals so external cancellation can interrupt
                # retry backoff quickly.
                wake_interval_s = max(0.05, min(0.25, backoff_s))
                sleep_deadline = time.time() + backoff_s
                while True:
                    if cancel_check():
                        raise InterruptedError("Interrupted during TTS retry backoff")
                    remaining = sleep_deadline - time.time()
                    if remaining <= 0:
                        break
                    time.sleep(min(wake_interval_s, remaining))
        if last_exc is not None:
            raise RuntimeError(f"TTS request failed for stage={stage}: {last_exc}") from last_exc
        raise RuntimeError(f"TTS request failed for stage={stage}")

