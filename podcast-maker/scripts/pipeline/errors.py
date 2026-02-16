#!/usr/bin/env python3
from __future__ import annotations

import re
import socket
import urllib.error
from typing import Dict, Iterable, List, Sequence

ERROR_KIND_TIMEOUT = "timeout"
ERROR_KIND_STUCK = "stuck"
ERROR_KIND_RATE_LIMIT = "rate_limit"
ERROR_KIND_NETWORK = "network"
ERROR_KIND_RESUME_BLOCKED = "resume_blocked"
ERROR_KIND_SOURCE_TOO_SHORT = "source_too_short"
ERROR_KIND_INVALID_SCHEMA = "invalid_schema"
ERROR_KIND_OPENAI_EMPTY_OUTPUT = "openai_empty_output"
ERROR_KIND_SCRIPT_COMPLETENESS = "script_completeness_failed"
ERROR_KIND_SCRIPT_QUALITY = "script_quality_rejected"
ERROR_KIND_RUN_MISMATCH = "run_mismatch"
ERROR_KIND_INTERRUPTED = "interrupted"
ERROR_KIND_UNKNOWN = "unknown"

STUCK_ERROR_KINDS = {
    ERROR_KIND_TIMEOUT,
    ERROR_KIND_STUCK,
}


def is_stuck_error_kind(kind: str) -> bool:
    return str(kind or "").strip().lower() in STUCK_ERROR_KINDS


class TTSOperationError(RuntimeError):
    def __init__(self, message: str, *, error_kind: str) -> None:
        super().__init__(message)
        self.error_kind = str(error_kind or ERROR_KIND_UNKNOWN).strip().lower()


class ScriptOperationError(RuntimeError):
    def __init__(self, message: str, *, error_kind: str) -> None:
        super().__init__(message)
        self.error_kind = str(error_kind or ERROR_KIND_UNKNOWN).strip().lower()


class TTSBatchError(RuntimeError):
    def __init__(
        self,
        *,
        manifest_path: str,
        failed_segments: Sequence[Dict[str, object]],
        failed_kinds: Sequence[str],
    ) -> None:
        self.manifest_path = manifest_path
        self.failed_segments = [dict(seg) for seg in failed_segments]
        dedup_kinds: List[str] = []
        for kind in failed_kinds:
            normalized = str(kind or ERROR_KIND_UNKNOWN).strip().lower()
            if not normalized:
                normalized = ERROR_KIND_UNKNOWN
            if normalized not in dedup_kinds:
                dedup_kinds.append(normalized)
        if not dedup_kinds:
            dedup_kinds = [ERROR_KIND_UNKNOWN]
        self.failed_kinds = dedup_kinds
        self.primary_kind = self.failed_kinds[0]
        self.stuck_abort = any(is_stuck_error_kind(kind) for kind in self.failed_kinds)
        kinds_preview = ", ".join(self.failed_kinds)
        super().__init__(
            f"TTS synthesis failed for {len(self.failed_segments)} segment(s). "
            f"kinds=[{kinds_preview}] manifest={self.manifest_path}"
        )


def _iter_exception_chain(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        next_exc = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        current = next_exc if isinstance(next_exc, BaseException) else None


def classify_tts_exception(exc: BaseException) -> str:
    messages: List[str] = []
    for item in _iter_exception_chain(exc):
        if isinstance(item, TTSOperationError):
            return item.error_kind
        if isinstance(item, InterruptedError):
            return ERROR_KIND_INTERRUPTED
        if isinstance(item, (TimeoutError, socket.timeout)):
            return ERROR_KIND_TIMEOUT
        if isinstance(item, urllib.error.HTTPError):
            code = int(getattr(item, "code", 0) or 0)
            if code == 429:
                return ERROR_KIND_RATE_LIMIT
            if code in {408, 504}:
                return ERROR_KIND_TIMEOUT
            if code >= 500:
                return ERROR_KIND_NETWORK
        if isinstance(item, urllib.error.URLError):
            reason = getattr(item, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                return ERROR_KIND_TIMEOUT
            return ERROR_KIND_NETWORK
        if isinstance(item, ConnectionError):
            return ERROR_KIND_NETWORK
        messages.append(str(item or ""))

    message = " ".join(messages).lower()
    if "resume blocked" in message:
        return ERROR_KIND_RESUME_BLOCKED
    if "429" in message or "rate limit" in message or "ratelimit" in message:
        return ERROR_KIND_RATE_LIMIT
    if re.search(r"\bstuck\b", message):
        return ERROR_KIND_STUCK
    if "timeout" in message or "timed out" in message:
        return ERROR_KIND_TIMEOUT
    if (
        "connection" in message
        or "network" in message
        or "name or service not known" in message
        or "temporary failure in name resolution" in message
        or "urlopen error" in message
    ):
        return ERROR_KIND_NETWORK
    return ERROR_KIND_UNKNOWN


def summarize_failure_kinds(kinds: Iterable[str]) -> List[str]:
    out: List[str] = []
    for kind in kinds:
        normalized = str(kind or ERROR_KIND_UNKNOWN).strip().lower()
        if not normalized:
            normalized = ERROR_KIND_UNKNOWN
        if normalized not in out:
            out.append(normalized)
    return out

