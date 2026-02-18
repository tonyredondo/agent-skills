#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

from .openai_client import OpenAIClient


CONTENT_TYPE_EXTENSION_MAP = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/flac": "flac",
}


def normalize_file_extension(
    file_extension: str = "",
    *,
    content_type: str = "",
    fallback: str = "mp3",
) -> str:
    ext = str(file_extension or "").strip().lower().lstrip(".")
    if ext and re.match(r"^[a-z0-9]+$", ext):
        return ext
    normalized_content_type = str(content_type or "").split(";", 1)[0].strip().lower()
    mapped = CONTENT_TYPE_EXTENSION_MAP.get(normalized_content_type)
    if mapped:
        return mapped
    fb = str(fallback or "mp3").strip().lower().lstrip(".")
    if fb and re.match(r"^[a-z0-9]+$", fb):
        return fb
    return "mp3"


@dataclass(frozen=True)
class TTSAudioResult:
    audio_bytes: bytes
    content_type: str
    file_extension: str
    provider: str
    model: str


@runtime_checkable
class TTSProvider(Protocol):
    """Provider contract for TTS adapters.

    Implementations must enforce budget and usage accounting internally, and
    protect shared counters with thread-safe synchronization.
    """

    provider_name: str
    model_name: str
    default_file_extension: str

    @property
    def requests_made(self) -> int:
        ...

    @property
    def estimated_cost_usd(self) -> float:
        ...

    @property
    def retries_total(self) -> int:
        ...

    def check_budget(self) -> None:
        ...

    def reserve_request_slot(self) -> None:
        ...

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
        ...


@dataclass
class OpenAITTSProvider:
    client: OpenAIClient
    provider_name: str = "openai"
    default_file_extension: str = "mp3"

    @property
    def model_name(self) -> str:
        return str(getattr(self.client, "tts_model", "")).strip() or "gpt-4o-mini-tts"

    @property
    def requests_made(self) -> int:
        return int(getattr(self.client, "tts_requests_made", getattr(self.client, "requests_made", 0)))

    @property
    def estimated_cost_usd(self) -> float:
        return float(getattr(self.client, "estimated_cost_usd", 0.0))

    @property
    def retries_total(self) -> int:
        return int(getattr(self.client, "tts_retries_total", 0))

    def check_budget(self) -> None:
        check = getattr(self.client, "check_budget", None)
        if callable(check):
            check()
            return
        fallback = getattr(self.client, "_check_budget", None)
        if callable(fallback):
            fallback()

    def reserve_request_slot(self) -> None:
        # OpenAIClient performs reservation internally in synthesize_speech.
        self.check_budget()

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
        audio_bytes = self.client.synthesize_speech(
            text=text,
            instructions=instructions,
            voice=voice,
            speed=speed,
            stage=stage,
            timeout_seconds_override=timeout_seconds_override,
            cancel_check=cancel_check,
        )
        return TTSAudioResult(
            audio_bytes=audio_bytes,
            content_type="audio/mpeg",
            file_extension="mp3",
            provider=self.provider_name,
            model=self.model_name,
        )
