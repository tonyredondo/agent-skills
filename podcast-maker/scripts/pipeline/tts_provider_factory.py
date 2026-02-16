#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

from .alibaba_tts_client import AlibabaInstructTTSProvider
from .config import AudioConfig, ReliabilityConfig
from .logging_utils import Logger
from .openai_client import OpenAIClient
from .tts_provider import OpenAITTSProvider, TTSProvider


def create_tts_provider(
    *,
    audio_cfg: AudioConfig,
    reliability: ReliabilityConfig,
    logger: Logger,
    openai_client: Optional[OpenAIClient] = None,
) -> TTSProvider:
    provider = str(audio_cfg.tts_provider or "openai").strip().lower()
    if provider == "alibaba":
        return AlibabaInstructTTSProvider.from_audio_config(
            audio_cfg=audio_cfg,
            reliability=reliability,
            logger=logger,
        )
    if provider != "openai":
        raise RuntimeError("Unsupported TTS_PROVIDER value. Use openai or alibaba.")
    if openai_client is None:
        raise RuntimeError("OpenAI TTS provider requires an OpenAIClient instance.")
    return OpenAITTSProvider(client=openai_client)
