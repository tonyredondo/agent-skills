#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


PHASE_INTRO = "intro"
PHASE_BODY = "body"
PHASE_CLOSING = "closing"
VALID_PHASES = {PHASE_INTRO, PHASE_BODY, PHASE_CLOSING}


@dataclass
class CanonicalExpressiveness:
    voice_affect: str = ""
    tone: str = ""
    pacing: str = ""
    emotion: str = ""
    pronunciation: str = ""
    pauses: str = ""
    energy: str = ""
    extras: List[str] = field(default_factory=list)


_DEFAULTS_BY_ROLE: Dict[str, CanonicalExpressiveness] = {
    "Host1": CanonicalExpressiveness(
        voice_affect="Warm and confident",
        tone="Conversational",
        pacing="Brisk",
        emotion="Curiosity",
        pronunciation="Clear",
        pauses="Brief",
        energy="High",
    ),
    "Host2": CanonicalExpressiveness(
        voice_affect="Bright and friendly",
        tone="Conversational",
        pacing="Measured",
        emotion="Enthusiasm",
        pronunciation="Clear",
        pauses="Brief",
        energy="High",
    ),
}

_PHASE_OVERRIDES: Dict[str, Dict[str, str]] = {
    PHASE_INTRO: {
        "tone": "Conversational and inviting",
        "pacing": "Brisk but clear",
        "emotion": "Enthusiasm",
        "pauses": "Brief",
        "energy": "High",
    },
    PHASE_BODY: {
        "tone": "Conversational and analytical",
        "pacing": "Measured",
        "emotion": "Focus",
        "pauses": "Balanced",
        "energy": "Medium",
    },
    PHASE_CLOSING: {
        "tone": "Warm and appreciative",
        "pacing": "Calm",
        "emotion": "Gratitude",
        "pauses": "Slightly longer",
        "energy": "Medium",
    },
}

_KEY_MAP = {
    "voiceaffect": "voice_affect",
    "voicea_ect": "voice_affect",
    "afectodevoz": "voice_affect",
    "afectovoz": "voice_affect",
    "tone": "tone",
    "tono": "tone",
    "pacing": "pacing",
    "ritmo": "pacing",
    "velocidad": "pacing",
    "emotion": "emotion",
    "emocion": "emotion",
    "pronunciation": "pronunciation",
    "pronunciacion": "pronunciation",
    "articulation": "pronunciation",
    "articulacion": "pronunciation",
    "pauses": "pauses",
    "pausas": "pauses",
    "energy": "energy",
    "energia": "energy",
}

_VALUE_WORD_MAP = {
    "entusiasmo": "enthusiasm",
    "entusiasta": "enthusiastic",
    "conversacional": "conversational",
    "rapido": "brisk",
    "rapida": "brisk",
    "pausado": "measured",
    "pausada": "measured",
    "medido": "measured",
    "medida": "measured",
    "calma": "calm",
    "curiosidad": "curiosity",
    "gratitud": "gratitude",
    "foco": "focus",
    "clara": "clear",
    "claro": "clear",
    "largas": "longer",
    "breves": "brief",
    "equilibradas": "balanced",
    "alta": "high",
    "media": "medium",
}


def normalize_phase(phase: str) -> str:
    normalized = str(phase or "").strip().lower()
    if normalized in VALID_PHASES:
        return normalized
    return PHASE_BODY


def _normalize_key(raw_key: str) -> str:
    key = re.sub(r"[^a-z]", "", str(raw_key or "").strip().lower())
    return _KEY_MAP.get(key, "")


def _default_style(role: str) -> CanonicalExpressiveness:
    base = _DEFAULTS_BY_ROLE.get(str(role or "").strip(), _DEFAULTS_BY_ROLE["Host1"])
    return CanonicalExpressiveness(
        voice_affect=base.voice_affect,
        tone=base.tone,
        pacing=base.pacing,
        emotion=base.emotion,
        pronunciation=base.pronunciation,
        pauses=base.pauses,
        energy=base.energy,
        extras=[],
    )


def _apply_phase_overrides(style: CanonicalExpressiveness, phase: str) -> None:
    for key, value in _PHASE_OVERRIDES.get(normalize_phase(phase), {}).items():
        setattr(style, key, value)


def _apply_raw_instructions(style: CanonicalExpressiveness, raw_instructions: str) -> None:
    for raw_part in str(raw_instructions or "").split("|"):
        part = raw_part.strip()
        if not part:
            continue
        if part.lower() in {"x", "-", "none", "n/a", "na"}:
            continue
        if ":" not in part:
            style.extras.append(part)
            continue
        key_raw, value_raw = part.split(":", 1)
        key = _normalize_key(key_raw)
        value = str(value_raw or "").strip()
        if not key or not value:
            style.extras.append(part)
            continue
        setattr(style, key, value)


def _to_english_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    words = re.split(r"(\W+)", text)
    translated: List[str] = []
    for token in words:
        lowered = str(token).strip().lower()
        if lowered in _VALUE_WORD_MAP:
            translated.append(_VALUE_WORD_MAP[lowered])
        else:
            translated.append(token)
    return "".join(translated).strip()


def _render_openai(style: CanonicalExpressiveness) -> str:
    parts = [
        f"Voice Affect: {style.voice_affect}",
        f"Tone: {style.tone}",
        f"Pacing: {style.pacing}",
        f"Emotion: {style.emotion}",
        f"Pronunciation: {style.pronunciation}",
        f"Pauses: {style.pauses}",
    ]
    if style.energy:
        parts.append(f"Energy: {style.energy}")
    parts.extend(item for item in style.extras if str(item).strip())
    return " | ".join(parts).strip()


def _render_alibaba(style: CanonicalExpressiveness) -> str:
    extras_en = [_to_english_text(item) for item in style.extras if str(item).strip()]
    prompt = (
        "Speak with Spanish (Spain) accent, high enthusiasm, clear articulation, and natural prosody. "
        f"Tone: {_to_english_text(style.tone)}. "
        f"Pacing: {_to_english_text(style.pacing)}. "
        f"Emotion: {_to_english_text(style.emotion)}. "
        f"Pronunciation: {_to_english_text(style.pronunciation)}. "
        f"Pauses: {_to_english_text(style.pauses)}. "
        f"Energy: {_to_english_text(style.energy)}."
    )
    if extras_en:
        prompt += " Additional style constraints: " + " ; ".join(extras_en) + "."
    return prompt.strip()


def build_provider_instructions(
    *,
    provider: str,
    role: str,
    phase: str,
    raw_instructions: str,
) -> str:
    style = _default_style(role)
    _apply_phase_overrides(style, phase)
    _apply_raw_instructions(style, raw_instructions)
    if str(provider or "").strip().lower() == "alibaba":
        return _render_alibaba(style)
    return _render_openai(style)
