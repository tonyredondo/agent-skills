#!/usr/bin/env python3
from __future__ import annotations

"""Schema normalization and salvage utilities for script payloads."""

import hashlib
import json
from typing import Any, Dict, List

DEFAULT_HOST1_INSTRUCTIONS = (
    "Speak in a warm, confident, conversational tone. "
    "Keep pacing measured and clear with brief pauses."
)
DEFAULT_HOST2_INSTRUCTIONS = (
    "Speak in a bright, friendly, conversational tone. "
    "Keep pacing measured and clear with brief pauses."
)
DEFAULT_INSTRUCTIONS = DEFAULT_HOST1_INSTRUCTIONS
MAX_INSTRUCTIONS_CHARS = 220
AMBIGUOUS_INSTRUCTION_PHRASES = (
    "speak naturally",
    "do your best",
    "good voice",
)
ACTIONABLE_INSTRUCTION_HINTS = (
    "tone",
    "pacing",
    "pace",
    "clarity",
    "clear",
    "pronounce",
    "pronunciation",
    "warm",
    "friendly",
    "confident",
    "analytical",
    "inviting",
    "energetic",
    "appreciative",
    "delivery",
    "cadence",
    "pause",
)


SCRIPT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "role": {"type": "string"},
                    "instructions": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "role", "instructions", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


def load_json_text(text: str) -> Dict[str, Any]:
    """Parse JSON text and require object root."""
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _default_instructions_for_role(role: str) -> str:
    """Return deterministic default instruction text for host role."""
    if str(role or "").strip() == "Host2":
        return DEFAULT_HOST2_INSTRUCTIONS
    return DEFAULT_HOST1_INSTRUCTIONS


def _normalize_instruction_whitespace(value: str) -> str:
    """Collapse whitespace into single spaces without regex-heavy parsing."""
    return " ".join(str(value or "").strip().split())


def _sentence_count(text: str) -> int:
    """Count coarse sentence boundaries using simple punctuation checks."""
    count = 0
    in_sentence = False
    for ch in str(text or ""):
        if ch.strip():
            in_sentence = True
        if ch in ".!?" and in_sentence:
            count += 1
            in_sentence = False
    if in_sentence:
        count += 1
    return count


def _has_actionable_instruction_detail(text: str) -> bool:
    """Heuristic check: instruction should include delivery/tone details."""
    lowered = str(text or "").lower()
    return any(token in lowered for token in ACTIONABLE_INSTRUCTION_HINTS)


def _looks_ambiguous_instruction(text: str) -> bool:
    """Detect vague template-like instructions lacking concrete guidance."""
    lowered = str(text or "").lower()
    if not lowered:
        return True
    for phrase in AMBIGUOUS_INSTRUCTION_PHRASES:
        if phrase in lowered and not _has_actionable_instruction_detail(lowered):
            return True
    return False


def _normalize_instructions_for_line(
    *,
    instructions: str,
    role: str,
) -> str:
    """Normalize instruction string and enforce hard-cut contract semantics."""
    cleaned = _normalize_instruction_whitespace(instructions)
    if not cleaned:
        return _default_instructions_for_role(role)

    if len(cleaned) > MAX_INSTRUCTIONS_CHARS:
        cleaned = cleaned[:MAX_INSTRUCTIONS_CHARS].rstrip()

    if _sentence_count(cleaned) > 2:
        return _default_instructions_for_role(role)

    if _looks_ambiguous_instruction(cleaned):
        return _default_instructions_for_role(role)

    if not _has_actionable_instruction_detail(cleaned):
        return _default_instructions_for_role(role)

    return cleaned


def normalize_line(
    raw: Dict[str, Any],
    idx: int,
) -> Dict[str, str]:
    """Normalize one dialogue line into canonical schema fields."""
    speaker = str(raw.get("speaker", "")).strip()
    role = str(raw.get("role", "")).strip()
    instructions = str(raw.get("instructions", "")).strip()
    text = str(raw.get("text", "")).strip()
    if not speaker:
        raise ValueError(f"line[{idx}] missing speaker")
    if not role:
        raise ValueError(f"line[{idx}] missing role")
    if role not in {"Host1", "Host2"}:
        role = "Host1" if idx % 2 == 0 else "Host2"
    if not text:
        raise ValueError(f"line[{idx}] missing text")
    instructions = _normalize_instructions_for_line(
        instructions=instructions,
        role=role,
    )
    return {
        "speaker": speaker,
        "role": role,
        "instructions": instructions,
        "text": text,
    }


def validate_script_payload(
    payload: Dict[str, Any],
) -> Dict[str, List[Dict[str, str]]]:
    """Validate/normalize payload and ensure non-empty `lines` list."""
    lines = payload.get("lines", [])
    if not isinstance(lines, list):
        raise ValueError("'lines' must be a list")
    out: List[Dict[str, str]] = []
    for idx, item in enumerate(lines):
        if not isinstance(item, dict):
            raise ValueError(f"line[{idx}] must be object")
        out.append(
            normalize_line(
                item,
                idx,
            )
        )
    if not out:
        raise ValueError("script has no lines")
    return {"lines": out}


def _coerce_text_value(value: Any) -> str:
    """Coerce heterogeneous value types into cleaned text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            text = _coerce_text_value(item)
            if text:
                parts.append(text)
        return " ".join(parts).strip()
    return str(value).strip()


def _extract_candidate_lines(payload: Dict[str, Any]) -> List[Any]:
    """Extract possible line arrays from common nested payload shapes."""
    candidates: List[Any] = []

    def _add_lines(value: Any) -> None:
        if isinstance(value, list):
            candidates.extend(list(value))

    _add_lines(payload.get("lines"))
    _add_lines(payload.get("dialogue"))
    _add_lines(payload.get("items"))
    _add_lines(payload.get("entries"))

    script_block = payload.get("script")
    if isinstance(script_block, dict):
        _add_lines(script_block.get("lines"))
        _add_lines(script_block.get("dialogue"))
        _add_lines(script_block.get("items"))
    elif isinstance(script_block, list):
        _add_lines(script_block)

    data_block = payload.get("data")
    if isinstance(data_block, dict):
        _add_lines(data_block.get("lines"))
        _add_lines(data_block.get("dialogue"))
    elif isinstance(data_block, list):
        _add_lines(data_block)

    result_block = payload.get("result")
    if isinstance(result_block, dict):
        _add_lines(result_block.get("lines"))
        _add_lines(result_block.get("dialogue"))
        _add_lines(result_block.get("items"))
    elif isinstance(result_block, list):
        _add_lines(result_block)

    output_block = payload.get("output")
    if isinstance(output_block, dict):
        _add_lines(output_block.get("lines"))
        _add_lines(output_block.get("dialogue"))
        _add_lines(output_block.get("items"))
    elif isinstance(output_block, list):
        _add_lines(output_block)

    return candidates


def salvage_script_payload(
    payload: Dict[str, Any],
) -> Dict[str, List[Dict[str, str]]]:
    """Best-effort salvage of malformed payload into canonical `lines`."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    raw_lines = _extract_candidate_lines(payload)
    if not raw_lines:
        raise ValueError("no candidate lines found for salvage")

    out: List[Dict[str, str]] = []
    seen_keys: set[str] = set()
    for idx, item in enumerate(raw_lines):
        if not isinstance(item, dict):
            continue
        speaker = _coerce_text_value(
            item.get("speaker")
            or item.get("name")
            or item.get("host")
            or item.get("speaker_name")
            or item.get("presenter")
        )
        role = _coerce_text_value(item.get("role") or item.get("slot") or item.get("label"))
        instructions = _coerce_text_value(
            item.get("instructions") or item.get("instruction") or item.get("style")
        )
        text = _coerce_text_value(
            item.get("text")
            or item.get("line")
            or item.get("content")
            or item.get("dialogue")
            or item.get("utterance")
            or item.get("message")
        )
        if not text:
            nested = item.get("line")
            if isinstance(nested, dict):
                text = _coerce_text_value(
                    nested.get("text")
                    or nested.get("content")
                    or nested.get("dialogue")
                )
                speaker = speaker or _coerce_text_value(
                    nested.get("speaker") or nested.get("name") or nested.get("host")
                )
                role = role or _coerce_text_value(nested.get("role") or nested.get("slot"))
                instructions = instructions or _coerce_text_value(
                    nested.get("instructions") or nested.get("instruction")
                )
        if not text:
            continue
        if not role or role not in {"Host1", "Host2"}:
            role = "Host1" if len(out) % 2 == 0 else "Host2"
        if not speaker:
            speaker = "Host One" if role == "Host1" else "Host Two"
        instructions = _normalize_instructions_for_line(
            instructions=instructions,
            role=role,
        )
        normalized = {
            "speaker": speaker,
            "role": role,
            "instructions": instructions,
            "text": text,
        }
        key = dedupe_key(normalized)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(normalized)
    if not out:
        raise ValueError("salvage could not recover any valid lines")
    return {"lines": out}


def count_words_from_lines(lines: List[Dict[str, str]]) -> int:
    """Count total words across normalized line texts."""
    total = 0
    for line in lines:
        total += len(line.get("text", "").split())
    return total


def count_words_from_payload(payload: Dict[str, Any]) -> int:
    """Validate payload then count words from normalized lines."""
    validated = validate_script_payload(payload)
    return count_words_from_lines(validated["lines"])


def canonical_json(payload: Dict[str, Any]) -> str:
    """Serialize payload to stable compact JSON representation."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def content_hash(text: str) -> str:
    """Compute SHA-256 hash for text input."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedupe_key(line: Dict[str, str]) -> str:
    """Compute content-based dedupe key for one line."""
    normalized = (
        f"{line.get('speaker','')}|"
        f"{line.get('role','')}|"
        f"{line.get('instructions','')}|"
        f"{line.get('text','')}"
    ).strip()
    return content_hash(normalized.lower())

