from __future__ import annotations

"""Helper utilities extracted from `script_generator`.

These helpers are intentionally side-effect light and easy to test in
isolation, while keeping the main generator module focused on orchestration.
"""

import math
import os
from typing import Any, Dict, List

from .script_chunker import context_tail


def atomic_write_text(path: str, value: str) -> None:
    """Write text atomically, always ending file with a newline."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(value)
        if not value.endswith("\n"):
            f.write("\n")
    os.replace(tmp, path)


def phase_seconds_with_generation(phase_seconds: Dict[str, float]) -> Dict[str, float]:
    """Normalize phase timings and add aggregate `generation` duration."""
    out: Dict[str, float] = {}
    for key, value in phase_seconds.items():
        try:
            out[str(key)] = round(float(value), 3)
        except (TypeError, ValueError):
            out[str(key)] = 0.0
    generation_components = (
        out.get("pre_summary", 0.0),
        out.get("chunk_generation", 0.0),
        out.get("continuations", 0.0),
        out.get("truncation_recovery", 0.0),
        out.get("postprocess", 0.0),
    )
    out["generation"] = round(sum(generation_components), 3)
    return out


def env_bool(name: str, default: bool = False) -> bool:
    """Read boolean env vars using common truthy values."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    """Read finite float env vars with defensive fallback."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def env_int(name: str, default: int) -> int:
    """Read integer env vars with defensive fallback."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def sum_int_maps(*maps: Dict[str, int]) -> Dict[str, int]:
    """Merge integer maps by summing values per normalized key."""
    out: Dict[str, int] = {}
    for payload in maps:
        for key, value in dict(payload or {}).items():
            try:
                out[str(key)] = int(out.get(str(key), 0)) + int(value)
            except (TypeError, ValueError):
                continue
    return out


def default_completeness_report(*, reason: str = "") -> Dict[str, object]:
    """Return a baseline completeness report payload."""
    reasons: List[str] = []
    if reason:
        reasons.append(str(reason))
    return {
        "pass": True,
        "reasons": reasons,
        "truncation_indices": [],
        "block_sequence": [],
    }


def recent_dialogue(lines: List[Dict[str, str]], max_lines: int) -> str:
    """Format the latest dialogue turns for prompt context windows."""
    tail = context_tail(lines, max_lines)
    rows: List[str] = []
    for line in tail:
        rows.append(f"{line['speaker']} ({line['role']}): {line['text']}")
    return "\n".join(rows).strip()


def migrate_checkpoint_lines(raw_lines: Any) -> List[Dict[str, str]]:
    """Migrate legacy checkpoint line structures to canonical schema."""
    if not isinstance(raw_lines, list):
        return []
    migrated: List[Dict[str, str]] = []
    for idx, item in enumerate(raw_lines):
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker") or item.get("name") or "").strip()
        text = str(
            item.get("text")
            or item.get("line")
            or item.get("content")
            or item.get("dialogue")
            or ""
        ).strip()
        if not speaker or not text:
            continue
        role = str(item.get("role") or "").strip() or ("Host1" if idx % 2 == 0 else "Host2")
        instructions = str(item.get("instructions") or "").strip()
        pace_hint = str(item.get("pace_hint") or "").strip().lower()
        migrated.append(
            {
                "speaker": speaker,
                "role": role,
                "instructions": instructions,
                "text": text,
                **({"pace_hint": pace_hint} if pace_hint in {"calm", "steady", "brisk"} else {}),
            }
        )
    return migrated
