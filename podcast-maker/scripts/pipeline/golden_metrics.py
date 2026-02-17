#!/usr/bin/env python3
from __future__ import annotations

"""Golden-regression metric extraction/comparison helpers."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List

from .schema import count_words_from_lines, validate_script_payload


def _normalized_text(value: str) -> str:
    """Normalize text for metric token matching."""
    lowered = str(value or "").strip().lower()
    deaccented = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", deaccented)


RECAP_TOKENS = (
    "en resumen",
    "resumen",
    "resumiendo",
    "en sintesis",
    "en conclusion",
    "nos quedamos con",
    "en pocas palabras",
    "idea central",
    "in summary",
    "to sum up",
    "overall",
    "recap",
    "key takeaway",
    "em resumo",
    "en bref",
)

META_LANGUAGE_RE = re.compile(
    r"(?:\bseg[uú]n\s+el\s+[ií]ndice\b|\ben\s+este\s+resumen\b|\ben\s+el\s+siguiente\s+tramo\b|\bruta\s+del\s+episodio\b|\btabla\s+de\s+contenidos?\b)",
    re.IGNORECASE,
)

FAREWELL_TOKENS = (
    "gracias por escuch",
    "hasta la proxima",
    "nos vemos",
    "nos escuchamos",
    "adios",
    "thank you",
    "thanks for listening",
    "see you",
    "goodbye",
    "next episode",
    "until next",
    "merci",
    "au revoir",
    "danke",
    "obrigad",
)


@dataclass(frozen=True)
class ScriptMetrics:
    """Compact script metrics used by golden suite comparisons."""

    line_count: int
    word_count: int
    unique_speakers: int
    has_recap_signal: bool
    farewell_in_last_3: bool
    meta_language_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass metrics into plain dict."""
        return {
            "line_count": self.line_count,
            "word_count": self.word_count,
            "unique_speakers": self.unique_speakers,
            "has_recap_signal": self.has_recap_signal,
            "farewell_in_last_3": self.farewell_in_last_3,
            "meta_language_ok": self.meta_language_ok,
        }


def compute_script_metrics(payload: Dict[str, Any]) -> ScriptMetrics:
    """Compute deterministic script metrics from validated payload."""
    validated = validate_script_payload(payload)
    lines = validated["lines"]
    speakers = {line["speaker"] for line in lines if line.get("speaker")}
    has_recap = any(
        any(token in _normalized_text(line["text"]) for token in RECAP_TOKENS)
        for line in lines
    )
    meta_language_hits = sum(
        1
        for line in lines
        if META_LANGUAGE_RE.search(str(line.get("text", "")))
    )
    tail = lines[-3:]
    farewell_in_tail = any(
        any(
            token in _normalized_text(line["text"])
            for token in FAREWELL_TOKENS
        )
        for line in tail
    )
    return ScriptMetrics(
        line_count=len(lines),
        word_count=count_words_from_lines(lines),
        unique_speakers=len(speakers),
        has_recap_signal=has_recap,
        farewell_in_last_3=farewell_in_tail,
        meta_language_ok=meta_language_hits == 0,
    )


def compare_metric_ratio(current: int, baseline: int) -> float:
    """Return ratio between current and baseline values."""
    if baseline <= 0:
        return 1.0
    return float(current) / float(baseline)


def compare_against_baseline(
    current: ScriptMetrics,
    baseline: ScriptMetrics,
    *,
    min_word_ratio: float = 0.7,
    max_word_ratio: float = 1.5,
    min_line_ratio: float = 0.7,
    max_line_ratio: float = 1.6,
) -> Dict[str, Any]:
    """Compare current metrics against baseline ratio thresholds."""
    word_ratio = compare_metric_ratio(current.word_count, baseline.word_count)
    line_ratio = compare_metric_ratio(current.line_count, baseline.line_count)
    return {
        "word_ratio": word_ratio,
        "line_ratio": line_ratio,
        "word_ratio_ok": min_word_ratio <= word_ratio <= max_word_ratio,
        "line_ratio_ok": min_line_ratio <= line_ratio <= max_line_ratio,
        "recap_ok": current.has_recap_signal,
        "summary_ok": current.has_recap_signal,
        "farewell_ok": current.farewell_in_last_3,
        "meta_language_ok": current.meta_language_ok,
    }

