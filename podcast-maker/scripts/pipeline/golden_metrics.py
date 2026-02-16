#!/usr/bin/env python3
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List

from .schema import count_words_from_lines, validate_script_payload


def _normalized_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    deaccented = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", deaccented)


SUMMARY_TOKENS = (
    "en resumen",
    "resumen",
    "resumiendo",
    "en sintesis",
    "en conclusion",
    "in summary",
    "to sum up",
    "overall",
    "recap",
    "key takeaway",
    "em resumo",
    "en bref",
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
    line_count: int
    word_count: int
    unique_speakers: int
    has_en_resumen: bool
    farewell_in_last_3: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_count": self.line_count,
            "word_count": self.word_count,
            "unique_speakers": self.unique_speakers,
            "has_en_resumen": self.has_en_resumen,
            "farewell_in_last_3": self.farewell_in_last_3,
        }


def compute_script_metrics(payload: Dict[str, Any]) -> ScriptMetrics:
    validated = validate_script_payload(payload)
    lines = validated["lines"]
    speakers = {line["speaker"] for line in lines if line.get("speaker")}
    has_summary = any(
        any(token in _normalized_text(line["text"]) for token in SUMMARY_TOKENS)
        for line in lines
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
        has_en_resumen=has_summary,
        farewell_in_last_3=farewell_in_tail,
    )


def compare_metric_ratio(current: int, baseline: int) -> float:
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
    word_ratio = compare_metric_ratio(current.word_count, baseline.word_count)
    line_ratio = compare_metric_ratio(current.line_count, baseline.line_count)
    return {
        "word_ratio": word_ratio,
        "line_ratio": line_ratio,
        "word_ratio_ok": min_word_ratio <= word_ratio <= max_word_ratio,
        "line_ratio_ok": min_line_ratio <= line_ratio <= max_line_ratio,
        "summary_ok": current.has_en_resumen,
        "farewell_ok": current.farewell_in_last_3,
    }

