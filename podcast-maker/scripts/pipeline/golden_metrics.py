#!/usr/bin/env python3
from __future__ import annotations

"""Golden-regression helpers aligned with the redesigned oral pipeline."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict

from .schema import count_words_from_lines, validate_script_payload

SCAFFOLD_PHRASES = (
    "por otro lado",
    "ahora bien",
    "dicho esto",
    "a partir de ahi",
    "pasando a",
    "en paralelo",
    "si lo conectamos",
    "en ese sentido",
    "por cierto",
)
META_LANGUAGE_RE = re.compile(
    r"(?:\bseg[uú]n\s+el\s+[ií]ndice\b|\ben\s+este\s+resumen\b|\ben\s+el\s+siguiente\s+tramo\b|\bruta\s+del\s+episodio\b|\btabla\s+de\s+contenidos?\b)",
    re.IGNORECASE,
)
HOST2_PUSH_RE = re.compile(
    r"(?:\?|por ejemplo|vale,? bajemos|en la practica|que coste|que riesgo|entonces|aterriza|concreto|tradeoff)",
    re.IGNORECASE,
)


def _normalized_text(value: str) -> str:
    lowered = str(value or '').strip().lower()
    deaccented = ''.join(
        ch for ch in unicodedata.normalize('NFKD', lowered) if not unicodedata.combining(ch)
    )
    return re.sub(r'\s+', ' ', deaccented)


@dataclass(frozen=True)
class ScriptMetrics:
    """Compact script metrics for redesigned golden checks."""

    line_count: int
    word_count: int
    unique_speakers: int
    alternating_ratio: float
    host2_turn_ratio: float
    host2_push_ratio: float
    scaffold_phrase_hits: int
    meta_language_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            'line_count': self.line_count,
            'word_count': self.word_count,
            'unique_speakers': self.unique_speakers,
            'alternating_ratio': self.alternating_ratio,
            'host2_turn_ratio': self.host2_turn_ratio,
            'host2_push_ratio': self.host2_push_ratio,
            'scaffold_phrase_hits': self.scaffold_phrase_hits,
            'meta_language_ok': self.meta_language_ok,
        }


def compute_script_metrics(payload: Dict[str, Any]) -> ScriptMetrics:
    validated = validate_script_payload(payload)
    lines = validated['lines']
    speakers = {line['speaker'] for line in lines if line.get('speaker')}
    roles = [str(line.get('role', '')).strip() for line in lines]
    alternations = 0
    transitions = 0
    for idx in range(1, len(roles)):
        if roles[idx] and roles[idx - 1]:
            transitions += 1
            if roles[idx] != roles[idx - 1]:
                alternations += 1
    alternating_ratio = round(float(alternations) / float(max(1, transitions)), 4)
    host2_lines = [line for line in lines if str(line.get('role', '')).strip() == 'Host2']
    host2_turn_ratio = round(float(len(host2_lines)) / float(max(1, len(lines))), 4)
    host2_push_hits = 0
    scaffold_phrase_hits = 0
    meta_language_hits = 0
    for line in lines:
        text = _normalized_text(str(line.get('text', '')))
        if META_LANGUAGE_RE.search(text):
            meta_language_hits += 1
        for phrase in SCAFFOLD_PHRASES:
            if phrase in text:
                scaffold_phrase_hits += 1
        if str(line.get('role', '')).strip() == 'Host2' and HOST2_PUSH_RE.search(text):
            host2_push_hits += 1
    host2_push_ratio = round(float(host2_push_hits) / float(max(1, len(host2_lines))), 4)
    return ScriptMetrics(
        line_count=len(lines),
        word_count=count_words_from_lines(lines),
        unique_speakers=len(speakers),
        alternating_ratio=alternating_ratio,
        host2_turn_ratio=host2_turn_ratio,
        host2_push_ratio=host2_push_ratio,
        scaffold_phrase_hits=scaffold_phrase_hits,
        meta_language_ok=meta_language_hits == 0,
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
        'word_ratio': word_ratio,
        'line_ratio': line_ratio,
        'word_ratio_ok': min_word_ratio <= word_ratio <= max_word_ratio,
        'line_ratio_ok': min_line_ratio <= line_ratio <= max_line_ratio,
        'speaker_count_ok': current.unique_speakers >= min(2, baseline.unique_speakers),
        'alternation_ok': current.alternating_ratio >= max(0.75, baseline.alternating_ratio - 0.15),
        'host2_presence_ok': current.host2_turn_ratio >= max(0.3, baseline.host2_turn_ratio - 0.1),
            'host2_push_ok': current.host2_push_ratio >= max(0.2, baseline.host2_push_ratio - 0.2),
        'scaffold_control_ok': current.scaffold_phrase_hits <= max(1, baseline.scaffold_phrase_hits + 1),
        'meta_language_ok': current.meta_language_ok,
    }
