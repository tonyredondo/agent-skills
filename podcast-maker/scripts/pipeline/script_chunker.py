#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from typing import Dict, List


def _split_very_long_sentence(sentence: str, target_words: int) -> List[str]:
    words = sentence.split()
    if not words:
        return []
    if len(words) <= target_words:
        return [" ".join(words)]
    out: List[str] = []
    step = max(40, target_words)
    for i in range(0, len(words), step):
        out.append(" ".join(words[i : i + step]).strip())
    return [chunk for chunk in out if chunk]


def _split_paragraph_safely(paragraph: str, target_words_per_chunk: int) -> List[str]:
    words_total = len(paragraph.split())
    if words_total <= target_words_per_chunk:
        return [paragraph.strip()]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
    if not sentences:
        return _split_very_long_sentence(paragraph, target_words_per_chunk)

    out: List[str] = []
    current: List[str] = []
    current_words = 0
    for sentence in sentences:
        sw = len(sentence.split())
        if sw > target_words_per_chunk:
            # Flush previous context, then split long sentence by words.
            if current:
                out.append(" ".join(current).strip())
                current = []
                current_words = 0
            out.extend(_split_very_long_sentence(sentence, target_words_per_chunk))
            continue

        if current and current_words + sw > target_words_per_chunk:
            out.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sw

    if current:
        out.append(" ".join(current).strip())
    return [chunk for chunk in out if chunk]


def split_source_chunks(
    source: str,
    *,
    target_minutes: float,
    chunk_target_minutes: float,
    words_per_min: float,
) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", source) if p.strip()]
    if not paragraphs:
        return []

    desired_chunks = target_chunk_count(target_minutes=target_minutes, chunk_target_minutes=chunk_target_minutes)
    target_words_total = max(120, int(max(1.0, target_minutes) * max(80.0, words_per_min)))
    source_words_total = max(1, len(source.split()))
    target_by_duration = int(math.ceil(target_words_total / float(max(1, desired_chunks))))
    target_by_source = int(math.ceil(source_words_total / float(max(1, desired_chunks))))
    target_words_per_chunk = max(120, min(target_by_duration, target_by_source))
    min_chunk_words = max(60, min(180, int(round(target_words_total * 0.12))))
    tail_merge_threshold = max(40, int(round(min_chunk_words * 0.35)))

    normalized_units: List[str] = []
    for paragraph in paragraphs:
        normalized_units.extend(_split_paragraph_safely(paragraph, target_words_per_chunk))

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for unit in normalized_units:
        uw = len(unit.split())
        if current and current_words + uw > target_words_per_chunk:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_words = 0
        current.append(unit)
        current_words += uw
    if current:
        chunks.append("\n\n".join(current).strip())

    # Keep chunks reasonably dense: merge tiny chunks into neighbors.
    idx = 1
    while idx < len(chunks):
        words_here = len(chunks[idx].split())
        if words_here < min_chunk_words and len(chunks) > desired_chunks:
            chunks[idx - 1] = (chunks[idx - 1] + "\n\n" + chunks[idx]).strip()
            del chunks[idx]
            continue
        idx += 1

    # Merge tiny tail chunks that usually add latency without improving quality.
    while len(chunks) > max(1, desired_chunks) and len(chunks[-1].split()) < tail_merge_threshold:
        chunks[-2] = (chunks[-2] + "\n\n" + chunks[-1]).strip()
        chunks = chunks[:-1]

    # Soft cap: if chunk count is still much larger than planned, merge from tail.
    soft_cap = max(1, desired_chunks + 1)
    while len(chunks) > soft_cap and len(chunks) > 1:
        chunks[-2] = (chunks[-2] + "\n\n" + chunks[-1]).strip()
        chunks = chunks[:-1]
    return chunks


def target_chunk_count(target_minutes: float, chunk_target_minutes: float) -> int:
    return max(1, int(math.ceil(max(1.0, target_minutes) / max(0.8, chunk_target_minutes))))


def context_tail(lines: List[Dict[str, str]], max_lines: int) -> List[Dict[str, str]]:
    if max_lines <= 0:
        return []
    return lines[-max_lines:]

