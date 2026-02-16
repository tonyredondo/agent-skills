#!/usr/bin/env python3
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

from .schema import dedupe_key


FAREWELL_PATTERNS = [
    "hasta la proxima",
    "hasta la próxima",
    "nos vemos",
    "nos escuchamos",
    "gracias por escuch",
    "adios",
    "adiós",
    "thanks for listening",
    "thank you",
    "see you",
    "goodbye",
    "next episode",
    "until next",
    "merci",
    "au revoir",
    "danke",
    "obrigad",
    "ate a proxima",
    "até a proxima",
    "cuidense",
    "desped",
]

SUMMARY_PATTERNS = [
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
]

SUMMARY_BY_LANG = {
    "es": "En Resumen: repasamos las ideas clave y los puntos mas utiles para aplicar desde hoy.",
    "en": "In summary: we recapped the key ideas and the most practical takeaways to apply today.",
    "pt": "Em resumo: revisamos as ideias-chave e os pontos mais praticos para aplicar hoje.",
    "fr": "En bref : nous avons resume les idees cles et les points les plus utiles a appliquer des aujourd'hui.",
}

FAREWELL_BY_LANG = {
    "es": "Gracias por escucharnos, nos vemos en el proximo episodio.",
    "en": "Thank you for listening, see you in the next episode.",
    "pt": "Obrigado por ouvir, ate a proxima edicao.",
    "fr": "Merci de nous avoir ecoutes, a tres bientot pour le prochain episode.",
}

BRIDGE_BY_LANG = {
    "es": (
        "Pasemos a la siguiente idea con un ejemplo concreto.",
        "Retomemos el punto central para aterrizarlo en decisiones prácticas.",
    ),
    "en": (
        "Let's continue with the next point.",
        "Back to the topic, there are more useful details to cover.",
    ),
    "pt": (
        "Vamos continuar com o proximo ponto.",
        "Voltando ao tema, ainda ha detalhes importantes para cobrir.",
    ),
    "fr": (
        "Passons au point suivant.",
        "Revenons au sujet, il reste des details importants a couvrir.",
    ),
}

LANG_SCORE_PATTERNS = {
    "pt": re.compile(
        r"\b(?:hoje|obrigad|em resumo|resumo|ate a proxima|edicao|com|vamos|continuar)\b",
        re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(?:merci|en bref|au revoir|avec|pour|passons|prochain)\b",
        re.IGNORECASE,
    ),
    "en": re.compile(
        r"\b(?:today|thank you|thanks|listening|in summary|with|for|and|we|next episode|continue)\b",
        re.IGNORECASE,
    ),
    "es": re.compile(
        r"\b(?:hoy|gracias|en resumen|resumen|episodio|sigamos|tema|con|para)\b",
        re.IGNORECASE,
    ),
}

TRAILING_CONNECTORS = (
    "y",
    "o",
    "pero",
    "porque",
    "aunque",
    "que",
    "and",
    "or",
    "but",
    "because",
    "although",
    "that",
    "et",
    "ou",
    "mais",
    "car",
    "e",
    "mas",
)
_CONNECTOR_PATTERN = "|".join(re.escape(token) for token in TRAILING_CONNECTORS)

BLOCK_MARKER_RE = re.compile(r"\b(Bloque|Block|Section|Part)\s+(\d+)\b", re.IGNORECASE)
ELLIPSIS_RE = re.compile(r"(?:\.\.\.|…)+\s*$")
TRAILING_CONNECTOR_RE = re.compile(rf"\b(?:{_CONNECTOR_PATTERN})(?:\.)?\s*$", re.IGNORECASE)
TAIL_TRUNCATION_RE = re.compile(
    rf"(?:"
    rf"(?:\.\.\.|…)+"
    rf"|[,;:\-—]"
    rf"|\b(?:{_CONNECTOR_PATTERN})(?:\.)?"
    rf")\s*$",
    re.IGNORECASE,
)
COMPLETE_SENTENCE_END_RE = re.compile(r"(?:[.!?…]|[.!?…][\"'”’)\]])\s*$")
SPANISH_OPENING_WORD_RE = re.compile(r"^\s*(?:[¡¿\"'(\[]\s*)*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)")
SPANISH_REPETITIVE_OPENER_ALTERNATIVES = {
    "y": ("Además,", "Por otro lado,", "A la vez,"),
    "ademas": ("Por otro lado,", "A la vez,", "También,"),
    "tambien": ("Además,", "En paralelo,", "Por otra parte,"),
}
SPANISH_OPENER_START_RE = {
    "y": re.compile(r"^\s*(?:[¡¿\"'(\[]\s*)*y\b[,\s]*", re.IGNORECASE),
    "ademas": re.compile(r"^\s*(?:[¡¿\"'(\[]\s*)*adem[aá]s\b[,\s]*", re.IGNORECASE),
    "tambien": re.compile(r"^\s*(?:[¡¿\"'(\[]\s*)*tambi[eé]n\b[,\s]*", re.IGNORECASE),
}
SPANISH_TECH_TERM_REPLACEMENTS = (
    (re.compile(r"\bdonor\s+extra\b", re.IGNORECASE), "donante adicional"),
)


def _normalized_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    deaccented = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", deaccented)


def _contains_non_latin_letters(value: str) -> bool:
    for ch in str(value or ""):
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if name and "LATIN" not in name:
            return True
    return False


def _is_farewell(text: str) -> bool:
    t = _normalized_text(text)
    return any(p in t for p in FAREWELL_PATTERNS)


def _infer_language_hint(lines: List[Dict[str, str]]) -> str:
    text = " ".join(_normalized_text(str(line.get("text") or "")) for line in lines[:12])
    raw_text = " ".join(str(line.get("text") or "") for line in lines[:12])
    best_lang = "es"
    best_score = 0
    for lang, pattern in LANG_SCORE_PATTERNS.items():
        score = len(pattern.findall(text))
        if score > best_score:
            best_score = score
            best_lang = lang
    if best_score > 0:
        return best_lang
    if _contains_non_latin_letters(raw_text):
        # Non-Latin scripts are safer with neutral English templates than
        # forcing Spanish summary/farewell text by default.
        return "en"
    return best_lang


def _resolve_language_hint(lines: List[Dict[str, str]], language_hint: str | None) -> str:
    hint = str(language_hint or "").strip().lower()
    if hint in {"es", "en", "pt", "fr"}:
        return hint
    return _infer_language_hint(lines)


def _line_opening_token(text: str) -> str:
    match = SPANISH_OPENING_WORD_RE.match(str(text or ""))
    if match is None:
        return ""
    return _normalized_text(match.group(1))


def normalize_spanish_technical_terms(
    lines: List[Dict[str, str]],
    *,
    language_hint: str | None = None,
) -> List[Dict[str, str]]:
    out = [dict(line) for line in lines]
    if _resolve_language_hint(out, language_hint) != "es":
        return out
    for line in out:
        text = str(line.get("text") or "")
        if not text:
            continue
        updated = text
        for pattern, replacement in SPANISH_TECH_TERM_REPLACEMENTS:
            updated = pattern.sub(replacement, updated)
        if updated != text:
            line["text"] = updated
    return out


def diversify_repetitive_openers(
    lines: List[Dict[str, str]],
    *,
    language_hint: str | None = None,
) -> List[Dict[str, str]]:
    out = [dict(line) for line in lines]
    if _resolve_language_hint(out, language_hint) != "es":
        return out
    previous_opener = ""
    repeat_streak = 0
    for idx, line in enumerate(out):
        text = str(line.get("text") or "").strip()
        if not text:
            previous_opener = ""
            repeat_streak = 0
            continue
        opener = _line_opening_token(text)
        if opener not in SPANISH_REPETITIVE_OPENER_ALTERNATIVES:
            previous_opener = opener
            repeat_streak = 1 if opener else 0
            continue
        if opener == previous_opener:
            repeat_streak += 1
            alternatives = SPANISH_REPETITIVE_OPENER_ALTERNATIVES[opener]
            replacement = alternatives[(idx + repeat_streak) % len(alternatives)]
            opener_pattern = SPANISH_OPENER_START_RE.get(opener)
            if opener_pattern is not None:
                replaced = opener_pattern.sub(f"{replacement} ", text, count=1)
                replaced = re.sub(r"\s+", " ", replaced).strip()
                if replaced:
                    line["text"] = replaced
        else:
            repeat_streak = 1
        previous_opener = _line_opening_token(str(line.get("text") or ""))
    return out


def _canonical_speaker_map(lines: List[Dict[str, str]]) -> Dict[str, str]:
    by_role: Dict[str, str] = {}
    fallback_names: List[str] = []
    for idx, line in enumerate(lines):
        role = str(line.get("role") or "").strip()
        if role not in {"Host1", "Host2"}:
            role = "Host1" if idx % 2 == 0 else "Host2"
        speaker = str(line.get("speaker") or "").strip()
        if speaker and role not in by_role:
            by_role[role] = speaker
            normalized = _normalized_text(speaker)
            if normalized and normalized not in fallback_names:
                fallback_names.append(normalized)

    host1 = by_role.get("Host1") or str(lines[0].get("speaker") or "").strip() if lines else "Host One"
    host2 = by_role.get("Host2", "")
    if not host2 and len(lines) > 1:
        host2 = str(lines[1].get("speaker") or "").strip()
    if not host1:
        host1 = "Host One"
    if not host2:
        host2 = "Host Two"
    if _normalized_text(host1) == _normalized_text(host2):
        host2 = "Host Two"
        if _normalized_text(host1) == _normalized_text(host2):
            host1 = "Host One"
            host2 = "Host Two"
    return {"Host1": host1, "Host2": host2}


def normalize_speaker_turns(
    lines: List[Dict[str, str]],
    *,
    max_consecutive_same_speaker: int = 2,
) -> List[Dict[str, str]]:
    if not lines:
        return []
    out = [dict(line) for line in lines]
    canonical = _canonical_speaker_map(out)
    max_run_allowed = max(1, int(max_consecutive_same_speaker))
    previous_speaker = ""
    run = 0
    for idx, line in enumerate(out):
        role = str(line.get("role") or "").strip()
        if role not in {"Host1", "Host2"}:
            role = "Host1" if idx % 2 == 0 else "Host2"
        speaker = canonical.get(role, "").strip() or str(line.get("speaker") or "").strip()
        if not speaker:
            speaker = "Host One" if role == "Host1" else "Host Two"
        normalized_speaker = _normalized_text(speaker)
        if normalized_speaker and normalized_speaker == previous_speaker:
            run += 1
        else:
            run = 1
        if run > max_run_allowed:
            role = "Host2" if role == "Host1" else "Host1"
            speaker = canonical.get(role, "").strip() or ("Host One" if role == "Host1" else "Host Two")
            normalized_speaker = _normalized_text(speaker)
            run = 1 if normalized_speaker != previous_speaker else max_run_allowed
        line["role"] = role
        line["speaker"] = speaker
        previous_speaker = normalized_speaker
    return out


def fix_mid_farewells(lines: List[Dict[str, str]], *, language_hint: str | None = None) -> List[Dict[str, str]]:
    out = [dict(line) for line in lines]
    resolved_lang = _resolve_language_hint(out, language_hint)
    bridge_templates = BRIDGE_BY_LANG.get(resolved_lang, BRIDGE_BY_LANG["es"])
    n = len(out)
    for idx, line in enumerate(out):
        text = line.get("text") or ""
        # Allow farewells only in the last two lines.
        if _is_farewell(text) and idx < max(0, n - 2):
            line["text"] = bridge_templates[0] if idx % 2 == 0 else bridge_templates[1]
    return out


def dedupe_append(base_lines: List[Dict[str, str]], new_lines: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    seen = {dedupe_key(line) for line in base_lines}
    out = list(base_lines)
    added = 0
    for line in new_lines:
        key = dedupe_key(line)
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        added += 1
    return out, added


def ensure_en_resumen(lines: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not lines:
        return lines
    has_summary = any(
        any(token in _normalized_text(line.get("text") or "") for token in SUMMARY_PATTERNS)
        for line in lines[-6:]
    )
    if has_summary:
        return [dict(line) for line in lines]
    has_summary_anywhere = any(
        any(token in _normalized_text(line.get("text") or "") for token in SUMMARY_PATTERNS)
        for line in lines
    )
    if has_summary_anywhere and len(lines) <= 6:
        return [dict(line) for line in lines]
    out = list(lines)
    resolved_lang = _resolve_language_hint(out, None)
    role = out[-1].get("role", "Host1")
    speaker = out[-1].get("speaker", "Host1")
    summary_line = {
        "speaker": speaker,
        "role": role,
        "instructions": out[-1].get("instructions", ""),
        "text": SUMMARY_BY_LANG.get(resolved_lang, SUMMARY_BY_LANG["es"]),
    }
    if _is_farewell(str(out[-1].get("text") or "")):
        out.insert(max(0, len(out) - 1), summary_line)
    else:
        out.append(summary_line)
    return out


def ensure_farewell_last(lines: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not lines:
        return lines
    out = list(lines)
    resolved_lang = _resolve_language_hint(out, None)
    has_farewell_near_end = any(_is_farewell(line.get("text") or "") for line in out[-3:])
    if has_farewell_near_end:
        out = fix_mid_farewells(out, language_hint=resolved_lang)
        tail_text = str(out[-1].get("text") or "").strip()
        if _is_farewell(tail_text) and _is_complete_sentence(tail_text):
            return out
        last = out[-1]
        out.append(
            {
                "speaker": last.get("speaker", "Host1"),
                "role": last.get("role", "Host1"),
                "instructions": last.get("instructions", ""),
                "text": FAREWELL_BY_LANG.get(resolved_lang, FAREWELL_BY_LANG["es"]),
            }
        )
        return out
    last = out[-1]
    out.append(
        {
            "speaker": last.get("speaker", "Host1"),
            "role": last.get("role", "Host1"),
            "instructions": last.get("instructions", ""),
            "text": FAREWELL_BY_LANG.get(resolved_lang, FAREWELL_BY_LANG["es"]),
        }
    )
    return fix_mid_farewells(out, language_hint=resolved_lang)


def normalize_block_numbering(lines: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = [dict(line) for line in lines]
    occurrences: List[Tuple[int, int]] = []
    for idx, line in enumerate(out):
        text = str(line.get("text") or "")
        match = BLOCK_MARKER_RE.search(text)
        if match is None:
            continue
        try:
            number = int(match.group(2))
        except (TypeError, ValueError):
            continue
        occurrences.append((idx, number))
    if len(occurrences) < 2:
        return out

    sequence: List[int] = []
    for _, number in occurrences:
        if not sequence or sequence[-1] != number:
            sequence.append(number)
    if len(sequence) < 2:
        return out

    # Renumber by transition order so we repair both monotonic gaps (1,2,4,6)
    # and non-monotonic drifts (1,3,2) in a deterministic way.
    normalized_by_occurrence_idx: Dict[int, int] = {}
    previous_number: int | None = None
    next_target = 0
    for idx, number in occurrences:
        if previous_number is None or number != previous_number:
            next_target += 1
            previous_number = number
        normalized_by_occurrence_idx[idx] = next_target
    if all(number == normalized_by_occurrence_idx.get(idx, number) for idx, number in occurrences):
        return out

    for idx, original in occurrences:
        target = normalized_by_occurrence_idx.get(idx, original)
        if target == original:
            continue
        text = str(out[idx].get("text") or "")
        match = BLOCK_MARKER_RE.search(text)
        if match is None:
            continue
        try:
            found = int(match.group(2))
        except (TypeError, ValueError):
            continue
        if found != original:
            continue
        start, end = match.span(2)
        out[idx]["text"] = f"{text[:start]}{target}{text[end:]}"
    return out


def sanitize_abrupt_tail(lines: List[Dict[str, str]], *, tail_window: int = 8) -> List[Dict[str, str]]:
    out = [dict(line) for line in lines]
    start = max(0, len(out) - max(1, int(tail_window)))
    for idx in range(start, len(out)):
        text = str(out[idx].get("text") or "").strip()
        if not text:
            continue
        updated = _sanitize_tail_text(text)
        updated = re.sub(r"\s+\.", ".", updated).strip()
        if updated != text:
            out[idx]["text"] = updated
    return out


def _sanitize_tail_text(text: str) -> str:
    updated = str(text or "")
    if ELLIPSIS_RE.search(updated):
        updated = ELLIPSIS_RE.sub(".", updated)
    if updated.endswith((",", ";", ":", "-", "—")):
        updated = updated.rstrip(" ,;:-—").strip() + "."
    updated = TRAILING_CONNECTOR_RE.sub(".", updated)
    return updated


def _is_complete_sentence(text: str) -> bool:
    sample = str(text or "").strip()
    if not sample:
        return False
    return COMPLETE_SENTENCE_END_RE.search(sample) is not None


def detect_truncation_indices(lines: List[Dict[str, str]]) -> List[int]:
    out: List[int] = []
    for idx, line in enumerate(lines):
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        if TAIL_TRUNCATION_RE.search(text):
            out.append(idx)
            continue
        if idx == (len(lines) - 1) and not _is_complete_sentence(text):
            out.append(idx)
    return out


def evaluate_script_completeness(lines: List[Dict[str, str]]) -> Dict[str, object]:
    truncation_indices = detect_truncation_indices(lines)
    reasons: List[str] = []
    if truncation_indices:
        reasons.append("script_contains_truncated_segments")

    sequence: List[int] = []
    for line in lines:
        text = str(line.get("text") or "")
        match = BLOCK_MARKER_RE.search(text)
        if match is None:
            continue
        try:
            number = int(match.group(2))
        except (TypeError, ValueError):
            continue
        if not sequence or sequence[-1] != number:
            sequence.append(number)
    if len(sequence) >= 2:
        expected = list(range(1, len(sequence) + 1))
        if sequence != expected:
            reasons.append("block_numbering_not_sequential")

    return {
        "pass": not reasons,
        "reasons": reasons,
        "truncation_indices": truncation_indices,
        "block_sequence": sequence,
    }


def repair_script_completeness(
    lines: List[Dict[str, str]],
    *,
    max_consecutive_same_speaker: int = 2,
) -> List[Dict[str, str]]:
    normalized_turns = normalize_speaker_turns(
        lines,
        max_consecutive_same_speaker=max_consecutive_same_speaker,
    )
    normalized = normalize_block_numbering(normalized_turns)
    normalized = normalize_spanish_technical_terms(normalized)
    normalized = diversify_repetitive_openers(normalized)
    return sanitize_abrupt_tail(normalized, tail_window=max(1, len(normalized)))


def harden_script_structure(
    lines: List[Dict[str, str]],
    *,
    max_consecutive_same_speaker: int = 2,
) -> List[Dict[str, str]]:
    normalized_turns = normalize_speaker_turns(
        lines,
        max_consecutive_same_speaker=max_consecutive_same_speaker,
    )
    normalized = normalize_block_numbering(normalized_turns)
    normalized = normalize_spanish_technical_terms(normalized)
    normalized = diversify_repetitive_openers(normalized)
    return sanitize_abrupt_tail(normalized, tail_window=8)

