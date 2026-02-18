#!/usr/bin/env python3
from __future__ import annotations

"""Deterministic post-processing for generated podcast scripts."""

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

RECAP_PATTERNS = [
    "en resumen",
    "resumen",
    "resumiendo",
    "en sintesis",
    "en conclusion",
    "nos quedamos con",
    "idea central",
    "en pocas palabras",
    "in summary",
    "to sum up",
    "overall",
    "recap",
    "key takeaway",
    "em resumo",
    "en bref",
]

RECAP_STRONG_PATTERNS = [
    "en resumen",
    "en sintesis",
    "en conclusion",
    "nos quedamos con",
    "en pocas palabras",
    "in summary",
    "to sum up",
    "em resumo",
    "en bref",
]

RECAP_BY_LANG = {
    "es": "Nos quedamos con dos ideas accionables: medir calidad con criterios claros y cerrar cada iteracion con ajustes concretos.",
    "en": "We close with two practical ideas: measure quality with clear criteria and end each iteration with concrete adjustments.",
    "pt": "Fechamos com duas ideias praticas: medir qualidade com criterios claros e encerrar cada iteracao com ajustes concretos.",
    "fr": "Nous retenons deux idees pratiques : mesurer la qualite avec des criteres clairs et conclure chaque iteration avec des ajustements concrets.",
}

FAREWELL_BY_LANG = {
    "es": "Gracias por escucharnos, nos vemos en el proximo episodio.",
    "en": "Thank you for listening, see you in the next episode.",
    "pt": "Obrigado por ouvir, ate a proxima edicao.",
    "fr": "Merci de nous avoir ecoutes, a tres bientot pour le prochain episode.",
}

TAIL_QUESTION_ANSWER_BY_LANG = {
    "es": "Buena pregunta. Si, y la clave es cerrarlo con evidencia concreta y una decision accionable.",
    "en": "Good question. Yes, and the key is to close it with concrete evidence and one actionable decision.",
    "pt": "Boa pergunta. Sim, e a chave e fechar isso com evidencia concreta e uma decisao acionavel.",
    "fr": "Bonne question. Oui, et l'essentiel est de conclure avec des preuves concretes et une decision actionnable.",
}

BRIDGE_BY_LANG = {
    "es": (
        "Sigamos con la siguiente idea conectandola con lo anterior.",
        "Antes de cerrar, aterricemos esto en una decision practica.",
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

TRANSITION_CONNECTOR_RE = re.compile(
    r"^(?:y\s+de\s+hecho|por\s+otro\s+lado|ahora\s+bien|dicho\s+esto|a\s+partir\s+de\s+ahi|pasando\s+a|en\s+paralelo|si\s+lo\s+conectamos|en\s+ese\s+sentido|por\s+cierto)\b",
    re.IGNORECASE,
)
SUMMARY_LABEL_RE = re.compile(
    r"^(?:summary|recap|takeaways?|conclusion|resumen|sintesis|síntesis|cierre)\s*[:\-]\s*",
    re.IGNORECASE,
)
QUESTION_PUNCT_RE = re.compile(r"[¿?]")
TRANSITION_WORD_RE = re.compile(r"[^\W_]{3,}", re.UNICODE)
CLAUSE_SPLIT_RE = re.compile(r"[.!?;:]+")
SPANISH_LEADING_FILLER_RE = re.compile(
    r"^\s*(?:(?:y|adem[aá]s|tambi[eé]n|entonces|pues|bueno|vale|o\s+sea|de\s+hecho)\b[\s,:;\-]*)+",
    re.IGNORECASE,
)
TRANSITION_PREFIX_TEXT_RE = re.compile(
    r"^\s*(?:por\s+otro\s+lado|ahora\s+bien|dicho\s+esto|en\s+ese\s+sentido|en\s+paralelo|pasando\s+a\s+otro\s+frente|a\s+partir\s+de\s+ahi)\s*,?\s*",
    re.IGNORECASE,
)
NATURAL_RESPONSE_OPENER_RE = re.compile(
    r"^(?:exacto|tal\s+cual|claro|de\s+acuerdo|totalmente|justo|vale|bien|ojo|ademas|tambien|de\s+hecho|correcto|cierto)\b",
    re.IGNORECASE,
)
RECAP_GENERIC_FRAGMENT_RE = re.compile(
    r"^(?:con\s+eso\s+nos\s+quedamos|me\s+quedo\s+con|antes\s+de\s+cerrar|en\s+ese\s+sentido|por\s+otro\s+lado|ahora\s+bien|dicho\s+esto|tal\s+cual|exacto|claro|vale)\b",
    re.IGNORECASE,
)
RECAP_WEAK_TAIL_TOKEN_RE = re.compile(
    r"\b(?:de|del|la|el|los|las|en|con|por|para|que|y|o|a|al|un|una)\b\s*$",
    re.IGNORECASE,
)
RECAP_WEAK_START_TOKEN_RE = re.compile(
    r"^(?:si|porque|que|cuando|aunque|como|pero|y)\b",
    re.IGNORECASE,
)
RECAP_HEDGE_FRAGMENT_RE = re.compile(
    r"\bme\s+(?:parece|gusta|interesa|suena|quedo|quedaria)\b",
    re.IGNORECASE,
)
RECAP_FIRST_PERSON_RE = re.compile(
    r"\b(?:yo|me|mi|mio|mia|nosotros|nosotras|nos|i|my|we|our)\b",
    re.IGNORECASE,
)
RECAP_FRAGMENT_STOPWORDS = {
    "hoy",
    "nos",
    "ideas",
    "idea",
    "esto",
    "eso",
    "tema",
    "cosa",
    "cosas",
    "parte",
    "bueno",
    "mejor",
    "aqui",
    "ahi",
    "alli",
    "luego",
    "tambien",
    "ademas",
    "porque",
    "cuando",
    "como",
    "para",
    "desde",
    "sobre",
    "entre",
    "hacia",
    "with",
    "from",
    "that",
    "this",
    "about",
}
TRANSITION_PREFIX_BY_LANG = {
    "es": (
        "Por otro lado,",
        "Ahora bien,",
        "Dicho esto,",
        "En ese sentido,",
        "En paralelo,",
        "Pasando a otro frente,",
    ),
    "en": (
        "On the other hand,",
        "That said,",
        "In that sense,",
        "In parallel,",
    ),
    "pt": (
        "Por outro lado,",
        "Dito isso,",
        "Nesse sentido,",
        "Em paralelo,",
    ),
    "fr": (
        "D'un autre cote,",
        "Cela dit,",
        "Dans ce sens,",
        "En parallele,",
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

META_PODCAST_LANGUAGE_REPLACEMENTS = (
    (re.compile(r"\bseg[uú]n\s+el\s+[ií]ndice\b", re.IGNORECASE), "si miramos el panorama general"),
    (re.compile(r"\ben\s+el\s+[ií]ndice\b", re.IGNORECASE), "en el panorama"),
    (re.compile(r"\b[ií]ndice\s+multidisciplinar\b", re.IGNORECASE), "panorama general"),
    (re.compile(r"\ben\s+este\s+resumen\b", re.IGNORECASE), "en este repaso"),
    (re.compile(r"\ben\s+el\s+siguiente\s+tramo\b", re.IGNORECASE), "a continuacion"),
    (re.compile(r"\bruta\s+breve\b", re.IGNORECASE), "recorrido breve"),
)

DECLARED_TEASE_REPLACEMENTS = (
    (
        re.compile(
            r"\bte\s+voy\s+a\s+(?:chinchar|pinchar|provocar|picar)\b",
            re.IGNORECASE,
        ),
        "te planteo una objecion directa",
    ),
    (
        re.compile(
            r"\bte\s+pincho\s+un\s+poco\b",
            re.IGNORECASE,
        ),
        "te planteo una objecion breve",
    ),
)


def _normalized_text(value: str) -> str:
    """Normalize text for robust token/pattern matching."""
    lowered = str(value or "").strip().lower()
    deaccented = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", deaccented)


def _contains_non_latin_letters(value: str) -> bool:
    """Return True when text includes non-Latin alphabet letters."""
    for ch in str(value or ""):
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if name and "LATIN" not in name:
            return True
    return False


def _is_farewell(text: str) -> bool:
    """Detect whether a line is a farewell/closing utterance."""
    t = _normalized_text(text)
    return any(p in t for p in FAREWELL_PATTERNS)


def _has_strong_recap_signal(text: str) -> bool:
    """Return True when text contains a clear recap phrase."""
    t = _normalized_text(text)
    return any(token in t for token in RECAP_STRONG_PATTERNS)


def _infer_language_hint(lines: List[Dict[str, str]]) -> str:
    """Infer coarse language hint from early dialogue lines."""
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
    """Resolve explicit language hint or infer from content."""
    hint = str(language_hint or "").strip().lower()
    if hint in {"es", "en", "pt", "fr"}:
        return hint
    return _infer_language_hint(lines)


def _line_opening_token(text: str) -> str:
    """Extract normalized first spoken token from a line."""
    match = SPANISH_OPENING_WORD_RE.match(str(text or ""))
    if match is None:
        return ""
    return _normalized_text(match.group(1))


def _strip_leading_discourse_fillers(text: str, *, language_hint: str) -> str:
    """Drop repeated spoken fillers at line start for cleaner transitions."""
    sample = str(text or "").strip()
    if not sample:
        return ""
    if language_hint != "es":
        return sample
    updated = sample
    for _ in range(3):
        trimmed = SPANISH_LEADING_FILLER_RE.sub("", updated, count=1).strip()
        if not trimmed or trimmed == updated:
            break
        updated = trimmed
    return updated


def _strip_transition_prefix_text(text: str) -> str:
    """Drop canned transition prefixes from a fragment start."""
    sample = str(text or "").strip()
    if not sample:
        return ""
    updated = sample
    for _ in range(2):
        trimmed = TRANSITION_PREFIX_TEXT_RE.sub("", updated, count=1).strip()
        if not trimmed or trimmed == updated:
            break
        updated = trimmed
    return updated


def _truncate_words(text: str, *, max_words: int) -> str:
    """Cap fragment length by words while preserving readability."""
    words = re.findall(r"\S+", str(text or "").strip(), re.UNICODE)
    if not words:
        return ""
    clipped = words[: max(1, int(max_words))]
    return " ".join(clipped).strip(" ,;:-")


def _recap_content_token_count(text: str) -> int:
    """Count distinct content tokens to avoid weak recap fragments."""
    normalized = _normalized_text(text)
    tokens = [
        token
        for token in TRANSITION_WORD_RE.findall(normalized)
        if token not in RECAP_FRAGMENT_STOPWORDS
    ]
    return len(set(tokens))


def _extract_recap_fragments(
    lines: List[Dict[str, str]],
    *,
    language_hint: str,
    max_fragments: int = 2,
    max_words_per_fragment: int = 12,
) -> List[str]:
    """Collect short, recent topical fragments for contextual recap text."""
    fragments: List[str] = []
    seen: set[str] = set()
    for line in reversed(lines[-14:]):
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        if _is_farewell(text) or _has_strong_recap_signal(text):
            continue
        clauses = [chunk.strip(" \"'()[]") for chunk in CLAUSE_SPLIT_RE.split(text) if chunk.strip()]
        if not clauses:
            continue
        clause = _strip_transition_prefix_text(clauses[0])
        clause = _strip_leading_discourse_fillers(clause, language_hint=language_hint)
        clause = re.sub(r"\s+", " ", clause).strip(" ,;:.!?-")
        if len(clause.split()) < 5:
            continue
        if RECAP_GENERIC_FRAGMENT_RE.search(clause):
            continue
        if RECAP_WEAK_START_TOKEN_RE.search(clause):
            continue
        if RECAP_HEDGE_FRAGMENT_RE.search(clause):
            continue
        if RECAP_FIRST_PERSON_RE.search(_normalized_text(clause)):
            continue
        if RECAP_WEAK_TAIL_TOKEN_RE.search(_normalized_text(clause)):
            continue
        clipped = _truncate_words(clause, max_words=max_words_per_fragment)
        if RECAP_WEAK_TAIL_TOKEN_RE.search(_normalized_text(clipped)):
            continue
        if _recap_content_token_count(clipped) < 4:
            continue
        key = _normalized_text(clipped)
        if not key or key in seen:
            continue
        seen.add(key)
        fragments.append(clipped)
        if len(fragments) >= max(1, int(max_fragments)):
            break
    return fragments


def _build_contextual_recap_text(lines: List[Dict[str, str]], *, language_hint: str) -> str:
    """Build recap line from recent content, with language fallback templates."""
    fallback = RECAP_BY_LANG.get(language_hint, RECAP_BY_LANG["es"])
    fragments = _extract_recap_fragments(lines, language_hint=language_hint)
    if len(fragments) < 2:
        return fallback
    first = fragments[0].rstrip(" .")
    second = fragments[1].rstrip(" .")
    if language_hint == "es":
        return f"Nos quedamos con dos ideas accionables: {first}. Tambien, {second}."
    if language_hint == "en":
        return f"We close with two practical takeaways: {first}. Also, {second}."
    if language_hint == "pt":
        return f"Fechamos com duas ideias praticas: {first}. Alem disso, {second}."
    if language_hint == "fr":
        return f"Nous retenons deux idees pratiques : {first}. En plus, {second}."
    return fallback


def _compose_transition_text(prefix: str, text: str, *, language_hint: str) -> str:
    """Compose transition + utterance avoiding duplicated discourse markers."""
    body = _strip_leading_discourse_fillers(text, language_hint=language_hint)
    if not body:
        body = str(text or "").strip()
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) >= 2 and body[0].isupper() and body[1].islower():
        body = body[0].lower() + body[1:]
    return f"{prefix} {body}".strip()


def normalize_spanish_technical_terms(
    lines: List[Dict[str, str]],
    *,
    language_hint: str | None = None,
) -> List[Dict[str, str]]:
    """Normalize known Spanish technical term preferences."""
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
    """Reduce repetitive Spanish line-openers for better cadence."""
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
    """Build canonical speaker names for Host1/Host2 roles."""
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
    """Normalize role/speaker fields and enforce alternation limits."""
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
    """Replace premature farewells with neutral bridge transitions."""
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


def sanitize_meta_podcast_language(
    lines: List[Dict[str, str]],
    *,
    language_hint: str | None = None,
) -> List[Dict[str, str]]:
    """Rewrite document-like narration into natural spoken phrasing."""
    out = [dict(line) for line in lines]
    if _resolve_language_hint(out, language_hint) != "es":
        return out
    for line in out:
        text = str(line.get("text") or "")
        if not text:
            continue
        updated = text
        for pattern, replacement in META_PODCAST_LANGUAGE_REPLACEMENTS:
            updated = pattern.sub(replacement, updated)
        updated = re.sub(r"\s+", " ", updated).strip()
        if updated:
            line["text"] = updated
    return out


def sanitize_declared_tease_intent(
    lines: List[Dict[str, str]],
    *,
    language_hint: str | None = None,
) -> List[Dict[str, str]]:
    """Remove explicit 'I will tease/challenge' declarations while keeping contrast."""
    out = [dict(line) for line in lines]
    if _resolve_language_hint(out, language_hint) != "es":
        return out
    for line in out:
        text = str(line.get("text") or "")
        if not text:
            continue
        updated = text
        for pattern, replacement in DECLARED_TEASE_REPLACEMENTS:
            updated = pattern.sub(replacement, updated)
        updated = re.sub(r"\s+", " ", updated).strip()
        if updated:
            line["text"] = updated
    return out


def dedupe_append(base_lines: List[Dict[str, str]], new_lines: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    """Append only non-duplicate lines and return (merged, added_count)."""
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


def _is_question_like(text: str) -> bool:
    """Detect whether text has direct question punctuation."""
    raw = str(text or "").strip()
    if not raw:
        return False
    return QUESTION_PUNCT_RE.search(raw) is not None


def _summary_line_index(lines: List[Dict[str, str]]) -> int:
    """Return first summary/recap index or -1 if not found."""
    for idx, line in enumerate(lines):
        raw = str(line.get("text") or "")
        text = _normalized_text(raw)
        if not text:
            continue
        if any(token in text for token in RECAP_PATTERNS):
            return idx
        if SUMMARY_LABEL_RE.search(text):
            return idx
    return -1


def _counterpart_role(role: str) -> str:
    """Map host role to counterpart role."""
    normalized = str(role or "").strip()
    if normalized == "Host1":
        return "Host2"
    if normalized == "Host2":
        return "Host1"
    return ""


def _speaker_for_role(lines: List[Dict[str, str]], *, role: str, fallback: str) -> str:
    """Find latest speaker name used for role, with fallback."""
    for line in reversed(lines):
        candidate_role = str(line.get("role") or "").strip()
        if candidate_role != role:
            continue
        speaker = str(line.get("speaker") or "").strip()
        if speaker:
            return speaker
    return fallback


def _instructions_for_role(lines: List[Dict[str, str]], *, role: str, fallback: str) -> str:
    """Find latest instruction style used for role, with fallback."""
    for line in reversed(lines):
        candidate_role = str(line.get("role") or "").strip()
        if candidate_role != role:
            continue
        instructions = str(line.get("instructions") or "").strip()
        if instructions:
            return instructions
    return fallback


def ensure_tail_questions_answered(
    lines: List[Dict[str, str]],
    *,
    language_hint: str | None = None,
    lookback_lines: int = 6,
) -> List[Dict[str, str]]:
    """Insert a concise counterpart answer before recap when tail question is unresolved."""
    out = [dict(line) for line in lines]
    if len(out) < 3:
        return out
    summary_idx = _summary_line_index(out)
    if summary_idx <= 0:
        return out

    window_start = max(0, summary_idx - max(1, int(lookback_lines)))
    last_question_idx = -1
    for idx in range(window_start, summary_idx):
        if _is_question_like(str(out[idx].get("text") or "")):
            last_question_idx = idx
    if last_question_idx < 0:
        return out

    question_role = str(out[last_question_idx].get("role") or "").strip()
    for idx in range(last_question_idx + 1, summary_idx):
        response_text = str(out[idx].get("text") or "").strip()
        if not response_text:
            continue
        if _is_question_like(response_text):
            continue
        response_role = str(out[idx].get("role") or "").strip()
        if question_role and response_role and response_role == question_role:
            continue
        return out

    answer_role = _counterpart_role(question_role)
    if answer_role not in {"Host1", "Host2"}:
        summary_role = str(out[summary_idx].get("role") or "").strip()
        answer_role = "Host2" if summary_role == "Host1" else "Host1"
    resolved_lang = _resolve_language_hint(out, language_hint)
    answer_text = TAIL_QUESTION_ANSWER_BY_LANG.get(
        resolved_lang,
        TAIL_QUESTION_ANSWER_BY_LANG["es"],
    )
    summary_line = out[summary_idx]
    fallback_speaker = str(summary_line.get("speaker") or "").strip()
    if not fallback_speaker:
        fallback_speaker = "Host One" if answer_role == "Host1" else "Host Two"
    fallback_instructions = str(summary_line.get("instructions") or "")
    answer_line = {
        "speaker": _speaker_for_role(out[:summary_idx], role=answer_role, fallback=fallback_speaker),
        "role": answer_role,
        "instructions": _instructions_for_role(
            out[:summary_idx],
            role=answer_role,
            fallback=fallback_instructions,
        ),
        "pace_hint": str(summary_line.get("pace_hint") or "").strip(),
        "text": answer_text,
    }
    out.insert(summary_idx, answer_line)
    return out


def ensure_recap_near_end(lines: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ensure a recap signal exists near the script ending."""
    if not lines:
        return lines
    has_recap_near_end = any(
        _has_strong_recap_signal(str(line.get("text") or ""))
        for line in lines[-6:]
    )
    if has_recap_near_end:
        return [dict(line) for line in lines]
    has_recap_anywhere = any(
        _has_strong_recap_signal(str(line.get("text") or ""))
        for line in lines
    )
    if has_recap_anywhere and len(lines) <= 6:
        return [dict(line) for line in lines]
    out = list(lines)
    resolved_lang = _resolve_language_hint(out, None)
    role = out[-1].get("role", "Host1")
    speaker = out[-1].get("speaker", "Host1")
    recap_line = {
        "speaker": speaker,
        "role": role,
        "instructions": out[-1].get("instructions", ""),
        "pace_hint": out[-1].get("pace_hint", ""),
        "text": _build_contextual_recap_text(out, language_hint=resolved_lang),
    }
    if _is_farewell(str(out[-1].get("text") or "")):
        out.insert(max(0, len(out) - 1), recap_line)
    else:
        out.append(recap_line)
    return out


def ensure_farewell_close(lines: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Ensure the final spoken turn is a complete farewell line."""
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
                "pace_hint": last.get("pace_hint", ""),
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
            "pace_hint": last.get("pace_hint", ""),
            "text": FAREWELL_BY_LANG.get(resolved_lang, FAREWELL_BY_LANG["es"]),
        }
    )
    return fix_mid_farewells(out, language_hint=resolved_lang)


def normalize_block_numbering(lines: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Normalize spoken block numbering to sequential order."""
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
    """Sanitize abrupt punctuation/connectors near script tail."""
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


def smooth_abrupt_transitions(
    lines: List[Dict[str, str]],
    *,
    language_hint: str | None = None,
    max_edits: int = 10,
    min_gap_between_edits: int = 2,
) -> List[Dict[str, str]]:
    """Inject subtle spoken connectors on abrupt topic jumps."""
    out = [dict(line) for line in lines]
    if len(out) < 3:
        return out
    resolved_lang = _resolve_language_hint(out, language_hint)
    prefixes = TRANSITION_PREFIX_BY_LANG.get(resolved_lang, TRANSITION_PREFIX_BY_LANG["es"])
    if not prefixes:
        return out

    effective_max_edits = min(
        max(0, int(max_edits)),
        max(2, len(out) // 12),
    )
    if effective_max_edits <= 0:
        return out

    edits = 0
    prefix_idx = 0
    last_edit_idx = -999
    for idx in range(1, len(out)):
        if edits >= effective_max_edits:
            break

        prev_text = str(out[idx - 1].get("text") or "").strip()
        curr_text = str(out[idx].get("text") or "").strip()
        if not prev_text or not curr_text:
            continue
        if idx - last_edit_idx <= max(0, int(min_gap_between_edits)):
            continue
        if _is_farewell(curr_text) or _has_strong_recap_signal(curr_text):
            continue
        if "?" in curr_text:
            continue
        normalized_prev = _normalized_text(prev_text)
        normalized_curr = _normalized_text(curr_text)
        if TRANSITION_CONNECTOR_RE.search(normalized_prev):
            continue
        if TRANSITION_CONNECTOR_RE.search(normalized_curr):
            continue
        if NATURAL_RESPONSE_OPENER_RE.search(normalized_curr):
            continue

        prev_tokens = set(TRANSITION_WORD_RE.findall(normalized_prev))
        curr_tokens = set(TRANSITION_WORD_RE.findall(normalized_curr))
        if len(prev_tokens) < 4 or len(curr_tokens) < 4:
            continue
        overlap_ratio = float(len(prev_tokens.intersection(curr_tokens))) / float(max(1, len(curr_tokens)))
        if overlap_ratio >= 0.08:
            continue

        prefix = prefixes[prefix_idx % len(prefixes)]
        prefix_idx += 1
        out[idx]["text"] = _compose_transition_text(prefix, curr_text, language_hint=resolved_lang)
        edits += 1
        last_edit_idx = idx
    return out


def _sanitize_tail_text(text: str) -> str:
    """Repair one possibly-abrupt tail text fragment."""
    updated = str(text or "")
    if ELLIPSIS_RE.search(updated):
        updated = ELLIPSIS_RE.sub(".", updated)
    if updated.endswith((",", ";", ":", "-", "—")):
        updated = updated.rstrip(" ,;:-—").strip() + "."
    updated = TRAILING_CONNECTOR_RE.sub(".", updated)
    return updated


def _is_complete_sentence(text: str) -> bool:
    """Return True when text appears to end as a complete sentence."""
    sample = str(text or "").strip()
    if not sample:
        return False
    return COMPLETE_SENTENCE_END_RE.search(sample) is not None


def detect_truncation_indices(lines: List[Dict[str, str]]) -> List[int]:
    """Return line indices that still look truncated."""
    out: List[int] = []
    for idx, line in enumerate(lines):
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        if TAIL_TRUNCATION_RE.search(text):
            out.append(idx)
    return out


def evaluate_script_completeness(lines: List[Dict[str, str]]) -> Dict[str, object]:
    """Evaluate deterministic completeness conditions and reasons."""
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
    """Run deterministic repair chain for completeness issues."""
    normalized_turns = normalize_speaker_turns(
        lines,
        max_consecutive_same_speaker=max_consecutive_same_speaker,
    )
    normalized = normalize_block_numbering(normalized_turns)
    normalized = normalize_spanish_technical_terms(normalized)
    normalized = sanitize_meta_podcast_language(normalized)
    normalized = sanitize_declared_tease_intent(normalized)
    normalized = diversify_repetitive_openers(normalized)
    normalized = smooth_abrupt_transitions(normalized)
    return sanitize_abrupt_tail(normalized, tail_window=max(1, len(normalized)))


def harden_script_structure(
    lines: List[Dict[str, str]],
    *,
    max_consecutive_same_speaker: int = 2,
) -> List[Dict[str, str]]:
    """Run structural hardening chain shared across pipeline stages."""
    normalized_turns = normalize_speaker_turns(
        lines,
        max_consecutive_same_speaker=max_consecutive_same_speaker,
    )
    normalized = normalize_block_numbering(normalized_turns)
    normalized = normalize_spanish_technical_terms(normalized)
    normalized = sanitize_meta_podcast_language(normalized)
    normalized = sanitize_declared_tease_intent(normalized)
    normalized = diversify_repetitive_openers(normalized)
    normalized = smooth_abrupt_transitions(normalized)
    return sanitize_abrupt_tail(normalized, tail_window=8)

