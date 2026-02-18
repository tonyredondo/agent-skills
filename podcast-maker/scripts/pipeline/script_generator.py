#!/usr/bin/env python3
from __future__ import annotations

"""Script generation engine for the podcast pipeline.

The generator converts source text into structured Host1/Host2 dialogue using
chunked prompting, schema-aware recovery, checkpoint resume, and deterministic
post-processing safeguards.
"""

import json
import math
import os
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import ReliabilityConfig, ScriptConfig, config_fingerprint
from .errors import (
    ERROR_KIND_INTERRUPTED,
    ERROR_KIND_INVALID_SCHEMA,
    ERROR_KIND_OPENAI_EMPTY_OUTPUT,
    ERROR_KIND_RESUME_BLOCKED,
    ERROR_KIND_SCRIPT_COMPLETENESS,
    ERROR_KIND_SOURCE_TOO_SHORT,
    ERROR_KIND_STUCK,
    ERROR_KIND_UNKNOWN,
    ScriptOperationError,
)
from .logging_utils import Logger
from .openai_client import OpenAIClient
from .schema import (
    SCRIPT_JSON_SCHEMA,
    canonical_json,
    content_hash,
    count_words_from_lines,
    salvage_script_payload,
    validate_script_payload,
)
from .script_checkpoint import ScriptCheckpointStore
from .script_chunker import split_source_chunks, target_chunk_count
from .script_postprocess import (
    dedupe_append,
    evaluate_script_completeness,
    ensure_farewell_close,
    ensure_recap_near_end,
    fix_mid_farewells,
    harden_script_structure,
    repair_script_completeness,
)
from .script_generator_helpers import (
    atomic_write_text as _atomic_write_text,
    default_completeness_report as _default_completeness_report,
    env_bool as _env_bool,
    env_float as _env_float,
    env_int as _env_int,
    migrate_checkpoint_lines as _migrate_checkpoint_lines,
    phase_seconds_with_generation as _phase_seconds_with_generation,
    recent_dialogue as _recent_dialogue,
    sum_int_maps as _sum_int_maps,
)
from .run_manifest import pipeline_summary_path, resolve_episode_id, run_manifest_path


SOURCE_AUTHOR_LINE_RE = re.compile(
    r"(?im)^\s*(?:autor(?:es)?|author(?:s)?|by|byline)\s*[:\-]\s*(.+?)\s*$"
)
SOURCE_AUTHOR_SPLIT_RE = re.compile(r"\s*(?:,|;|/|&|\by\b|\be\b|\band\b)\s*", re.IGNORECASE)
SOURCE_AUTHOR_STOPWORDS = {
    "n/a",
    "na",
    "anonimo",
    "anónimo",
    "anonymous",
    "desconocido",
    "unknown",
    "equipo editorial",
    "editorial team",
}
SOURCE_INDEX_HEADER_RE = re.compile(
    r"(?i)^\s*(?:indice|índice|agenda|temario|tabla de contenido|contents|outline)\s*:?\s*$"
)
SOURCE_INDEX_ITEM_RE = re.compile(
    r"(?i)^\s*(?:[-*•]\s+|\d{1,2}[.)]\s+|(?:tema|topic)\s+\d+\s*[:\-]\s+)(.+?)\s*$"
)
SOURCE_TIMELINE_INDEX_ITEM_RE = re.compile(
    r"^\s*-\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+·\s+([^·]+)\s+·\s+(.+?)\s*(?:\([^)]*\))?\s*$"
)
CONTEXTUAL_FALLBACK_SOURCE_MAX_CHARS = 7000
CONTEXTUAL_FALLBACK_RECENT_LINES = 14
CONTEXTUAL_TAIL_MIN_WORD_RATIO = 0.9


@dataclass
class ScriptGenerationResult:
    """Final artifact pointers and key script-stage metrics."""

    episode_id: str
    output_path: str
    line_count: int
    word_count: int
    checkpoint_path: str
    run_summary_path: str
    script_retry_rate: float
    invalid_schema_rate: float
    schema_validation_failures: int


@dataclass
class ScriptGenerator:
    """Stateful script generation coordinator with recovery ladders."""

    config: ScriptConfig
    reliability: ReliabilityConfig
    logger: Logger
    client: OpenAIClient

    def _mark_fallback_mode(self, *, stage: str, mode: str) -> None:
        by_stage = dict(getattr(self, "_fallback_modes_by_stage", {}))
        key = str(stage)
        values = list(by_stage.get(key, []))
        normalized_mode = str(mode).strip().lower()
        if normalized_mode and normalized_mode not in values:
            values.append(normalized_mode)
        by_stage[key] = values
        self._fallback_modes_by_stage = by_stage

    def _tts_speed_hints_enabled(self) -> bool:
        """Return whether optional pace-hint prompting is enabled."""
        return _env_bool("TTS_SPEED_HINTS_ENABLED", False)

    def _line_schema_fields_prompt(self) -> str:
        """Describe expected JSON line keys for current runtime mode."""
        if self._tts_speed_hints_enabled():
            return "speaker, role, instructions, optional pace_hint, text"
        return "speaker, role, instructions, text"

    def _pace_hint_prompt_guidance(self) -> str:
        """Prompt block for optional pace_hint generation when enabled."""
        if not self._tts_speed_hints_enabled():
            return ""
        return (
            "- Optional field `pace_hint`: calm|steady|brisk.\n"
            "- Prefer `steady` by default; use `calm`/`brisk` only for clear narrative intent.\n"
            "- Keep adjacent turns coherent; avoid oscillating pace_hint values without a clear reason.\n"
            "- If there is no clear pace signal, omit `pace_hint`."
        )

    def _can_use_contextual_fallback_llm(self) -> bool:
        """Return True when client supports contextual schema rewrites."""
        return (
            self.client is not None
            and hasattr(self.client, "generate_script_json")
        )

    def _compact_source_context(self, source_context: str, *, max_chars: int) -> str:
        """Trim source context to bounded size while preserving head/tail."""
        text = str(source_context or "").strip()
        if not text:
            return ""
        limit = max(400, int(max_chars))
        if len(text) <= limit:
            return text
        head = max(260, int(limit * 0.72))
        tail = max(120, limit - head - 44)
        return f"{text[:head]}\n...[source truncated]...\n{text[-tail:]}"

    def _tail_has_recap_and_farewell(self, lines: List[Dict[str, str]]) -> bool:
        """Lightweight lexical check for recap + farewell near the end."""
        if not lines:
            return False
        tail = " ".join(str(line.get("text") or "").lower() for line in lines[-5:])
        recap_tokens = (
            "en resumen",
            "nos quedamos con",
            "in summary",
            "to sum up",
            "em resumo",
            "en bref",
        )
        farewell_tokens = (
            "gracias por escuch",
            "nos vemos",
            "hasta la proxima",
            "nos escuchamos",
            "thank you for listening",
            "see you",
            "obrigado por ouvir",
            "merci",
            "au revoir",
        )
        has_recap = any(token in tail for token in recap_tokens)
        has_farewell = any(token in tail for token in farewell_tokens)
        return has_recap and has_farewell

    def _build_contextual_continuation_fallback_prompt(
        self,
        *,
        lines_so_far: List[Dict[str, str]],
        min_words: int,
        max_words: int,
        source_context: str,
        mode: str,
    ) -> str:
        """Build contextual fallback prompt for continuation recovery."""
        current_words = count_words_from_lines(lines_so_far)
        remaining_to_min = max(0, int(min_words) - int(current_words))
        line_fields = self._line_schema_fields_prompt()
        pace_hint_guidance = self._pace_hint_prompt_guidance()
        recent = _recent_dialogue(
            lines_so_far,
            max(8, min(self.config.max_context_lines, CONTEXTUAL_FALLBACK_RECENT_LINES)),
        )
        source_snippet = self._compact_source_context(
            source_context,
            max_chars=CONTEXTUAL_FALLBACK_SOURCE_MAX_CHARS,
        )
        if mode == "closure":
            goal_block = (
                "- You are near the target length and must close naturally.\n"
                "- Add 2-4 new lines that resolve pending ideas, include a concise recap, and end with a short farewell.\n"
                "- If there is a direct unresolved question near the tail, answer it explicitly before recap/farewell."
            )
        else:
            goal_block = (
                "- Add 2-4 meaningful lines that extend the conversation with source-grounded substance.\n"
                "- If this extension reaches the target zone, close naturally with recap + farewell.\n"
                "- If not yet near closure, keep continuity and avoid early farewell."
            )
        return textwrap.dedent(
            f"""
            Continue this podcast script with NEW lines only.
            Return ONLY JSON with key "lines" and fields: {line_fields}.

            Hard constraints:
            - Keep spoken text in the same language/tone as recent context (Spanish by default).
            - Keep role values Host1/Host2 and strict alternation.
            - Keep instructions in English as short natural-language guidance (1-2 sentences).
            - Keep instructions specific and actionable (tone, clarity, delivery, optional pronunciation hints).
            {pace_hint_guidance}
            - Keep both hosts lively and engaging: Host1 warm/confident with upbeat energy, Host2 bright/friendly with expressive curiosity.
            - Avoid generic filler and prefab lines; every line must be contextual and specific to this episode.
            - Preserve factual consistency with source context.
            - Do not invent names, dates, numbers, or tools absent from source/context.
            - Keep each line concise (1-2 sentences).
            - Use smooth bridges between turns and avoid abrupt topic jumps.
            - Avoid repeating transition templates or sentence openers.
            - Do not mention internal tooling/workflow.

            Fallback goal:
            {goal_block}

            Current words: {current_words}
            Target range: {min_words}-{max_words}
            Remaining words to minimum target: {remaining_to_min}

            RECENT CONTEXT:
            {recent if recent else "(no recent lines)"}

            SOURCE CONTEXT:
            {source_snippet if source_snippet else "(not provided)"}
            """
        ).strip()

    def _request_contextual_continuation_fallback_lines(
        self,
        *,
        lines_so_far: List[Dict[str, str]],
        min_words: int,
        max_words: int,
        source_context: str,
        continuation_stage: str,
        mode: str,
    ) -> List[Dict[str, str]]:
        """Ask LLM for contextual continuation fallback lines."""
        if not self._can_use_contextual_fallback_llm():
            return []
        try:
            prompt = self._build_contextual_continuation_fallback_prompt(
                lines_so_far=lines_so_far,
                min_words=min_words,
                max_words=max_words,
                source_context=source_context,
                mode=mode,
            )
            stage = f"{continuation_stage}_contextual_{mode}_fallback"
            max_output_tokens = max(700, min(2400, int(self.config.max_output_tokens_continuation)))
            repaired_raw = self.client.generate_script_json(
                prompt=prompt,
                schema=SCRIPT_JSON_SCHEMA,
                max_output_tokens=max_output_tokens,
                stage=stage,
            )
            candidate_lines = validate_script_payload(repaired_raw)["lines"]
            merged, added = dedupe_append(lines_so_far, candidate_lines)
            if added <= 0:
                return []
            merged = fix_mid_farewells(merged)
            merged = harden_script_structure(
                merged,
                max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
            )
            if mode == "closure" and not self._tail_has_recap_and_farewell(merged):
                return []
            if len(merged) <= len(lines_so_far):
                return []
            return merged[len(lines_so_far) :]
        except Exception as exc:  # noqa: BLE001
            self.logger.warn(
                "continuation_contextual_fallback_failed",
                stage=continuation_stage,
                mode=mode,
                error=str(exc),
            )
            return []

    def _build_contextual_tail_finalize_prompt(
        self,
        *,
        lines: List[Dict[str, str]],
        min_words: int,
        max_words: int,
        source_context: str,
    ) -> str:
        """Build postprocess tail-finalization prompt with source context."""
        payload = canonical_json({"lines": lines})
        line_fields = self._line_schema_fields_prompt()
        pace_hint_guidance = self._pace_hint_prompt_guidance()
        source_snippet = self._compact_source_context(
            source_context,
            max_chars=CONTEXTUAL_FALLBACK_SOURCE_MAX_CHARS,
        )
        tail_focus = canonical_json({"lines": lines[-CONTEXTUAL_FALLBACK_RECENT_LINES:]})
        return textwrap.dedent(
            f"""
            Refine this podcast script JSON with minimal edits, focusing on the ending quality.
            Return ONLY JSON object with key "lines" and fields: {line_fields}.

            Goals:
            - Keep the script natural, specific, and contextual (no prefab filler).
            - If needed, resolve open tail questions before closing.
            - Ensure ending flow: brief synthesis/recap + short natural farewell.
            - Keep smooth transitions and avoid abrupt topic jumps.
            - Keep most earlier lines intact; prioritize tail-focused edits.

            Constraints:
            - Keep spoken text in the script language and instructions in English as short natural-language guidance (1-2 sentences).
            - Keep instructions specific and actionable (tone, clarity, delivery, optional pronunciation hints).
            {pace_hint_guidance}
            - Keep both hosts lively and engaging: Host1 warm/confident with upbeat energy, Host2 bright/friendly with expressive curiosity.
            - Keep Host1/Host2 alternation and consistent speakers.
            - Preserve source-grounded facts and avoid fabricated details.
            - Keep each line concise (1-2 sentences).
            - Do not mention internal tooling/workflow details.
            - Do not expose source-availability caveats or editorial process disclaimers in spoken text (for example: "en el material que tenemos hoy", "sin inventar especificaciones", "no tenemos ese dato aqui", "con lo que tenemos").
            - Maintain target size around {min_words}-{max_words} words.

            SOURCE CONTEXT:
            {source_snippet if source_snippet else "(not provided)"}

            ENDING FOCUS:
            {tail_focus}

            FULL SCRIPT JSON:
            {payload}
            """
        ).strip()

    def _request_contextual_tail_finalize(
        self,
        *,
        lines: List[Dict[str, str]],
        min_words: int,
        max_words: int,
        source_context: str,
    ) -> List[Dict[str, str]] | None:
        """Ask LLM to rewrite tail naturally before deterministic closure fallback."""
        if not self._can_use_contextual_fallback_llm():
            return None
        prompt = self._build_contextual_tail_finalize_prompt(
            lines=lines,
            min_words=min_words,
            max_words=max_words,
            source_context=source_context,
        )
        max_output_tokens = max(900, min(3200, int(self.config.max_output_tokens_chunk)))
        repaired_raw = self.client.generate_script_json(
            prompt=prompt,
            schema=SCRIPT_JSON_SCHEMA,
            max_output_tokens=max_output_tokens,
            stage="postprocess_contextual_tail_finalize",
        )
        candidate_lines = validate_script_payload(repaired_raw)["lines"]
        candidate_lines = harden_script_structure(
            candidate_lines,
            max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
        )
        if not candidate_lines:
            return None
        baseline_wc = count_words_from_lines(lines)
        candidate_wc = count_words_from_lines(candidate_lines)
        min_ratio = max(0.5, min(1.0, _env_float("SCRIPT_CONTEXTUAL_TAIL_MIN_WORD_RATIO", CONTEXTUAL_TAIL_MIN_WORD_RATIO)))
        retained_floor = int(math.floor(float(max(0, baseline_wc)) * float(min_ratio)))
        if baseline_wc >= int(min_words) and candidate_wc < int(min_words):
            return None
        if baseline_wc > 0 and candidate_wc < retained_floor:
            return None
        if not self._tail_has_recap_and_farewell(candidate_lines):
            return None
        return candidate_lines

    def _is_empty_output_error(self, exc: BaseException) -> bool:
        message = str(exc or "").strip().lower()
        if not message:
            return False
        return (
            "parse_failure_kind=empty_output" in message
            or "openai returned empty text" in message
        )

    def _is_invalid_schema_error(self, exc: BaseException) -> bool:
        message = str(exc or "").strip().lower()
        if not message:
            return False
        return (
            "schema" in message
            or "failed to parse json output" in message
            or "parse_failure_kind" in message
        )

    def _current_truncation_pressure(self) -> float:
        truncation_failures = int(
            dict(getattr(self.client, "script_json_parse_failures_by_kind", {})).get("truncation", 0) or 0
        )
        script_requests = int(getattr(self.client, "script_requests_made", getattr(self.client, "requests_made", 0)))
        if script_requests <= 0:
            return 0.0
        return float(truncation_failures) / float(script_requests)

    def _adaptive_token_budget(self, *, base_tokens: int, stage: str) -> int:
        """Scale token budgets up/down when truncation pressure increases."""
        budget = max(128, int(base_tokens))
        if not _env_bool("SCRIPT_TRUNCATION_PRESSURE_ADAPTIVE", True):
            return budget
        pressure = self._current_truncation_pressure()
        threshold = max(0.0, _env_float("SCRIPT_TRUNCATION_PRESSURE_THRESHOLD", 0.18))
        self._truncation_pressure_peak = max(float(getattr(self, "_truncation_pressure_peak", 0.0)), pressure)
        if pressure < threshold:
            return budget
        continuation_stage = str(stage or "").startswith("continuation_") or str(stage or "").startswith(
            "truncation_recovery_"
        )
        if continuation_stage:
            scale_up = max(1.0, _env_float("SCRIPT_TRUNCATION_PRESSURE_TOKEN_SCALE_UP", 1.18))
            cap = max(
                budget,
                _env_int(
                    "SCRIPT_TRUNCATION_PRESSURE_TOKEN_SCALE_UP_CAP",
                    int(max(self.config.max_output_tokens_initial, self.config.max_output_tokens_chunk) * 1.4),
                ),
            )
            adapted_budget = min(cap, max(256, int(round(float(budget) * scale_up))))
        else:
            scale = max(0.35, min(1.0, _env_float("SCRIPT_TRUNCATION_PRESSURE_TOKEN_SCALE", 0.82)))
            adapted_budget = max(256, int(round(float(budget) * scale)))
        self._truncation_pressure_adaptive_events = int(
            getattr(self, "_truncation_pressure_adaptive_events", 0)
        ) + 1
        self.logger.warn(
            "truncation_pressure_adaptive_tokens",
            stage=stage,
            pressure=round(pressure, 4),
            threshold=round(threshold, 4),
            base_tokens=budget,
            adapted_tokens=adapted_budget,
        )
        return adapted_budget

    def _effective_expected_tokens_per_chunk(self) -> int:
        base = max(128, int(self.config.expected_tokens_per_chunk))
        if not _env_bool("SCRIPT_TRUNCATION_PRESSURE_ADAPTIVE", True):
            return base
        pressure = self._current_truncation_pressure()
        threshold = max(0.0, _env_float("SCRIPT_TRUNCATION_PRESSURE_THRESHOLD", 0.18))
        if pressure < threshold:
            return base
        scale = max(0.35, min(1.0, _env_float("SCRIPT_TRUNCATION_PRESSURE_TOKEN_SCALE", 0.82)))
        return max(256, int(round(float(base) * scale)))

    def _max_consecutive_same_speaker(self) -> int:
        gate_profile = str(os.environ.get("SCRIPT_QUALITY_GATE_PROFILE", "default") or "default").strip().lower()
        strict_alternation = _env_bool("SCRIPT_STRICT_HOST_ALTERNATION", True)
        default_limit = 1 if strict_alternation else (2 if gate_profile == "production_strict" else 3)
        return max(1, _env_int("SCRIPT_QUALITY_MAX_CONSECUTIVE_SAME_SPEAKER", default_limit))

    def _script_tone_profile(self) -> str:
        raw = str(os.environ.get("SCRIPT_TONE_PROFILE", "balanced") or "").strip().lower()
        if raw in {"balanced", "energetic", "broadcast"}:
            return raw
        return "balanced"

    def _script_transition_style(self) -> str:
        raw = str(os.environ.get("SCRIPT_TRANSITION_STYLE", "subtle") or "").strip().lower()
        if raw in {"subtle", "explicit"}:
            return raw
        return "subtle"

    def _script_precision_profile(self) -> str:
        raw = str(os.environ.get("SCRIPT_PRECISION_PROFILE", "strict") or "").strip().lower()
        if raw in {"strict", "balanced"}:
            return raw
        return "strict"

    def _script_closing_style(self) -> str:
        raw = str(os.environ.get("SCRIPT_CLOSING_STYLE", "brief") or "").strip().lower()
        if raw in {"brief", "warm"}:
            return raw
        return "brief"

    def _tone_guidance(self) -> str:
        profile = self._script_tone_profile()
        if profile == "balanced":
            return "Keep tone conversational and confident with moderate energy (avoid hype or dryness)."
        if profile == "broadcast":
            return "Keep tone highly engaging and energetic, but credible and natural for spoken audio."
        return "Keep tone lively, warm, and enthusiastic, but natural (not theatrical)."

    def _transition_guidance(self) -> str:
        style = self._script_transition_style()
        if style == "explicit":
            return (
                "Treat transition smoothness as a hard requirement: on every topic switch, "
                "use an explicit but elegant bridge that links the previous point to the new one."
            )
        return (
            "Treat transition smoothness as a hard requirement: even with subtle style, "
            "every topic pivot must keep a clear bridge to the previous turn."
        )

    def _precision_guidance(self) -> str:
        precision = self._script_precision_profile()
        if precision == "balanced":
            return (
                "Prefer source-grounded claims; if uncertain, phrase as hypothesis/recommendation and avoid made-up figures."
            )
        return (
            "Use only source-grounded claims; do not invent names, numbers, dates, or tools not present in context. "
            "If detail is uncertain, use neutral listener-facing uncertainty (for example 'todavia no esta claro') "
            "and never mention source/document limitations (for example 'en el material que tenemos hoy', "
            "'sin inventar especificaciones', 'no tenemos ese dato aqui')."
        )

    def _closing_guidance(self) -> str:
        style = self._script_closing_style()
        if style == "brief":
            return "In the final turns, include a concise recap and a short natural farewell (1-2 lines)."
        return "In the final turns, include a concise recap and a warm natural farewell."

    def _extract_source_authors(self, source_text: str) -> List[str]:
        candidates: List[str] = []
        for match in SOURCE_AUTHOR_LINE_RE.finditer(str(source_text or "")):
            raw_value = str(match.group(1) or "").strip()
            if not raw_value:
                continue
            normalized_value = re.sub(r"\([^)]*\)", "", raw_value).strip()
            if not normalized_value:
                continue
            parts = SOURCE_AUTHOR_SPLIT_RE.split(normalized_value)
            for part in parts:
                cleaned = str(part or "").strip(" .:-")
                if not cleaned:
                    continue
                lowered = cleaned.lower()
                if lowered in SOURCE_AUTHOR_STOPWORDS:
                    continue
                if cleaned not in candidates:
                    candidates.append(cleaned)
            if len(candidates) >= 6:
                break
        return candidates[:4]

    def _author_reference_guidance(self, *, source_text: Optional[str] = None) -> str:
        if source_text:
            parsed = self._extract_source_authors(source_text)
            if parsed:
                self._source_authors_detected = list(parsed)
        authors = list(getattr(self, "_source_authors_detected", []))
        if authors:
            if len(authors) == 1:
                return (
                    f"Source metadata includes author: {authors[0]}. "
                    "Reference this author naturally once in the dialogue and do not invent other authors."
                )
            return (
                "Source metadata includes authors: "
                + ", ".join(authors)
                + ". Reference these authors naturally once in the dialogue and do not invent additional authors."
            )
        return (
            "If source metadata includes author names, reference them naturally once in the dialogue "
            "and do not invent author names."
        )

    def _normalize_intro_topic(self, raw_topic: str) -> str:
        cleaned = str(raw_topic or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"^\s*(?:[-*•]\s+|\d{1,2}[.)]\s+)", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .:-")
        words = cleaned.split()
        if not words:
            return ""
        if len(words) > 12:
            cleaned = " ".join(words[:12]).strip(" ,;:-")
        return cleaned

    def _extract_source_index_entries(self, source_text: str) -> List[Dict[str, str]]:
        """Extract category/topic entries from timeline-like source indexes."""
        lines = str(source_text or "").splitlines()
        if not lines:
            return []
        entries: List[Dict[str, str]] = []
        for raw_line in lines[:320]:
            line = str(raw_line or "").strip()
            if not line:
                continue
            match = SOURCE_TIMELINE_INDEX_ITEM_RE.match(line)
            if match is None:
                # Once timeline entries are collected, stop when prose starts.
                if entries and len(line.split()) > 16 and not line.startswith("-"):
                    break
                continue
            category = re.sub(r"\s+", " ", str(match.group(1) or "").strip())
            title = re.sub(r"\s+", " ", str(match.group(2) or "").strip())
            if not category or not title:
                continue
            topic_hint = self._normalize_intro_topic(
                title.replace("_", " ").replace("-", " ")
            )
            if not topic_hint:
                topic_hint = self._normalize_intro_topic(title)
            entries.append(
                {
                    "category": category,
                    "title": title,
                    "topic": topic_hint or "tema general",
                }
            )
            if len(entries) >= 120:
                break
        return entries

    def _build_source_topic_plan(
        self,
        *,
        entries: List[Dict[str, str]],
        total_chunks: int,
        target_words: int,
    ) -> List[Dict[str, Any]]:
        """Build per-chunk topic/category plan from source index density."""
        if not entries or total_chunks <= 0:
            return []
        category_counts: Dict[str, int] = {}
        category_topics: Dict[str, List[str]] = {}
        for item in entries:
            category = str(item.get("category", "")).strip() or "General"
            topic = str(item.get("topic", "")).strip() or "tema general"
            category_counts[category] = int(category_counts.get(category, 0)) + 1
            topics = category_topics.setdefault(category, [])
            if topic and topic not in topics:
                topics.append(topic)

        ordered_categories = sorted(
            category_counts.keys(),
            key=lambda key: (-int(category_counts.get(key, 0)), key.lower()),
        )
        if not ordered_categories:
            return []

        chunk_budget: Dict[str, int] = {}
        if len(ordered_categories) <= total_chunks:
            for category in ordered_categories:
                chunk_budget[category] = 1
            remaining_slots = max(0, int(total_chunks) - len(ordered_categories))
            if remaining_slots > 0:
                total_count = float(sum(category_counts.values()) or 1)
                extras_allocated = 0
                for category in ordered_categories:
                    raw_extra = (float(category_counts[category]) / total_count) * float(remaining_slots)
                    extra = int(math.floor(raw_extra))
                    if extra > 0:
                        chunk_budget[category] += extra
                        extras_allocated += extra
                remainder = max(0, remaining_slots - extras_allocated)
                if remainder > 0:
                    remainders = sorted(
                        ordered_categories,
                        key=lambda key: (
                            -(
                                (float(category_counts[key]) / float(total_count)) * float(remaining_slots)
                                - math.floor(
                                    (float(category_counts[key]) / float(total_count))
                                    * float(remaining_slots)
                                )
                            ),
                            -int(category_counts[key]),
                            key.lower(),
                        ),
                    )
                    for category in remainders[:remainder]:
                        chunk_budget[category] = int(chunk_budget.get(category, 0)) + 1
        else:
            # If there are more categories than chunks, prioritize densest ones.
            selected = ordered_categories[:total_chunks]
            for category in selected:
                chunk_budget[category] = 1

        sequence: List[str] = []
        previous_category = ""
        while len(sequence) < total_chunks:
            candidates = [cat for cat, remaining in chunk_budget.items() if int(remaining) > 0]
            if not candidates:
                break
            candidates = sorted(
                candidates,
                key=lambda key: (-int(chunk_budget.get(key, 0)), -int(category_counts.get(key, 0)), key.lower()),
            )
            picked = candidates[0]
            if len(candidates) > 1 and picked == previous_category:
                picked = candidates[1]
            sequence.append(picked)
            chunk_budget[picked] = int(chunk_budget.get(picked, 0)) - 1
            previous_category = picked

        if not sequence:
            return []
        while len(sequence) < total_chunks:
            sequence.append(sequence[-1])
        if len(sequence) > total_chunks:
            sequence = sequence[:total_chunks]

        total_count = float(sum(category_counts.values()) or 1.0)
        category_word_budget: Dict[str, int] = {}
        for category, count in category_counts.items():
            share = float(count) / total_count
            category_word_budget[category] = max(90, int(round(float(target_words) * share)))

        topic_cursor: Dict[str, int] = {category: 0 for category in category_topics}
        per_category_chunk_count: Dict[str, int] = {}
        for category in sequence:
            per_category_chunk_count[category] = int(per_category_chunk_count.get(category, 0)) + 1

        plan: List[Dict[str, Any]] = []
        for idx, category in enumerate(sequence, start=1):
            topics = category_topics.get(category, [])
            cursor = int(topic_cursor.get(category, 0))
            if topics:
                topic = topics[cursor % len(topics)]
                topic_cursor[category] = cursor + 1
            else:
                topic = "tema general"
            chunks_for_category = max(1, int(per_category_chunk_count.get(category, 1)))
            target_for_chunk = max(
                80,
                int(round(float(category_word_budget.get(category, 120)) / float(chunks_for_category))),
            )
            plan.append(
                {
                    "chunk_idx": idx,
                    "category": category,
                    "topic": topic,
                    "objective": (
                        "Develop this category with source-grounded precision, concrete detail, and a smooth bridge "
                        "from the previous segment."
                    ),
                    "target_words": target_for_chunk,
                    "category_source_count": int(category_counts.get(category, 0)),
                }
            )
        return plan

    def _outline_category_coverage_ratio(
        self,
        *,
        outline: List[Dict[str, Any]],
        chunks_done: int,
    ) -> float:
        """Estimate coverage ratio of planned categories by completed chunks."""
        planned_categories = {
            str(section.get("category", "")).strip()
            for section in list(outline or [])
            if str(section.get("category", "")).strip()
        }
        if not planned_categories:
            return 1.0
        limit = max(0, min(int(chunks_done), len(outline)))
        covered_categories = {
            str(section.get("category", "")).strip()
            for section in list(outline or [])[:limit]
            if str(section.get("category", "")).strip()
        }
        return float(len(covered_categories)) / float(max(1, len(planned_categories)))

    def _extract_source_agenda_topics(self, source_text: str) -> List[str]:
        lines = [line.strip() for line in str(source_text or "").splitlines() if str(line or "").strip()]
        if not lines:
            return []
        topics: List[str] = []
        header_idx = -1
        for idx, line in enumerate(lines[:80]):
            if SOURCE_INDEX_HEADER_RE.match(line):
                header_idx = idx
                break
        if header_idx < 0:
            return []
        for line in lines[header_idx + 1 : header_idx + 25]:
            match = SOURCE_INDEX_ITEM_RE.match(line)
            if match is None:
                # Stop once prose begins after collecting enough index topics.
                if len(topics) >= 2 and len(line.split()) > 12:
                    break
                continue
            normalized = self._normalize_intro_topic(str(match.group(1) or ""))
            if normalized and normalized not in topics:
                topics.append(normalized)
            if len(topics) >= 5:
                break
        return topics

    def _extract_outline_agenda_topics(self, outline: List[Dict[str, Any]]) -> List[str]:
        topics: List[str] = []
        for section in list(outline or []):
            normalized = self._normalize_intro_topic(str(section.get("topic", "")))
            if normalized and normalized not in topics:
                topics.append(normalized)
            if len(topics) >= 5:
                break
        return topics

    def _opening_agenda_guidance(self, *, chunk_idx: int, chunk_total: int) -> str:
        source_topics = list(getattr(self, "_source_agenda_topics", []))
        outline_topics = list(getattr(self, "_outline_agenda_topics", []))
        has_multi_topic = bool(
            len(source_topics) >= 2
            or len(outline_topics) >= 2
            or int(chunk_total) > 1
        )
        if int(chunk_idx) != 1 or not has_multi_topic:
            return ""
        topic_list = source_topics if len(source_topics) >= 2 else outline_topics
        if len(topic_list) >= 2:
            topic_preview = "; ".join(topic_list[:4])
            return (
                "In the opening turns, include a brief natural roadmap of the episode "
                "(for example: 'hoy hablaremos de...') covering these topics: "
                + topic_preview
                + ". Then pivot smoothly to the first topic with a phrase like 'comenzamos con...'."
            )
        return (
            "If the source covers multiple topics, include a brief natural roadmap in the opening turns "
            "(for example: 'hoy hablaremos de...') and then pivot smoothly to the first topic "
            "with a phrase like 'comenzamos con...'."
        )

    def _build_chunk_prompt(
        self,
        *,
        source_chunk: str,
        chunk_idx: int,
        chunk_total: int,
        section_plan: Dict[str, Any],
        lines_so_far: List[Dict[str, str]],
        min_words: int,
        max_words: int,
    ) -> str:
        """Build a chunk prompt with continuity and target-size guidance."""
        recent = _recent_dialogue(lines_so_far, self.config.max_context_lines)
        current_words = count_words_from_lines(lines_so_far)
        remaining_to_min = max(0, min_words - current_words)
        remaining_to_max = max(0, max_words - current_words)
        chunks_left = max(1, int(chunk_total) - int(chunk_idx) + 1)
        section_target_raw = int(section_plan.get("target_words", 0) or 0)
        section_target = max(80, section_target_raw) if section_target_raw > 0 else max(
            80, int(round(float(max(1, remaining_to_min)) / float(chunks_left)))
        )
        remaining_to_min_per_chunk = max(80, int(round(float(max(1, remaining_to_min)) / float(chunks_left))))
        remaining_to_max_per_chunk = max(90, int(round(float(max(1, remaining_to_max)) / float(chunks_left))))
        chunk_goal = max(80, min(section_target, remaining_to_max_per_chunk))
        chunk_soft_cap = max(chunk_goal, int(round(float(remaining_to_max_per_chunk) * 1.1)))
        chunk_soft_cap = max(100, min(max(100, remaining_to_max), chunk_soft_cap))
        chunk_min_guidance = max(70, min(chunk_goal, int(round(float(chunk_goal) * 0.85))))
        tone_guidance = self._tone_guidance()
        transition_guidance = self._transition_guidance()
        precision_guidance = self._precision_guidance()
        author_guidance = self._author_reference_guidance(source_text=source_chunk)
        line_fields = self._line_schema_fields_prompt()
        pace_hint_guidance = self._pace_hint_prompt_guidance()
        opening_agenda_guidance = self._opening_agenda_guidance(
            chunk_idx=chunk_idx,
            chunk_total=chunk_total,
        )
        opening_agenda_line = f"- {opening_agenda_guidance}" if opening_agenda_guidance else ""
        if chunk_idx >= chunk_total:
            ending_rule = (
                "- This is the final chunk: add a coherent ending with explicit recap and farewell in the last turns.\n"
                f"- {self._closing_guidance()}\n"
                "- Preferred ending shape: penultimate turn gives a practical mini-recap (2-4 concrete actions), final turn is a short farewell only.\n"
                "- End with complete sentences only; do not leave trailing ellipsis or dangling connectors."
            )
        else:
            ending_rule = "- This is not the final chunk: do not add farewell/closing yet."
        return textwrap.dedent(
            f"""
            You are writing a Spanish podcast script with two presenters (Host1 and Host2).
            Requirements:
            - Output JSON object with key "lines" (array), each line includes: {line_fields}.
            - speaker must be a real person name and consistent through the episode.
            - role must be Host1 or Host2.
            - Alternate turns strictly between Host1 and Host2 (no consecutive turns by the same role).
            - Spoken text must be in Spanish.
            - instructions MUST be in English as short natural-language guidance (1-2 sentences).
            - Keep instructions specific and actionable (tone, clarity, delivery, optional pronunciation hints).
            {pace_hint_guidance}
            - Keep both hosts lively and engaging: Host1 warm/confident with upbeat energy, Host2 bright/friendly with expressive curiosity.
            - Keep instructions consistent per speaker and avoid contradictory speed cues.
            - Good examples:
              "Speak in a warm, confident, conversational Spanish tone. Keep pacing measured and clear with brief pauses."
              "Use a calm, analytical delivery. Prioritize clarity over drama and keep a steady rhythm."
            - Keep facts accurate to source.
            - {precision_guidance}
            - {author_guidance}
            {opening_agenda_line}
            - Avoid placeholders.
            - Do not mention internal tooling or research workflow details (for example script paths, shell commands, "DailyRead pipeline", "Tavily", "Serper").
            - Do not expose source-availability caveats or editorial process disclaimers in spoken text (for example: "en el material que tenemos hoy", "sin inventar especificaciones", "no tenemos ese dato aqui", "con lo que tenemos").
            - Do not speak as if reading a document structure (avoid phrases such as "segun el indice", "en este resumen", "en el siguiente tramo", "ruta del episodio", "tabla de contenidos").
            - Do not use explicit section labels in spoken text (for example: "Bloque 1", "Bloque 2", "Section 3", "Part 4").
            - Use elegant spoken transitions between topics instead of numeric labels.
            - Transition smoothness is a hard quality constraint: each topic switch must include a brief bridge that links one concrete element from the previous turn to the new angle.
            - Avoid abrupt jumps between unrelated ideas; on pivots, keep lexical carry-over or an explicit connector before introducing the next detail.
            - Avoid opening consecutive turns with the same connector (especially repeated "Y ...").
            - Prefer natural Spanish technical phrasing and avoid unnecessary anglicisms (for example, use "donante adicional" instead of "donor extra").
            - Keep each spoken line concise (usually 1-2 sentences).
            - Avoid repeating the same transition template over and over.
            - Avoid semantic repetition: each turn should add a new angle, example, or decision detail.
            - Avoid "two parallel monologues": each turn should react to the previous turn (question, challenge, clarification, or concrete follow-up).
            - Include direct host-to-host questions regularly but without overuse (roughly 1 question every 4-6 turns).
            - Avoid interview-like cadence: not every turn should end with a question; mix questions with assertions, reactions, and mini-conclusions.
            - Introduce brief, respectful tension when relevant (contrast viewpoints, probe assumptions, then resolve with evidence).
            - Respectful disagreement, light humor, and uncomfortable questions are allowed when they improve clarity.
            - Never pre-announce tension with lines like "te voy a chinchar/pinchar", "te voy a provocar", or similar meta-intent phrasing.
            - Use occasional everyday analogies to explain complex technical ideas more naturally.
            - Do not leave direct questions unresolved near the ending.
            - If a host asks a direct question in the final stretch, the counterpart host must answer explicitly before recap or farewell.
            - In the final 3 turns before recap/farewell, do not introduce new questions.
            - Right before recap, prefer short declarative synthesis/answer turns (no interrogative endings).
            - Hard cap: keep total episode length at or below {max_words} words.
            - If you are near the cap, stop adding new branches and move to concise synthesis.
            - Reserve roughly 60-120 words for a natural ending: mini recap plus warm farewell.
            - Keep recap digestible: no overloaded single-line summary with too many clauses; if needed, split recap across two short turns before farewell.
            {ending_rule}

            Episode target: {min_words}-{max_words} words.
            Current words so far: {current_words}. Remaining to max target: about {remaining_to_max}.
            Chunk {chunk_idx}/{chunk_total}. Aim to contribute about {chunk_goal} useful words.
            Keep this chunk roughly in the {chunk_min_guidance}-{chunk_soft_cap} word range.
            Remaining required words to hit minimum target: about {remaining_to_min}.
            Planning hint per remaining chunk: at least ~{remaining_to_min_per_chunk}, ideally <= ~{remaining_to_max_per_chunk}.
            Section objective: {section_plan.get("objective", "develop next core topic")}
            Section category hint: {section_plan.get("category", "general")}
            Section topic hint: {section_plan.get("topic", "general")}
            Section target words: {section_target}
            {tone_guidance}
            {transition_guidance}
            Prioritize concrete examples and actionable detail over generic filler.

            RECENT CONTEXT:
            {recent if recent else "(start of episode)"}

            SOURCE CHUNK:
            {source_chunk}

            Output ONLY JSON.
            """
        ).strip()

    def _extract_topic(self, chunk: str) -> str:
        first_sentence = re.split(r"(?<=[.!?])\s+", chunk.strip())[0] if chunk.strip() else ""
        words = first_sentence.split()[:12]
        if not words:
            return "tema general"
        return " ".join(words)

    def _build_outline(
        self,
        *,
        chunks: List[str],
        min_words: int,
        max_words: int,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        total_chunks = len(chunks)
        target_words = max(min_words, int((min_words + max_words) / 2))
        baseline_chunks = target_chunk_count(
            target_minutes=self.config.target_minutes,
            chunk_target_minutes=self.config.chunk_target_minutes,
        )
        source_plan = self._build_source_topic_plan(
            entries=list(getattr(self, "_source_index_entries", [])),
            total_chunks=total_chunks,
            target_words=target_words,
        )
        per_chunk = max(80, int(target_words / max(1, total_chunks)))
        remainder = max(0, target_words - (per_chunk * total_chunks))

        outline: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            bonus = 1 if idx <= remainder else 0
            target = per_chunk + bonus
            topic = self._extract_topic(chunk)
            category = ""
            source_topic = ""
            if idx <= len(source_plan):
                source_section = dict(source_plan[idx - 1])
                category = str(source_section.get("category", "")).strip()
                source_topic = str(source_section.get("topic", "")).strip()
                target = max(80, int(source_section.get("target_words", target) or target))
            if source_topic:
                topic = source_topic
            outline.append(
                {
                    "chunk_idx": idx,
                    "category": category,
                    "topic": topic,
                    "objective": (
                        "Introduce and expand this section with source-grounded precision, concrete examples, and smooth transitions"
                    ),
                    "target_words": target,
                    "baseline_chunks_estimate": baseline_chunks,
                }
            )
        return outline

    def _build_schema_repair_prompt(self, payload: Dict[str, Any], error_text: str) -> str:
        compact = json.dumps(payload, ensure_ascii=False)
        if len(compact) > 120000:
            compact = compact[:120000]
        line_fields = self._line_schema_fields_prompt()
        pace_hint_guidance = self._pace_hint_prompt_guidance()
        return textwrap.dedent(
            f"""
            Repair this JSON object so it strictly follows the schema with key "lines".
            Each line must contain: {line_fields}.
            Keep instructions in English as short natural-language guidance (1-2 sentences).
            {pace_hint_guidance}
            Keep semantics and wording as much as possible.
            Output ONLY JSON.

            Validation error:
            {error_text}

            INPUT JSON:
            {compact}
            """
        ).strip()

    def _request_validated_lines(
        self,
        *,
        prompt: str,
        stage: str,
        max_output_tokens: int,
    ) -> List[Dict[str, str]]:
        self._last_stage = str(stage)
        payload = self.client.generate_script_json(
            prompt=prompt,
            schema=SCRIPT_JSON_SCHEMA,
            max_output_tokens=max_output_tokens,
            stage=stage,
        )
        try:
            return validate_script_payload(payload)["lines"]
        except Exception as exc:  # noqa: BLE001
            self._schema_validation_failures = int(getattr(self, "_schema_validation_failures", 0)) + 1
            by_stage = dict(getattr(self, "_schema_validation_failures_by_stage", {}))
            by_stage[stage] = int(by_stage.get(stage, 0)) + 1
            self._schema_validation_failures_by_stage = by_stage
            self._schema_salvage_attempts = int(getattr(self, "_schema_salvage_attempts", 0)) + 1
            salvage_attempts_by_stage = dict(getattr(self, "_schema_salvage_attempts_by_stage", {}))
            salvage_attempts_by_stage[stage] = int(salvage_attempts_by_stage.get(stage, 0)) + 1
            self._schema_salvage_attempts_by_stage = salvage_attempts_by_stage
            self.logger.warn("schema_validation_failed", stage=stage, error=str(exc))
            try:
                salvaged_payload = salvage_script_payload(payload)
                salvaged_lines = validate_script_payload(salvaged_payload)["lines"]
                self._schema_salvage_successes = int(getattr(self, "_schema_salvage_successes", 0)) + 1
                salvage_success_by_stage = dict(getattr(self, "_schema_salvage_successes_by_stage", {}))
                salvage_success_by_stage[stage] = int(salvage_success_by_stage.get(stage, 0)) + 1
                self._schema_salvage_successes_by_stage = salvage_success_by_stage
                self._mark_fallback_mode(stage=stage, mode="schema_salvage")
                self.logger.info("schema_salvage_ok", stage=stage, salvaged_lines=len(salvaged_lines))
                return salvaged_lines
            except Exception as salvage_exc:  # noqa: BLE001
                self._schema_salvage_failures = int(getattr(self, "_schema_salvage_failures", 0)) + 1
                salvage_fail_by_stage = dict(getattr(self, "_schema_salvage_failures_by_stage", {}))
                salvage_fail_by_stage[stage] = int(salvage_fail_by_stage.get(stage, 0)) + 1
                self._schema_salvage_failures_by_stage = salvage_fail_by_stage
                self.logger.warn("schema_salvage_failed", stage=stage, error=str(salvage_exc))
            self._mark_fallback_mode(stage=stage, mode="schema_repair")
            last_exc: Exception = exc
            repaired_payload = payload
            for attempt in range(1, self.config.repair_max_attempts + 1):
                repair_prompt = self._build_schema_repair_prompt(repaired_payload, str(last_exc))
                repaired_payload = self.client.generate_script_json(
                    prompt=repair_prompt,
                    schema=SCRIPT_JSON_SCHEMA,
                    max_output_tokens=max_output_tokens,
                    stage=f"{stage}_schema_repair_{attempt}",
                )
                try:
                    validated = validate_script_payload(repaired_payload)["lines"]
                    self._schema_repair_successes = int(getattr(self, "_schema_repair_successes", 0)) + 1
                    repair_success_by_stage = dict(getattr(self, "_schema_repair_successes_by_stage", {}))
                    repair_success_by_stage[stage] = int(repair_success_by_stage.get(stage, 0)) + 1
                    self._schema_repair_successes_by_stage = repair_success_by_stage
                    self.logger.info("schema_repair_ok", stage=stage, attempt=attempt)
                    return validated
                except Exception as repair_exc:  # noqa: BLE001
                    last_exc = repair_exc
                    self._schema_repair_failures = int(getattr(self, "_schema_repair_failures", 0)) + 1
                    repair_fail_by_stage = dict(getattr(self, "_schema_repair_failures_by_stage", {}))
                    repair_fail_by_stage[stage] = int(repair_fail_by_stage.get(stage, 0)) + 1
                    self._schema_repair_failures_by_stage = repair_fail_by_stage
                    self.logger.warn(
                        "schema_repair_failed",
                        stage=stage,
                        attempt=attempt,
                        error=str(repair_exc),
                    )
            raise RuntimeError(f"Schema validation failed after repairs at stage={stage}: {last_exc}")

    def _split_chunk_for_recovery(self, source_chunk: str) -> List[str]:
        words = [w for w in str(source_chunk or "").split() if w]
        if len(words) < 120:
            return [source_chunk]
        midpoint = len(words) // 2
        first = " ".join(words[:midpoint]).strip()
        second = " ".join(words[midpoint:]).strip()
        out = [part for part in (first, second) if part]
        return out if len(out) >= 2 else [source_chunk]

    def _request_chunk_part_with_recovery(
        self,
        *,
        source_chunk: str,
        chunk_idx: int,
        chunk_total: int,
        section_plan: Dict[str, Any],
        lines_so_far: List[Dict[str, str]],
        min_words: int,
        max_words: int,
        max_output_tokens: int,
        stage: str,
    ) -> List[Dict[str, str]]:
        prompt = self._build_chunk_prompt(
            source_chunk=source_chunk,
            chunk_idx=chunk_idx,
            chunk_total=chunk_total,
            section_plan=section_plan,
            lines_so_far=lines_so_far,
            min_words=min_words,
            max_words=max_words,
        )
        try:
            return self._request_validated_lines(
                prompt=prompt,
                stage=stage,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            recoverable_error = self._is_empty_output_error(exc) or self._is_invalid_schema_error(exc)
            if not _env_bool("SCRIPT_ADAPTIVE_SUBPART_RECOVERY_V2", True) or not recoverable_error:
                raise
            self._adaptive_subpart_failures = int(getattr(self, "_adaptive_subpart_failures", 0)) + 1
            self._mark_fallback_mode(stage=stage, mode="adaptive_subpart_retry")
            self.logger.warn(
                "adaptive_subpart_recovery_start",
                stage=stage,
                error=str(exc),
            )
            last_exc: BaseException = exc
            retry_tokens = max(192, int(max_output_tokens * 0.8))
            for attempt in range(1, 3):
                retry_prompt = self._build_chunk_prompt(
                    source_chunk=source_chunk,
                    chunk_idx=chunk_idx,
                    chunk_total=chunk_total,
                    section_plan=section_plan,
                    lines_so_far=lines_so_far,
                    min_words=min_words,
                    max_words=max_words,
                )
                try:
                    recovered_lines = self._request_validated_lines(
                        prompt=retry_prompt,
                        stage=f"{stage}_retry_{attempt}",
                        max_output_tokens=retry_tokens,
                    )
                    self._adaptive_subpart_recoveries = int(
                        getattr(self, "_adaptive_subpart_recoveries", 0)
                    ) + 1
                    self.logger.info(
                        "adaptive_subpart_recovery_ok",
                        stage=stage,
                        attempt=attempt,
                        recovered_lines=len(recovered_lines),
                    )
                    return recovered_lines
                except Exception as retry_exc:  # noqa: BLE001
                    last_exc = retry_exc
                    retry_tokens = max(160, int(retry_tokens * 0.9))

            nested_parts = self._split_chunk_for_recovery(source_chunk)
            nested_recovered: List[Dict[str, str]] = []
            if len(nested_parts) >= 2 and len(str(source_chunk or "").split()) >= 180:
                self._mark_fallback_mode(stage=stage, mode="adaptive_subpart_subsplit")
                for nested_idx, nested_part in enumerate(nested_parts, start=1):
                    nested_prompt = self._build_chunk_prompt(
                        source_chunk=nested_part,
                        chunk_idx=chunk_idx,
                        chunk_total=chunk_total,
                        section_plan=section_plan,
                        lines_so_far=lines_so_far + nested_recovered,
                        min_words=min_words,
                        max_words=max_words,
                    )
                    nested_tokens = max(160, int(max_output_tokens * 0.65))
                    try:
                        nested_lines = self._request_validated_lines(
                            prompt=nested_prompt,
                            stage=f"{stage}_subsplit_{nested_idx}",
                            max_output_tokens=nested_tokens,
                        )
                        nested_recovered, _ = dedupe_append(nested_recovered, nested_lines)
                    except Exception as nested_exc:  # noqa: BLE001
                        last_exc = nested_exc
                        if not _env_bool("SCRIPT_ADAPTIVE_SUBPART_ALLOW_SKIP", True):
                            raise
                        self._adaptive_subpart_skips = int(getattr(self, "_adaptive_subpart_skips", 0)) + 1
                        self.logger.warn(
                            "adaptive_subpart_subsplit_partial_skip",
                            stage=stage,
                            nested_idx=nested_idx,
                            error=str(nested_exc),
                        )
                if nested_recovered:
                    self._adaptive_subpart_recoveries = int(
                        getattr(self, "_adaptive_subpart_recoveries", 0)
                    ) + 1
                    self.logger.info(
                        "adaptive_subpart_subsplit_ok",
                        stage=stage,
                        recovered_lines=len(nested_recovered),
                    )
                    return nested_recovered

            if _env_bool("SCRIPT_ADAPTIVE_SUBPART_ALLOW_SKIP", True):
                self._adaptive_subpart_skips = int(getattr(self, "_adaptive_subpart_skips", 0)) + 1
                self._mark_fallback_mode(stage=stage, mode="adaptive_subpart_skip")
                self.logger.warn(
                    "adaptive_subpart_recovery_skip",
                    stage=stage,
                    error=str(last_exc),
                )
                return []
            raise RuntimeError(
                f"Adaptive subpart recovery failed at stage={stage}: {last_exc}"
            ) from last_exc

    def _request_chunk_with_recovery(
        self,
        *,
        source_chunk: str,
        chunk_idx: int,
        chunk_total: int,
        section_plan: Dict[str, Any],
        lines_so_far: List[Dict[str, str]],
        min_words: int,
        max_words: int,
        max_output_tokens: int,
    ) -> List[Dict[str, str]]:
        """Generate one chunk, falling back to adaptive split recovery when needed."""
        stage = f"chunk_{chunk_idx}"
        adaptive_tokens = self._adaptive_token_budget(base_tokens=max_output_tokens, stage=stage)
        if _env_bool("SCRIPT_TRUNCATION_PRESSURE_ADAPTIVE", True):
            pressure = self._current_truncation_pressure()
            presplit_threshold = max(0.0, _env_float("SCRIPT_TRUNCATION_PRESSURE_PRESPLIT_THRESHOLD", 0.28))
            parts = self._split_chunk_for_recovery(source_chunk)
            if pressure >= presplit_threshold and len(parts) >= 2:
                # When truncation pressure is high, pre-splitting reduces parse
                # risk before we even spend a full-chunk request.
                self._truncation_pressure_presplit_events = int(
                    getattr(self, "_truncation_pressure_presplit_events", 0)
                ) + 1
                self._mark_fallback_mode(stage=stage, mode="adaptive_presplit")
                self.logger.warn(
                    "chunk_adaptive_presplit_start",
                    stage=stage,
                    pressure=round(pressure, 4),
                    threshold=round(presplit_threshold, 4),
                    parts=len(parts),
                )
                recovered: List[Dict[str, str]] = []
                part_tokens = max(256, int(adaptive_tokens * 0.85))
                for part_idx, part in enumerate(parts, start=1):
                    sub_lines = self._request_chunk_part_with_recovery(
                        source_chunk=part,
                        chunk_idx=chunk_idx,
                        chunk_total=chunk_total,
                        section_plan=section_plan,
                        lines_so_far=lines_so_far + recovered,
                        min_words=min_words,
                        max_words=max_words,
                        max_output_tokens=part_tokens,
                        stage=f"{stage}_adaptive_{part_idx}",
                    )
                    recovered, _ = dedupe_append(recovered, sub_lines)
                if recovered:
                    self.logger.info(
                        "chunk_adaptive_presplit_ok",
                        stage=stage,
                        recovered_lines=len(recovered),
                    )
                    return recovered
                self._mark_fallback_mode(stage=stage, mode="adaptive_presplit_empty_fallback")
                self.logger.warn(
                    "chunk_adaptive_presplit_empty_fallback",
                    stage=stage,
                    pressure=round(pressure, 4),
                    threshold=round(presplit_threshold, 4),
                )
        prompt = self._build_chunk_prompt(
            source_chunk=source_chunk,
            chunk_idx=chunk_idx,
            chunk_total=chunk_total,
            section_plan=section_plan,
            lines_so_far=lines_so_far,
            min_words=min_words,
            max_words=max_words,
        )
        try:
            return self._request_validated_lines(
                prompt=prompt,
                stage=stage,
                max_output_tokens=adaptive_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            if not _env_bool("SCRIPT_RECOVERY_LADDER_V2", True):
                raise
            parts = self._split_chunk_for_recovery(source_chunk)
            if len(parts) < 2:
                # Unsplittable chunk falls back to a cheaper whole-chunk retry
                # before escalating to a hard failure.
                self._mark_fallback_mode(stage=stage, mode="whole_chunk_retry")
                self.logger.warn(
                    "chunk_recovery_whole_retry_start",
                    stage=stage,
                    error=str(exc),
                )
                retry_tokens = max(192, int(adaptive_tokens * 0.75))
                try:
                    recovered_lines = self._request_chunk_part_with_recovery(
                        source_chunk=source_chunk,
                        chunk_idx=chunk_idx,
                        chunk_total=chunk_total,
                        section_plan=section_plan,
                        lines_so_far=lines_so_far,
                        min_words=min_words,
                        max_words=max_words,
                        max_output_tokens=retry_tokens,
                        stage=f"{stage}_whole_retry",
                    )
                    self.logger.info(
                        "chunk_recovery_whole_retry_ok",
                        stage=stage,
                        recovered_lines=len(recovered_lines),
                    )
                    return recovered_lines
                except Exception as whole_exc:  # noqa: BLE001
                    self.logger.warn(
                        "chunk_recovery_whole_retry_failed",
                        stage=stage,
                        error=str(whole_exc),
                    )
                    raise
            self._chunk_subsplit_recoveries = int(getattr(self, "_chunk_subsplit_recoveries", 0)) + 1
            self._mark_fallback_mode(stage=stage, mode="subsplit")
            self.logger.warn(
                "chunk_recovery_subsplit_start",
                stage=stage,
                parts=len(parts),
                error=str(exc),
            )
            recovered: List[Dict[str, str]] = []
            for part_idx, part in enumerate(parts, start=1):
                part_tokens = max(256, int(max_output_tokens * 0.75))
                sub_lines = self._request_chunk_part_with_recovery(
                    source_chunk=part,
                    chunk_idx=chunk_idx,
                    chunk_total=chunk_total,
                    section_plan=section_plan,
                    lines_so_far=lines_so_far + recovered,
                    min_words=min_words,
                    max_words=max_words,
                    max_output_tokens=part_tokens,
                    stage=f"{stage}_subsplit_{part_idx}",
                )
                recovered, _ = dedupe_append(recovered, sub_lines)
            self.logger.info(
                "chunk_recovery_subsplit_ok",
                stage=stage,
                recovered_lines=len(recovered),
            )
            return recovered

    def _maybe_pre_summarize_source(
        self,
        *,
        source: str,
        cancel_check: Optional[Callable[[], bool]],
    ) -> str:
        words = len(source.split())
        if words <= self.config.pre_summary_trigger_words:
            return source

        self.logger.warn(
            "pre_summary_start",
            source_words=words,
            trigger=self.config.pre_summary_trigger_words,
        )
        working = source
        previous_words = words
        for round_idx in range(1, self.config.pre_summary_max_rounds + 1):
            if cancel_check and cancel_check():
                raise InterruptedError("Interrupted during pre-summary stage")
            current_words = len(working.split())
            if current_words <= self.config.pre_summary_target_words:
                break

            chunks = split_source_chunks(
                working,
                target_minutes=max(self.config.target_minutes, self.config.pre_summary_chunk_target_minutes),
                chunk_target_minutes=self.config.pre_summary_chunk_target_minutes,
                words_per_min=self.config.words_per_min,
            )
            if not chunks:
                break

            summaries: List[str] = []
            target_words_per_chunk = max(120, int(self.config.pre_summary_target_words / max(1, len(chunks))))
            if self.config.pre_summary_parallel and len(chunks) > 1:
                workers = min(self.config.pre_summary_parallel_workers, len(chunks))
                self.logger.info(
                    "pre_summary_parallel_round",
                    round=round_idx,
                    chunks=len(chunks),
                    workers=workers,
                )
                summary_map: Dict[int, str] = {}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {}
                    for ci, chunk in enumerate(chunks, start=1):
                        if cancel_check and cancel_check():
                            raise InterruptedError("Interrupted during pre-summary stage")
                        prompt = textwrap.dedent(
                            f"""
                            Summarize the source content below into a concise Spanish briefing.
                            Keep key facts and actionable points.
                            Avoid introduction and farewell.
                            Keep around {target_words_per_chunk} words.

                            SOURCE CHUNK:
                            {chunk}
                            """
                        ).strip()
                        fut = executor.submit(
                            self.client.generate_freeform_text,
                            prompt=prompt,
                            max_output_tokens=self.config.max_output_tokens_chunk,
                            stage=f"presummary_r{round_idx}_c{ci}",
                        )
                        futures[fut] = ci
                    for fut in as_completed(futures):
                        if cancel_check and cancel_check():
                            for pending in futures:
                                pending.cancel()
                            raise InterruptedError("Interrupted during pre-summary stage")
                        ci = futures[fut]
                        summary_map[ci] = str(fut.result() or "").strip()
                summaries = [summary_map.get(ci, "") for ci in range(1, len(chunks) + 1)]
            else:
                for ci, chunk in enumerate(chunks, start=1):
                    if cancel_check and cancel_check():
                        raise InterruptedError("Interrupted during pre-summary stage")
                    prompt = textwrap.dedent(
                        f"""
                        Summarize the source content below into a concise Spanish briefing.
                        Keep key facts and actionable points.
                        Avoid introduction and farewell.
                        Keep around {target_words_per_chunk} words.

                        SOURCE CHUNK:
                        {chunk}
                        """
                    ).strip()
                    summary = self.client.generate_freeform_text(
                        prompt=prompt,
                        max_output_tokens=self.config.max_output_tokens_chunk,
                        stage=f"presummary_r{round_idx}_c{ci}",
                    )
                    summaries.append(summary.strip())
            working = "\n\n".join(summaries).strip()
            new_words = len(working.split())
            self.logger.info(
                "pre_summary_round_done",
                round=round_idx,
                chunks=len(chunks),
                words=new_words,
            )
            if new_words >= previous_words:
                self.logger.warn("pre_summary_not_shrinking", round=round_idx, words=new_words)
                break
            previous_words = new_words
        return working

    def _deterministic_continuation_fallback_lines(
        self,
        *,
        lines_so_far: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        expanded = ensure_recap_near_end(list(lines_so_far))
        expanded = ensure_farewell_close(expanded)
        expanded = harden_script_structure(
            expanded,
            max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
        )
        if len(expanded) <= len(lines_so_far):
            return []
        return expanded[len(lines_so_far) :]

    def _continuation_extension_language(self, *, lines_so_far: List[Dict[str, str]]) -> str:
        text = " ".join(str(line.get("text") or "").lower() for line in lines_so_far[-10:])
        if any(token in text for token in ("merci", "prochain", "en bref", "au revoir")):
            return "fr"
        if any(token in text for token in ("obrigad", "edicao", "em resumo", "ate a proxima")):
            return "pt"
        if any(token in text for token in ("thank you", "in summary", "next episode", "today")):
            return "en"
        return "es"

    def _deterministic_continuation_extension_lines(
        self,
        *,
        lines_so_far: List[Dict[str, str]],
        min_words: int,
    ) -> List[Dict[str, str]]:
        if not lines_so_far:
            return []
        if count_words_from_lines(lines_so_far) >= int(min_words):
            return []
        templates_by_lang = {
            "es": (
                "Vale, bajemos esto a tierra: que experimento pequeno haria el equipo esta semana para validar la idea sin riesgo?",
                "Empezaria con una metrica clara, una prueba acotada y una decision predefinida segun el resultado.",
            ),
            "en": (
                "Let's make this concrete: what small experiment should the team run this week to validate the idea safely?",
                "Start with one clear metric, a scoped trial, and a predefined decision based on the result.",
            ),
            "pt": (
                "Vamos tornar isso concreto: qual experimento pequeno a equipe pode rodar nesta semana para validar a ideia com seguranca?",
                "Comecaria com uma metrica clara, um teste limitado e uma decisao predefinida com base no resultado.",
            ),
            "fr": (
                "Rendons cela concret : quel petit test l'equipe peut lancer cette semaine pour valider l'idee sans risque ?",
                "Je commencerais par une metrique claire, un test limite et une decision predefinie selon le resultat.",
            ),
        }
        language = self._continuation_extension_language(lines_so_far=lines_so_far)
        templates = templates_by_lang.get(language, templates_by_lang["en"])
        role_to_speaker: Dict[str, str] = {}
        for line in reversed(lines_so_far):
            role = str(line.get("role") or "").strip()
            speaker = str(line.get("speaker") or "").strip()
            if role in {"Host1", "Host2"} and speaker and role not in role_to_speaker:
                role_to_speaker[role] = speaker

        expanded = list(lines_so_far)
        for idx, text in enumerate(templates, start=1):
            if count_words_from_lines(expanded) >= int(min_words):
                break
            previous = expanded[-1] if expanded else {}
            role = str(previous.get("role") or "").strip()
            if role == "Host1":
                role = "Host2"
            elif role == "Host2":
                role = "Host1"
            else:
                role = "Host1" if idx % 2 == 1 else "Host2"
            speaker = role_to_speaker.get(role) or str(previous.get("speaker") or "").strip() or role
            instructions = str(previous.get("instructions") or "").strip()
            expanded.append(
                {
                    "speaker": speaker,
                    "role": role,
                    "instructions": instructions,
                    "text": text,
                }
            )
        expanded = ensure_recap_near_end(expanded)
        expanded = ensure_farewell_close(expanded)
        expanded = harden_script_structure(
            expanded,
            max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
        )
        if len(expanded) <= len(lines_so_far):
            return []
        return expanded[len(lines_so_far) :]

    def _request_continuation_with_recovery(
        self,
        *,
        lines_so_far: List[Dict[str, str]],
        continuation_round: int,
        min_words: int,
        max_words: int,
        source_context: str,
    ) -> List[Dict[str, str]]:
        """Generate continuation lines with targeted recovery on parse failures."""
        continuation_stage = f"continuation_{continuation_round}"
        continuation_tokens = self._adaptive_token_budget(
            base_tokens=self.config.max_output_tokens_continuation,
            stage=continuation_stage,
        )
        prompt = self._build_continuation_prompt(
            lines_so_far=lines_so_far,
            min_words=min_words,
            max_words=max_words,
        )
        try:
            return self._request_validated_lines(
                prompt=prompt,
                stage=continuation_stage,
                max_output_tokens=continuation_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            recoverable = self._is_empty_output_error(exc) or self._is_invalid_schema_error(exc)
            if not _env_bool("SCRIPT_CONTINUATION_RECOVERY_V2", True) or not recoverable:
                raise
            self._continuation_recovery_attempts = int(
                getattr(self, "_continuation_recovery_attempts", 0)
            ) + 1
            self._mark_fallback_mode(stage=continuation_stage, mode="continuation_recovery")
            self.logger.warn(
                "continuation_recovery_start",
                stage=continuation_stage,
                error=str(exc),
            )
            last_exc: BaseException = exc
            retry_tokens = max(192, int(continuation_tokens * 0.85))
            for attempt in range(1, 3):
                retry_prompt = self._build_continuation_prompt(
                    lines_so_far=lines_so_far,
                    min_words=min_words,
                    max_words=max_words,
                )
                try:
                    recovered_lines = self._request_validated_lines(
                        prompt=retry_prompt,
                        stage=f"{continuation_stage}_recovery_{attempt}",
                        max_output_tokens=retry_tokens,
                    )
                    self._continuation_recovery_successes = int(
                        getattr(self, "_continuation_recovery_successes", 0)
                    ) + 1
                    self.logger.info(
                        "continuation_recovery_ok",
                        stage=continuation_stage,
                        attempt=attempt,
                        recovered_lines=len(recovered_lines),
                    )
                    return recovered_lines
                except Exception as retry_exc:  # noqa: BLE001
                    last_exc = retry_exc
                    retry_tokens = max(160, int(retry_tokens * 0.9))

            fallback_min_ratio = max(0.0, min(1.0, _env_float("SCRIPT_CONTINUATION_FALLBACK_MIN_RATIO", 0.82)))
            fallback_secondary_ratio = max(
                0.0,
                min(
                    fallback_min_ratio,
                    _env_float("SCRIPT_CONTINUATION_FALLBACK_SECONDARY_MIN_RATIO", 0.70),
                ),
            )
            current_words = count_words_from_lines(lines_so_far)
            primary_threshold = int(round(float(min_words) * fallback_min_ratio))
            secondary_threshold = int(round(float(min_words) * fallback_secondary_ratio))
            if current_words >= primary_threshold:
                # Near-target path prefers contextual LLM closure fallback first,
                # then deterministic closure as a safety net.
                contextual_closure_lines = self._request_contextual_continuation_fallback_lines(
                    lines_so_far=lines_so_far,
                    min_words=min_words,
                    max_words=max_words,
                    source_context=source_context,
                    continuation_stage=continuation_stage,
                    mode="closure",
                )
                if contextual_closure_lines:
                    self._continuation_fallback_closures = int(
                        getattr(self, "_continuation_fallback_closures", 0)
                    ) + 1
                    self._mark_fallback_mode(stage=continuation_stage, mode="continuation_closure_fallback")
                    self._mark_fallback_mode(
                        stage=continuation_stage,
                        mode="continuation_closure_contextual_llm_fallback",
                    )
                    self.logger.warn(
                        "continuation_closure_fallback_applied",
                        stage=continuation_stage,
                        variant="contextual_llm",
                        added_lines=len(contextual_closure_lines),
                        current_words=current_words,
                    )
                    return contextual_closure_lines
                fallback_lines = self._deterministic_continuation_fallback_lines(lines_so_far=lines_so_far)
                if fallback_lines:
                    self._continuation_fallback_closures = int(
                        getattr(self, "_continuation_fallback_closures", 0)
                    ) + 1
                    self._mark_fallback_mode(stage=continuation_stage, mode="continuation_closure_fallback")
                    self.logger.warn(
                        "continuation_closure_fallback_applied",
                        stage=continuation_stage,
                        added_lines=len(fallback_lines),
                        current_words=current_words,
                    )
                    return fallback_lines
                contextual_extension_lines = self._request_contextual_continuation_fallback_lines(
                    lines_so_far=lines_so_far,
                    min_words=min_words,
                    max_words=max_words,
                    source_context=source_context,
                    continuation_stage=continuation_stage,
                    mode="extension",
                )
                if contextual_extension_lines:
                    self._continuation_fallback_extensions = int(
                        getattr(self, "_continuation_fallback_extensions", 0)
                    ) + 1
                    self._mark_fallback_mode(stage=continuation_stage, mode="continuation_extension_fallback")
                    self._mark_fallback_mode(
                        stage=continuation_stage,
                        mode="continuation_extension_contextual_llm_fallback",
                    )
                    self.logger.warn(
                        "continuation_extension_fallback_applied",
                        stage=continuation_stage,
                        tier="primary",
                        variant="contextual_llm",
                        added_lines=len(contextual_extension_lines),
                        current_words=current_words,
                    )
                    return contextual_extension_lines
                extension_lines = self._deterministic_continuation_extension_lines(
                    lines_so_far=lines_so_far,
                    min_words=min_words,
                )
                if extension_lines:
                    self._continuation_fallback_extensions = int(
                        getattr(self, "_continuation_fallback_extensions", 0)
                    ) + 1
                    self._mark_fallback_mode(stage=continuation_stage, mode="continuation_extension_fallback")
                    self.logger.warn(
                        "continuation_extension_fallback_applied",
                        stage=continuation_stage,
                        tier="primary",
                        added_lines=len(extension_lines),
                        current_words=current_words,
                    )
                    return extension_lines
            elif current_words >= secondary_threshold:
                # Mid-tier threshold tries contextual extension fallback before
                # deterministic extension.
                contextual_extension_lines = self._request_contextual_continuation_fallback_lines(
                    lines_so_far=lines_so_far,
                    min_words=min_words,
                    max_words=max_words,
                    source_context=source_context,
                    continuation_stage=continuation_stage,
                    mode="extension",
                )
                if contextual_extension_lines:
                    self._continuation_fallback_extensions = int(
                        getattr(self, "_continuation_fallback_extensions", 0)
                    ) + 1
                    self._mark_fallback_mode(stage=continuation_stage, mode="continuation_extension_fallback")
                    self._mark_fallback_mode(
                        stage=continuation_stage,
                        mode="continuation_extension_contextual_llm_fallback",
                    )
                    self.logger.warn(
                        "continuation_extension_fallback_applied",
                        stage=continuation_stage,
                        tier="secondary",
                        variant="contextual_llm",
                        added_lines=len(contextual_extension_lines),
                        current_words=current_words,
                    )
                    return contextual_extension_lines
                extension_lines = self._deterministic_continuation_extension_lines(
                    lines_so_far=lines_so_far,
                    min_words=min_words,
                )
                if extension_lines:
                    self._continuation_fallback_extensions = int(
                        getattr(self, "_continuation_fallback_extensions", 0)
                    ) + 1
                    self._mark_fallback_mode(stage=continuation_stage, mode="continuation_extension_fallback")
                    self.logger.warn(
                        "continuation_extension_fallback_applied",
                        stage=continuation_stage,
                        tier="secondary",
                        added_lines=len(extension_lines),
                        current_words=current_words,
                    )
                    return extension_lines
            raise RuntimeError(
                f"Continuation recovery failed at stage={continuation_stage}: {last_exc}"
            ) from last_exc

    def _build_continuation_prompt(
        self,
        *,
        lines_so_far: List[Dict[str, str]],
        min_words: int,
        max_words: int,
    ) -> str:
        current_wc = count_words_from_lines(lines_so_far)
        remaining = max(0, min_words - current_wc)
        recent = _recent_dialogue(lines_so_far, self.config.max_context_lines)
        tone_guidance = self._tone_guidance()
        transition_guidance = self._transition_guidance()
        precision_guidance = self._precision_guidance()
        author_guidance = self._author_reference_guidance()
        pace_hint_guidance = self._pace_hint_prompt_guidance()
        return textwrap.dedent(
            f"""
            Continue the same Spanish podcast episode.
            Return ONLY JSON with key "lines", same schema as before.
            Do not repeat prior lines.
            Expand with clarifying examples and useful Q&A.
            Keep spoken text in Spanish and instructions in English as short natural-language guidance (1-2 sentences).
            Keep instructions specific and actionable (tone, clarity, delivery, optional pronunciation hints).
            {pace_hint_guidance}
            Keep both hosts lively and engaging: Host1 warm/confident with upbeat energy, Host2 bright/friendly with expressive curiosity.
            Alternate turns strictly between Host1 and Host2 (no consecutive turns by the same role).
            Do not mention internal tooling or research workflow details (for example script paths, shell commands, "DailyRead pipeline", "Tavily", "Serper").
            Do not expose source-availability caveats or editorial process disclaimers in spoken text (for example: "en el material que tenemos hoy", "sin inventar especificaciones", "no tenemos ese dato aqui", "con lo que tenemos").
            Do not speak as if reading a document structure (avoid phrases such as "segun el indice", "en este resumen", "en el siguiente tramo", "ruta del episodio", "tabla de contenidos").
            Avoid opening consecutive turns with the same connector (especially repeated "Y ...").
            Prefer natural Spanish technical phrasing and avoid unnecessary anglicisms (for example, use "donante adicional" instead of "donor extra").
            Do not use explicit section labels in spoken text (for example: "Bloque 1", "Bloque 2", "Section 3", "Part 4").
            Use elegant spoken transitions to move between ideas naturally.
            Transition smoothness is a hard quality constraint: each topic switch must add a brief bridge that links one concrete element from the previous turn to the new one.
            Avoid abrupt jumps between unrelated ideas; on pivots, keep lexical carry-over or an explicit connector before adding the next detail.
            {precision_guidance}
            {author_guidance}
            Avoid semantic repetition: do not restate the same thesis with different wording.
            Do NOT close the episode unless total words are at least {min_words}.
            If current words are near {max_words}, prioritize concise synthesis and avoid adding new branches.
            Hard cap: do not exceed {max_words} total words.
            Reserve roughly 60-120 words for the final mini recap plus natural farewell.
            If you are entering the final stretch, close in two turns: practical mini-recap, then short farewell-only line.
            End each new line with complete sentences only (no trailing ellipsis or dangling connectors).
            Keep each spoken line concise (usually 1-2 sentences).
            Avoid repeating the same transition template in consecutive turns.
            Keep the exchange genuinely interactive: one host should ask, probe, or challenge and the other should answer or refine.
            Include direct questions regularly but without overuse (roughly 1 question every 4-6 turns).
            Avoid interview-like rhythm: combine questions with concise assertions and reactions.
            Add occasional respectful contrast between hosts instead of fully aligned monologues.
            Use occasional everyday analogies to make technical points easier to visualize.
            Light humor and respectful teasing are acceptable only when fully organic.
            Never pre-announce tension with lines like "te voy a chinchar/pinchar", "te voy a provocar", or similar meta-intent phrasing.
            Do not leave open questions unresolved before the recap/farewell.
            If a direct question appears near the ending, include an explicit answer in the next 1-2 turns before recap/farewell.
            In the final 3 turns before recap/farewell, avoid introducing new questions.
            Keep pre-recap turns declarative and conclusive (not interrogative).
            Keep the final recap concise and easy to hear; split dense recap content across two turns if needed.
            {tone_guidance}
            {transition_guidance}

            Current words: {current_wc}
            Remaining target words: about {remaining}
            Final target range: {min_words}-{max_words}

            RECENT CONTEXT:
            {recent if recent else "(no recent lines)"}
            """
        ).strip()

    def _build_truncation_recovery_prompt(
        self,
        *,
        lines_so_far: List[Dict[str, str]],
        min_words: int,
        max_words: int,
    ) -> str:
        current_wc = count_words_from_lines(lines_so_far)
        recent = _recent_dialogue(lines_so_far, self.config.max_context_lines)
        tone_guidance = self._tone_guidance()
        transition_guidance = self._transition_guidance()
        precision_guidance = self._precision_guidance()
        author_guidance = self._author_reference_guidance()
        line_fields = self._line_schema_fields_prompt()
        pace_hint_guidance = self._pace_hint_prompt_guidance()
        return textwrap.dedent(
            f"""
            Continue and safely complete this Spanish podcast script without rewriting previous lines.
            Return ONLY JSON with key "lines" and fields: {line_fields}.
            Keep spoken text in Spanish and instructions in English as short natural-language guidance (1-2 sentences).
            Keep instructions specific and actionable (tone, clarity, delivery, optional pronunciation hints).
            {pace_hint_guidance}
            Keep both hosts lively and engaging: Host1 warm/confident with upbeat energy, Host2 bright/friendly with expressive curiosity.
            Alternate turns strictly between Host1 and Host2 (no consecutive turns by the same role).
            Fix abrupt/incomplete ending and provide a coherent closing flow.
            Do not mention internal tooling or research workflow details (for example script paths, shell commands, "DailyRead pipeline", "Tavily", "Serper").
            Do not expose source-availability caveats or editorial process disclaimers in spoken text (for example: "en el material que tenemos hoy", "sin inventar especificaciones", "no tenemos ese dato aqui", "con lo que tenemos").
            Do not speak as if reading a document structure (avoid phrases such as "segun el indice", "en este resumen", "en el siguiente tramo", "ruta del episodio", "tabla de contenidos").
            Avoid opening consecutive turns with the same connector (especially repeated "Y ...").
            Prefer natural Spanish technical phrasing and avoid unnecessary anglicisms (for example, use "donante adicional" instead of "donor extra").
            Do not use explicit section labels in spoken text (for example: "Bloque 1", "Bloque 2", "Section 3", "Part 4").
            Use elegant spoken transitions to connect final ideas.
            Transition smoothness is a hard quality constraint: each pivot must include a brief bridge that links one concrete element from the previous turn to the new one.
            Avoid abrupt jumps between unrelated ideas; if changing angle near the ending, add an explicit connector before introducing the next point.
            {precision_guidance}
            {author_guidance}
            Keep style conversational, warm, and engaging, and avoid duplicated lines.
            Do not repeat previous ideas: closing lines must add synthesis, not restate earlier paragraphs.
            Keep each spoken line concise (usually 1-2 sentences).
            Keep interactive flow alive in recovery too: include at least one direct question and one answer/refinement.
            Avoid making all recovery lines interrogative; keep a natural mix of question and answer/statement turns.
            Avoid flat turn-taking where both hosts only explain; add a brief challenge/probe before closing.
            Prefer at least one concrete analogy in the recovery lines when it helps clarity.
            If using humor, keep it brief, respectful, and context-relevant (never forced).
            Light teasing is acceptable only if organic; never announce it explicitly with lines like "te voy a chinchar/pinchar".
            Do not jump from an open question directly into recap/farewell.
            If a question appears near the end, add an explicit counterpart answer before the mini recap.
            In the final 2-3 turns, do not add new questions; switch to concise declarative closure.
            Hard cap: keep total words at or below {max_words} while still closing naturally.
            Reserve enough space for a complete mini recap plus natural farewell in the final turns.
            Return 2-5 lines where recap is concise (optionally split in two short turns) and the final line is a short farewell-only close.
            {tone_guidance}
            {transition_guidance}
            End each new line with complete sentences only (no trailing ellipsis or dangling connectors).

            Current words: {current_wc}
            Target range: {min_words}-{max_words}

            RECENT CONTEXT:
            {recent if recent else "(no recent lines)"}
            """
        ).strip()

    def _looks_truncated(self, lines: List[Dict[str, str]]) -> bool:
        if not lines:
            return True
        tail = str(lines[-1].get("text", "")).strip()
        if not tail:
            return True
        lower_tail = tail.lower()
        if lower_tail.endswith(("...", "…", ",", ";", ":", "-", "—")):
            return True
        if lower_tail.endswith(
            (
                " y",
                " o",
                " pero",
                " porque",
                " aunque",
                " que",
                " and",
                " or",
                " but",
                " because",
                " although",
                " that",
                " et",
                " ou",
                " mais",
                " car",
                " e",
                " mas",
            )
        ):
            return True
        if re.search(r"(?:[.!?…]|[.!?…][\"'”’)\]])\s*$", tail) is None:
            return True
        if (tail.count('"') % 2) == 1:
            return True
        window = " ".join(str(line.get("text", "")) for line in lines[-8:])
        if window.count("(") != window.count(")"):
            return True
        if window.count("¿") > window.count("?"):
            return True
        if window.count("¡") > window.count("!"):
            return True
        return False

    def _validate_source_length(
        self,
        *,
        source_word_count: int,
        min_words: int,
        max_words: int,
    ) -> Dict[str, Any]:
        """Evaluate source sufficiency for requested target duration."""
        target_word_count = max(min_words, int((min_words + max_words) / 2))
        source_to_target_ratio = float(source_word_count) / float(max(1, target_word_count))
        recommended_min_words = max(
            120,
            int(math.ceil(target_word_count * self.config.source_validation_warn_ratio)),
        )
        required_min_words = int(math.ceil(target_word_count * self.config.source_validation_enforce_ratio))
        validation_status = "ok"
        validation_reason = ""
        validation_blocked = False
        blocked_message = ""

        if self.config.target_minutes < 2.0 or self.config.target_minutes > 60.0:
            self.logger.warn(
                "target_minutes_outside_recommended_range",
                target_minutes=self.config.target_minutes,
                recommended_range="2-60",
            )

        if self.config.source_validation_mode != "off":
            if source_to_target_ratio < self.config.source_validation_enforce_ratio:
                validation_status = "warn"
                validation_reason = "source_ratio_below_enforce_threshold"
                self.logger.warn(
                    "source_validation_low_ratio",
                    mode=self.config.source_validation_mode,
                    source_word_count=source_word_count,
                    target_word_count=target_word_count,
                    source_to_target_ratio=round(source_to_target_ratio, 4),
                    required_min_words=required_min_words,
                    recommended_min_words=recommended_min_words,
                )
                if self.config.source_validation_mode == "enforce":
                    validation_blocked = True
                    blocked_message = (
                        "Source is too short for the requested target length. "
                        f"Provide at least ~{required_min_words} words "
                        f"(recommended ~{recommended_min_words})."
                    )
            elif source_to_target_ratio < self.config.source_validation_warn_ratio:
                validation_status = "warn"
                validation_reason = "source_ratio_below_warn_threshold"
                self.logger.warn(
                    "source_validation_warn_ratio",
                    mode=self.config.source_validation_mode,
                    source_word_count=source_word_count,
                    target_word_count=target_word_count,
                    source_to_target_ratio=round(source_to_target_ratio, 4),
                    recommended_min_words=recommended_min_words,
                )

        return {
            "source_word_count": source_word_count,
            "target_word_range": [int(min_words), int(max_words)],
            "target_word_count": target_word_count,
            "source_to_target_ratio": round(source_to_target_ratio, 4),
            "source_validation_status": validation_status,
            "source_validation_reason": validation_reason,
            "source_validation_blocked": validation_blocked,
            "source_validation_blocked_message": blocked_message,
            "source_recommended_min_words": recommended_min_words,
            "source_required_min_words": required_min_words,
        }

    def _resolve_generation_source(
        self,
        *,
        source: str,
        state: Dict[str, Any],
        store: ScriptCheckpointStore,
        resume: bool,
        resume_force: bool,
        cancel_check: Optional[Callable[[], bool]],
    ) -> str:
        """Resolve source text used for generation, including pre-summary cache."""
        cached = str(state.get("generation_source", "")).strip()
        if cached:
            return cached
        if resume and int(state.get("chunks_done", 0) or 0) > 0 and not resume_force:
            raise ScriptOperationError(
                "Resume blocked because checkpoint is missing generation_source. "
                "Use --resume-force to recompute and continue.",
                error_kind=ERROR_KIND_RESUME_BLOCKED,
            )
        if resume and int(state.get("chunks_done", 0) or 0) > 0 and resume_force:
            self.logger.warn("resume_force_recomputing_generation_source")
        generated = self._maybe_pre_summarize_source(source=source, cancel_check=cancel_check)
        state["generation_source"] = generated
        state["generation_source_hash"] = content_hash(generated)
        state["generation_source_words"] = len(generated.split())
        state["last_success_at"] = int(time.time())
        store.save(state)
        return generated

    def _load_or_init_state(
        self,
        *,
        store: ScriptCheckpointStore,
        source_hash: str,
        cfg_fp: str,
        resume: bool,
        resume_force: bool,
    ) -> Dict[str, Any]:
        """Load resume state or initialize a fresh checkpoint state."""
        existing = store.load()
        if resume:
            if existing is None:
                if store.last_corrupt_backup_path:
                    if resume_force:
                        self.logger.warn(
                            "resume_force_after_corrupt_checkpoint",
                            backup_path=store.last_corrupt_backup_path,
                            error=store.last_corrupt_error,
                        )
                        initial = store.create_initial_state(
                            source_hash=source_hash,
                            config_fingerprint=cfg_fp,
                        )
                        store.save(initial)
                        return initial
                    raise ScriptOperationError(
                        "Resume requested but checkpoint was corrupt and moved to "
                        f"{store.last_corrupt_backup_path}. "
                        "Fix or discard that backup, then rerun with --resume-force "
                        "or without --resume.",
                        error_kind=ERROR_KIND_RESUME_BLOCKED,
                    )
                raise ScriptOperationError(
                    "Resume requested but no script checkpoint exists",
                    error_kind=ERROR_KIND_RESUME_BLOCKED,
                )
            try:
                migrated = store.validate_resume(
                    existing,
                    source_hash=source_hash,
                    config_fingerprint=cfg_fp,
                    resume_force=resume_force,
                )
            except RuntimeError as exc:
                raise ScriptOperationError(
                    str(exc),
                    error_kind=ERROR_KIND_RESUME_BLOCKED,
                ) from exc
            if migrated:
                existing["last_success_at"] = int(time.time())
                store.save(existing)
                self.logger.info("checkpoint_version_migrated", checkpoint=store.checkpoint_path)
            self.logger.info(
                "resume_script_checkpoint",
                chunks_done=existing.get("chunks_done", 0),
                word_count=existing.get("current_word_count", 0),
            )
            return existing
        if existing is not None:
            self.logger.warn("existing_checkpoint_will_be_overwritten", checkpoint=store.checkpoint_path)
        elif store.last_corrupt_backup_path:
            self.logger.warn(
                "script_checkpoint_corrupt_quarantined",
                backup_path=store.last_corrupt_backup_path,
                error=store.last_corrupt_error,
            )
        initial = store.create_initial_state(source_hash=source_hash, config_fingerprint=cfg_fp)
        store.save(initial)
        return initial

    def generate(
        self,
        *,
        source_text: str,
        output_path: str,
        episode_id: str | None = None,
        resume: bool = False,
        resume_force: bool = False,
        force_unlock: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
        run_token: Optional[str] = None,
    ) -> ScriptGenerationResult:
        """Generate final script JSON with checkpointing and resilience controls."""
        source = source_text.strip()
        if not source:
            raise RuntimeError("Input source is empty. Provide enough content to build a podcast.")

        min_words = self.config.min_words
        max_words = self.config.max_words
        if max_words < min_words:
            max_words = min_words

        resolved_episode_id = resolve_episode_id(output_path=output_path, override=episode_id)
        store = ScriptCheckpointStore(
            base_dir=self.config.checkpoint_dir,
            episode_id=resolved_episode_id,
            reliability=self.reliability,
        )
        store.acquire_lock(force_unlock=force_unlock)
        started = time.time()
        run_summary_path = os.path.join(store.run_dir, "run_summary.json")
        source_word_count = len(source.split())
        source_validation = {
            "source_word_count": source_word_count,
            "target_word_range": [int(min_words), int(max_words)],
            "target_word_count": max(min_words, int((min_words + max_words) / 2)),
            "source_to_target_ratio": round(
                float(source_word_count) / float(max(1, max(min_words, int((min_words + max_words) / 2)))),
                4,
            ),
            "source_validation_status": "ok",
            "source_validation_reason": "",
            "source_validation_blocked": False,
            "source_validation_blocked_message": "",
            "source_recommended_min_words": 0,
            "source_required_min_words": 0,
        }
        phase_seconds: Dict[str, float] = {
            "pre_summary": 0.0,
            "chunk_generation": 0.0,
            "continuations": 0.0,
            "truncation_recovery": 0.0,
            "postprocess": 0.0,
        }
        truncation_recovery_triggered = False
        truncation_recovery_added_words = 0
        state: Dict[str, Any] = {}
        continuation_round = 0
        total_chunks = 0
        completeness_check_enabled = _env_bool("SCRIPT_COMPLETENESS_CHECK_V2", True)
        completeness_before_repair: Dict[str, object] = _default_completeness_report()
        completeness_after_repair: Dict[str, object] = _default_completeness_report()

        try:
            # Reset per-run counters so resumes/retries report clean deltas.
            self._schema_validation_failures = 0
            self._schema_repair_successes = 0
            self._schema_repair_failures = 0
            self._schema_validation_failures_by_stage = {}
            self._schema_repair_successes_by_stage = {}
            self._schema_repair_failures_by_stage = {}
            self._schema_salvage_attempts = 0
            self._schema_salvage_successes = 0
            self._schema_salvage_failures = 0
            self._schema_salvage_attempts_by_stage = {}
            self._schema_salvage_successes_by_stage = {}
            self._schema_salvage_failures_by_stage = {}
            self._no_progress_abort = False
            self._chunk_subsplit_recoveries = 0
            self._adaptive_subpart_failures = 0
            self._adaptive_subpart_recoveries = 0
            self._adaptive_subpart_skips = 0
            self._continuation_recovery_attempts = 0
            self._continuation_recovery_successes = 0
            self._continuation_fallback_closures = 0
            self._continuation_fallback_extensions = 0
            self._fallback_modes_by_stage = {}
            self._source_authors_detected = []
            self._source_agenda_topics = []
            self._outline_agenda_topics = []
            self._truncation_recovery_fallback_used = False
            self._truncation_pressure_peak = 0.0
            self._truncation_pressure_adaptive_events = 0
            self._truncation_pressure_presplit_events = 0
            self._last_stage = "startup"
            cfg_fp = config_fingerprint(
                script_cfg=self.config,
                reliability_cfg=self.reliability,
                extra={"component": "script_generator"},
            )
            source_digest = content_hash(source)
            state = self._load_or_init_state(
                store=store,
                source_hash=source_digest,
                cfg_fp=cfg_fp,
                resume=resume,
                resume_force=resume_force,
            )
            # Validate source suitability before spending API budget.
            source_validation = self._validate_source_length(
                source_word_count=source_word_count,
                min_words=min_words,
                max_words=max_words,
            )
            if bool(source_validation.get("source_validation_blocked", False)):
                raise ScriptOperationError(
                    str(source_validation.get("source_validation_blocked_message") or "Source validation blocked run"),
                    error_kind=ERROR_KIND_SOURCE_TOO_SHORT,
                )

            raw_lines = state.get("lines", [])
            lines: List[Dict[str, str]] = []
            if raw_lines:
                try:
                    lines = validate_script_payload({"lines": raw_lines})["lines"]
                except Exception:
                    migrated_lines = _migrate_checkpoint_lines(raw_lines)
                    if not migrated_lines:
                        raise
                    lines = validate_script_payload({"lines": migrated_lines})["lines"]
                    state["lines"] = lines
                    state["current_word_count"] = count_words_from_lines(lines)
                    state["last_success_at"] = int(time.time())
                    store.save(state)
                    self.logger.warn(
                        "resume_lines_migrated",
                        previous_lines=len(raw_lines) if isinstance(raw_lines, list) else 0,
                        migrated_lines=len(lines),
                    )
            lines = fix_mid_farewells(lines)
            pre_summary_started = time.time()
            self._last_stage = "pre_summary"
            try:
                # Source may be pre-summarized/cached to stabilize long inputs.
                source_for_generation = self._resolve_generation_source(
                    source=source,
                    state=state,
                    store=store,
                    resume=resume,
                    resume_force=resume_force,
                    cancel_check=cancel_check,
                )
                self._source_authors_detected = self._extract_source_authors(source_for_generation)
                self._source_agenda_topics = self._extract_source_agenda_topics(source)
                if not self._source_agenda_topics:
                    self._source_agenda_topics = self._extract_source_agenda_topics(source_for_generation)
                self._source_index_entries = self._extract_source_index_entries(source)
                if not self._source_index_entries:
                    self._source_index_entries = self._extract_source_index_entries(source_for_generation)
            finally:
                phase_seconds["pre_summary"] = round(time.time() - pre_summary_started, 3)
            chunks = split_source_chunks(
                source_for_generation,
                target_minutes=self.config.target_minutes,
                chunk_target_minutes=self.config.chunk_target_minutes,
                words_per_min=self.config.words_per_min,
            )
            if not chunks:
                raise RuntimeError("Could not split source into chunks")
            outline = self._build_outline(chunks=chunks, min_words=min_words, max_words=max_words)
            self._outline_agenda_topics = self._extract_outline_agenda_topics(outline)
            self.logger.info("outline_planned", sections=len(outline))

            chunk_start = int(state.get("chunks_done", 0))
            if chunk_start > len(chunks):
                chunk_start = len(chunks)
            total_chunks = len(chunks)

            self.logger.info(
                "script_generation_start",
                chunks=total_chunks,
                from_chunk=chunk_start,
                min_words=min_words,
                max_words=max_words,
                profile=self.config.profile_name,
            )

            with self.logger.heartbeat(
                "script_generation",
                status_fn=lambda: {
                    "chunks_done": state.get("chunks_done", 0),
                    "word_count": count_words_from_lines(lines),
                },
            ):
                chunk_phase_started = time.time()
                self._last_stage = "chunk_generation"
                try:
                    for chunk_idx in range(chunk_start, total_chunks):
                        if cancel_check and cancel_check():
                            state["status"] = "interrupted"
                            state["last_success_at"] = int(time.time())
                            store.save(state)
                            raise InterruptedError("Interrupted by signal during chunk generation")
                        chunk_tokens = (
                            self.config.max_output_tokens_initial
                            if chunk_idx == 0
                            else self.config.max_output_tokens_chunk
                        )
                        new_lines = self._request_chunk_with_recovery(
                            source_chunk=chunks[chunk_idx],
                            chunk_idx=chunk_idx + 1,
                            chunk_total=total_chunks,
                            section_plan=outline[chunk_idx] if chunk_idx < len(outline) else {},
                            lines_so_far=lines,
                            min_words=min_words,
                            max_words=max_words,
                            max_output_tokens=chunk_tokens,
                        )
                        merged, added = dedupe_append(lines, new_lines)
                        lines = fix_mid_farewells(merged)
                        wc = count_words_from_lines(lines)
                        self.logger.info(
                            "chunk_completed",
                            chunk=chunk_idx + 1,
                            added_lines=added,
                            total_lines=len(lines),
                            word_count=wc,
                        )
                        state.update(
                            {
                                "chunks_done": chunk_idx + 1,
                                "lines": lines,
                                "current_word_count": wc,
                                "continuation_round": 0,
                                "no_progress_rounds": 0,
                                "last_success_at": int(time.time()),
                                "status": "running",
                            }
                        )
                        store.save(state)
                        if wc >= max_words and chunk_idx < (total_chunks - 1):
                            # Soft gate: avoid hard-stop when category coverage is
                            # still too narrow in multi-topic runs.
                            coverage_ratio = self._outline_category_coverage_ratio(
                                outline=outline,
                                chunks_done=chunk_idx + 1,
                            )
                            min_coverage_ratio = max(
                                0.0,
                                min(1.0, _env_float("SCRIPT_TOPIC_COVERAGE_MIN_RATIO", 0.85)),
                            )
                            if coverage_ratio >= min_coverage_ratio:
                                self.logger.warn(
                                    "chunk_generation_early_stop_max_words",
                                    chunk=chunk_idx + 1,
                                    word_count=wc,
                                    max_words=max_words,
                                    remaining_chunks=max(0, total_chunks - (chunk_idx + 1)),
                                    category_coverage_ratio=round(float(coverage_ratio), 4),
                                    category_coverage_min_ratio=round(float(min_coverage_ratio), 4),
                                )
                                break
                            self.logger.warn(
                                "chunk_generation_skip_early_stop_for_coverage",
                                chunk=chunk_idx + 1,
                                word_count=wc,
                                max_words=max_words,
                                remaining_chunks=max(0, total_chunks - (chunk_idx + 1)),
                                category_coverage_ratio=round(float(coverage_ratio), 4),
                                category_coverage_min_ratio=round(float(min_coverage_ratio), 4),
                            )
                finally:
                    phase_seconds["chunk_generation"] = round(time.time() - chunk_phase_started, 3)

                # Continuations for minimum word target.
                no_progress_rounds = int(state.get("no_progress_rounds", 0) or 0)
                continuation_round = int(state.get("continuation_round", 0) or 0)
                max_rounds = max(1, total_chunks * self.config.max_continuations_per_chunk)
                continuation_phase_started = time.time()
                self._last_stage = "continuations"
                try:
                    while count_words_from_lines(lines) < min_words and continuation_round < max_rounds:
                        if cancel_check and cancel_check():
                            state["status"] = "interrupted"
                            state["last_success_at"] = int(time.time())
                            store.save(state)
                            raise InterruptedError("Interrupted by signal during continuations")
                        continuation_round += 1
                        prev_wc = count_words_from_lines(lines)
                        new_lines = self._request_continuation_with_recovery(
                            lines_so_far=lines,
                            continuation_round=continuation_round,
                            min_words=min_words,
                            max_words=max_words,
                            source_context=source_for_generation,
                        )
                        merged, _ = dedupe_append(lines, new_lines)
                        lines = fix_mid_farewells(merged)
                        new_wc = count_words_from_lines(lines)
                        delta = new_wc - prev_wc
                        self.logger.info(
                            "continuation_completed",
                            round=continuation_round,
                            word_delta=delta,
                            word_count=new_wc,
                        )
                        if delta < self.config.min_word_delta:
                            no_progress_rounds += 1
                        else:
                            no_progress_rounds = 0
                        if no_progress_rounds >= self.config.no_progress_rounds:
                            # Safety brake for "alive but not progressing" loops.
                            self._no_progress_abort = True
                            raise RuntimeError(
                                "No progress while expanding script. Aborting to avoid long hang."
                            )
                        state.update(
                            {
                                "lines": lines,
                                "current_word_count": new_wc,
                                "continuation_round": continuation_round,
                                "no_progress_rounds": no_progress_rounds,
                                "last_success_at": int(time.time()),
                                "status": "running",
                            }
                        )
                        store.save(state)
                finally:
                    phase_seconds["continuations"] = round(time.time() - continuation_phase_started, 3)

                truncation_phase_started = time.time()
                self._last_stage = "truncation_recovery"
                try:
                    if self._looks_truncated(lines):
                        # Final-tail recovery runs once after chunking/continuations
                        # if deterministic heuristics still detect truncation.
                        prev_wc = count_words_from_lines(lines)
                        self.logger.warn(
                            "script_truncation_detected",
                            word_count=prev_wc,
                            continuation_round=continuation_round,
                        )
                        recovery_prompt = self._build_truncation_recovery_prompt(
                            lines_so_far=lines,
                            min_words=min_words,
                            max_words=max_words,
                        )
                        recovery_tokens = self._adaptive_token_budget(
                            base_tokens=self.config.max_output_tokens_continuation,
                            stage="truncation_recovery_1",
                        )
                        try:
                            recovery_lines = self._request_validated_lines(
                                prompt=recovery_prompt,
                                stage="truncation_recovery_1",
                                max_output_tokens=recovery_tokens,
                            )
                            merged, _ = dedupe_append(lines, recovery_lines)
                            lines = fix_mid_farewells(merged)
                        except Exception as exc:  # noqa: BLE001
                            empty_output_error = self._is_empty_output_error(exc)
                            invalid_schema_error = self._is_invalid_schema_error(exc)
                            if not empty_output_error and not invalid_schema_error:
                                raise
                            # Preserve partial output with deterministic closure when
                            # recovery call fails due to empty/schema issues.
                            self._truncation_recovery_fallback_used = True
                            fallback_mode = (
                                "empty_output_preserve_partial"
                                if empty_output_error
                                else "invalid_schema_preserve_partial"
                            )
                            fallback_event = (
                                "script_truncation_recovery_empty_output_fallback"
                                if empty_output_error
                                else "script_truncation_recovery_invalid_schema_fallback"
                            )
                            self._mark_fallback_mode(
                                stage="truncation_recovery_1",
                                mode=fallback_mode,
                            )
                            self.logger.warn(
                                fallback_event,
                                error=str(exc),
                            )
                            lines = ensure_recap_near_end(lines)
                            lines = ensure_farewell_close(lines)
                            lines = harden_script_structure(
                                lines,
                                max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
                            )
                        new_wc = count_words_from_lines(lines)
                        truncation_recovery_added_words = max(0, new_wc - prev_wc)
                        truncation_recovery_triggered = True
                        state.update(
                            {
                                "lines": lines,
                                "current_word_count": new_wc,
                                "last_success_at": int(time.time()),
                                "status": "running",
                            }
                        )
                        store.save(state)
                        self.logger.info(
                            "script_truncation_recovery_done",
                            added_words=truncation_recovery_added_words,
                            word_count=new_wc,
                            fallback_used=bool(getattr(self, "_truncation_recovery_fallback_used", False)),
                        )
                finally:
                    phase_seconds["truncation_recovery"] = round(time.time() - truncation_phase_started, 3)

            postprocess_started = time.time()
            self._last_stage = "postprocess"
            try:
                # Final deterministic hardening pass before writing output.
                lines = harden_script_structure(
                    lines,
                    max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
                )
                if completeness_check_enabled:
                    completeness_before_repair = evaluate_script_completeness(lines)
                    lines = repair_script_completeness(
                        lines,
                        max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
                    )
                else:
                    completeness_before_repair = _default_completeness_report(reason="check_disabled")
                tentative_tail = ensure_recap_near_end(lines)
                tentative_tail = ensure_farewell_close(tentative_tail)
                tail_repair_needed = tentative_tail != lines
                if tail_repair_needed:
                    try:
                        contextual_tail = self._request_contextual_tail_finalize(
                            lines=lines,
                            min_words=min_words,
                            max_words=max_words,
                            source_context=source_for_generation,
                        )
                        if contextual_tail:
                            lines = contextual_tail
                            self._mark_fallback_mode(
                                stage="postprocess",
                                mode="postprocess_contextual_tail_finalize",
                            )
                    except Exception as exc:  # noqa: BLE001
                        self.logger.warn(
                            "postprocess_contextual_tail_finalize_failed",
                            error=str(exc),
                        )
                lines = ensure_recap_near_end(lines)
                lines = ensure_farewell_close(lines)
                lines = harden_script_structure(
                    lines,
                    max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
                )
                # Run a final closing pass after structural hardening so the tail
                # always ends with a complete mini-summary + farewell.
                lines = ensure_recap_near_end(lines)
                lines = ensure_farewell_close(lines)
                if completeness_check_enabled:
                    completeness_after_repair = evaluate_script_completeness(lines)
                    if not bool(completeness_after_repair.get("pass", False)):
                        raise ScriptOperationError(
                            "Script completeness check failed: "
                            + ", ".join(
                                str(reason) for reason in completeness_after_repair.get("reasons", []) or []
                            ),
                            error_kind=ERROR_KIND_SCRIPT_COMPLETENESS,
                        )
                else:
                    completeness_after_repair = _default_completeness_report(reason="check_disabled")
                final_wc = count_words_from_lines(lines)
                if final_wc < min_words:
                    raise ScriptOperationError(
                        f"Generated script below minimum words target ({final_wc} < {min_words}). "
                        "Increase source detail or continuation limits.",
                        error_kind=ERROR_KIND_SCRIPT_COMPLETENESS,
                    )
                final_payload = {"lines": lines}
                _atomic_write_text(output_path, canonical_json(final_payload))
            finally:
                phase_seconds["postprocess"] = round(time.time() - postprocess_started, 3)

            state.update(
                {
                    "lines": lines,
                    "current_word_count": final_wc,
                    "continuation_round": continuation_round,
                    "no_progress_rounds": no_progress_rounds,
                    "status": "completed",
                    "completed_at": int(time.time()),
                }
            )
            store.save(state)

            run_summary = {
                "component": "script_generator",
                "episode_id": resolved_episode_id,
                "run_token": run_token,
                "profile": self.config.profile_name,
                "word_count": final_wc,
                "line_count": len(lines),
                "chunks_done": state.get("chunks_done", 0),
                "chunks_planned": total_chunks,
                "continuation_rounds": continuation_round,
                "requests_made": self.client.requests_made,
                "estimated_cost_usd": round(self.client.estimated_cost_usd, 4),
                "elapsed_seconds": round(time.time() - started, 2),
                "output_path": output_path,
                "status": "completed",
                "script_started_at": int(started),
                "script_completed_at": int(time.time()),
                "failed_stage": None,
                "phase_seconds": _phase_seconds_with_generation(phase_seconds),
                "truncation_recovery_triggered": truncation_recovery_triggered,
                "truncation_recovery_added_words": truncation_recovery_added_words,
                "truncation_recovery_fallback_used": bool(
                    getattr(self, "_truncation_recovery_fallback_used", False)
                ),
                "expected_words_per_chunk": self.config.expected_words_per_chunk,
                "expected_tokens_per_chunk": self.config.expected_tokens_per_chunk,
                "expected_tokens_per_chunk_effective": self._effective_expected_tokens_per_chunk(),
                "source_validation_mode": self.config.source_validation_mode,
                "source_index_entry_count": int(len(list(getattr(self, "_source_index_entries", [])))),
                "outline_category_coverage_ratio": round(
                    float(
                        self._outline_category_coverage_ratio(
                            outline=outline,
                            chunks_done=int(state.get("chunks_done", 0)),
                        )
                    ),
                    4,
                ),
                "outline_categories_planned": [
                    str(section.get("category", "")).strip()
                    for section in list(outline or [])
                    if str(section.get("category", "")).strip()
                ],
                "script_retry_rate": round(
                    float(getattr(self.client, "script_retries_total", 0))
                    / float(max(1, getattr(self.client, "requests_made", 0))),
                    4,
                ),
                "schema_validation_failures": int(self._schema_validation_failures),
                "schema_repair_successes": int(self._schema_repair_successes),
                "schema_repair_failures": int(self._schema_repair_failures),
                "schema_salvage_attempts": int(getattr(self, "_schema_salvage_attempts", 0)),
                "schema_salvage_successes": int(getattr(self, "_schema_salvage_successes", 0)),
                "schema_salvage_failures": int(getattr(self, "_schema_salvage_failures", 0)),
                "script_json_parse_failures": int(getattr(self.client, "script_json_parse_failures", 0)),
                "script_empty_output_events": int(getattr(self.client, "script_empty_output_events", 0)),
                "script_empty_output_retries": int(getattr(self.client, "script_empty_output_retries", 0)),
                "script_empty_output_failures": int(getattr(self.client, "script_empty_output_failures", 0)),
                "script_empty_output_by_stage": dict(getattr(self.client, "script_empty_output_by_stage", {})),
                "schema_validation_failures_by_stage": dict(
                    getattr(self, "_schema_validation_failures_by_stage", {})
                ),
                "schema_repair_successes_by_stage": dict(
                    getattr(self, "_schema_repair_successes_by_stage", {})
                ),
                "schema_repair_failures_by_stage": dict(
                    getattr(self, "_schema_repair_failures_by_stage", {})
                ),
                "schema_salvage_attempts_by_stage": dict(
                    getattr(self, "_schema_salvage_attempts_by_stage", {})
                ),
                "schema_salvage_successes_by_stage": dict(
                    getattr(self, "_schema_salvage_successes_by_stage", {})
                ),
                "schema_salvage_failures_by_stage": dict(
                    getattr(self, "_schema_salvage_failures_by_stage", {})
                ),
                "script_json_parse_failures_by_stage": dict(
                    getattr(self.client, "script_json_parse_failures_by_stage", {})
                ),
                "script_json_parse_failures_by_kind": dict(
                    getattr(self.client, "script_json_parse_failures_by_kind", {})
                ),
                "script_json_parse_repair_attempts_by_stage": _sum_int_maps(
                    dict(getattr(self.client, "script_json_parse_repair_successes_by_stage", {})),
                    dict(getattr(self.client, "script_json_parse_repair_failures_by_stage", {})),
                ),
                "script_json_parse_repair_successes_by_kind": dict(
                    getattr(self.client, "script_json_parse_repair_successes_by_kind", {})
                ),
                "script_json_parse_repair_failures_by_kind": dict(
                    getattr(self.client, "script_json_parse_repair_failures_by_kind", {})
                ),
                "script_json_parse_repair_attempts_by_kind": _sum_int_maps(
                    dict(getattr(self.client, "script_json_parse_repair_successes_by_kind", {})),
                    dict(getattr(self.client, "script_json_parse_repair_failures_by_kind", {})),
                ),
                "schema_repair_attempts_by_stage": _sum_int_maps(
                    dict(getattr(self, "_schema_repair_successes_by_stage", {})),
                    dict(getattr(self, "_schema_repair_failures_by_stage", {})),
                ),
                "repair_attempts_by_stage": _sum_int_maps(
                    _sum_int_maps(
                        dict(getattr(self.client, "script_json_parse_repair_successes_by_stage", {})),
                        dict(getattr(self.client, "script_json_parse_repair_failures_by_stage", {})),
                    ),
                    _sum_int_maps(
                        dict(getattr(self, "_schema_repair_successes_by_stage", {})),
                        dict(getattr(self, "_schema_repair_failures_by_stage", {})),
                    ),
                ),
                "invalid_schema_rate": round(
                    float(
                        int(self._schema_validation_failures)
                        + int(getattr(self.client, "script_json_parse_failures", 0))
                    )
                    / float(max(1, self.client.requests_made)),
                    4,
                ),
                "stuck_abort": False,
                "chunk_subsplit_recoveries": int(getattr(self, "_chunk_subsplit_recoveries", 0)),
                "adaptive_subpart_failures": int(getattr(self, "_adaptive_subpart_failures", 0)),
                "adaptive_subpart_recoveries": int(getattr(self, "_adaptive_subpart_recoveries", 0)),
                "adaptive_subpart_skips": int(getattr(self, "_adaptive_subpart_skips", 0)),
                "continuation_recovery_attempts": int(getattr(self, "_continuation_recovery_attempts", 0)),
                "continuation_recovery_successes": int(
                    getattr(self, "_continuation_recovery_successes", 0)
                ),
                "continuation_fallback_closures": int(
                    getattr(self, "_continuation_fallback_closures", 0)
                ),
                "continuation_fallback_extensions": int(
                    getattr(self, "_continuation_fallback_extensions", 0)
                ),
                "script_completeness_before_repair": dict(completeness_before_repair),
                "script_completeness_after_repair": dict(completeness_after_repair),
                "script_completeness_pass": bool(completeness_after_repair.get("pass", False)),
                "script_completeness_reasons": list(completeness_after_repair.get("reasons", [])),
                "fallback_modes_by_stage": dict(getattr(self, "_fallback_modes_by_stage", {})),
                "parse_truncation_pressure": round(self._current_truncation_pressure(), 4),
                "parse_truncation_pressure_peak": round(
                    float(getattr(self, "_truncation_pressure_peak", 0.0)),
                    4,
                ),
                "truncation_pressure_adaptive_events": int(
                    getattr(self, "_truncation_pressure_adaptive_events", 0)
                ),
                "truncation_pressure_presplit_events": int(
                    getattr(self, "_truncation_pressure_presplit_events", 0)
                ),
                "run_manifest_path": run_manifest_path(
                    checkpoint_dir=self.config.checkpoint_dir,
                    episode_id=resolved_episode_id,
                ),
                "pipeline_summary_path": pipeline_summary_path(
                    checkpoint_dir=self.config.checkpoint_dir,
                    episode_id=resolved_episode_id,
                ),
            }
            run_summary.update(source_validation)
            # Persist rich telemetry for debugging and orchestrated retries.
            _atomic_write_text(run_summary_path, json.dumps(run_summary, indent=2, ensure_ascii=False))
            self.logger.info("script_generation_done", output_path=output_path, word_count=final_wc)
            return ScriptGenerationResult(
                episode_id=resolved_episode_id,
                output_path=output_path,
                line_count=len(lines),
                word_count=final_wc,
                checkpoint_path=store.checkpoint_path,
                run_summary_path=run_summary_path,
                script_retry_rate=run_summary["script_retry_rate"],
                invalid_schema_rate=run_summary["invalid_schema_rate"],
                schema_validation_failures=run_summary["schema_validation_failures"],
            )
        except InterruptedError:
            if state:
                try:
                    # Persist interruption metadata before bubbling up so
                    # orchestrators can resume without ambiguity.
                    state["status"] = "interrupted"
                    state["failure_kind"] = ERROR_KIND_INTERRUPTED
                    state["failed_stage"] = str(getattr(self, "_last_stage", "") or "unknown")
                    state["failed_at"] = int(time.time())
                    state["last_success_at"] = int(time.time())
                    store.save(state)
                except Exception:
                    pass
            failure_summary = {
                "component": "script_generator",
                "episode_id": resolved_episode_id,
                "run_token": run_token,
                "status": "interrupted",
                "script_started_at": int(started),
                "script_completed_at": None,
                "elapsed_seconds": round(time.time() - started, 2),
                "requests_made": self.client.requests_made,
                "estimated_cost_usd": round(self.client.estimated_cost_usd, 4),
                "script_retry_rate": round(
                    float(getattr(self.client, "script_retries_total", 0))
                    / float(max(1, getattr(self.client, "requests_made", 0))),
                    4,
                ),
                "phase_seconds": _phase_seconds_with_generation(phase_seconds),
                "source_validation_mode": self.config.source_validation_mode,
                "schema_validation_failures": int(getattr(self, "_schema_validation_failures", 0)),
                "schema_repair_successes": int(getattr(self, "_schema_repair_successes", 0)),
                "schema_repair_failures": int(getattr(self, "_schema_repair_failures", 0)),
                "schema_salvage_attempts": int(getattr(self, "_schema_salvage_attempts", 0)),
                "schema_salvage_successes": int(getattr(self, "_schema_salvage_successes", 0)),
                "schema_salvage_failures": int(getattr(self, "_schema_salvage_failures", 0)),
                "script_json_parse_failures": int(getattr(self.client, "script_json_parse_failures", 0)),
                "script_empty_output_events": int(getattr(self.client, "script_empty_output_events", 0)),
                "script_empty_output_retries": int(getattr(self.client, "script_empty_output_retries", 0)),
                "script_empty_output_failures": int(getattr(self.client, "script_empty_output_failures", 0)),
                "script_empty_output_by_stage": dict(getattr(self.client, "script_empty_output_by_stage", {})),
                "schema_validation_failures_by_stage": dict(
                    getattr(self, "_schema_validation_failures_by_stage", {})
                ),
                "schema_repair_successes_by_stage": dict(
                    getattr(self, "_schema_repair_successes_by_stage", {})
                ),
                "schema_repair_failures_by_stage": dict(
                    getattr(self, "_schema_repair_failures_by_stage", {})
                ),
                "schema_salvage_attempts_by_stage": dict(
                    getattr(self, "_schema_salvage_attempts_by_stage", {})
                ),
                "schema_salvage_successes_by_stage": dict(
                    getattr(self, "_schema_salvage_successes_by_stage", {})
                ),
                "schema_salvage_failures_by_stage": dict(
                    getattr(self, "_schema_salvage_failures_by_stage", {})
                ),
                "script_json_parse_failures_by_stage": dict(
                    getattr(self.client, "script_json_parse_failures_by_stage", {})
                ),
                "script_json_parse_failures_by_kind": dict(
                    getattr(self.client, "script_json_parse_failures_by_kind", {})
                ),
                "script_json_parse_repair_attempts_by_stage": _sum_int_maps(
                    dict(getattr(self.client, "script_json_parse_repair_successes_by_stage", {})),
                    dict(getattr(self.client, "script_json_parse_repair_failures_by_stage", {})),
                ),
                "script_json_parse_repair_successes_by_kind": dict(
                    getattr(self.client, "script_json_parse_repair_successes_by_kind", {})
                ),
                "script_json_parse_repair_failures_by_kind": dict(
                    getattr(self.client, "script_json_parse_repair_failures_by_kind", {})
                ),
                "script_json_parse_repair_attempts_by_kind": _sum_int_maps(
                    dict(getattr(self.client, "script_json_parse_repair_successes_by_kind", {})),
                    dict(getattr(self.client, "script_json_parse_repair_failures_by_kind", {})),
                ),
                "schema_repair_attempts_by_stage": _sum_int_maps(
                    dict(getattr(self, "_schema_repair_successes_by_stage", {})),
                    dict(getattr(self, "_schema_repair_failures_by_stage", {})),
                ),
                "repair_attempts_by_stage": _sum_int_maps(
                    _sum_int_maps(
                        dict(getattr(self.client, "script_json_parse_repair_successes_by_stage", {})),
                        dict(getattr(self.client, "script_json_parse_repair_failures_by_stage", {})),
                    ),
                    _sum_int_maps(
                        dict(getattr(self, "_schema_repair_successes_by_stage", {})),
                        dict(getattr(self, "_schema_repair_failures_by_stage", {})),
                    ),
                ),
                "invalid_schema_rate": round(
                    float(
                        int(getattr(self, "_schema_validation_failures", 0))
                        + int(getattr(self.client, "script_json_parse_failures", 0))
                    )
                    / float(max(1, self.client.requests_made)),
                    4,
                ),
                "stuck_abort": bool(getattr(self, "_no_progress_abort", False)),
                "failure_kind": ERROR_KIND_INTERRUPTED,
                "failed_stage": str(getattr(self, "_last_stage", "") or "unknown"),
                "chunk_subsplit_recoveries": int(getattr(self, "_chunk_subsplit_recoveries", 0)),
                "adaptive_subpart_failures": int(getattr(self, "_adaptive_subpart_failures", 0)),
                "adaptive_subpart_recoveries": int(getattr(self, "_adaptive_subpart_recoveries", 0)),
                "adaptive_subpart_skips": int(getattr(self, "_adaptive_subpart_skips", 0)),
                "continuation_recovery_attempts": int(getattr(self, "_continuation_recovery_attempts", 0)),
                "continuation_recovery_successes": int(
                    getattr(self, "_continuation_recovery_successes", 0)
                ),
                "continuation_fallback_closures": int(
                    getattr(self, "_continuation_fallback_closures", 0)
                ),
                "continuation_fallback_extensions": int(
                    getattr(self, "_continuation_fallback_extensions", 0)
                ),
                "script_completeness_before_repair": dict(completeness_before_repair),
                "script_completeness_after_repair": dict(completeness_after_repair),
                "script_completeness_pass": bool(completeness_after_repair.get("pass", False)),
                "script_completeness_reasons": list(completeness_after_repair.get("reasons", [])),
                "fallback_modes_by_stage": dict(getattr(self, "_fallback_modes_by_stage", {})),
                "parse_truncation_pressure": round(self._current_truncation_pressure(), 4),
                "parse_truncation_pressure_peak": round(
                    float(getattr(self, "_truncation_pressure_peak", 0.0)),
                    4,
                ),
                "truncation_pressure_adaptive_events": int(
                    getattr(self, "_truncation_pressure_adaptive_events", 0)
                ),
                "truncation_pressure_presplit_events": int(
                    getattr(self, "_truncation_pressure_presplit_events", 0)
                ),
                "truncation_recovery_fallback_used": bool(
                    getattr(self, "_truncation_recovery_fallback_used", False)
                ),
                "run_manifest_path": run_manifest_path(
                    checkpoint_dir=self.config.checkpoint_dir,
                    episode_id=resolved_episode_id,
                ),
                "pipeline_summary_path": pipeline_summary_path(
                    checkpoint_dir=self.config.checkpoint_dir,
                    episode_id=resolved_episode_id,
                ),
            }
            failure_summary.update(source_validation)
            _atomic_write_text(run_summary_path, json.dumps(failure_summary, indent=2, ensure_ascii=False))
            raise
        except Exception as exc:
            # Store a failure summary for post-mortem before re-raising.
            schema_validation_failures = int(getattr(self, "_schema_validation_failures", 0))
            script_json_parse_failures = int(getattr(self.client, "script_json_parse_failures", 0))
            empty_output_failures = int(getattr(self.client, "script_empty_output_failures", 0))
            no_progress_abort = bool(getattr(self, "_no_progress_abort", False))
            failure_kind = ERROR_KIND_UNKNOWN
            if isinstance(exc, ScriptOperationError):
                failure_kind = exc.error_kind
            elif empty_output_failures > 0 and self._is_empty_output_error(exc):
                failure_kind = ERROR_KIND_OPENAI_EMPTY_OUTPUT
            elif no_progress_abort:
                failure_kind = ERROR_KIND_STUCK
            elif empty_output_failures > 0:
                failure_kind = ERROR_KIND_OPENAI_EMPTY_OUTPUT
            elif (schema_validation_failures + script_json_parse_failures) > 0:
                # Group parse/schema drift under a stable invalid-schema bucket
                # for retry policy and incident trend analysis.
                failure_kind = ERROR_KIND_INVALID_SCHEMA
            if state:
                try:
                    state["status"] = "failed"
                    state["failure_kind"] = failure_kind
                    state["failed_stage"] = str(getattr(self, "_last_stage", "") or "unknown")
                    state["failed_at"] = int(time.time())
                    state["last_success_at"] = int(time.time())
                    store.save(state)
                except Exception:
                    pass
            failure_summary = {
                "component": "script_generator",
                "episode_id": resolved_episode_id,
                "run_token": run_token,
                "status": "failed",
                "script_started_at": int(started),
                "script_completed_at": None,
                "elapsed_seconds": round(time.time() - started, 2),
                "requests_made": self.client.requests_made,
                "estimated_cost_usd": round(self.client.estimated_cost_usd, 4),
                "script_retry_rate": round(
                    float(getattr(self.client, "script_retries_total", 0))
                    / float(max(1, getattr(self.client, "requests_made", 0))),
                    4,
                ),
                "phase_seconds": _phase_seconds_with_generation(phase_seconds),
                "source_validation_mode": self.config.source_validation_mode,
                "schema_validation_failures": schema_validation_failures,
                "schema_repair_successes": int(getattr(self, "_schema_repair_successes", 0)),
                "schema_repair_failures": int(getattr(self, "_schema_repair_failures", 0)),
                "schema_salvage_attempts": int(getattr(self, "_schema_salvage_attempts", 0)),
                "schema_salvage_successes": int(getattr(self, "_schema_salvage_successes", 0)),
                "schema_salvage_failures": int(getattr(self, "_schema_salvage_failures", 0)),
                "script_json_parse_failures": script_json_parse_failures,
                "script_empty_output_events": int(getattr(self.client, "script_empty_output_events", 0)),
                "script_empty_output_retries": int(getattr(self.client, "script_empty_output_retries", 0)),
                "script_empty_output_failures": int(getattr(self.client, "script_empty_output_failures", 0)),
                "script_empty_output_by_stage": dict(getattr(self.client, "script_empty_output_by_stage", {})),
                "schema_validation_failures_by_stage": dict(
                    getattr(self, "_schema_validation_failures_by_stage", {})
                ),
                "schema_repair_successes_by_stage": dict(
                    getattr(self, "_schema_repair_successes_by_stage", {})
                ),
                "schema_repair_failures_by_stage": dict(
                    getattr(self, "_schema_repair_failures_by_stage", {})
                ),
                "schema_salvage_attempts_by_stage": dict(
                    getattr(self, "_schema_salvage_attempts_by_stage", {})
                ),
                "schema_salvage_successes_by_stage": dict(
                    getattr(self, "_schema_salvage_successes_by_stage", {})
                ),
                "schema_salvage_failures_by_stage": dict(
                    getattr(self, "_schema_salvage_failures_by_stage", {})
                ),
                "script_json_parse_failures_by_stage": dict(
                    getattr(self.client, "script_json_parse_failures_by_stage", {})
                ),
                "script_json_parse_failures_by_kind": dict(
                    getattr(self.client, "script_json_parse_failures_by_kind", {})
                ),
                "script_json_parse_repair_attempts_by_stage": _sum_int_maps(
                    dict(getattr(self.client, "script_json_parse_repair_successes_by_stage", {})),
                    dict(getattr(self.client, "script_json_parse_repair_failures_by_stage", {})),
                ),
                "script_json_parse_repair_successes_by_kind": dict(
                    getattr(self.client, "script_json_parse_repair_successes_by_kind", {})
                ),
                "script_json_parse_repair_failures_by_kind": dict(
                    getattr(self.client, "script_json_parse_repair_failures_by_kind", {})
                ),
                "script_json_parse_repair_attempts_by_kind": _sum_int_maps(
                    dict(getattr(self.client, "script_json_parse_repair_successes_by_kind", {})),
                    dict(getattr(self.client, "script_json_parse_repair_failures_by_kind", {})),
                ),
                "schema_repair_attempts_by_stage": _sum_int_maps(
                    dict(getattr(self, "_schema_repair_successes_by_stage", {})),
                    dict(getattr(self, "_schema_repair_failures_by_stage", {})),
                ),
                "repair_attempts_by_stage": _sum_int_maps(
                    _sum_int_maps(
                        dict(getattr(self.client, "script_json_parse_repair_successes_by_stage", {})),
                        dict(getattr(self.client, "script_json_parse_repair_failures_by_stage", {})),
                    ),
                    _sum_int_maps(
                        dict(getattr(self, "_schema_repair_successes_by_stage", {})),
                        dict(getattr(self, "_schema_repair_failures_by_stage", {})),
                    ),
                ),
                "invalid_schema_rate": round(
                    float(
                        schema_validation_failures + script_json_parse_failures
                    )
                    / float(max(1, self.client.requests_made)),
                    4,
                ),
                "stuck_abort": no_progress_abort,
                "failure_kind": failure_kind,
                "failed_stage": str(getattr(self, "_last_stage", "") or "unknown"),
                "chunk_subsplit_recoveries": int(getattr(self, "_chunk_subsplit_recoveries", 0)),
                "adaptive_subpart_failures": int(getattr(self, "_adaptive_subpart_failures", 0)),
                "adaptive_subpart_recoveries": int(getattr(self, "_adaptive_subpart_recoveries", 0)),
                "adaptive_subpart_skips": int(getattr(self, "_adaptive_subpart_skips", 0)),
                "continuation_recovery_attempts": int(getattr(self, "_continuation_recovery_attempts", 0)),
                "continuation_recovery_successes": int(
                    getattr(self, "_continuation_recovery_successes", 0)
                ),
                "continuation_fallback_closures": int(
                    getattr(self, "_continuation_fallback_closures", 0)
                ),
                "continuation_fallback_extensions": int(
                    getattr(self, "_continuation_fallback_extensions", 0)
                ),
                "script_completeness_before_repair": dict(completeness_before_repair),
                "script_completeness_after_repair": dict(completeness_after_repair),
                "script_completeness_pass": bool(completeness_after_repair.get("pass", False)),
                "script_completeness_reasons": list(completeness_after_repair.get("reasons", [])),
                "fallback_modes_by_stage": dict(getattr(self, "_fallback_modes_by_stage", {})),
                "parse_truncation_pressure": round(self._current_truncation_pressure(), 4),
                "parse_truncation_pressure_peak": round(
                    float(getattr(self, "_truncation_pressure_peak", 0.0)),
                    4,
                ),
                "truncation_pressure_adaptive_events": int(
                    getattr(self, "_truncation_pressure_adaptive_events", 0)
                ),
                "truncation_pressure_presplit_events": int(
                    getattr(self, "_truncation_pressure_presplit_events", 0)
                ),
                "truncation_recovery_fallback_used": bool(
                    getattr(self, "_truncation_recovery_fallback_used", False)
                ),
                "run_manifest_path": run_manifest_path(
                    checkpoint_dir=self.config.checkpoint_dir,
                    episode_id=resolved_episode_id,
                ),
                "pipeline_summary_path": pipeline_summary_path(
                    checkpoint_dir=self.config.checkpoint_dir,
                    episode_id=resolved_episode_id,
                ),
            }
            failure_summary.update(source_validation)
            _atomic_write_text(run_summary_path, json.dumps(failure_summary, indent=2, ensure_ascii=False))
            raise
        finally:
            store.release_lock()

