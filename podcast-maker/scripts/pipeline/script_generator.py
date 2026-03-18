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
    ERROR_KIND_SCRIPT_QUALITY,
    ERROR_KIND_SCRIPT_COMPLETENESS,
    ERROR_KIND_SOURCE_TOO_SHORT,
    ERROR_KIND_STUCK,
    ERROR_KIND_UNKNOWN,
    ScriptOperationError,
)
from .dialogue_drafter import DialogueDrafter
from .editorial_gate import EditorialGate
from .editorial_rewriter import EditorialRewriter
from .episode_planner import EpisodePlanner
from .evidence_map import EvidenceMapBuilder
from .fact_guard import FactGuard
from .logging_utils import Logger
from .openai_client import OpenAIClient
from .podcast_artifacts import (
    RESUME_COMPAT_VERSION,
    build_script_artifact,
    build_script_artifact_paths,
    build_public_script_payload,
    public_lines_from_script_artifact,
    read_json_artifact,
    rewrite_round_path,
    validate_editorial_report,
    validate_episode_plan,
    validate_evidence_map,
    validate_fact_guard_report,
    validate_script_artifact,
    write_json_artifact,
    write_script_payload,
)
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
from .structural_gate import StructuralGate


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
    quality_report_path: str = ""
    artifact_paths: Dict[str, str] | None = None


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
        return _env_bool("TTS_SPEED_HINTS_ENABLED", True)

    def _line_schema_fields_prompt(self) -> str:
        """Describe expected JSON line keys for current runtime mode."""
        if self._tts_speed_hints_enabled():
            return "speaker, role, instructions, pace_hint, text"
        return "speaker, role, instructions, text"

    def _pace_hint_prompt_guidance(self) -> str:
        """Prompt block for optional pace_hint generation when enabled."""
        if not self._tts_speed_hints_enabled():
            return ""
        return (
            "- Required field `pace_hint`: calm|steady|brisk|null.\n"
            "- Keep delivery dynamic: do not keep all lines as `steady`.\n"
            "- For scripts with >= 8 lines, include at least one `brisk` and one `calm` where narratively natural.\n"
            "- Prefer `brisk` for energetic openings, transitions, and actionable calls.\n"
            "- Prefer `calm` for nuanced explanations, cautions, and closing recap/farewell.\n"
            "- Keep adjacent turns coherent; avoid abrupt oscillations unless there is a clear narrative shift.\n"
            "- Avoid long flat runs: do not keep more than 3 consecutive lines with the same non-null pace_hint unless justified.\n"
            "- If there is no clear pace signal, set `pace_hint` to null."
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

    def _beat_has_close_language(self, *, turns: List[Dict[str, Any]]) -> bool:
        """Return True when turns already sound like a closing recap."""
        if not turns:
            return False
        joined = " ".join(str(turn.get("text") or "").lower() for turn in turns)
        close_tokens = (
            "al final",
            "la idea es simple",
            "nos quedamos con",
            "en resumen",
            "gracias por escuch",
            "nos vemos",
            "bundle de depuracion",
            "bundle de depuración",
        )
        hits = sum(1 for token in close_tokens if token in joined)
        return hits >= 2

    def _select_underlength_expansion_beat(
        self,
        *,
        script_artifact: Dict[str, Any],
        episode_plan: Dict[str, Any],
    ) -> str:
        """Pick deterministic beat target for in-place underlength expansion."""
        beats = [beat for beat in list(episode_plan.get("beats", []) or []) if isinstance(beat, dict)]
        if not beats:
            return ""
        turns = [dict(turn) for turn in list(script_artifact.get("turns", []) or []) if isinstance(turn, dict)]
        coverage = dict(script_artifact.get("coverage", {}) or {})
        line_counts_by_beat = dict(coverage.get("line_counts_by_beat", {}) or {})
        for beat in beats:
            beat_id = str(beat.get("beat_id", "") or "").strip()
            if not beat_id:
                continue
            min_turns = 2 if int(beat.get("target_words", 0) or 0) >= 45 else 1
            if int(line_counts_by_beat.get(beat_id, 0) or 0) < min_turns:
                return beat_id
        deficits: List[tuple[int, int, str]] = []
        for idx, beat in enumerate(beats):
            beat_id = str(beat.get("beat_id", "") or "").strip()
            if not beat_id:
                continue
            beat_turns = [turn for turn in turns if str(turn.get("beat_id", "")).strip() == beat_id]
            beat_words = sum(count_words_from_lines([turn]) for turn in beat_turns)
            deficit = max(0, int(beat.get("target_words", 0) or 0) - beat_words)
            deficits.append((deficit, -idx, beat_id))
        positive_deficits = [item for item in deficits if item[0] > 0]
        if positive_deficits:
            positive_deficits.sort(reverse=True)
            return positive_deficits[0][2]
        closing_beat_id = str(beats[-1].get("beat_id", "") or "").strip()
        closing_turns = [turn for turn in turns if str(turn.get("beat_id", "")).strip() == closing_beat_id]
        if closing_beat_id and self._beat_has_close_language(turns=closing_turns):
            for beat in beats:
                beat_id = str(beat.get("beat_id", "") or "").strip()
                if beat_id and beat_id != closing_beat_id:
                    return beat_id
        return str(beats[0].get("beat_id", "") or "").strip()

    def _build_contextual_beat_expansion_prompt(
        self,
        *,
        script_artifact: Dict[str, Any],
        episode_plan: Dict[str, Any],
        source_context: str,
        target_beat_id: str,
        min_words: int,
        max_words: int,
    ) -> str:
        """Build contextual prompt to enrich one beat without appending a new ending."""
        turns = [dict(turn) for turn in list(script_artifact.get("turns", []) or []) if isinstance(turn, dict)]
        target_turns = [turn for turn in turns if str(turn.get("beat_id", "")).strip() == target_beat_id]
        target_beat = next(
            (dict(beat) for beat in list(episode_plan.get("beats", []) or []) if str(beat.get("beat_id", "")).strip() == target_beat_id),
            {},
        )
        target_indexes = [
            idx for idx, turn in enumerate(turns) if str(turn.get("beat_id", "")).strip() == target_beat_id
        ]
        before_context = []
        after_context = []
        if target_indexes:
            start_idx = target_indexes[0]
            end_idx = target_indexes[-1]
            before_context = turns[max(0, start_idx - 2):start_idx]
            after_context = turns[end_idx + 1:end_idx + 3]
        current_words = count_words_from_lines(list(script_artifact.get("lines", []) or []))
        remaining_to_min = max(0, int(min_words) - int(current_words))
        source_snippet = self._compact_source_context(
            source_context,
            max_chars=CONTEXTUAL_FALLBACK_SOURCE_MAX_CHARS,
        )
        line_fields = self._line_schema_fields_prompt()
        pace_hint_guidance = self._pace_hint_prompt_guidance()
        return textwrap.dedent(
            f"""
            Expand ONLY one beat inside this podcast script.
            Return ONLY JSON with key "lines" and fields: {line_fields}.

            Hard constraints:
            - Add 1-3 NEW lines that belong inside beat `{target_beat_id}`.
            - These lines will be inserted before the next beat, not appended as a new ending.
            - Do not add recap, farewell, big synthesis, or "al final" style closing language.
            - Do not restate the full thesis; add a new move inside this beat: example, objection, cost, consequence, grounding, or decision.
            - Preserve factual consistency with source context and current beat claims.
            - If the beat asks for cost or consequence but the source context does not state that effect directly, switch to grounding, example, tradeoff, or decision instead.
            - Keep rules, defaults, thresholds, paths, and presets procedural; do not upgrade them into unsupported downstream impact.
            - Keep Host1/Host2 alternation and keep spoken text in Spanish.
            - Keep instructions in English as short, actionable delivery guidance.
            {pace_hint_guidance}
            - Keep each line concise (1-2 sentences).
            - Avoid stock openers and transition filler.
            - Prefer factual paraphrases such as "sirve para", "indica", "te deja", "se usa cuando", or "marca el umbral".
            - Avoid unsupported effect phrasing such as "causa", "encarece", "sale mas caro", "siempre hay alguien", or "te rompe".
            - Do not turn "strict" or production presets into superlatives unless the source says that verbatim.
            - Split a risky idea into separate lines if one clause is factual and another clause is inferential.

            Expansion target:
            - Current total words: {current_words}
            - Remaining words to minimum target: {remaining_to_min}
            - Focus beat: {json.dumps(target_beat, ensure_ascii=False, indent=2)}

            PREVIOUS CONTEXT:
            {json.dumps({"turns": before_context}, ensure_ascii=False, indent=2)}

            TARGET BEAT CURRENT TURNS:
            {json.dumps({"turns": target_turns}, ensure_ascii=False, indent=2)}

            NEXT CONTEXT:
            {json.dumps({"turns": after_context}, ensure_ascii=False, indent=2)}

            SOURCE CONTEXT:
            {source_snippet if source_snippet else "(not provided)"}
            """
        ).strip()

    def _insert_lines_into_beat(
        self,
        *,
        script_artifact: Dict[str, Any],
        episode_plan: Dict[str, Any],
        beat_id: str,
        new_lines: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Insert validated lines after the last turn of one beat and rebuild artifact."""
        target_beat_id = str(beat_id or "").strip()
        turns = [dict(turn) for turn in list(script_artifact.get("turns", []) or []) if isinstance(turn, dict)]
        if not turns or not target_beat_id:
            return script_artifact
        insert_after = max(
            (idx for idx, turn in enumerate(turns) if str(turn.get("beat_id", "")).strip() == target_beat_id),
            default=-1,
        )
        if insert_after < 0:
            return script_artifact
        beat_meta = next(
            (dict(beat) for beat in list(episode_plan.get("beats", []) or []) if str(beat.get("beat_id", "")).strip() == target_beat_id),
            {},
        )
        can_cut = bool(beat_meta.get("can_cut", False))
        validated_lines = validate_script_payload({"lines": list(new_lines or [])}).get("lines", [])
        if not validated_lines:
            return script_artifact
        raw_turns = list(turns)
        insert_idx = insert_after + 1
        for line in validated_lines:
            raw_turns.insert(
                insert_idx,
                {
                    **line,
                    "beat_id": target_beat_id,
                    "can_cut": can_cut,
                },
            )
            insert_idx += 1
        rebuilt_lines: List[Dict[str, Any]] = []
        for turn in raw_turns:
            payload = {
                "speaker": str(turn.get("speaker", "")).strip(),
                "role": str(turn.get("role", "")).strip(),
                "instructions": str(turn.get("instructions", "")).strip(),
                "text": str(turn.get("text", "")).strip(),
            }
            pace_hint = str(turn.get("pace_hint", "") or "").strip()
            if pace_hint:
                payload["pace_hint"] = pace_hint
            rebuilt_lines.append(payload)
        rebuilt_lines = harden_script_structure(
            rebuilt_lines,
            max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
        )
        return build_script_artifact(
            stage=str(script_artifact.get("stage") or "rewritten"),
            episode_id=str(script_artifact.get("episode_id") or ""),
            run_token=str(script_artifact.get("run_token") or ""),
            source_digest=str(script_artifact.get("source_digest") or ""),
            plan_ref=str(script_artifact.get("plan_ref") or ""),
            plan_digest=str(script_artifact.get("plan_digest") or ""),
            lines=rebuilt_lines,
            episode_plan=episode_plan,
            raw_turns=raw_turns,
            prior_artifact=script_artifact,
            target_word_count=script_artifact.get("target_word_count"),
        )

    def _harden_script_artifact(
        self,
        *,
        script_artifact: Dict[str, Any],
        episode_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize rewritten/finalized lines before writing or evaluating."""
        hardened_lines = harden_script_structure(
            list(script_artifact.get("lines", []) or []),
            max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
        )
        return build_script_artifact(
            stage=str(script_artifact.get("stage") or "rewritten"),
            episode_id=str(script_artifact.get("episode_id") or ""),
            run_token=str(script_artifact.get("run_token") or ""),
            source_digest=str(script_artifact.get("source_digest") or ""),
            plan_ref=str(script_artifact.get("plan_ref") or ""),
            plan_digest=str(script_artifact.get("plan_digest") or ""),
            lines=hardened_lines,
            episode_plan=episode_plan,
            prior_artifact=script_artifact,
            target_word_count=script_artifact.get("target_word_count"),
        )

    def _request_contextual_beat_expansion_lines(
        self,
        *,
        script_artifact: Dict[str, Any],
        episode_plan: Dict[str, Any],
        min_words: int,
        max_words: int,
        source_context: str,
        target_beat_id: str,
        continuation_stage: str,
    ) -> List[Dict[str, Any]]:
        """Ask LLM for beat-local expansion lines instead of tail padding."""
        if not self._can_use_contextual_fallback_llm() or not str(target_beat_id or "").strip():
            return []
        try:
            prompt = self._build_contextual_beat_expansion_prompt(
                script_artifact=script_artifact,
                episode_plan=episode_plan,
                source_context=source_context,
                target_beat_id=target_beat_id,
                min_words=min_words,
                max_words=max_words,
            )
            stage = f"{continuation_stage}_contextual_beat_expand"
            max_output_tokens = max(500, min(1800, int(self.config.max_output_tokens_continuation)))
            repaired_raw = self.client.generate_script_json(
                prompt=prompt,
                schema=SCRIPT_JSON_SCHEMA,
                max_output_tokens=max_output_tokens,
                stage=stage,
            )
            candidate_lines = validate_script_payload(repaired_raw)["lines"]
            filtered_lines: List[Dict[str, Any]] = []
            seen_texts: set[str] = set()
            for line in candidate_lines:
                text = str(line.get("text", "") or "").strip()
                normalized = re.sub(r"\s+", " ", text.lower())
                if not text:
                    continue
                if any(token in normalized for token in ("gracias por escuch", "nos vemos", "al final", "en resumen", "nos quedamos con")):
                    continue
                if normalized in seen_texts:
                    continue
                seen_texts.add(normalized)
                filtered_lines.append(line)
            return filtered_lines
        except Exception as exc:  # noqa: BLE001
            self.logger.warn(
                "continuation_contextual_beat_expand_failed",
                stage=continuation_stage,
                beat_id=target_beat_id,
                error=str(exc),
            )
            return []

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
        suggested_target_minutes = max(
            1.0,
            float(source_word_count) / float(max(80.0, self.config.words_per_min) * max(0.05, self.config.source_validation_enforce_ratio)),
        )
        suggested_profile = "short"
        if suggested_target_minutes >= 25.0:
            suggested_profile = "long"
        elif suggested_target_minutes >= 10.0:
            suggested_profile = "standard"

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
                        f"(recommended ~{recommended_min_words}). "
                        f"Suggested fallback: profile `{suggested_profile}` or target_minutes <= {round(suggested_target_minutes, 1)}."
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
        phase_cursor = str(state.get("phase_cursor", "") or "").strip().lower()
        phase_status = dict(state.get("phase_status", {}) or {})
        has_pipeline_progress = bool(phase_status) or phase_cursor not in {"", "startup"}
        if resume and has_pipeline_progress and not resume_force:
            raise ScriptOperationError(
                "Resume blocked because checkpoint is missing generation_source. "
                "Use --resume-force to recompute and continue.",
                error_kind=ERROR_KIND_RESUME_BLOCKED,
            )
        if resume and has_pipeline_progress and resume_force:
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
        run_token: str,
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
                            run_token=run_token,
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
                phase_cursor=existing.get("phase_cursor", "startup"),
                completed_phases=len(
                    [
                        name
                        for name, status in dict(existing.get("phase_status", {}) or {}).items()
                        if str(status).strip().lower() == "completed"
                    ]
                ),
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
        initial = store.create_initial_state(
            source_hash=source_hash,
            config_fingerprint=cfg_fp,
            run_token=run_token,
        )
        store.save(initial)
        return initial

    def _artifact_identity_matches(
        self,
        *,
        payload: Dict[str, Any],
        expected: Dict[str, Any] | None,
    ) -> bool:
        if not expected:
            return True
        for key, value in dict(expected or {}).items():
            expected_value = str(value or "").strip()
            if not expected_value:
                continue
            if str(payload.get(key, "") or "").strip() != expected_value:
                return False
        return True

    def _load_validated_artifact(
        self,
        *,
        state: Dict[str, Any],
        phase_name: str,
        path: str,
        validator: Callable[[Dict[str, Any]], Dict[str, Any]],
        expected_identity: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """Load one artifact only when phase state marks it as completed."""
        phase_status = dict(state.get("phase_status", {}) or {})
        if str(phase_status.get(phase_name, "")).strip().lower() != "completed":
            return None
        if not os.path.exists(path):
            return None
        try:
            payload = read_json_artifact(path)
            validated = validator(payload)
            if not self._artifact_identity_matches(payload=validated, expected=expected_identity):
                self.logger.warn("artifact_resume_identity_mismatch", phase=phase_name, path=path)
                return None
            return validated
        except Exception as exc:  # noqa: BLE001
            self.logger.warn("artifact_resume_reload_failed", phase=phase_name, path=path, error=str(exc))
            return None

    def _mark_phase_state(
        self,
        *,
        state: Dict[str, Any],
        store: ScriptCheckpointStore,
        phase_name: str,
        status: str,
    ) -> None:
        phase_status = dict(state.get("phase_status", {}) or {})
        phase_status[str(phase_name)] = str(status)
        state["phase_status"] = phase_status
        state["phase_cursor"] = str(phase_name)
        state["last_success_at"] = int(time.time())
        store.save(state)

    def _build_quality_report_payload(
        self,
        *,
        output_path: str,
        final_artifact: Dict[str, Any],
        structural_report: Dict[str, Any],
        editorial_report: Dict[str, Any],
        fact_guard_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        final_lines = list(final_artifact.get("lines", []) or [])
        editorial_scores = dict(editorial_report.get("scores", {}) or {})
        reasons: List[str] = []
        reasons.extend(str(item).strip() for item in list(structural_report.get("notes", []) or []) if str(item).strip())
        for failure in list(editorial_report.get("failures", []) or []):
            if isinstance(failure, dict):
                reason = str(failure.get("failure_type", "") or "").strip()
                if reason:
                    reasons.append(reason)
        for issue in list(fact_guard_report.get("issues", []) or []):
            if isinstance(issue, dict):
                issue_type = str(issue.get("issue_type", "") or "").strip()
                if issue_type:
                    reasons.append(issue_type)
        dedup_reasons: List[str] = []
        for reason in reasons:
            if reason and reason not in dedup_reasons:
                dedup_reasons.append(reason)
        fact_guard_blocking = self._fact_guard_blocks(fact_guard_report)
        coverage = dict(final_artifact.get("coverage", {}) or {})
        passed = bool(
            structural_report.get("pass", False)
            and editorial_report.get("pass", False)
            and not fact_guard_blocking
        )
        return {
            "component": "script_quality_gate",
            "resume_compat_version": RESUME_COMPAT_VERSION,
            "status": "passed" if passed else "failed",
            "pass": passed,
            "action": "enforce",
            "evaluator": "editorial_pipeline",
            "profile": self.config.profile_name,
            "run_token": final_artifact.get("run_token", ""),
            "source_digest": final_artifact.get("source_digest", ""),
            "plan_digest": final_artifact.get("plan_digest", ""),
            "internal_artifact_digest": final_artifact.get("internal_artifact_digest", ""),
            "public_payload_digest": final_artifact.get("public_payload_digest", ""),
            "script_path": output_path,
            "line_count": len(final_lines),
            "word_count": count_words_from_lines(final_lines),
            "rules": dict(structural_report.get("checks", {}) or {}),
            "scores": {
                "overall_score": editorial_scores.get("listener_engagement"),
                "cadence_score": editorial_scores.get("orality"),
                "logic_score": editorial_scores.get("progression"),
                "clarity_score": editorial_scores.get("density_control"),
                "editorial_scores": editorial_scores,
            },
            "reasons_structural": list(structural_report.get("notes", []) or []),
            "reasons_llm": [str(item.get("failure_type")) for item in list(editorial_report.get("failures", []) or []) if isinstance(item, dict)],
            "llm_score_failures": [],
            "evidence_structural": {
                "failed_rules": [
                    name for name, ok in dict(structural_report.get("checks", {}) or {}).items() if not bool(ok)
                ],
                "structural_report": structural_report,
                "editorial_deterministic_metrics": dict(editorial_report.get("deterministic_metrics", {}) or {}),
            },
            "llm_called": True,
            "llm_sampled": True,
            "llm_error": False,
            "llm_explicit_fail": not bool(editorial_report.get("pass", False)),
            "llm_editorial_fail": not bool(editorial_report.get("pass", False)),
            "llm_truncation_claims_filtered": 0,
            "llm_rule_judgments": {},
            "llm_rule_judgments_applied": False,
            "llm_rule_judgments_confidence": None,
            "llm_degraded_to_rules": False,
            "semantic_rule_fallback": {"enabled": False},
            "source_topic_balance": {},
            "coverage": coverage,
            "fact_guard_warning_count": sum(
                1
                for issue in list(fact_guard_report.get("issues", []) or [])
                if isinstance(issue, dict) and str(issue.get("action", "")).strip().lower() not in {"block", "rewrite_local"}
            ),
            "fact_guard_repairable_count": sum(
                1
                for issue in list(fact_guard_report.get("issues", []) or [])
                if isinstance(issue, dict) and str(issue.get("action", "")).strip().lower() == "rewrite_local"
            ),
            "fact_guard_blocking": fact_guard_blocking,
            "hard_fail_eligible": not passed,
            "score_blocking_enabled": True,
            "score_blocking_critical_threshold": 3.6,
            "score_blocking_critical_failed": not passed,
            "hard_fail_structural_only_rollout": False,
            "editorial_warn_only": False,
            "reasons": dedup_reasons,
            "failure_kind": (ERROR_KIND_SCRIPT_QUALITY if not passed else None),
            "structural_report_path": "",
            "editorial_report_path": "",
            "fact_guard_report_path": "",
            "script_quality_report_path": "",
        }

    def _fact_guard_blocks(self, report: Dict[str, Any]) -> bool:
        """Only high-severity or explicit block actions stop the pipeline."""
        if bool(report.get("pass", False)):
            return False
        stage_name = str(report.get("stage", "") or "").strip().lower()
        if stage_name == "final" and list(report.get("issues", []) or []):
            return True
        for issue in list(report.get("issues", []) or []):
            if not isinstance(issue, dict):
                continue
            if str(issue.get("action", "")).strip().lower() == "block":
                return True
            if str(issue.get("severity", "")).strip().lower() == "high":
                return True
        return False

    def _fact_guard_blocking_issue_count(self, report: Dict[str, Any]) -> int:
        count = 0
        for issue in list(report.get("issues", []) or []):
            if not isinstance(issue, dict):
                continue
            action = str(issue.get("action", "")).strip().lower()
            severity = str(issue.get("severity", "")).strip().lower()
            if action == "block" or severity == "high":
                count += 1
        return count

    def _fact_guard_severity_score(self, report: Dict[str, Any]) -> int:
        weights = {"low": 1, "medium": 2, "high": 3}
        return sum(
            weights.get(str(issue.get("severity", "")).strip().lower(), 0)
            for issue in list(report.get("issues", []) or [])
            if isinstance(issue, dict)
        )

    def _fact_guard_report_escalated(self, *, before: Dict[str, Any], after: Dict[str, Any]) -> bool:
        if self._fact_guard_blocking_issue_count(after) > self._fact_guard_blocking_issue_count(before):
            return True
        if self._fact_guard_severity_score(after) > self._fact_guard_severity_score(before):
            return True
        return len(list(after.get("issues", []) or [])) > len(list(before.get("issues", []) or []))

    def _generate_redesigned(
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
        """Execute the redesigned planner->drafter->rewriter pipeline."""
        source = source_text.strip()
        if not source:
            raise RuntimeError("Input source is empty. Provide enough content to build a podcast.")

        min_words = self.config.min_words
        max_words = max(self.config.max_words, min_words)
        resolved_episode_id = resolve_episode_id(output_path=output_path, override=episode_id)
        store = ScriptCheckpointStore(
            base_dir=self.config.checkpoint_dir,
            episode_id=resolved_episode_id,
            reliability=self.reliability,
        )
        store.acquire_lock(force_unlock=force_unlock)
        started = time.time()
        run_summary_path = os.path.join(store.run_dir, "run_summary.json")
        effective_run_token = str(run_token or f"run_{int(started)}").strip()
        source_word_count = len(source.split())
        phase_seconds: Dict[str, float] = {
            "pre_summary": 0.0,
            "evidence_map": 0.0,
            "episode_plan": 0.0,
            "draft_dialogue": 0.0,
            "fact_guard_draft": 0.0,
            "editorial_rewrite": 0.0,
            "editorial_gate": 0.0,
            "fact_guard_final": 0.0,
            "postprocess": 0.0,
            "chunk_generation": 0.0,
            "continuations": 0.0,
            "truncation_recovery": 0.0,
        }
        source_validation = {
            "source_word_count": source_word_count,
            "target_word_range": [int(min_words), int(max_words)],
            "target_word_count": max(min_words, int((min_words + max_words) / 2)),
            "source_to_target_ratio": 0.0,
            "source_validation_status": "ok",
            "source_validation_reason": "",
            "source_validation_blocked": False,
            "source_validation_blocked_message": "",
            "source_recommended_min_words": 0,
            "source_required_min_words": 0,
        }
        state: Dict[str, Any] = {}
        quality_report_path = ""
        artifact_paths: Dict[str, str] = {}
        try:
            cfg_fp = config_fingerprint(
                script_cfg=self.config,
                reliability_cfg=self.reliability,
                extra={"component": "script_generator_redesigned"},
            )
            source_digest = content_hash(source)
            state = self._load_or_init_state(
                store=store,
                source_hash=source_digest,
                cfg_fp=cfg_fp,
                run_token=effective_run_token,
                resume=resume,
                resume_force=resume_force,
            )
            existing_run_token = str(state.get("run_token", "") or "").strip()
            if existing_run_token:
                effective_run_token = existing_run_token
            else:
                state["run_token"] = effective_run_token
            state["resume_compat_version"] = RESUME_COMPAT_VERSION
            artifact_paths = build_script_artifact_paths(run_dir=store.run_dir)
            state["artifact_paths"] = artifact_paths
            if "phase_status" not in state or not isinstance(state.get("phase_status"), dict):
                state["phase_status"] = {}
            store.save(state)
            if resume and str(state.get("status", "")).strip().lower() == "completed" and os.path.exists(output_path):
                existing_quality_report_path = str(
                    state.get("quality_report_path")
                    or state.get("script_quality_report_path")
                    or artifact_paths.get("quality_report", "")
                ).strip()
                existing_summary: Dict[str, Any] = {}
                if os.path.exists(run_summary_path):
                    try:
                        existing_summary = read_json_artifact(run_summary_path)
                    except Exception:  # noqa: BLE001
                        existing_summary = {}
                return ScriptGenerationResult(
                    episode_id=resolved_episode_id,
                    output_path=output_path,
                    line_count=int(existing_summary.get("line_count", len(list(state.get("lines", []) or [])))),
                    word_count=int(existing_summary.get("word_count", int(state.get("current_word_count", 0) or 0))),
                    checkpoint_path=store.checkpoint_path,
                    run_summary_path=run_summary_path,
                    script_retry_rate=float(existing_summary.get("script_retry_rate", 0.0) or 0.0),
                    invalid_schema_rate=float(existing_summary.get("invalid_schema_rate", 0.0) or 0.0),
                    schema_validation_failures=int(existing_summary.get("schema_validation_failures", 0) or 0),
                    quality_report_path=existing_quality_report_path,
                    artifact_paths=artifact_paths,
                )
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
            pre_summary_started = time.time()
            self._last_stage = "pre_summary"
            source_for_generation = self._resolve_generation_source(
                source=source,
                state=state,
                store=store,
                resume=resume,
                resume_force=resume_force,
                cancel_check=cancel_check,
            )
            phase_seconds["pre_summary"] = round(time.time() - pre_summary_started, 3)

            evidence_map = self._load_validated_artifact(
                state=state,
                phase_name="evidence_map",
                path=artifact_paths["evidence_map"],
                validator=validate_evidence_map,
                expected_identity={
                    "episode_id": resolved_episode_id,
                    "run_token": effective_run_token,
                    "source_digest": source_digest,
                },
            )
            if evidence_map is None:
                self._last_stage = "evidence_map"
                phase_started = time.time()
                evidence_map = EvidenceMapBuilder(client=self.client, logger=self.logger).build(
                    source_text=source,
                    source_digest=source_digest,
                    episode_id=resolved_episode_id,
                    run_token=effective_run_token,
                    target_minutes=self.config.target_minutes,
                    chunk_target_minutes=self.config.chunk_target_minutes,
                    words_per_min=self.config.words_per_min,
                )
                write_json_artifact(path=artifact_paths["evidence_map"], payload=evidence_map)
                phase_seconds["evidence_map"] = round(time.time() - phase_started, 3)
                self._mark_phase_state(state=state, store=store, phase_name="evidence_map", status="completed")

            episode_plan = self._load_validated_artifact(
                state=state,
                phase_name="episode_plan",
                path=artifact_paths["episode_plan"],
                validator=validate_episode_plan,
                expected_identity={
                    "episode_id": resolved_episode_id,
                    "run_token": effective_run_token,
                },
            )
            if episode_plan is None:
                self._last_stage = "episode_plan"
                phase_started = time.time()
                episode_plan = EpisodePlanner(client=self.client, logger=self.logger).build(
                    evidence_map=evidence_map,
                    episode_id=resolved_episode_id,
                    run_token=effective_run_token,
                    profile_name=self.config.profile_name,
                    min_words=min_words,
                    max_words=max_words,
                )
                write_json_artifact(path=artifact_paths["episode_plan"], payload=episode_plan)
                phase_seconds["episode_plan"] = round(time.time() - phase_started, 3)
                self._mark_phase_state(state=state, store=store, phase_name="episode_plan", status="completed")
            plan_digest = content_hash(canonical_json(episode_plan))
            state["plan_digest"] = plan_digest

            draft_artifact = self._load_validated_artifact(
                state=state,
                phase_name="draft_dialogue",
                path=artifact_paths["draft_script"],
                validator=lambda payload: validate_script_artifact(
                    payload,
                    expected_stage="draft",
                    episode_plan=episode_plan,
                ),
                expected_identity={
                    "episode_id": resolved_episode_id,
                    "run_token": effective_run_token,
                    "source_digest": source_digest,
                    "plan_digest": plan_digest,
                },
            )
            if draft_artifact is None:
                self._last_stage = "draft_dialogue"
                phase_started = time.time()
                draft_artifact = DialogueDrafter(client=self.client, logger=self.logger).draft(
                    evidence_map=evidence_map,
                    episode_plan=episode_plan,
                    episode_id=resolved_episode_id,
                    run_token=effective_run_token,
                    source_digest=source_digest,
                    plan_digest=plan_digest,
                    profile_name=self.config.profile_name,
                    min_words=min_words,
                    max_words=max_words,
                )
                write_json_artifact(path=artifact_paths["draft_script_raw"], payload=draft_artifact)
                phase_seconds["draft_dialogue"] = round(time.time() - phase_started, 3)
                self._mark_phase_state(state=state, store=store, phase_name="draft_dialogue", status="completed")

            fact_guard = FactGuard(client=self.client, logger=self.logger)
            fact_guard_draft_report = self._load_validated_artifact(
                state=state,
                phase_name="fact_guard_draft",
                path=artifact_paths["fact_guard_report_draft"],
                validator=validate_fact_guard_report,
                expected_identity={
                    "run_token": effective_run_token,
                    "source_digest": source_digest,
                    "plan_digest": plan_digest,
                    "internal_artifact_digest": str(draft_artifact.get("internal_artifact_digest", "")),
                },
            )
            if fact_guard_draft_report is None:
                self._last_stage = "fact_guard_draft"
                phase_started = time.time()
                fact_guard_draft_report = fact_guard.validate(
                    script_artifact=draft_artifact,
                    evidence_map=evidence_map,
                    episode_plan=episode_plan,
                    stage_name="draft",
                )
                if not bool(fact_guard_draft_report.get("pass", False)):
                    draft_artifact = fact_guard.repair(
                        script_artifact=draft_artifact,
                        evidence_map=evidence_map,
                        episode_plan=episode_plan,
                        report=fact_guard_draft_report,
                        stage_name="draft",
                    )
                    fact_guard_draft_report = fact_guard.validate(
                        script_artifact=draft_artifact,
                        evidence_map=evidence_map,
                        episode_plan=episode_plan,
                        stage_name="draft",
                    )
                write_json_artifact(path=artifact_paths["draft_script_fact_checked"], payload=draft_artifact)
                write_json_artifact(path=artifact_paths["draft_script"], payload=draft_artifact)
                write_json_artifact(
                    path=artifact_paths["fact_guard_report_draft"],
                    payload=fact_guard_draft_report,
                )
                phase_seconds["fact_guard_draft"] = round(time.time() - phase_started, 3)
                self._mark_phase_state(state=state, store=store, phase_name="fact_guard_draft", status="completed")
            else:
                write_json_artifact(path=artifact_paths["draft_script"], payload=draft_artifact)
            if self._fact_guard_blocks(fact_guard_draft_report):
                raise ScriptOperationError(
                    "Draft failed factual validation",
                    error_kind=ERROR_KIND_SCRIPT_QUALITY,
                )

            rewritten_artifact = self._load_validated_artifact(
                state=state,
                phase_name="editorial_rewrite",
                path=artifact_paths["rewritten_script"],
                validator=lambda payload: validate_script_artifact(
                    payload,
                    expected_stage="rewritten",
                    episode_plan=episode_plan,
                    prior_artifact=draft_artifact,
                ),
                expected_identity={
                    "episode_id": resolved_episode_id,
                    "run_token": effective_run_token,
                    "source_digest": source_digest,
                    "plan_digest": plan_digest,
                },
            )
            editorial_gate = EditorialGate(client=self.client, logger=self.logger)
            editorial_report = None
            if rewritten_artifact is not None:
                editorial_report = self._load_validated_artifact(
                    state=state,
                    phase_name="editorial_gate",
                    path=artifact_paths["editorial_report"],
                    validator=validate_editorial_report,
                    expected_identity={
                        "run_token": effective_run_token,
                        "source_digest": source_digest,
                        "plan_digest": plan_digest,
                        "internal_artifact_digest": str(rewritten_artifact.get("internal_artifact_digest", "")),
                    },
                )
            max_rewrite_rounds = 2
            round_used = 0
            if editorial_report is not None and bool(editorial_report.get("pass", False)):
                self._mark_phase_state(state=state, store=store, phase_name="editorial_gate", status="completed")
            else:
                if rewritten_artifact is None:
                    self._last_stage = "editorial_rewrite"
                    phase_started = time.time()
                    round_used = 1
                    rewritten_artifact = EditorialRewriter(client=self.client, logger=self.logger).rewrite(
                        script_artifact=draft_artifact,
                        evidence_map=evidence_map,
                        episode_plan=episode_plan,
                        editorial_report=None,
                        round_idx=round_used,
                    )
                    rewritten_artifact = self._harden_script_artifact(
                        script_artifact=rewritten_artifact,
                        episode_plan=episode_plan,
                    )
                    write_json_artifact(
                        path=rewrite_round_path(run_dir=store.run_dir, round_idx=round_used),
                        payload=rewritten_artifact,
                    )
                    write_json_artifact(path=artifact_paths["rewritten_script_final"], payload=rewritten_artifact)
                    write_json_artifact(path=artifact_paths["rewritten_script"], payload=rewritten_artifact)
                    phase_seconds["editorial_rewrite"] += round(time.time() - phase_started, 3)
                    self._mark_phase_state(state=state, store=store, phase_name="editorial_rewrite", status="completed")
                while True:
                    if count_words_from_lines(list(rewritten_artifact.get("lines", []))) < int(min_words):
                        target_beat_id = self._select_underlength_expansion_beat(
                            script_artifact=rewritten_artifact,
                            episode_plan=episode_plan,
                        )
                        expansion_lines = self._request_contextual_beat_expansion_lines(
                            script_artifact=rewritten_artifact,
                            episode_plan=episode_plan,
                            min_words=min_words,
                            max_words=max_words,
                            source_context=source_for_generation,
                            target_beat_id=target_beat_id,
                            continuation_stage="editorial_underlength",
                        )
                        if expansion_lines:
                            rewritten_artifact = self._insert_lines_into_beat(
                                script_artifact=rewritten_artifact,
                                episode_plan=episode_plan,
                                beat_id=target_beat_id,
                                new_lines=expansion_lines,
                            )
                            write_json_artifact(path=artifact_paths["rewritten_script_final"], payload=rewritten_artifact)
                            write_json_artifact(path=artifact_paths["rewritten_script"], payload=rewritten_artifact)
                    self._last_stage = "editorial_gate"
                    phase_started = time.time()
                    editorial_report = editorial_gate.evaluate(
                        script_artifact=rewritten_artifact,
                        script_lines=list(rewritten_artifact.get("lines", [])),
                        episode_plan=episode_plan,
                        evidence_map=evidence_map,
                        profile_name=self.config.profile_name,
                        min_words=min_words,
                        max_words=max_words,
                        round_used=round_used,
                        max_rounds=max_rewrite_rounds,
                    )
                    editorial_report.update(
                        {
                            "run_token": effective_run_token,
                            "source_digest": source_digest,
                            "plan_digest": plan_digest,
                            "internal_artifact_digest": rewritten_artifact.get("internal_artifact_digest", ""),
                            "public_payload_digest": rewritten_artifact.get("public_payload_digest", ""),
                        }
                    )
                    write_json_artifact(path=artifact_paths["editorial_report"], payload=editorial_report)
                    phase_seconds["editorial_gate"] += round(time.time() - phase_started, 3)
                    if bool(editorial_report.get("pass", False)):
                        break
                    if round_used >= max_rewrite_rounds:
                        break
                    self._last_stage = "editorial_rewrite"
                    phase_started = time.time()
                    round_used += 1
                    rewritten_artifact = EditorialRewriter(client=self.client, logger=self.logger).rewrite(
                        script_artifact=rewritten_artifact,
                        evidence_map=evidence_map,
                        episode_plan=episode_plan,
                        editorial_report=editorial_report,
                        round_idx=round_used,
                    )
                    rewritten_artifact = self._harden_script_artifact(
                        script_artifact=rewritten_artifact,
                        episode_plan=episode_plan,
                    )
                    write_json_artifact(
                        path=rewrite_round_path(run_dir=store.run_dir, round_idx=round_used),
                        payload=rewritten_artifact,
                    )
                    write_json_artifact(path=artifact_paths["rewritten_script_final"], payload=rewritten_artifact)
                    write_json_artifact(path=artifact_paths["rewritten_script"], payload=rewritten_artifact)
                    phase_seconds["editorial_rewrite"] += round(time.time() - phase_started, 3)
                    self._mark_phase_state(state=state, store=store, phase_name="editorial_rewrite", status="completed")
            if not bool(editorial_report.get("pass", False)):
                raise ScriptOperationError(
                    "Editorial gate rejected generated script",
                    error_kind=ERROR_KIND_SCRIPT_QUALITY,
                )
            write_json_artifact(path=artifact_paths["rewritten_script_final"], payload=rewritten_artifact)
            write_json_artifact(path=artifact_paths["rewritten_script"], payload=rewritten_artifact)
            self._last_stage = "structural_finalize"
            phase_started = time.time()
            structural_gate = StructuralGate(max_consecutive_same_speaker=self._max_consecutive_same_speaker())
            final_lines = structural_gate.finalize(lines=list(rewritten_artifact.get("lines", [])))
            structural_report = structural_gate.evaluate(lines=final_lines)
            write_json_artifact(path=artifact_paths["structural_report"], payload=structural_report)
            phase_seconds["postprocess"] = round(time.time() - phase_started, 3)
            if not bool(structural_report.get("pass", False)):
                repaired_final_lines = repair_script_completeness(
                    final_lines,
                    max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
                )
                repaired_structural_report = structural_gate.evaluate(lines=repaired_final_lines)
                if not bool(repaired_structural_report.get("pass", False)):
                    raise ScriptOperationError(
                        "Script failed structural validation",
                        error_kind=ERROR_KIND_SCRIPT_COMPLETENESS,
                    )
                final_lines = structural_gate.finalize(lines=repaired_final_lines)
                structural_report = repaired_structural_report
                write_json_artifact(path=artifact_paths["structural_report"], payload=structural_report)
            final_artifact = build_script_artifact(
                stage="final",
                episode_id=resolved_episode_id,
                run_token=effective_run_token,
                source_digest=source_digest,
                plan_ref=str(rewritten_artifact.get("plan_ref")),
                plan_digest=plan_digest,
                lines=final_lines,
                episode_plan=episode_plan,
                prior_artifact=rewritten_artifact,
                target_word_count=rewritten_artifact.get("target_word_count"),
            )
            fact_guard_final_report = self._load_validated_artifact(
                state=state,
                phase_name="fact_guard_final",
                path=artifact_paths["fact_guard_report_final"],
                validator=validate_fact_guard_report,
                expected_identity={
                    "run_token": effective_run_token,
                    "source_digest": source_digest,
                    "plan_digest": plan_digest,
                    "internal_artifact_digest": str(final_artifact.get("internal_artifact_digest", "")),
                },
            )
            if fact_guard_final_report is None:
                self._last_stage = "fact_guard_final"
                phase_started = time.time()
                write_json_artifact(
                    path=artifact_paths["final_evaluated_pre_fact_repair"],
                    payload=final_artifact,
                )
                initial_fact_guard_final_report = fact_guard.validate(
                    script_artifact=final_artifact,
                    evidence_map=evidence_map,
                    episode_plan=episode_plan,
                    stage_name="final",
                )
                fact_guard_final_report = initial_fact_guard_final_report
                if not bool(initial_fact_guard_final_report.get("pass", False)):
                    repaired_final_artifact = fact_guard.repair(
                        script_artifact=final_artifact,
                        evidence_map=evidence_map,
                        episode_plan=episode_plan,
                        report=initial_fact_guard_final_report,
                        stage_name="final",
                    )
                    if str(repaired_final_artifact.get("internal_artifact_digest", "")) != str(final_artifact.get("internal_artifact_digest", "")):
                        candidate_final_lines = structural_gate.finalize(lines=list(repaired_final_artifact.get("lines", [])))
                        candidate_structural_report = structural_gate.evaluate(lines=candidate_final_lines)
                        if not bool(candidate_structural_report.get("pass", False)):
                            raise ScriptOperationError(
                                "Script failed structural validation after final fact repair",
                                error_kind=ERROR_KIND_SCRIPT_COMPLETENESS,
                            )
                        candidate_final_artifact = build_script_artifact(
                            stage="final",
                            episode_id=resolved_episode_id,
                            run_token=effective_run_token,
                            source_digest=source_digest,
                            plan_ref=str(repaired_final_artifact.get("plan_ref")),
                            plan_digest=plan_digest,
                            lines=candidate_final_lines,
                            episode_plan=episode_plan,
                            prior_artifact=repaired_final_artifact,
                            target_word_count=repaired_final_artifact.get("target_word_count"),
                        )
                        candidate_fact_guard_final_report = fact_guard.validate(
                            script_artifact=candidate_final_artifact,
                            evidence_map=evidence_map,
                            episode_plan=episode_plan,
                            stage_name="final",
                        )
                        if self._fact_guard_report_escalated(
                            before=initial_fact_guard_final_report,
                            after=candidate_fact_guard_final_report,
                        ):
                            self.logger.warn(
                                "fact_guard_final_repair_escalated",
                                before_issues=len(list(initial_fact_guard_final_report.get("issues", []) or [])),
                                after_issues=len(list(candidate_fact_guard_final_report.get("issues", []) or [])),
                                before_blocking=self._fact_guard_blocking_issue_count(initial_fact_guard_final_report),
                                after_blocking=self._fact_guard_blocking_issue_count(candidate_fact_guard_final_report),
                            )
                        else:
                            final_artifact = candidate_final_artifact
                            final_lines = candidate_final_lines
                            structural_report = candidate_structural_report
                            fact_guard_final_report = candidate_fact_guard_final_report
                            write_json_artifact(path=artifact_paths["structural_report"], payload=structural_report)
                write_json_artifact(
                    path=artifact_paths["final_evaluated_post_fact_repair"],
                    payload=final_artifact,
                )
                write_json_artifact(
                    path=artifact_paths["fact_guard_report_final"],
                    payload=fact_guard_final_report,
                )
                phase_seconds["fact_guard_final"] = round(time.time() - phase_started, 3)
                self._mark_phase_state(state=state, store=store, phase_name="fact_guard_final", status="completed")
            phase_started = time.time()
            quality_editorial_report = EditorialGate(client=self.client, logger=self.logger).evaluate(
                script_artifact=final_artifact,
                script_lines=list(final_artifact.get("lines", [])),
                episode_plan=episode_plan,
                evidence_map=evidence_map,
                profile_name=self.config.profile_name,
                min_words=min_words,
                max_words=max_words,
                round_used=round_used,
                max_rounds=max_rewrite_rounds,
            )
            quality_editorial_report.update(
                {
                    "run_token": effective_run_token,
                    "source_digest": source_digest,
                    "plan_digest": plan_digest,
                    "internal_artifact_digest": final_artifact.get("internal_artifact_digest", ""),
                    "public_payload_digest": final_artifact.get("public_payload_digest", ""),
                }
            )
            write_json_artifact(path=artifact_paths["editorial_report"], payload=quality_editorial_report)
            phase_seconds["editorial_gate"] += round(time.time() - phase_started, 3)
            self._mark_phase_state(state=state, store=store, phase_name="editorial_gate", status="completed")
            if not bool(quality_editorial_report.get("pass", False)):
                stabilization_lines = list(final_lines)
                closing_beat_id = ""
                failure_types = {
                    str(item.get("failure_type", "")).strip()
                    for item in list(quality_editorial_report.get("failures", []) or [])
                    if isinstance(item, dict)
                }
                if "repeated_closing_tail" in failure_types:
                    closing_beat_id = str(
                        quality_editorial_report.get("deterministic_metrics", {}).get("closing_beat_id")
                        or (
                            list(episode_plan.get("beats", []) or [])[-1].get("beat_id")
                            if list(episode_plan.get("beats", []) or [])
                            else ""
                        )
                    ).strip()
                    trimmed_lines: List[Dict[str, Any]] = []
                    closing_seen = 0
                    for turn in list(final_artifact.get("turns", []) or []):
                        if not isinstance(turn, dict):
                            continue
                        beat_id = str(turn.get("beat_id", "")).strip()
                        if closing_beat_id and beat_id == closing_beat_id:
                            closing_seen += 1
                            if closing_seen > 2:
                                continue
                        payload = {
                            "speaker": str(turn.get("speaker", "")).strip(),
                            "role": str(turn.get("role", "")).strip(),
                            "instructions": str(turn.get("instructions", "")).strip(),
                            "text": str(turn.get("text", "")).strip(),
                        }
                        pace_hint = str(turn.get("pace_hint", "") or "").strip()
                        if pace_hint:
                            payload["pace_hint"] = pace_hint
                        trimmed_lines.append(payload)
                    if trimmed_lines:
                        stabilization_lines = trimmed_lines
                stabilized_artifact = build_script_artifact(
                    stage="final",
                    episode_id=resolved_episode_id,
                    run_token=effective_run_token,
                    source_digest=source_digest,
                    plan_ref=str(final_artifact.get("plan_ref")),
                    plan_digest=plan_digest,
                    lines=repair_script_completeness(
                        stabilization_lines,
                        max_consecutive_same_speaker=self._max_consecutive_same_speaker(),
                    ),
                    episode_plan=episode_plan,
                    prior_artifact=final_artifact,
                    target_word_count=final_artifact.get("target_word_count"),
                )
                if count_words_from_lines(list(stabilized_artifact.get("lines", []))) < int(min_words):
                    target_beat_id = self._select_underlength_expansion_beat(
                        script_artifact=stabilized_artifact,
                        episode_plan=episode_plan,
                    )
                    if closing_beat_id and target_beat_id == closing_beat_id:
                        for beat in list(episode_plan.get("beats", []) or []):
                            candidate_beat_id = str(dict(beat).get("beat_id", "")).strip()
                            if candidate_beat_id and candidate_beat_id != closing_beat_id:
                                target_beat_id = candidate_beat_id
                                break
                    expansion_lines = self._request_contextual_beat_expansion_lines(
                        script_artifact=stabilized_artifact,
                        episode_plan=episode_plan,
                        min_words=min_words,
                        max_words=max_words,
                        source_context=source_for_generation,
                        target_beat_id=target_beat_id,
                        continuation_stage="final_underlength",
                    )
                    if expansion_lines:
                        stabilized_artifact = self._insert_lines_into_beat(
                            script_artifact=stabilized_artifact,
                            episode_plan=episode_plan,
                            beat_id=target_beat_id,
                            new_lines=expansion_lines,
                        )
                stabilized_lines = structural_gate.finalize(lines=list(stabilized_artifact.get("lines", [])))
                stabilized_structural_report = structural_gate.evaluate(lines=stabilized_lines)
                if bool(stabilized_structural_report.get("pass", False)):
                    stabilized_artifact = build_script_artifact(
                        stage="final",
                        episode_id=resolved_episode_id,
                        run_token=effective_run_token,
                        source_digest=source_digest,
                        plan_ref=str(stabilized_artifact.get("plan_ref")),
                        plan_digest=plan_digest,
                        lines=stabilized_lines,
                        episode_plan=episode_plan,
                        prior_artifact=stabilized_artifact,
                        target_word_count=stabilized_artifact.get("target_word_count"),
                    )
                    stabilized_fact_guard_report = fact_guard.validate(
                        script_artifact=stabilized_artifact,
                        evidence_map=evidence_map,
                        episode_plan=episode_plan,
                        stage_name="final",
                    )
                    stabilized_editorial_report = EditorialGate(client=self.client, logger=self.logger).evaluate(
                        script_artifact=stabilized_artifact,
                        script_lines=list(stabilized_artifact.get("lines", [])),
                        episode_plan=episode_plan,
                        evidence_map=evidence_map,
                        profile_name=self.config.profile_name,
                        min_words=min_words,
                        max_words=max_words,
                        round_used=round_used,
                        max_rounds=max_rewrite_rounds,
                    )
                    stabilized_editorial_report.update(
                        {
                            "run_token": effective_run_token,
                            "source_digest": source_digest,
                            "plan_digest": plan_digest,
                            "internal_artifact_digest": stabilized_artifact.get("internal_artifact_digest", ""),
                            "public_payload_digest": stabilized_artifact.get("public_payload_digest", ""),
                        }
                    )
                    if bool(stabilized_editorial_report.get("pass", False)) and not self._fact_guard_blocks(
                        stabilized_fact_guard_report
                    ):
                        final_artifact = stabilized_artifact
                        final_lines = stabilized_lines
                        structural_report = stabilized_structural_report
                        fact_guard_final_report = stabilized_fact_guard_report
                        quality_editorial_report = stabilized_editorial_report
                        write_json_artifact(
                            path=artifact_paths["final_evaluated_post_fact_repair"],
                            payload=final_artifact,
                        )
                        write_json_artifact(path=artifact_paths["structural_report"], payload=structural_report)
                        write_json_artifact(
                            path=artifact_paths["fact_guard_report_final"],
                            payload=fact_guard_final_report,
                        )
                        write_json_artifact(path=artifact_paths["editorial_report"], payload=quality_editorial_report)
            public_payload = build_public_script_payload(artifact=final_artifact)
            write_script_payload(path=output_path, lines=final_lines)
            quality_report = self._build_quality_report_payload(
                output_path=output_path,
                final_artifact=final_artifact,
                structural_report=structural_report,
                editorial_report=quality_editorial_report,
                fact_guard_report=fact_guard_final_report,
            )
            quality_report["structural_report_path"] = artifact_paths["structural_report"]
            quality_report["editorial_report_path"] = artifact_paths["editorial_report"]
            quality_report["fact_guard_report_path"] = artifact_paths["fact_guard_report_final"]
            quality_report["final_evaluated_pre_fact_repair_path"] = artifact_paths["final_evaluated_pre_fact_repair"]
            quality_report["final_evaluated_post_fact_repair_path"] = artifact_paths["final_evaluated_post_fact_repair"]
            quality_report_path = artifact_paths["quality_report"]
            quality_report["script_quality_report_path"] = quality_report_path
            write_json_artifact(path=quality_report_path, payload=quality_report)
            self._mark_phase_state(state=state, store=store, phase_name="structural_finalize", status="completed")
            if not bool(quality_editorial_report.get("pass", False)):
                raise ScriptOperationError(
                    "Final script failed editorial validation",
                    error_kind=ERROR_KIND_SCRIPT_QUALITY,
                )
            if self._fact_guard_blocks(fact_guard_final_report):
                raise ScriptOperationError(
                    "Final script failed factual validation",
                    error_kind=ERROR_KIND_SCRIPT_QUALITY,
                )
            if not bool(quality_report.get("pass", False)):
                raise ScriptOperationError(
                    "Final script failed quality validation",
                    error_kind=ERROR_KIND_SCRIPT_QUALITY,
                )

            state.update(
                {
                    "lines": public_payload.get("lines", []),
                    "current_word_count": count_words_from_lines(final_lines),
                    "status": "completed",
                    "completed_at": int(time.time()),
                    "phase_cursor": "completed",
                    "quality_report_path": quality_report_path,
                    "script_quality_report_path": quality_report_path,
                    "public_payload_digest": final_artifact.get("public_payload_digest", ""),
                    "internal_artifact_digest": final_artifact.get("internal_artifact_digest", ""),
                    "artifact_paths": artifact_paths,
                }
            )
            state["status"] = "completed"
            store.save(state)
            run_summary = {
                "component": "script_generator",
                "episode_id": resolved_episode_id,
                "run_token": effective_run_token,
                "profile": self.config.profile_name,
                "status": "completed",
                "word_count": count_words_from_lines(final_lines),
                "line_count": len(final_lines),
                "source_segments_count": int(len(evidence_map.get("source_segments", []))),
                "beats_planned": int(len(episode_plan.get("beats", []))),
                "phase_cursor": "completed",
                "phase_status": dict(state.get("phase_status", {}) or {}),
                "requests_made": self.client.requests_made,
                "estimated_cost_usd": round(self.client.estimated_cost_usd, 4),
                "elapsed_seconds": round(time.time() - started, 2),
                "output_path": output_path,
                "script_started_at": int(started),
                "script_completed_at": int(time.time()),
                "failed_stage": None,
                "phase_seconds": _phase_seconds_with_generation(phase_seconds),
                "script_retry_rate": round(
                    float(getattr(self.client, "script_retries_total", 0))
                    / float(max(1, getattr(self.client, "requests_made", 0))),
                    4,
                ),
                "invalid_schema_rate": round(
                    float(getattr(self.client, "script_json_parse_failures", 0))
                    / float(max(1, getattr(self.client, "requests_made", 0))),
                    4,
                ),
                "schema_validation_failures": int(getattr(self, "_schema_validation_failures", 0)),
                "script_completeness_before_repair": dict(structural_report.get("completeness", {})),
                "script_completeness_after_repair": dict(structural_report.get("completeness", {})),
                "script_completeness_pass": bool(structural_report.get("pass", False)),
                "script_completeness_reasons": list(structural_report.get("notes", [])),
                "source_validation_mode": self.config.source_validation_mode,
                "quality_report_path": quality_report_path,
                "script_quality_report_path": quality_report_path,
                "quality_gate_pass": bool(quality_report.get("pass", False)),
                "editorial_pass": bool(quality_editorial_report.get("pass", False)),
                "fact_guard_pass": bool(fact_guard_final_report.get("pass", False)),
                "fact_guard_warning_count": int(quality_report.get("fact_guard_warning_count", 0)),
                "fact_guard_repairable_count": int(quality_report.get("fact_guard_repairable_count", 0)),
                "coverage_metrics": dict(final_artifact.get("coverage", {}) or {}),
                "artifact_paths": artifact_paths,
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
            _atomic_write_text(run_summary_path, json.dumps(run_summary, indent=2, ensure_ascii=False))
            return ScriptGenerationResult(
                episode_id=resolved_episode_id,
                output_path=output_path,
                line_count=len(public_payload.get("lines", [])),
                word_count=count_words_from_lines(public_payload.get("lines", [])),
                checkpoint_path=store.checkpoint_path,
                run_summary_path=run_summary_path,
                script_retry_rate=run_summary["script_retry_rate"],
                invalid_schema_rate=run_summary["invalid_schema_rate"],
                schema_validation_failures=int(run_summary["schema_validation_failures"]),
                quality_report_path=quality_report_path,
                artifact_paths=artifact_paths,
            )
        except InterruptedError:
            if state:
                state["status"] = "interrupted"
                state["failure_kind"] = ERROR_KIND_INTERRUPTED
                state["failed_stage"] = str(getattr(self, "_last_stage", "") or "unknown")
                state["last_success_at"] = int(time.time())
                store.save(state)
            failure_summary = {
                "component": "script_generator",
                "episode_id": resolved_episode_id,
                "run_token": effective_run_token,
                "status": "interrupted",
                "failed_stage": str(getattr(self, "_last_stage", "") or "unknown"),
                "failure_kind": ERROR_KIND_INTERRUPTED,
                "elapsed_seconds": round(time.time() - started, 2),
                "phase_seconds": _phase_seconds_with_generation(phase_seconds),
                "quality_report_path": quality_report_path,
                "artifact_paths": artifact_paths,
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
            failure_kind = ERROR_KIND_UNKNOWN
            if isinstance(exc, ScriptOperationError):
                failure_kind = exc.error_kind
            elif int(getattr(self.client, "script_json_parse_failures", 0)) > 0:
                failure_kind = ERROR_KIND_INVALID_SCHEMA
            if state:
                state["status"] = "failed"
                state["failure_kind"] = failure_kind
                state["failed_stage"] = str(getattr(self, "_last_stage", "") or "unknown")
                state["last_success_at"] = int(time.time())
                store.save(state)
            failure_summary = {
                "component": "script_generator",
                "episode_id": resolved_episode_id,
                "run_token": effective_run_token,
                "status": "failed",
                "failed_stage": str(getattr(self, "_last_stage", "") or "unknown"),
                "failure_kind": failure_kind,
                "elapsed_seconds": round(time.time() - started, 2),
                "phase_seconds": _phase_seconds_with_generation(phase_seconds),
                "quality_report_path": quality_report_path,
                "artifact_paths": artifact_paths,
                "script_retry_rate": round(
                    float(getattr(self.client, "script_retries_total", 0))
                    / float(max(1, getattr(self.client, "requests_made", 0))),
                    4,
                ),
                "invalid_schema_rate": round(
                    float(getattr(self.client, "script_json_parse_failures", 0))
                    / float(max(1, getattr(self.client, "requests_made", 0))),
                    4,
                ),
                "schema_validation_failures": int(getattr(self, "_schema_validation_failures", 0)),
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
        """Generate final script JSON with the redesigned pipeline only."""
        return self._generate_redesigned(
            source_text=source_text,
            output_path=output_path,
            episode_id=episode_id,
            resume=resume,
            resume_force=resume_force,
            force_unlock=force_unlock,
            cancel_check=cancel_check,
            run_token=run_token,
        )
