#!/usr/bin/env python3
from __future__ import annotations

"""Editorial validation for redesigned podcast scripts."""

import json
import re
import textwrap
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List

from .logging_utils import Logger
from .podcast_artifacts import validate_editorial_report
from .schema import count_words_from_lines
from .script_postprocess import FAREWELL_BY_LANG, RECAP_BY_LANG, TAIL_QUESTION_ANSWER_BY_LANG

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
_STOCK_OPENERS = (
    "por otro lado",
    "ahora bien",
    "dicho esto",
    "a partir de ahi",
    "pasando a otro frente",
    "pasando a",
    "en paralelo",
    "si lo conectamos",
    "en ese sentido",
    "por cierto",
    "exacto",
    "tal cual",
    "claro",
    "totalmente",
    "de acuerdo",
    "correcto",
    "cierto",
)
_WORD_RE = re.compile(r"[^\W_]{3,}", re.UNICODE)
_QUESTION_PROMPT_RE = re.compile(
    r"(?:por ejemplo|ejemplo|coste|costo|tradeoff|contrapartida|bajemos esto a tierra|en la practica|vale, bajemos esto a tierra)",
    re.IGNORECASE,
)
_HOST2_PUSH_RE = re.compile(
    r"(?:pero|vale,? bajemos|en la practica|por ejemplo|cuanto cuesta|que coste|que costo|que tradeoff|que riesgo|que ejemplo|aterriza|concreto|humo|exacto|claro,? porque|entonces|si falta|la barra|el orden es|autoenga[nñ]o|no me dice|no me voy|por eso|lo que cambia|ese orden importa|no miras solo|no alcanza|eso obliga|si mezclas|ese marco|si fallan|no es lo mismo|dame el orden|si sospecho|no hay misterio|conviene usar|antes de compartir|tesis completa|la diferencia practica|no conviene mezclar|una cosa es|para separar|la secuencia correcta|distingues si|si quieres una referencia|cuando ya no quieres|no decidir por intuicion|te deja separar|sirve para inspeccionar|se ve rapido en una se[nñ]al)",
    re.IGNORECASE,
)
_THESIS_REPEAT_RE = re.compile(r"(?:en el fondo|la idea central|la tesis|esto va de|lo importante es)", re.IGNORECASE)
_BRIEFING_TONE_RE = re.compile(r"(?:en resumen|a continuacion|en este episodio|vamos a recorrer|comenzamos con)", re.IGNORECASE)
_CLOSE_SIGNAL_RE = re.compile(
    r"(?:en resumen|nos quedamos con|al final|la clave es|lo importante es|si algo deja|la idea central)",
    re.IGNORECASE,
)
_TAIL_TRANSITION_RE = re.compile(
    r"^(?:con eso|aun asi|visto asi|y ahi|de paso|por otro lado|ahora bien|dicho esto)\b",
    re.IGNORECASE,
)

_LLM_EDITORIAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "orality": {"type": "number"},
                "host_distinction": {"type": "number"},
                "progression": {"type": "number"},
                "freshness": {"type": "number"},
                "listener_engagement": {"type": "number"},
                "density_control": {"type": "number"},
            },
            "required": [
                "orality",
                "host_distinction",
                "progression",
                "freshness",
                "listener_engagement",
                "density_control",
            ],
            "additionalProperties": False,
        },
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["scores", "reasons"],
    "additionalProperties": False,
}


def _normalize_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    deaccented = "".join(ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", deaccented)


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(_normalize_text(text))


def _leading_stock_opener(text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    for phrase in sorted(_STOCK_OPENERS, key=len, reverse=True):
        if normalized == phrase or normalized.startswith(f"{phrase} ") or normalized.startswith(f"{phrase},"):
            return phrase
    return ""


def _lexical_overlap(a: str, b: str) -> float:
    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return float(len(tokens_a.intersection(tokens_b))) / float(max(1, len(tokens_b)))


def _clause_count(text: str) -> int:
    parts = [chunk.strip() for chunk in re.split(r"[,:;.!?]+", str(text or "")) if chunk.strip()]
    return len(parts)


def _normalized_template_values() -> List[str]:
    out: List[str] = []
    for mapping in (RECAP_BY_LANG, FAREWELL_BY_LANG, TAIL_QUESTION_ANSWER_BY_LANG):
        for value in mapping.values():
            normalized = _normalize_text(str(value or ""))
            if normalized and normalized not in out:
                out.append(normalized)
    return out


@dataclass(frozen=True)
class EditorialGate:
    """Blend deterministic metrics with structured LLM judgment."""

    client: Any | None
    logger: Logger

    def evaluate(
        self,
        *,
        script_artifact: Dict[str, Any] | None = None,
        script_lines: List[Dict[str, Any]],
        episode_plan: Dict[str, Any],
        evidence_map: Dict[str, Any],
        profile_name: str,
        min_words: int,
        max_words: int,
        round_used: int = 0,
        max_rounds: int = 2,
    ) -> Dict[str, Any]:
        llm_scores, llm_reasons = self._judge_with_llm(
            script_lines=script_lines,
            episode_plan=episode_plan,
            evidence_map=evidence_map,
        )
        deterministic = self._deterministic_metrics(
            script_artifact=script_artifact,
            script_lines=script_lines,
            episode_plan=episode_plan,
            profile_name=profile_name,
            min_words=min_words,
            max_words=max_words,
        )
        failures = self._build_failures(
            script_artifact=script_artifact,
            script_lines=script_lines,
            llm_scores=llm_scores,
            llm_reasons=llm_reasons,
            deterministic=deterministic,
            profile_name=profile_name,
            min_words=min_words,
        )
        passed = not failures
        report = validate_editorial_report(
            {
                "artifact_version": 1,
                "stage": str((script_artifact or {}).get("stage") or "rewritten"),
                "profile": profile_name,
                "pass": passed,
                "scores": llm_scores,
                "deterministic_metrics": deterministic,
                "failures": failures,
                "rewrite_budget": {
                    "max_rounds": max_rounds,
                    "round_used": round_used,
                    "stop_reason": "" if passed else "editorial_fail",
                },
            }
        )
        self.logger.info(
            "editorial_gate_evaluated",
            passed=bool(report.get("pass", False)),
            failures=len(report.get("failures", [])),
            profile=profile_name,
        )
        return report

    def _judge_with_llm(
        self,
        *,
        script_lines: List[Dict[str, Any]],
        episode_plan: Dict[str, Any],
        evidence_map: Dict[str, Any],
    ) -> tuple[Dict[str, float], List[str]]:
        fallback_scores = {
            "orality": 4.0,
            "host_distinction": 4.0,
            "progression": 4.0,
            "freshness": 4.0,
            "listener_engagement": 4.0,
            "density_control": 4.0,
        }
        if self.client is None:
            return fallback_scores, []
        prompt = textwrap.dedent(
            f"""
            Evaluate this Spanish podcast script editorially.
            Return ONLY JSON.

            Score from 1 to 5:
            - orality
            - host_distinction
            - progression
            - freshness
            - listener_engagement
            - density_control

            Reasons should be short failure-oriented notes only when something is weak.
            Penalize scripts that keep opening turns with stock fillers such as
            "por otro lado", "ahora bien", "exacto", "claro", "tal cual", or "totalmente".

            EPISODE PLAN:
            {json.dumps(episode_plan, ensure_ascii=False, indent=2)}

            EVIDENCE MAP:
            {json.dumps(evidence_map, ensure_ascii=False, indent=2)}

            SCRIPT:
            {json.dumps({"lines": script_lines}, ensure_ascii=False, indent=2)}
            """
        ).strip()
        try:
            payload = self.client.generate_script_json(
                prompt=prompt,
                schema=_LLM_EDITORIAL_SCHEMA,
                max_output_tokens=1400,
                stage="editorial_gate_eval",
            )
            scores_raw = payload.get("scores", {})
            reasons = [str(item or "").strip() for item in list(payload.get("reasons", []) or []) if str(item or "").strip()]
            scores = {
                key: max(1.0, min(5.0, float(scores_raw.get(key, 4.0))))
                for key in fallback_scores
            }
            return scores, reasons
        except Exception as exc:  # noqa: BLE001
            self.logger.warn("editorial_gate_llm_failed", error=str(exc))
            return fallback_scores, []

    def _deterministic_metrics(
        self,
        *,
        script_artifact: Dict[str, Any] | None,
        script_lines: List[Dict[str, Any]],
        episode_plan: Dict[str, Any],
        profile_name: str,
        min_words: int,
        max_words: int,
    ) -> Dict[str, Any]:
        normalized_templates = _normalized_template_values()
        joined = " ".join(str(line.get("text", "") or "") for line in script_lines)
        normalized_joined = _normalize_text(joined)
        scaffold_phrase_hits = 0
        repeated_scaffolds: Dict[str, int] = {}
        stock_opener_cluster_hits = 0
        recent_stock_hit_indexes: List[int] = []
        for line in script_lines:
            normalized = _normalize_text(str(line.get("text", "") or ""))
            opener = _leading_stock_opener(normalized)
            if opener:
                scaffold_phrase_hits += 1
                repeated_scaffolds[opener] = repeated_scaffolds.get(opener, 0) + 1
        recent_stock_hit_indexes = []
        for idx, line in enumerate(script_lines):
            opener = _leading_stock_opener(str(line.get("text", "") or ""))
            if not opener:
                continue
            recent_stock_hit_indexes = [prev_idx for prev_idx in recent_stock_hit_indexes if idx - prev_idx <= 5]
            if recent_stock_hit_indexes:
                stock_opener_cluster_hits += 1
            recent_stock_hit_indexes.append(idx)
        long_turn_limits = {"short": 48, "standard": 58, "long": 64}
        long_turn_limit = long_turn_limits.get(str(profile_name or "").strip().lower(), 58)
        long_turn_count = 0
        dense_turn_indexes: List[int] = []
        for idx, line in enumerate(script_lines):
            text = str(line.get("text", "") or "")
            word_count = len(text.split())
            clause_count = _clause_count(text)
            if word_count > long_turn_limit:
                long_turn_count += 1
            if word_count >= max(34, long_turn_limit - 10) and clause_count >= 3:
                dense_turn_indexes.append(idx)
        abrupt_transition_count = 0
        for idx in range(1, len(script_lines)):
            prev_text = str(script_lines[idx - 1].get("text", "") or "")
            curr_text = str(script_lines[idx].get("text", "") or "")
            overlap = _lexical_overlap(prev_text, curr_text)
            shares_plan_context = self._shares_plan_context(idx=idx, episode_plan=episode_plan)
            if overlap < 0.08 and not shares_plan_context:
                abrupt_transition_count += 1
        question_ratio = 0.0
        max_question_streak = 0
        question_streak = 0
        if script_lines:
            question_lines = 0
            for line in script_lines:
                text = str(line.get("text", "") or "")
                if "?" in text:
                    question_lines += 1
                    question_streak += 1
                    max_question_streak = max(max_question_streak, question_streak)
                else:
                    question_streak = 0
            question_ratio = float(question_lines) / float(max(1, len(script_lines)))
        host2_lines = [line for line in script_lines if str(line.get("role", "")).strip() == "Host2"]
        host2_push_count = 0
        for line in host2_lines:
            text = str(line.get("text", "") or "")
            if _HOST2_PUSH_RE.search(text) or _QUESTION_PROMPT_RE.search(text):
                host2_push_count += 1
        host2_push_ratio = (
            float(host2_push_count) / float(max(1, len(host2_lines))) if host2_lines else 0.0
        )
        template_reuse_hits = 0
        for template in normalized_templates:
            if template and template in normalized_joined:
                template_reuse_hits += 1
        coverage = dict((script_artifact or {}).get("coverage", {}) or {})
        turns = [dict(turn) for turn in list((script_artifact or {}).get("turns", []) or []) if isinstance(turn, dict)]
        overlong_for_profile = False
        word_count = count_words_from_lines(script_lines)
        line_count = len(script_lines)
        beats = [beat for beat in list(episode_plan.get("beats", []) or []) if isinstance(beat, dict)]
        non_cuttable_beats = [beat for beat in beats if not bool(beat.get("can_cut", False))]
        missing_non_cuttable = list(coverage.get("missing_non_cuttable_beat_ids", []) or [])
        host2_missing_non_cuttable = list(coverage.get("host2_missing_non_cuttable_beat_ids", []) or [])
        host2_turn_ratio = float(coverage.get("host2_turn_ratio", host2_push_ratio) or 0.0)
        line_counts_by_beat = dict(coverage.get("line_counts_by_beat", {}) or {})
        compressed_beat_count = 0
        for beat in beats:
            beat_id = str(beat.get("beat_id", "") or "").strip()
            if not beat_id:
                continue
            min_turns = 2 if int(beat.get("target_words", 0) or 0) >= 45 else 1
            if int(line_counts_by_beat.get(beat_id, 0) or 0) < min_turns:
                compressed_beat_count += 1
        if str(profile_name or "").strip().lower() == "short":
            overlong_for_profile = word_count > int(max_words)
            if not overlong_for_profile:
                for beat in list(episode_plan.get("beats", []) or []):
                    if bool(beat.get("can_cut", False)):
                        overlong_for_profile = word_count > int(round(max(min_words, max_words) * 0.92))
                        if overlong_for_profile:
                            break
        thesis_repetition_hits = sum(1 for line in script_lines if _THESIS_REPEAT_RE.search(str(line.get("text", "") or "")))
        briefing_tone_hits = sum(1 for line in script_lines if _BRIEFING_TONE_RE.search(str(line.get("text", "") or "")))
        repeated_closing_tail_indexes: List[int] = []
        tail_transition_count = 0
        tail_start = max(0, len(script_lines) - 6)
        prior_close_idx = -1
        closing_beat_id = ""
        if turns:
            closing_beat_id = str(turns[-1].get("beat_id", "") or "").strip()
        for idx in range(tail_start, len(script_lines)):
            text = str(script_lines[idx].get("text", "") or "")
            if idx > tail_start and _TAIL_TRANSITION_RE.search(text):
                tail_transition_count += 1
            if not _CLOSE_SIGNAL_RE.search(text):
                continue
            if prior_close_idx >= 0:
                repeated_closing_tail_indexes.append(idx)
            prior_close_idx = idx
        for idx in range(max(tail_start + 1, 1), len(script_lines)):
            curr_text = str(script_lines[idx].get("text", "") or "")
            prev_text = str(script_lines[idx - 1].get("text", "") or "")
            overlap = _lexical_overlap(prev_text, curr_text)
            if overlap >= 0.58 and (
                _CLOSE_SIGNAL_RE.search(curr_text) or _CLOSE_SIGNAL_RE.search(prev_text)
            ):
                repeated_closing_tail_indexes.append(idx)
        if closing_beat_id:
            closing_beat_indexes = [
                idx for idx, turn in enumerate(turns) if str(turn.get("beat_id", "")).strip() == closing_beat_id
            ]
            closing_signal_indexes = [
                idx for idx in closing_beat_indexes if idx < len(script_lines) and _CLOSE_SIGNAL_RE.search(str(script_lines[idx].get("text", "") or ""))
            ]
            if len(closing_signal_indexes) > 1:
                repeated_closing_tail_indexes.extend(closing_signal_indexes[1:])
        repeated_closing_tail_indexes = sorted(set(idx for idx in repeated_closing_tail_indexes if idx >= tail_start))
        return {
            "scaffold_phrase_hits": scaffold_phrase_hits,
            "repeated_scaffolds": repeated_scaffolds,
            "stock_opener_cluster_hits": stock_opener_cluster_hits,
            "long_turn_count": long_turn_count,
            "dense_turn_indexes": dense_turn_indexes,
            "abrupt_transition_count": abrupt_transition_count,
            "tail_transition_count": tail_transition_count,
            "question_ratio": round(question_ratio, 4),
            "max_question_streak": max_question_streak,
            "host2_push_ratio": round(host2_push_ratio, 4),
            "host2_turn_ratio": round(host2_turn_ratio, 4),
            "template_reuse_hits": template_reuse_hits,
            "overlong_for_profile": bool(overlong_for_profile),
            "thesis_repetition_hits": thesis_repetition_hits,
            "briefing_tone_hits": briefing_tone_hits,
            "repeated_closing_tail_indexes": repeated_closing_tail_indexes,
            "closing_beat_id": closing_beat_id,
            "word_count": word_count,
            "line_count": line_count,
            "missing_non_cuttable_beat_count": len(missing_non_cuttable),
            "host2_missing_non_cuttable_beat_count": len(host2_missing_non_cuttable),
            "compressed_beat_count": compressed_beat_count,
        }

    def _shares_plan_context(self, *, idx: int, episode_plan: Dict[str, Any]) -> bool:
        beats = list(episode_plan.get("beats", []) or [])
        if len(beats) < 2:
            return True
        beat_idx = min(len(beats) - 1, int(idx / max(1, round(len(beats)))))
        prev_beat = beats[max(0, beat_idx - 1)]
        curr_beat = beats[beat_idx]
        prev_claims = set(str(item or "").strip() for item in list(prev_beat.get("claim_ids", []) or []))
        curr_claims = set(str(item or "").strip() for item in list(curr_beat.get("claim_ids", []) or []))
        prev_topics = set(str(item or "").strip() for item in list(prev_beat.get("topic_ids", []) or []))
        curr_topics = set(str(item or "").strip() for item in list(curr_beat.get("topic_ids", []) or []))
        return bool(prev_claims.intersection(curr_claims) or prev_topics.intersection(curr_topics))

    def _scores_above_thresholds(self, scores: Dict[str, float]) -> bool:
        return bool(
            scores.get("orality", 0.0) >= 3.8
            and scores.get("host_distinction", 0.0) >= 3.7
            and scores.get("progression", 0.0) >= 3.7
            and scores.get("freshness", 0.0) >= 3.6
            and scores.get("listener_engagement", 0.0) >= 3.6
            and scores.get("density_control", 0.0) >= 3.7
        )

    def _build_failures(
        self,
        *,
        script_artifact: Dict[str, Any] | None,
        script_lines: List[Dict[str, Any]],
        llm_scores: Dict[str, float],
        llm_reasons: List[str],
        deterministic: Dict[str, Any],
        profile_name: str,
        min_words: int,
    ) -> List[Dict[str, Any]]:
        failures: List[Dict[str, Any]] = []
        turns = [dict(turn) for turn in list((script_artifact or {}).get("turns", []) or []) if isinstance(turn, dict)]
        if int(deterministic.get("scaffold_phrase_hits", 0)) >= 2 or int(deterministic.get("stock_opener_cluster_hits", 0)) > 0:
            failures.append(
                self._failure(
                    "scaffold_phrase_repetition",
                    "medium",
                    script_lines=script_lines,
                    reason="se repiten muletillas o arranques mecanicos de turno",
                    action="rewrite_turn_openers_for_variety",
                )
            )
        if int(deterministic.get("thesis_repetition_hits", 0)) >= 2:
            failures.append(
                self._failure(
                    "thesis_repetition",
                    "medium",
                    script_lines=script_lines,
                    reason="la tesis se reformula demasiadas veces",
                    action="compress_repeated_thesis",
                )
            )
        long_turn_allowance = {"short": 2, "standard": 3, "long": 4}.get(
            str(profile_name or "").strip().lower(),
            3,
        )
        dense_indexes = [max(0, int(idx)) for idx in list(deterministic.get("dense_turn_indexes", []) or [])]
        dense_by_structure = (
            int(deterministic.get("long_turn_count", 0)) > long_turn_allowance
            or len(dense_indexes) >= max(1, long_turn_allowance)
        )
        dense_by_llm = dense_by_structure and llm_scores.get("density_control", 5.0) < 3.4
        if dense_by_structure or dense_by_llm:
            failures.append(
                self._failure(
                    "dense_turns",
                    "high",
                    script_lines=script_lines,
                    reason="hay turnos densos o con mini-ensayo oralizado",
                    action="decompress_dense_turn",
                    line_indexes=dense_indexes,
                    beat_ids=self._beat_ids_for_line_indexes(turns=turns, line_indexes=dense_indexes),
                )
            )
        repeated_tail_indexes = [
            max(0, int(idx)) for idx in list(deterministic.get("repeated_closing_tail_indexes", []) or [])
        ]
        if repeated_tail_indexes or int(deterministic.get("tail_transition_count", 0)) >= 3:
            failures.append(
                self._failure(
                    "repeated_closing_tail",
                    "high",
                    script_lines=script_lines,
                    reason="el cierre vuelve a sintetizar o reabrir transiciones en vez de cerrar una sola vez",
                    action="rewrite_closing_for_single_earned_close",
                    line_indexes=repeated_tail_indexes,
                    beat_ids=self._beat_ids_for_line_indexes(turns=turns, line_indexes=repeated_tail_indexes),
                )
            )
        if int(deterministic.get("word_count", 0)) < int(min_words):
            failures.append(
                self._failure(
                    "underlength",
                    "high",
                    script_lines=script_lines,
                    reason="el guion queda por debajo del minimo de palabras",
                    action="expand_missing_beats",
                )
            )
        if int(deterministic.get("missing_non_cuttable_beat_count", 0)) > 0:
            failures.append(
                self._failure(
                    "undercoverage",
                    "high",
                    script_lines=script_lines,
                    reason="faltan beats no recortables en el artefacto terminal",
                    action="restore_missing_beats",
                )
            )
        if int(deterministic.get("compressed_beat_count", 0)) > 0:
            failures.append(
                self._failure(
                    "dense_turns",
                    "medium",
                    script_lines=script_lines,
                    reason="hay beats demasiado comprimidos para su presupuesto de palabras",
                    action="expand_compressed_beats",
                )
            )
        host2_threshold = 0.30 if str(profile_name or "").strip().lower() == "short" else 0.25
        if float(deterministic.get("host2_push_ratio", 0.0)) < host2_threshold:
            failures.append(
                self._failure(
                    "host2_not_pushing",
                    "high",
                    script_lines=script_lines,
                    reason="Host2 no mete suficiente fricción o aterrizaje",
                    action="rewrite_host2_for_friction",
                )
            )
        if float(deterministic.get("host2_turn_ratio", 0.0)) < host2_threshold:
            failures.append(
                self._failure(
                    "host2_not_pushing",
                    "medium",
                    script_lines=script_lines,
                    reason="Host2 interviene demasiado poco en el total del episodio",
                    action="increase_host2_turn_ratio",
                )
            )
        if int(deterministic.get("host2_missing_non_cuttable_beat_count", 0)) > 0:
            failures.append(
                self._failure(
                    "host2_missing_beat",
                    "high",
                    script_lines=script_lines,
                    reason="Host2 desaparece en beats no recortables",
                    action="restore_host2_presence",
                )
            )
        if int(deterministic.get("briefing_tone_hits", 0)) > 0 or llm_scores.get("orality", 5.0) < 3.8:
            failures.append(
                self._failure(
                    "briefing_tone",
                    "medium",
                    script_lines=script_lines,
                    reason="el texto sigue sonando demasiado a briefing o documento",
                    action="debriefingize_span",
                )
            )
        if bool(deterministic.get("overlong_for_profile", False)):
            failures.append(
                self._failure(
                    "overlong_for_profile",
                    "high",
                    script_lines=script_lines,
                    reason="el episodio es demasiado largo para el valor narrativo del perfil",
                    action="cut_low_value_beat",
                )
            )
        if int(deterministic.get("template_reuse_hits", 0)) > 0:
            failures.append(
                self._failure(
                    "earned_closing_missing",
                    "medium",
                    script_lines=script_lines,
                    reason="aparecen plantillas legacy de recap o farewell en el camino normal",
                    action="rewrite_closing",
                )
            )
        for reason in llm_reasons:
            normalized = _normalize_text(reason)
            if "opening" in normalized and "meta" in normalized:
                failures.append(
                    self._failure(
                        "meta_opening",
                        "high",
                        script_lines=script_lines,
                        reason=reason,
                        action="remove_scaffold_phrases",
                    )
                )
        deduped: List[Dict[str, Any]] = []
        seen_types: set[str] = set()
        for failure in failures:
            failure_type = str(failure.get("failure_type", "")).strip()
            if failure_type and failure_type not in seen_types:
                seen_types.add(failure_type)
                deduped.append(failure)
        return deduped

    def _failure(
        self,
        failure_type: str,
        severity: str,
        *,
        script_lines: List[Dict[str, Any]],
        reason: str,
        action: str,
        line_indexes: List[int] | None = None,
        beat_ids: List[str] | None = None,
    ) -> Dict[str, Any]:
        resolved_line_indexes = [idx for idx in list(line_indexes or []) if 0 <= int(idx) < len(script_lines)]
        if not resolved_line_indexes:
            resolved_line_indexes = list(range(max(0, len(script_lines) - 4), len(script_lines)))
        resolved_beat_ids = [str(item or "").strip() for item in list(beat_ids or []) if str(item or "").strip()]
        return {
            "failure_type": failure_type,
            "severity": severity,
            "line_indexes": resolved_line_indexes,
            "beat_ids": resolved_beat_ids,
            "reason": reason,
            "recommended_action": action,
        }

    def _beat_ids_for_line_indexes(self, *, turns: List[Dict[str, Any]], line_indexes: List[int]) -> List[str]:
        beat_ids: List[str] = []
        for idx in line_indexes:
            if idx < 0 or idx >= len(turns):
                continue
            beat_id = str(turns[idx].get("beat_id", "") or "").strip()
            if beat_id and beat_id not in beat_ids:
                beat_ids.append(beat_id)
        return beat_ids
