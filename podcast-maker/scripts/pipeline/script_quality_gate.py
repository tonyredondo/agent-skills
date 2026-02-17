#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import textwrap
import time
import unicodedata
from typing import Any, Callable, Dict, List

from .config import ScriptConfig
from .errors import ERROR_KIND_SCRIPT_QUALITY
from .logging_utils import Logger
from .openai_client import OpenAIClient
from .script_quality_gate_config import (
    ScriptQualityGateConfig,
    critical_score_threshold as _critical_score_threshold,
    hard_fail_structural_only_enabled as _hard_fail_structural_only_enabled,
    strict_score_blocking_enabled as _strict_score_blocking_enabled,
)
from .schema import (
    SCRIPT_JSON_SCHEMA,
    canonical_json,
    content_hash,
    count_words_from_lines,
    validate_script_payload,
)
from .script_postprocess import (
    detect_truncation_indices,
    ensure_en_resumen,
    ensure_farewell_last,
    harden_script_structure,
)

SUMMARY_TOKENS = (
    "en resumen",
    "resumen",
    "resumiendo",
    "en sintesis",
    "en conclusion",
    "in summary",
    "to sum up",
    "overall",
    "in short",
    "recap",
    "key takeaway",
    "takeaway",
    "tl;dr",
    "em resumo",
    "concluindo",
    "en bref",
    "pour resumer",
)

CLOSING_TOKENS = (
    "gracias por escuch",
    "hasta la proxima",
    "nos escuchamos",
    "nos vemos",
    "adios",
    "cuidense",
    "desped",
    "thanks for listening",
    "thank you for listening",
    "thanks for joining",
    "thank you",
    "see you",
    "next episode",
    "next time",
    "until next",
    "goodbye",
    "farewell",
    "merci",
    "au revoir",
    "danke",
    "bis bald",
    "obrigad",
    "ate a proxima",
    "até a proxima",
)

SUMMARY_LABEL_RE = re.compile(
    r"^(?:summary|recap|takeaways?|conclusion|resumen|sintesis|síntesis|cierre)\s*[:\-]\s*",
    re.IGNORECASE,
)
WORD_TOKEN_RE = re.compile(r"[^\W_]{3,}", re.UNICODE)
COMPLETE_SENTENCE_END_RE = re.compile(r"(?:[.!?…]|[.!?…][\"'”’)\]])\s*$")
LLM_TRUNCATION_REASON_HINTS = (
    "trunc",
    "trunca",
    "truncado",
    "truncate",
    "abrupt",
    "incomplete",
    "incompleto",
    "incompleta",
    "inconclus",
    "sin remate",
    "sin cierre",
    "queda a medias",
    "no cierra",
    "cierre abrupt",
    "ending abruptly",
)
INTERNAL_WORKFLOW_HINT_RE = re.compile(
    r"(?:\bdailyread\b|\bnota de transparencia\b|\bse elaboro\b|\busando el script\b|scripts/[a-z0-9._/-]+\.sh\b|\btavily\b|\bserper\b)",
    re.IGNORECASE,
)
QUESTION_PUNCT_RE = re.compile(r"[¿?]")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


class ScriptQualityGateError(RuntimeError):
    def __init__(self, message: str, *, report: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.failure_kind = ERROR_KIND_SCRIPT_QUALITY
        self.report = dict(report or {})


def _normalized_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    deaccented = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", deaccented)


def _has_summary(lines: List[Dict[str, str]]) -> bool:
    for line in lines:
        raw = str(line.get("text", ""))
        text = _normalized_text(raw)
        if any(token in text for token in SUMMARY_TOKENS):
            return True
        if SUMMARY_LABEL_RE.search(text):
            return True
    return _has_recap_overlap(lines)


def _has_closing(lines: List[Dict[str, str]]) -> bool:
    tail = lines[-4:] if len(lines) >= 4 else lines
    for line in tail:
        text = _normalized_text(line.get("text", ""))
        if any(token in text for token in CLOSING_TOKENS):
            return True
    return False


def _internal_workflow_line_count(lines: List[Dict[str, str]]) -> int:
    count = 0
    for line in lines:
        text = _normalized_text(str(line.get("text", "")))
        if text and INTERNAL_WORKFLOW_HINT_RE.search(text):
            count += 1
    return count


def _summary_line_index(lines: List[Dict[str, str]]) -> int:
    for idx, line in enumerate(lines):
        raw = str(line.get("text", ""))
        text = _normalized_text(raw)
        if not text:
            continue
        if any(token in text for token in SUMMARY_TOKENS):
            return idx
        if SUMMARY_LABEL_RE.search(text):
            return idx
    return -1


def _is_question_like(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    return QUESTION_PUNCT_RE.search(raw) is not None


def _has_unanswered_tail_question(lines: List[Dict[str, str]]) -> bool:
    if len(lines) < 3:
        return False
    summary_idx = _summary_line_index(lines)
    if summary_idx <= 0:
        return False
    window_start = max(0, summary_idx - 6)
    last_question_idx = -1
    for idx in range(window_start, summary_idx):
        if _is_question_like(str(lines[idx].get("text", ""))):
            last_question_idx = idx
    if last_question_idx < 0:
        return False

    question_role = _normalized_text(str(lines[last_question_idx].get("role", "")))
    for idx in range(last_question_idx + 1, summary_idx):
        response_text = str(lines[idx].get("text", "")).strip()
        if not response_text:
            continue
        if _is_question_like(response_text):
            continue
        response_role = _normalized_text(str(lines[idx].get("role", "")))
        # Prefer answer turn from the counterpart host before summary.
        if question_role and response_role and response_role == question_role:
            continue
        return False
    return True


def _tail_is_structurally_complete(lines: List[Dict[str, str]]) -> bool:
    if not lines:
        return False
    if detect_truncation_indices(lines):
        return False
    if not _has_summary(lines):
        return False
    if not _has_closing(lines):
        return False
    tail_text = str(lines[-1].get("text", "")).strip()
    if not tail_text:
        return False
    return COMPLETE_SENTENCE_END_RE.search(tail_text) is not None


def _is_llm_tail_truncation_claim(reason: str) -> bool:
    normalized = _normalized_text(reason)
    if not normalized:
        return False
    return any(token in normalized for token in LLM_TRUNCATION_REASON_HINTS)


def _tokenize_content(text: str) -> List[str]:
    return WORD_TOKEN_RE.findall(_normalized_text(text))


def _has_recap_overlap(lines: List[Dict[str, str]]) -> bool:
    if len(lines) < 4:
        return False
    body = lines[:-2]
    tail = lines[-4:]
    freq: Dict[str, int] = {}
    for line in body:
        for token in _tokenize_content(str(line.get("text", ""))):
            freq[token] = int(freq.get(token, 0)) + 1
    if len(freq) < 10:
        return False
    ranked = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    top_tokens = {token for token, count in ranked[:24] if count >= 2}
    if len(top_tokens) < 4:
        return False
    for line in tail:
        raw = str(line.get("text", ""))
        tokens = _tokenize_content(raw)
        if len(tokens) < 8:
            continue
        unique_tokens = set(tokens)
        overlap = unique_tokens.intersection(top_tokens)
        overlap_ratio = float(len(overlap)) / float(max(1, len(unique_tokens)))
        punctuation_hint = raw.count(",") + raw.count(";") + raw.count(":")
        if len(overlap) >= 3 and overlap_ratio >= 0.25 and punctuation_hint >= 1:
            return True
    return False


def _max_consecutive_speaker_run(lines: List[Dict[str, str]]) -> int:
    max_run = 0
    current_run = 0
    previous = ""
    for line in lines:
        speaker = _normalized_text(line.get("speaker", ""))
        if not speaker:
            continue
        if speaker == previous:
            current_run += 1
        else:
            previous = speaker
            current_run = 1
        max_run = max(max_run, current_run)
    return max_run


def _repeat_line_ratio(lines: List[Dict[str, str]]) -> float:
    if not lines:
        return 0.0
    seen: Dict[str, int] = {}
    for line in lines:
        text = _normalized_text(line.get("text", ""))
        if not text:
            continue
        seen[text] = seen.get(text, 0) + 1
    if not seen:
        return 0.0
    repeated = sum(max(0, count - 1) for count in seen.values())
    return float(repeated) / float(max(1, len(lines)))


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("empty_llm_response")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("invalid_llm_json")


def _score(payload: Dict[str, Any], key: str) -> float | None:
    raw = payload.get(key)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return _clamp(value, 0.0, 5.0)


def _score_confidence(payload: Dict[str, Any], key: str) -> float | None:
    raw = payload.get(key)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return _clamp(value, 0.0, 1.0)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "y", "si", "sí", "on"}


def _should_sample_llm(lines: List[Dict[str, str]], sample_rate: float) -> bool:
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    digest = content_hash(canonical_json({"lines": lines}))
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return bucket < sample_rate


def _build_llm_prompt(
    *,
    lines: List[Dict[str, str]],
    script_cfg: ScriptConfig,
    quality_cfg: ScriptQualityGateConfig,
    word_count: int,
) -> str:
    rows: List[str] = []
    for line in lines:
        speaker = line.get("speaker", "Host")
        text = line.get("text", "")
        rows.append(f"{speaker}: {text}")
    dialogue = "\n".join(rows)
    if len(dialogue) > quality_cfg.llm_max_prompt_chars:
        dialogue = dialogue[: quality_cfg.llm_max_prompt_chars]
    return (
        "Evaluate this podcast script quality (language may vary).\n"
        "Return ONLY JSON object with keys:\n"
        "overall_score, cadence_score, logic_score, clarity_score, pass, reasons.\n"
        "Scores are numbers from 0 to 5.\n"
        "Set pass=true only if cadence, logic and clarity are coherent and useful.\n\n"
        f"Target range words: {script_cfg.min_words}-{script_cfg.max_words}\n"
        f"Actual words: {word_count}\n\n"
        "SCRIPT:\n"
        f"{dialogue}\n"
    )


def _build_semantic_rule_prompt(
    *,
    lines: List[Dict[str, str]],
    need_summary: bool,
    need_closing: bool,
    quality_cfg: ScriptQualityGateConfig,
) -> str:
    tail = lines[-quality_cfg.semantic_tail_lines :]
    rows: List[str] = []
    for line in tail:
        speaker = line.get("speaker", "Host")
        text = line.get("text", "")
        rows.append(f"{speaker}: {text}")
    dialogue = "\n".join(rows)
    checks: List[str] = []
    if need_summary:
        checks.append(
            "- summary_semantic: true if there is a recap/summary idea near the ending, even with paraphrases."
        )
    if need_closing:
        checks.append(
            "- closing_semantic: true if there is a farewell/ending intent (for example, thanks to audience, wrap-up, next episode invitation), even if wording is not exact."
        )
    checks_text = "\n".join(checks) if checks else "- no_checks"
    return (
        "Classify semantic ending signals in this podcast tail (language may vary).\n"
        "Return ONLY JSON object with keys:\n"
        "summary_semantic, closing_semantic, confidence, evidence.\n"
        "confidence is a number 0..1.\n"
        "evidence is an array of short strings.\n\n"
        "Checks:\n"
        f"{checks_text}\n\n"
        "TAIL DIALOGUE:\n"
        f"{dialogue}\n"
    )


def evaluate_script_quality(
    *,
    validated_payload: Dict[str, List[Dict[str, str]]],
    script_cfg: ScriptConfig,
    quality_cfg: ScriptQualityGateConfig,
    script_path: str,
    client: OpenAIClient | None = None,
    logger: Logger | None = None,
) -> Dict[str, Any]:
    started = time.time()
    raw_lines = list(validated_payload.get("lines", []))
    lines = harden_script_structure(
        raw_lines,
        max_consecutive_same_speaker=quality_cfg.max_consecutive_same_speaker,
    )
    structural_hardening_applied = lines != raw_lines
    word_count = count_words_from_lines(lines)
    line_count = len(lines)
    summary_ok = _has_summary(lines)
    closing_ok = _has_closing(lines)
    internal_workflow_count = _internal_workflow_line_count(lines)
    max_run = _max_consecutive_speaker_run(lines)
    repeat_ratio = _repeat_line_ratio(lines)
    min_words_required = int(round(float(script_cfg.min_words) * quality_cfg.min_words_ratio))
    max_words_allowed = int(round(float(script_cfg.max_words) * quality_cfg.max_words_ratio))

    rules: Dict[str, bool] = {
        "word_count_min_ok": word_count >= max(1, min_words_required),
        "word_count_max_ok": word_count <= max(1, max_words_allowed),
        "max_consecutive_speaker_ok": max_run <= quality_cfg.max_consecutive_same_speaker,
        "repeat_line_ratio_ok": repeat_ratio <= quality_cfg.max_repeat_line_ratio,
        "summary_ok": (not quality_cfg.require_summary) or summary_ok,
        "closing_ok": (not quality_cfg.require_closing) or closing_ok,
        "open_questions_resolved_ok": not _has_unanswered_tail_question(lines),
        "no_internal_workflow_meta_ok": internal_workflow_count == 0,
    }
    semantic_info: Dict[str, Any] = {
        "enabled": bool(quality_cfg.semantic_rule_fallback),
        "called": False,
        "used": False,
        "error": False,
        "skipped_reason": "",
        "summary_semantic": None,
        "closing_semantic": None,
        "confidence": None,
        "confidence_gate_passed": None,
        "evidence": [],
    }
    if quality_cfg.semantic_rule_fallback:
        need_summary = bool(quality_cfg.require_summary and not rules.get("summary_ok", True))
        need_closing = bool(quality_cfg.require_closing and not rules.get("closing_ok", True))
        if need_summary or need_closing:
            semantic_info["called"] = True
            if client is None or not hasattr(client, "generate_freeform_text"):
                semantic_info["skipped_reason"] = "semantic_client_unavailable"
            else:
                prompt = _build_semantic_rule_prompt(
                    lines=lines,
                    need_summary=need_summary,
                    need_closing=need_closing,
                    quality_cfg=quality_cfg,
                )
                try:
                    raw = client.generate_freeform_text(
                        prompt=prompt,
                        max_output_tokens=quality_cfg.semantic_max_output_tokens,
                        stage="script_quality_semantic_rules",
                    )
                    payload = _extract_json_object(raw)
                    summary_semantic = _to_bool(payload.get("summary_semantic", False))
                    closing_semantic = _to_bool(payload.get("closing_semantic", False))
                    confidence = _score_confidence(payload, "confidence")
                    confidence_gate_passed = confidence is None or confidence >= quality_cfg.semantic_min_confidence
                    evidence_raw = payload.get("evidence", [])
                    evidence: List[str] = []
                    if isinstance(evidence_raw, list):
                        for item in evidence_raw[:6]:
                            text = str(item or "").strip()
                            if text:
                                evidence.append(text)
                    semantic_info["summary_semantic"] = summary_semantic
                    semantic_info["closing_semantic"] = closing_semantic
                    semantic_info["confidence"] = confidence
                    semantic_info["confidence_gate_passed"] = confidence_gate_passed
                    semantic_info["evidence"] = evidence
                    if confidence_gate_passed:
                        used = False
                        if need_summary and summary_semantic:
                            rules["summary_ok"] = True
                            used = True
                        if need_closing and closing_semantic:
                            rules["closing_ok"] = True
                            used = True
                        semantic_info["used"] = used
                except Exception as exc:  # noqa: BLE001
                    semantic_info["error"] = True
                    semantic_info["skipped_reason"] = str(exc)

    reasons_structural: List[str] = [name for name, ok in rules.items() if not ok]
    reasons_llm: List[str] = []
    llm_score_failures: List[str] = []
    rules_pass = all(rules.values())
    llm_called = False
    llm_sampled = False
    llm_error = False
    llm_explicit_fail = False
    llm_editorial_fail = False
    llm_truncation_claims_filtered = 0
    tail_structurally_complete = _tail_is_structurally_complete(lines)
    scores: Dict[str, float | None] = {
        "overall_score": None,
        "cadence_score": None,
        "logic_score": None,
        "clarity_score": None,
    }
    structural_only_rollout = _hard_fail_structural_only_enabled()
    strict_score_blocking = _strict_score_blocking_enabled()
    critical_score_threshold = _critical_score_threshold()

    if quality_cfg.evaluator in {"llm", "hybrid"}:
        llm_sampled = quality_cfg.evaluator == "llm" or (rules_pass and _should_sample_llm(lines, quality_cfg.llm_sample_rate))
        if llm_sampled:
            llm_called = True
            if client is None:
                reasons_llm.append("llm_client_unavailable")
            else:
                prompt = _build_llm_prompt(
                    lines=lines,
                    script_cfg=script_cfg,
                    quality_cfg=quality_cfg,
                    word_count=word_count,
                )
                try:
                    raw = client.generate_freeform_text(
                        prompt=prompt,
                        max_output_tokens=quality_cfg.llm_max_output_tokens,
                        stage="script_quality_eval",
                    )
                    payload = _extract_json_object(raw)
                    scores["overall_score"] = _score(payload, "overall_score")
                    scores["cadence_score"] = _score(payload, "cadence_score")
                    scores["logic_score"] = _score(payload, "logic_score")
                    scores["clarity_score"] = _score(payload, "clarity_score")
                    llm_reasons = payload.get("reasons", [])
                    if isinstance(llm_reasons, list):
                        for item in llm_reasons:
                            reason = str(item or "").strip()
                            if reason:
                                reasons_llm.append(reason)
                    raw_llm_reasons = list(reasons_llm)
                    if tail_structurally_complete and reasons_llm:
                        reasons_llm = [
                            reason
                            for reason in reasons_llm
                            if not _is_llm_tail_truncation_claim(reason)
                        ]
                        llm_truncation_claims_filtered = max(
                            0, len(raw_llm_reasons) - len(reasons_llm)
                        )
                    if scores["overall_score"] is None or scores["overall_score"] < quality_cfg.min_overall_score:
                        llm_score_failures.append("overall_score_below_threshold")
                    if scores["cadence_score"] is None or scores["cadence_score"] < quality_cfg.min_cadence_score:
                        llm_score_failures.append("cadence_score_below_threshold")
                    if scores["logic_score"] is None or scores["logic_score"] < quality_cfg.min_logic_score:
                        llm_score_failures.append("logic_score_below_threshold")
                    if scores["clarity_score"] is None or scores["clarity_score"] < quality_cfg.min_clarity_score:
                        llm_score_failures.append("clarity_score_below_threshold")
                    explicit_pass = payload.get("pass")
                    if explicit_pass is False:
                        llm_explicit_fail = True
                    if (
                        tail_structurally_complete
                        and llm_explicit_fail
                        and raw_llm_reasons
                        and all(_is_llm_tail_truncation_claim(reason) for reason in raw_llm_reasons)
                        and llm_truncation_claims_filtered >= len(raw_llm_reasons)
                    ):
                        # Ignore explicit-fail if the evaluator only reported tail truncation
                        # and deterministic checks confirm the ending is complete.
                        llm_explicit_fail = False
                    llm_editorial_fail = bool(llm_score_failures or llm_explicit_fail or reasons_llm)
                except Exception as exc:  # noqa: BLE001
                    llm_error = True
                    # Degrade to deterministic rules when evaluator transport/parsing fails.
                    reasons_llm.append(f"llm_evaluator_error:{exc}")
                    llm_editorial_fail = False

    structural_fail = bool(reasons_structural)
    score_values = [score for score in scores.values() if score is not None]
    critical_score_failed = bool(score_values) and any(
        float(score) <= critical_score_threshold for score in score_values
    )
    score_hard_fail_eligible = bool(
        strict_score_blocking
        and llm_editorial_fail
        and critical_score_failed
    )
    hard_fail_eligible = bool(structural_fail or score_hard_fail_eligible)
    if not structural_only_rollout and llm_editorial_fail and not llm_error:
        hard_fail_eligible = True
    final_pass = not hard_fail_eligible

    reasons: List[str] = []
    reasons.extend(reasons_structural)
    reasons.extend(reasons_llm)
    reasons.extend(llm_score_failures)
    if llm_explicit_fail:
        reasons.append("llm_explicit_fail")

    dedup_reasons: List[str] = []
    for reason in reasons:
        normalized = str(reason or "").strip()
        if normalized and normalized not in dedup_reasons:
            dedup_reasons.append(normalized)

    report: Dict[str, Any] = {
        "component": "script_quality_gate",
        "status": "passed" if final_pass else "failed",
        "pass": bool(final_pass),
        "action": quality_cfg.action,
        "evaluator": quality_cfg.evaluator,
        "profile": script_cfg.profile_name,
        "script_path": script_path,
        "line_count": line_count,
        "word_count": word_count,
        "structural_hardening_applied": structural_hardening_applied,
        "min_words_required": max(1, min_words_required),
        "max_words_allowed": max(1, max_words_allowed),
        "max_consecutive_same_speaker": max_run,
        "repeat_line_ratio": round(float(repeat_ratio), 4),
        "rules": rules,
        "scores": scores,
        "reasons_structural": [reason for reason in dedup_reasons if reason in set(reasons_structural)],
        "reasons_llm": [
            reason
            for reason in dedup_reasons
            if reason in set(reasons_llm + llm_score_failures + (["llm_explicit_fail"] if llm_explicit_fail else []))
        ],
        "llm_score_failures": list(dict.fromkeys(llm_score_failures)),
        "evidence_structural": {
            "failed_rules": [name for name, ok in rules.items() if not ok],
            "max_consecutive_speaker": max_run,
            "repeat_line_ratio": round(float(repeat_ratio), 4),
            "word_count": int(word_count),
            "min_words_required": int(max(1, min_words_required)),
            "max_words_allowed": int(max(1, max_words_allowed)),
            "internal_workflow_meta_lines": int(internal_workflow_count),
            "semantic_fallback_used": bool(semantic_info.get("used", False)),
        },
        "llm_called": llm_called,
        "llm_sampled": llm_sampled,
        "llm_error": llm_error,
        "llm_explicit_fail": bool(llm_explicit_fail),
        "llm_editorial_fail": bool(llm_editorial_fail),
        "llm_truncation_claims_filtered": int(llm_truncation_claims_filtered),
        "llm_degraded_to_rules": bool(llm_error and quality_cfg.evaluator in {"llm", "hybrid"}),
        "semantic_rule_fallback": semantic_info,
        "hard_fail_eligible": bool(hard_fail_eligible),
        "score_blocking_enabled": bool(strict_score_blocking),
        "score_blocking_critical_threshold": float(critical_score_threshold),
        "score_blocking_critical_failed": bool(critical_score_failed),
        "hard_fail_structural_only_rollout": bool(structural_only_rollout),
        "editorial_warn_only": bool((not hard_fail_eligible) and llm_editorial_fail),
        "reasons": dedup_reasons,
        "failure_kind": (ERROR_KIND_SCRIPT_QUALITY if not final_pass else None),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if logger is not None:
        if semantic_info.get("called"):
            if semantic_info.get("error"):
                logger.warn(
                    "script_quality_semantic_fallback_failed",
                    error=semantic_info.get("skipped_reason"),
                )
            else:
                logger.info(
                    "script_quality_semantic_fallback_evaluated",
                    used=bool(semantic_info.get("used", False)),
                    confidence=semantic_info.get("confidence"),
                    summary_semantic=semantic_info.get("summary_semantic"),
                    closing_semantic=semantic_info.get("closing_semantic"),
                )
        logger.info(
            "script_quality_gate_evaluated",
            status=report["status"],
            action=quality_cfg.action,
            evaluator=quality_cfg.evaluator,
            hard_fail_eligible=bool(report.get("hard_fail_eligible", False)),
            editorial_warn_only=bool(report.get("editorial_warn_only", False)),
            reasons=dedup_reasons,
        )
    return report


def write_quality_report(path: str, report: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _with_repair_metadata(
    report: Dict[str, Any],
    *,
    initial_pass: bool,
    attempted: bool,
    succeeded: bool,
    changed: bool,
    attempts_used: int,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out = dict(report)
    out["initial_pass"] = bool(initial_pass)
    out["repair_attempted"] = bool(attempted)
    out["repair_succeeded"] = bool(succeeded)
    out["repair_changed_script"] = bool(changed)
    out["repair_attempts_used"] = int(max(0, attempts_used))
    out["repair_history"] = history
    return out


def _build_repair_prompt(
    *,
    payload: Dict[str, List[Dict[str, str]]],
    report: Dict[str, Any],
    script_cfg: ScriptConfig,
    quality_cfg: ScriptQualityGateConfig,
) -> str:
    reasons = report.get("reasons", [])
    reason_lines: List[str] = []
    if isinstance(reasons, list):
        for item in reasons[:12]:
            text = str(item or "").strip()
            if text:
                reason_lines.append(f"- {text}")
    compact = canonical_json(payload)
    if len(compact) > quality_cfg.repair_max_input_chars:
        compact = compact[: quality_cfg.repair_max_input_chars]
    reasons_text = "\n".join(reason_lines) if reason_lines else "- generic_quality_improvement"
    return textwrap.dedent(
        f"""
        Rewrite and improve this Spanish podcast script JSON.
        Goal: satisfy quality checks while preserving core meaning and facts.

        Constraints:
        - Return ONLY JSON object with key "lines".
        - Keep each line fields: speaker, role, instructions, text.
        - role must be Host1 or Host2.
        - Keep spoken text in Spanish.
        - Keep instructions in English single-line format.
        - Preserve conversation flow and avoid generic filler.
        - Do not include internal workflow/tooling disclosures in spoken text (for example script paths, shell commands, DailyRead pipeline notes, Tavily, Serper).
        - Alternate turns strictly between Host1 and Host2 (no consecutive turns by the same role).
        - Avoid repetitive line openers across consecutive turns (especially repeated "Y ...").
        - Prefer natural Spanish technical phrasing and avoid unnecessary anglicisms.
        - Do not use explicit section labels in spoken text (for example: "Bloque 1", "Bloque 2", "Section 3", "Part 4").
        - Use elegant transitions between topics so the dialogue feels natural.
        - Keep each spoken line concise (usually 1-2 sentences).
        - Avoid repeating the same transition template in consecutive turns.
        - Keep the conversation interactive: include direct host-to-host questions and answers, not two parallel monologues.
        - Keep question cadence natural (avoid chains of back-to-back questions without breathing turns).
        - Add occasional respectful contrast/challenge between hosts to improve engagement, then resolve clearly.
        - Use occasional concrete analogies to explain technical points naturally.
        - Humor is optional: if present, keep it brief, respectful, and relevant to the topic.
        - Do not leave open questions unresolved before the summary/closing.
        - If a host asks a direct question near the end, include an explicit answer from the counterpart host before "En Resumen" or farewell.
        - In the final 2-3 turns before summary/farewell, avoid introducing new questions.
        - Keep pre-summary closure turns declarative and conclusive.
        - Keep recap lines concise and listener-friendly; split overloaded closing lists into two short turns if needed.
        - Avoid abrupt endings (no trailing ellipsis, dangling conjunctions, or clipped phrases).
        - Maintain target episode size around {script_cfg.min_words}-{script_cfg.max_words} words.
        - Ensure natural summary and closing near the end.

        Issues to fix:
        {reasons_text}

        INPUT JSON:
        {compact}
        """
    ).strip()


def _deterministic_repair_payload(
    payload: Dict[str, List[Dict[str, str]]],
    *,
    quality_cfg: ScriptQualityGateConfig,
) -> Dict[str, List[Dict[str, str]]]:
    base_lines = list(payload.get("lines", []))
    lines = ensure_en_resumen(base_lines)
    lines = ensure_farewell_last(lines)
    lines = harden_script_structure(
        lines,
        max_consecutive_same_speaker=quality_cfg.max_consecutive_same_speaker,
    )
    return {"lines": lines}


def _repair_output_token_budget(
    *,
    payload: Dict[str, List[Dict[str, str]]],
    quality_cfg: ScriptQualityGateConfig,
) -> int:
    compact = canonical_json(payload)
    # Heuristic: JSON-heavy payloads need higher output budget than plain word count.
    approx_needed = int(math.ceil(float(len(compact)) / 3.5)) + 160
    hard_cap = max(
        quality_cfg.repair_max_output_tokens,
        _env_int("SCRIPT_QUALITY_GATE_REPAIR_OUTPUT_TOKENS_HARD_CAP", 6400),
    )
    return max(
        quality_cfg.repair_max_output_tokens,
        min(hard_cap, approx_needed),
    )


def attempt_script_quality_repair(
    *,
    validated_payload: Dict[str, List[Dict[str, str]]],
    initial_report: Dict[str, Any],
    script_cfg: ScriptConfig,
    quality_cfg: ScriptQualityGateConfig,
    script_path: str,
    client: OpenAIClient | None,
    logger: Logger | None = None,
    stage_prefix: str = "script_quality_repair",
    cancel_check: Callable[[], bool] | None = None,
    total_timeout_seconds: float | None = None,
    attempt_timeout_seconds: int | None = None,
) -> Dict[str, Any]:
    initial_pass = bool(initial_report.get("pass", False))
    history: List[Dict[str, Any]] = []
    validated_input_payload = validate_script_payload(validated_payload)
    current_payload = validated_input_payload
    current_payload = {
        "lines": harden_script_structure(
            list(current_payload.get("lines", [])),
            max_consecutive_same_speaker=quality_cfg.max_consecutive_same_speaker,
        )
    }
    current_report = dict(initial_report)
    initial_hash = content_hash(canonical_json(current_payload))
    initial_word_count = count_words_from_lines(list(current_payload.get("lines", [])))
    min_words_required = max(1, int(initial_report.get("min_words_required", script_cfg.min_words)))
    repair_timeout_reached = False
    hard_fail_eligible = bool(initial_report.get("hard_fail_eligible", not initial_pass))
    deadline: float | None = None
    if total_timeout_seconds is not None:
        try:
            parsed_timeout = float(total_timeout_seconds)
            if math.isfinite(parsed_timeout) and parsed_timeout > 0.0:
                deadline = time.time() + parsed_timeout
        except (TypeError, ValueError):
            deadline = None

    def _stage_timed_out() -> bool:
        return deadline is not None and time.time() >= deadline

    def _check_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Interrupted during script quality repair")

    if initial_pass:
        return {
            "payload": current_payload,
            "report": _with_repair_metadata(
                current_report,
                initial_pass=True,
                attempted=False,
                succeeded=False,
                changed=False,
                attempts_used=0,
                history=history,
            ),
            "repaired": False,
        }

    if not hard_fail_eligible:
        history.append({"attempt": 0, "status": "skipped", "reason": "hard_fail_not_eligible"})
        return {
            "payload": current_payload,
            "report": _with_repair_metadata(
                current_report,
                initial_pass=False,
                attempted=False,
                succeeded=False,
                changed=False,
                attempts_used=0,
                history=history,
            ),
            "repaired": False,
        }

    if not quality_cfg.auto_repair or quality_cfg.repair_attempts <= 0:
        return {
            "payload": current_payload,
            "report": _with_repair_metadata(
                current_report,
                initial_pass=False,
                attempted=False,
                succeeded=False,
                changed=False,
                attempts_used=0,
                history=history,
            ),
            "repaired": False,
        }

    _check_cancelled()
    deterministic_payload = _deterministic_repair_payload(
        current_payload,
        quality_cfg=quality_cfg,
    )
    deterministic_hash = content_hash(canonical_json(deterministic_payload))
    if deterministic_hash != initial_hash:
        if _stage_timed_out():
            history.append({"attempt": 0, "status": "timeout", "reason": "quality_repair_timeout"})
            current_report = dict(current_report)
            timeout_reasons = list(current_report.get("reasons", []))
            if "quality_repair_timeout" not in timeout_reasons:
                timeout_reasons.append("quality_repair_timeout")
            current_report["reasons"] = timeout_reasons
            current_report["pass"] = False
            current_report["status"] = "failed"
            current_report["hard_fail_eligible"] = True
            current_report["editorial_warn_only"] = False
            current_report["failure_kind"] = ERROR_KIND_SCRIPT_QUALITY
            current_report["repair_timeout_reached"] = True
            return {
                "payload": current_payload,
                "report": _with_repair_metadata(
                    current_report,
                    initial_pass=False,
                    attempted=True,
                    succeeded=False,
                    changed=False,
                    attempts_used=0,
                    history=history,
                ),
                "repaired": False,
            }
        deterministic_report = evaluate_script_quality(
            validated_payload=deterministic_payload,
            script_cfg=script_cfg,
            quality_cfg=quality_cfg,
            script_path=script_path,
            client=client,
            logger=logger,
        )
        deterministic_pass = bool(deterministic_report.get("pass", False))
        history.append(
            {
                "attempt": 0,
                "status": ("deterministic_passed" if deterministic_pass else "deterministic_failed"),
                "reasons": list(deterministic_report.get("reasons", [])),
                "word_count": deterministic_report.get("word_count"),
            }
        )
        if deterministic_pass:
            changed = deterministic_hash != initial_hash
            return {
                "payload": deterministic_payload,
                "report": _with_repair_metadata(
                    deterministic_report,
                    initial_pass=False,
                    attempted=True,
                    succeeded=True,
                    changed=changed,
                    attempts_used=0,
                    history=history,
                ),
                "repaired": changed,
            }
        if not quality_cfg.repair_revert_on_fail:
            current_payload = deterministic_payload
            current_report = deterministic_report

    if client is None or not hasattr(client, "generate_script_json"):
        history.append({"attempt": 0, "status": "skipped", "error": "repair_client_unavailable"})
        return {
            "payload": current_payload,
            "report": _with_repair_metadata(
                current_report,
                initial_pass=False,
                attempted=True,
                succeeded=False,
                changed=False,
                attempts_used=0,
                history=history,
            ),
            "repaired": False,
        }

    attempts_used = 0
    for attempt in range(1, quality_cfg.repair_attempts + 1):
        _check_cancelled()
        if _stage_timed_out():
            repair_timeout_reached = True
            history.append({"attempt": attempt, "status": "timeout", "reason": "quality_repair_timeout"})
            current_report = dict(current_report)
            timeout_reasons = list(current_report.get("reasons", []))
            if "quality_repair_timeout" not in timeout_reasons:
                timeout_reasons.append("quality_repair_timeout")
            current_report["reasons"] = timeout_reasons
            current_report["pass"] = False
            current_report["status"] = "failed"
            current_report["hard_fail_eligible"] = True
            current_report["editorial_warn_only"] = False
            current_report["failure_kind"] = ERROR_KIND_SCRIPT_QUALITY
            current_report["repair_timeout_reached"] = True
            break
        attempts_used = attempt
        repair_max_output_tokens = _repair_output_token_budget(
            payload=current_payload,
            quality_cfg=quality_cfg,
        )
        if logger is not None and repair_max_output_tokens > quality_cfg.repair_max_output_tokens:
            logger.info(
                "script_quality_repair_budget_increased",
                attempt=attempt,
                max_output_tokens=repair_max_output_tokens,
            )
        prompt = _build_repair_prompt(
            payload=current_payload,
            report=current_report,
            script_cfg=script_cfg,
            quality_cfg=quality_cfg,
        )
        try:
            call_kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "schema": SCRIPT_JSON_SCHEMA,
                "max_output_tokens": repair_max_output_tokens,
                "stage": f"{stage_prefix}_{attempt}",
            }
            if attempt_timeout_seconds is not None and int(attempt_timeout_seconds) > 0:
                call_kwargs["timeout_seconds_override"] = int(attempt_timeout_seconds)
            try:
                repaired_raw = client.generate_script_json(**call_kwargs)
            except TypeError as exc:
                if "timeout_seconds_override" not in str(exc):
                    raise
                call_kwargs.pop("timeout_seconds_override", None)
                repaired_raw = client.generate_script_json(**call_kwargs)
            candidate_payload = validate_script_payload(repaired_raw)
            candidate_payload = {
                "lines": harden_script_structure(
                    list(candidate_payload.get("lines", [])),
                    max_consecutive_same_speaker=quality_cfg.max_consecutive_same_speaker,
                )
            }
            candidate_report = evaluate_script_quality(
                validated_payload=candidate_payload,
                script_cfg=script_cfg,
                quality_cfg=quality_cfg,
                script_path=script_path,
                client=client,
                logger=logger,
            )
            candidate_word_count = int(
                candidate_report.get(
                    "word_count",
                    count_words_from_lines(list(candidate_payload.get("lines", []))),
                )
            )
            min_allowed_after_repair = max(
                min_words_required,
                int(round(float(initial_word_count) * float(quality_cfg.repair_min_word_ratio))),
            )
            if candidate_word_count < min_allowed_after_repair:
                history.append(
                    {
                        "attempt": attempt,
                        "status": "rejected_guardrail",
                        "word_count": candidate_word_count,
                        "min_allowed_after_repair": min_allowed_after_repair,
                        "reasons": list(candidate_report.get("reasons", [])),
                    }
                )
                if logger is not None:
                    logger.warn(
                        "script_quality_repair_rejected_guardrail",
                        attempt=attempt,
                        word_count=candidate_word_count,
                        min_allowed_after_repair=min_allowed_after_repair,
                    )
                continue
            history.append(
                {
                    "attempt": attempt,
                    "status": ("passed" if bool(candidate_report.get("pass", False)) else "failed"),
                    "reasons": list(candidate_report.get("reasons", [])),
                    "word_count": candidate_report.get("word_count"),
                }
            )
            if bool(candidate_report.get("pass", False)):
                final_hash = content_hash(canonical_json(candidate_payload))
                changed = final_hash != initial_hash
                return {
                    "payload": candidate_payload,
                    "report": _with_repair_metadata(
                        candidate_report,
                        initial_pass=False,
                        attempted=True,
                        succeeded=True,
                        changed=changed,
                        attempts_used=attempt,
                        history=history,
                    ),
                    "repaired": changed,
                }
            if not quality_cfg.repair_revert_on_fail:
                current_payload = candidate_payload
                current_report = candidate_report
        except Exception as exc:  # noqa: BLE001
            history.append({"attempt": attempt, "status": "error", "error": str(exc)})
            if logger is not None:
                logger.warn("script_quality_repair_attempt_failed", attempt=attempt, error=str(exc))

    if quality_cfg.repair_revert_on_fail:
        final_payload = validated_input_payload
        final_report = dict(initial_report)
    else:
        final_payload = current_payload
        final_report = current_report
    if repair_timeout_reached:
        final_report = current_report
        if not quality_cfg.repair_revert_on_fail:
            final_payload = current_payload

    final_hash = content_hash(canonical_json(final_payload))
    baseline_hash = (
        content_hash(canonical_json(validated_input_payload))
        if quality_cfg.repair_revert_on_fail
        else initial_hash
    )
    final_changed = final_hash != baseline_hash
    repaired = final_changed and (
        bool(final_report.get("pass", False)) or not quality_cfg.repair_revert_on_fail
    )
    return {
        "payload": final_payload,
        "report": _with_repair_metadata(
            final_report,
            initial_pass=False,
            attempted=True,
            succeeded=False,
            changed=final_changed,
            attempts_used=attempts_used,
            history=history,
        ),
        "repaired": repaired,
    }

