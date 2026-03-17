#!/usr/bin/env python3
from __future__ import annotations

"""Script quality evaluation using structural and editorial gates."""

import json
import os
from typing import Any, Dict, List

from .config import ScriptConfig
from .editorial_gate import EditorialGate
from .errors import ERROR_KIND_SCRIPT_QUALITY
from .logging_utils import Logger
from .openai_client import OpenAIClient
from .schema import count_words_from_lines, validate_script_payload
from .script_quality_gate_config import (
    ScriptQualityGateConfig,
    critical_score_threshold,
    hard_fail_structural_only_enabled,
    strict_score_blocking_enabled,
)
from .structural_gate import StructuralGate


class ScriptQualityGateError(RuntimeError):
    """Raised when the script quality gate rejects a script."""

    def __init__(self, message: str, *, report: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.failure_kind = ERROR_KIND_SCRIPT_QUALITY
        self.report = dict(report or {})


class _NullLogger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warn(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _resolve_logger(logger: Logger | None) -> Logger | _NullLogger:
    return logger if logger is not None else _NullLogger()


def _should_sample_llm(quality_cfg: ScriptQualityGateConfig, client: OpenAIClient | None) -> bool:
    if client is None:
        return False
    if quality_cfg.evaluator not in {"hybrid", "llm"}:
        return False
    return float(quality_cfg.llm_sample_rate) > 0.0


def _average_editorial_score(scores: Dict[str, float]) -> float:
    values = [float(value) for value in scores.values()]
    if not values:
        return 0.0
    return round(sum(values) / float(len(values)), 4)


def _build_reasons(
    *,
    structural_report: Dict[str, Any],
    editorial_report: Dict[str, Any],
    score_failures: List[str],
) -> tuple[List[str], List[str], List[str]]:
    structural_reasons = [
        str(item).strip()
        for item in list(structural_report.get("notes", []) or [])
        if str(item).strip()
    ]
    editorial_reasons = []
    for failure in list(editorial_report.get("failures", []) or []):
        if not isinstance(failure, dict):
            continue
        failure_type = str(
            failure.get("failure_type", failure.get("type", ""))
        ).strip()
        if failure_type:
            editorial_reasons.append(failure_type)
    combined: List[str] = []
    for reason in structural_reasons + editorial_reasons + score_failures:
        if reason and reason not in combined:
            combined.append(reason)
    return combined, structural_reasons, editorial_reasons


def evaluate_script_quality(
    *,
    validated_payload: Dict[str, List[Dict[str, Any]]],
    script_cfg: ScriptConfig,
    quality_cfg: ScriptQualityGateConfig,
    script_path: str,
    client: OpenAIClient | None,
    logger: Logger | None = None,
    source_context: str | None = None,
    episode_plan: Dict[str, Any] | None = None,
    evidence_map: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate script quality using the redesigned structural/editorial gates."""

    resolved_logger = _resolve_logger(logger)
    validated = validate_script_payload(validated_payload)
    lines = list(validated.get("lines", []))

    structural_gate = StructuralGate(
        max_consecutive_same_speaker=max(1, int(quality_cfg.max_consecutive_same_speaker))
    )
    structural_report = structural_gate.evaluate(lines=lines)

    llm_sampled = _should_sample_llm(quality_cfg, client)
    editorial_gate = EditorialGate(
        client=client if llm_sampled else None,
        logger=resolved_logger,
    )
    editorial_report = editorial_gate.evaluate(
        script_lines=lines,
        episode_plan=dict(episode_plan or {"beats": []}),
        evidence_map=dict(evidence_map or {}),
        profile_name=script_cfg.profile_name,
        min_words=script_cfg.min_words,
        max_words=script_cfg.max_words,
    )
    editorial_scores = dict(editorial_report.get("scores", {}) or {})
    score_failures: List[str] = []
    if strict_score_blocking_enabled():
        threshold = float(critical_score_threshold())
        for key, value in editorial_scores.items():
            if float(value) < threshold:
                score_failures.append(f"critical_editorial_score:{key}")

    combined_reasons, structural_reasons, editorial_reasons = _build_reasons(
        structural_report=structural_report,
        editorial_report=editorial_report,
        score_failures=score_failures,
    )

    structural_pass = bool(structural_report.get("pass", False))
    editorial_pass = bool(editorial_report.get("pass", False)) and not score_failures
    passed = structural_pass and editorial_pass
    failure_kind = ERROR_KIND_SCRIPT_QUALITY if not passed else None
    report = {
        "component": "script_quality_gate",
        "status": "passed" if passed else "failed",
        "pass": passed,
        "action": quality_cfg.action,
        "evaluator": quality_cfg.evaluator,
        "profile": script_cfg.profile_name,
        "script_path": script_path,
        "line_count": int(structural_report.get("line_count", len(lines))),
        "word_count": int(structural_report.get("word_count", count_words_from_lines(lines))),
        "rules": dict(structural_report.get("checks", {}) or {}),
        "scores": {
            "overall_score": _average_editorial_score(editorial_scores),
            "editorial_scores": editorial_scores,
        },
        "reasons_structural": structural_reasons,
        "reasons_llm": editorial_reasons,
        "llm_score_failures": score_failures,
        "evidence_structural": {"structural_report": structural_report},
        "evidence_editorial": {"editorial_report": editorial_report},
        "llm_called": bool(llm_sampled),
        "llm_sampled": bool(llm_sampled),
        "llm_error": False,
        "llm_explicit_fail": bool(editorial_reasons),
        "llm_editorial_fail": bool(not editorial_pass),
        "hard_fail_eligible": bool(not passed),
        "editorial_warn_only": bool(structural_pass and not editorial_pass and quality_cfg.action == "warn"),
        "reasons": combined_reasons,
        "failure_kind": failure_kind,
        "hard_fail_structural_only_enabled": bool(hard_fail_structural_only_enabled()),
    }
    resolved_logger.info(
        "script_quality_gate_evaluated",
        passed=passed,
        structural_pass=structural_pass,
        editorial_pass=editorial_pass,
        llm_called=bool(llm_sampled),
        script_path=script_path,
    )
    return report


def write_quality_report(path: str, report: Dict[str, Any]) -> None:
    """Write quality report JSON with stable formatting."""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
