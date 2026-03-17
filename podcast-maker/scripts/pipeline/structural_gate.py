#!/usr/bin/env python3
from __future__ import annotations

"""Structural validation for redesigned podcast scripts."""

from dataclasses import dataclass
from typing import Any, Dict, List

from .schema import count_words_from_lines, validate_script_payload
from .script_postprocess import evaluate_script_completeness, harden_script_structure


@dataclass(frozen=True)
class StructuralGate:
    """Validate mechanical script integrity independent of editorial taste."""

    max_consecutive_same_speaker: int = 1

    def finalize(self, *, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated = validate_script_payload({"lines": lines})["lines"]
        return harden_script_structure(
            list(validated),
            max_consecutive_same_speaker=max(1, int(self.max_consecutive_same_speaker)),
        )

    def evaluate(self, *, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = self.finalize(lines=lines)
        completeness = evaluate_script_completeness(normalized)
        roles = [str(item.get("role", "")).strip() for item in normalized]
        alternating = True
        notes: List[str] = []
        for idx in range(1, len(roles)):
            if roles[idx] == roles[idx - 1]:
                alternating = False
                notes.append(f"role repetition at indexes {idx-1}/{idx}")
                break
        report = {
            "pass": bool(
                normalized
                and alternating
                and bool(completeness.get("pass", False))
            ),
            "checks": {
                "json_valid": True,
                "non_empty_lines": bool(normalized),
                "alternating_roles": bool(alternating),
                "completeness_ok": bool(completeness.get("pass", False)),
            },
            "line_count": len(normalized),
            "word_count": count_words_from_lines(normalized),
            "notes": notes,
            "completeness": completeness,
        }
        if not report["checks"]["non_empty_lines"]:
            report["notes"].append("script contains no lines")
        if not report["checks"]["completeness_ok"]:
            report["notes"].extend(str(item) for item in list(completeness.get("reasons", [])) if str(item).strip())
        return report
