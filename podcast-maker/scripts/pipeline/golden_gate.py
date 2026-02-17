#!/usr/bin/env python3
from __future__ import annotations

"""Golden-suite gate evaluation over candidate script artifacts."""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .golden_metrics import ScriptMetrics, compare_against_baseline, compute_script_metrics


@dataclass(frozen=True)
class GoldenCaseResult:
    """Result payload for one golden case comparison."""

    name: str
    pass_case: bool
    current: Dict[str, Any]
    baseline: Dict[str, Any]
    comparison: Dict[str, Any]


def _load_json(path: str) -> Dict[str, Any]:
    """Load JSON object payload from disk."""
    with open(path, "r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def load_baseline_metrics(path: str) -> Dict[str, Dict[str, Any]]:
    """Load baseline metrics map keyed by case name."""
    payload = _load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            out[str(key)] = value
    return out


def evaluate_golden_suite(
    *,
    baseline_path: str,
    candidate_dir: str,
    fixtures_dir: Optional[str] = None,
    min_word_ratio: float = 0.6,
    max_word_ratio: float = 1.6,
    min_line_ratio: float = 0.6,
    max_line_ratio: float = 1.7,
) -> Dict[str, Any]:
    """Evaluate all baseline cases against candidate scripts."""
    baseline = load_baseline_metrics(baseline_path)
    case_results: List[GoldenCaseResult] = []
    candidate_dir_abs = os.path.abspath(candidate_dir)

    for name, baseline_metrics in sorted(baseline.items()):
        candidate_path = os.path.join(candidate_dir_abs, f"{name}.json")
        if not os.path.exists(candidate_path):
            # Backward-compatible fallback for local checks where candidates live in fixtures.
            if fixtures_dir:
                fallback_path = os.path.join(os.path.abspath(fixtures_dir), f"{name}.json")
                if os.path.exists(fallback_path):
                    candidate_path = fallback_path
                else:
                    case_results.append(
                        GoldenCaseResult(
                            name=name,
                            pass_case=False,
                            current={},
                            baseline=baseline_metrics,
                            comparison={"error": "missing_candidate"},
                        )
                    )
                    continue
            else:
                case_results.append(
                    GoldenCaseResult(
                        name=name,
                        pass_case=False,
                        current={},
                        baseline=baseline_metrics,
                        comparison={"error": "missing_candidate"},
                    )
                )
                continue
        payload = _load_json(candidate_path)
        current = compute_script_metrics(payload)
        baseline_obj = ScriptMetrics(
            line_count=int(baseline_metrics.get("line_count", 1)),
            word_count=int(baseline_metrics.get("word_count", 1)),
            unique_speakers=int(baseline_metrics.get("unique_speakers", 1)),
            has_recap_signal=bool(
                baseline_metrics.get(
                    "has_recap_signal",
                    baseline_metrics.get("has_en_resumen", True),
                )
            ),
            farewell_in_last_3=bool(baseline_metrics.get("farewell_in_last_3", True)),
            meta_language_ok=bool(baseline_metrics.get("meta_language_ok", True)),
        )
        cmp = compare_against_baseline(
            current,
            baseline_obj,
            min_word_ratio=min_word_ratio,
            max_word_ratio=max_word_ratio,
            min_line_ratio=min_line_ratio,
            max_line_ratio=max_line_ratio,
        )
        pass_case = bool(
            cmp.get("word_ratio_ok")
            and cmp.get("line_ratio_ok")
            and cmp.get("recap_ok")
            and cmp.get("farewell_ok")
            and cmp.get("meta_language_ok")
        )
        case_results.append(
            GoldenCaseResult(
                name=name,
                pass_case=pass_case,
                current=current.to_dict(),
                baseline=baseline_metrics,
                comparison=cmp,
            )
        )

    overall_pass = all(case.pass_case for case in case_results) if case_results else False
    return {
        "overall_pass": overall_pass,
        "cases": [case.__dict__ for case in case_results],
        "candidate_dir": candidate_dir_abs,
        "fixtures_dir": os.path.abspath(fixtures_dir) if fixtures_dir else "",
        "baseline_path": baseline_path,
    }

