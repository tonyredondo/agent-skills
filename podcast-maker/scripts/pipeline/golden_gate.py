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
    with open(path, 'r', encoding='utf-8') as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object at {path}')
    return value


def load_baseline_metrics(path: str) -> Dict[str, Dict[str, Any]]:
    payload = _load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            out[str(key)] = value
    return out


def _baseline_metrics_from_dict(baseline_metrics: Dict[str, Any]) -> ScriptMetrics:
    return ScriptMetrics(
        line_count=int(baseline_metrics.get('line_count', 1)),
        word_count=int(baseline_metrics.get('word_count', 1)),
        unique_speakers=int(baseline_metrics.get('unique_speakers', 2)),
        alternating_ratio=float(baseline_metrics.get('alternating_ratio', 1.0)),
        host2_turn_ratio=float(baseline_metrics.get('host2_turn_ratio', 0.5)),
        host2_push_ratio=float(baseline_metrics.get('host2_push_ratio', 0.5)),
        scaffold_phrase_hits=int(baseline_metrics.get('scaffold_phrase_hits', 0)),
        meta_language_ok=bool(baseline_metrics.get('meta_language_ok', True)),
    )


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
    baseline = load_baseline_metrics(baseline_path)
    case_results: List[GoldenCaseResult] = []
    candidate_dir_abs = os.path.abspath(candidate_dir)

    for name, baseline_metrics in sorted(baseline.items()):
        candidate_path = os.path.join(candidate_dir_abs, f'{name}.json')
        if not os.path.exists(candidate_path):
            if fixtures_dir:
                fallback_path = os.path.join(os.path.abspath(fixtures_dir), f'{name}.json')
                if os.path.exists(fallback_path):
                    candidate_path = fallback_path
                else:
                    case_results.append(
                        GoldenCaseResult(
                            name=name,
                            pass_case=False,
                            current={},
                            baseline=baseline_metrics,
                            comparison={'error': 'missing_candidate'},
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
                        comparison={'error': 'missing_candidate'},
                    )
                )
                continue

        payload = _load_json(candidate_path)
        current = compute_script_metrics(payload)
        baseline_obj = _baseline_metrics_from_dict(baseline_metrics)
        cmp = compare_against_baseline(
            current,
            baseline_obj,
            min_word_ratio=min_word_ratio,
            max_word_ratio=max_word_ratio,
            min_line_ratio=min_line_ratio,
            max_line_ratio=max_line_ratio,
        )
        pass_case = bool(
            cmp.get('word_ratio_ok')
            and cmp.get('line_ratio_ok')
            and cmp.get('speaker_count_ok')
            and cmp.get('alternation_ok')
            and cmp.get('host2_presence_ok')
            and cmp.get('host2_push_ok')
            and cmp.get('scaffold_control_ok')
            and cmp.get('meta_language_ok')
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
        'overall_pass': overall_pass,
        'cases': [case.__dict__ for case in case_results],
        'candidate_dir': candidate_dir_abs,
        'fixtures_dir': os.path.abspath(fixtures_dir) if fixtures_dir else '',
        'baseline_path': baseline_path,
    }
