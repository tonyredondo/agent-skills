#!/usr/bin/env python3
from __future__ import annotations

"""SLO/KPI tracking and window-based rollback gate evaluation."""

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class SLOTarget:
    """Per-profile SLO objectives used by rolling window checks."""

    success_rate_min: float
    p95_runtime_max_seconds: float
    p95_resume_runtime_max_seconds: float


SLO_TARGETS: Dict[str, SLOTarget] = {
    "short": SLOTarget(
        success_rate_min=0.98,
        p95_runtime_max_seconds=8 * 60,
        p95_resume_runtime_max_seconds=4 * 60,
    ),
    "standard": SLOTarget(
        success_rate_min=0.97,
        p95_runtime_max_seconds=20 * 60,
        p95_resume_runtime_max_seconds=10 * 60,
    ),
    "long": SLOTarget(
        success_rate_min=0.95,
        p95_runtime_max_seconds=40 * 60,
        p95_resume_runtime_max_seconds=18 * 60,
    ),
}

TECH_KPI_THRESHOLDS: Dict[str, float] = {
    "retry_rate_max": 0.15,
    "stuck_abort_rate_max": 0.02,
    "invalid_schema_rate_max": 0.03,
    "script_quality_rejected_rate_max": 0.05,
    "cost_error_p90_max_pct": 25.0,
}


def _default_history_path() -> str:
    """Resolve JSONL history path used for SLO telemetry events."""
    return os.environ.get("SLO_HISTORY_PATH", "./.podcast_slo_history.jsonl")


def append_slo_event(
    *,
    profile: str,
    component: str,
    status: str,
    elapsed_seconds: float,
    output_path: str = "",
    is_resume: bool = False,
    retry_rate: float | None = None,
    stuck_abort: bool = False,
    invalid_schema: bool = False,
    failure_kind: str | None = None,
    cost_estimation_error_pct: float | None = None,
    history_path: str | None = None,
) -> None:
    """Append one normalized SLO event to history JSONL file."""
    path = history_path or _default_history_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    event = {
        "ts": int(time.time()),
        "profile": profile,
        "component": component,
        "status": status,
        "elapsed_seconds": float(max(0.0, elapsed_seconds)),
        "output_path": output_path,
        "is_resume": bool(is_resume),
        "stuck_abort": bool(stuck_abort),
        "invalid_schema": bool(invalid_schema),
    }
    if retry_rate is not None:
        event["retry_rate"] = float(max(0.0, retry_rate))
    if failure_kind is not None and str(failure_kind).strip():
        event["failure_kind"] = str(failure_kind).strip().lower()
    if cost_estimation_error_pct is not None:
        event["cost_estimation_error_pct"] = float(abs(cost_estimation_error_pct))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _load_events(path: str) -> List[Dict[str, Any]]:
    """Load SLO history JSONL into dict events."""
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                out.append(payload)
    return out


def _p95(values: List[float]) -> float:
    """Compute empirical P95 from numeric sample list."""
    if not values:
        return math.inf
    data = sorted(values)
    idx = int(math.ceil(0.95 * len(data))) - 1
    idx = max(0, min(idx, len(data) - 1))
    return data[idx]


def _p90(values: List[float]) -> float:
    """Compute empirical P90 from numeric sample list."""
    if not values:
        return math.inf
    data = sorted(values)
    idx = int(math.ceil(0.90 * len(data))) - 1
    idx = max(0, min(idx, len(data) - 1))
    return data[idx]


def evaluate_slo_windows(
    *,
    profile: str,
    component: str | None = None,
    history_path: str | None = None,
    window_size: int = 20,
    required_failed_windows: int = 2,
) -> Dict[str, Any]:
    """Evaluate rolling SLO windows and decide rollback signal."""
    path = history_path or _default_history_path()
    events = [
        e
        for e in _load_events(path)
        if e.get("profile") == profile and (component is None or e.get("component") == component)
    ]
    target = SLO_TARGETS.get(profile, SLO_TARGETS["standard"])
    if not events:
        return {
            "has_data": False,
            "should_rollback": False,
            "reason": "no_data",
            "profile": profile,
            "component": component or "all",
        }

    windows: List[List[Dict[str, Any]]] = []
    max_needed = max(1, window_size * required_failed_windows)
    tail = events[-max_needed:]
    # Split into fixed-size windows from the end.
    while tail:
        windows.append(tail[-window_size:])
        tail = tail[:-window_size]
        if len(windows) >= required_failed_windows:
            break
    if len(windows) < required_failed_windows or any(len(w) < window_size for w in windows):
        return {
            "has_data": True,
            "should_rollback": False,
            "reason": "insufficient_window_data",
            "events_considered": len(events),
            "profile": profile,
            "component": component or "all",
        }

    failed_windows = 0
    window_reports = []
    for w in windows:
        # Compute reliability and technical KPIs per window.
        completed = [e for e in w if str(e.get("status")) == "completed"]
        success_rate = len(completed) / float(len(w))
        p95_runtime = _p95([float(e.get("elapsed_seconds", 0.0)) for e in completed]) if completed else math.inf
        resume_completed = [e for e in completed if bool(e.get("is_resume", False))]
        p95_resume_runtime = (
            _p95([float(e.get("elapsed_seconds", 0.0)) for e in resume_completed])
            if resume_completed
            else None
        )
        resume_runtime_ok = (
            p95_resume_runtime is None
            or p95_resume_runtime <= target.p95_resume_runtime_max_seconds
        )
        retry_samples = [float(e.get("retry_rate", 0.0)) for e in w if e.get("retry_rate") is not None]
        retry_rate_avg = (
            sum(retry_samples) / float(len(retry_samples))
            if retry_samples
            else None
        )
        retry_rate_ok = (
            retry_rate_avg is None
            or retry_rate_avg < TECH_KPI_THRESHOLDS["retry_rate_max"]
        )

        stuck_abort_rate = (
            sum(1 for e in w if bool(e.get("stuck_abort", False))) / float(len(w))
            if w
            else 0.0
        )
        stuck_abort_ok = stuck_abort_rate < TECH_KPI_THRESHOLDS["stuck_abort_rate_max"]

        invalid_candidates = [
            e for e in w if e.get("component") == "script" or e.get("invalid_schema") is not None
        ]
        invalid_schema_rate = (
            sum(1 for e in invalid_candidates if bool(e.get("invalid_schema", False)))
            / float(len(invalid_candidates))
            if invalid_candidates
            else 0.0
        )
        invalid_schema_ok = invalid_schema_rate < TECH_KPI_THRESHOLDS["invalid_schema_rate_max"]

        script_quality_rejected_rate = (
            sum(1 for e in w if str(e.get("failure_kind", "")).strip().lower() == "script_quality_rejected")
            / float(len(w))
            if w
            else 0.0
        )
        script_quality_rejected_ok = (
            script_quality_rejected_rate < TECH_KPI_THRESHOLDS["script_quality_rejected_rate_max"]
        )

        cost_error_samples = [
            float(e.get("cost_estimation_error_pct", 0.0))
            for e in w
            if e.get("cost_estimation_error_pct") is not None
        ]
        cost_error_p90 = _p90(cost_error_samples) if cost_error_samples else None
        cost_error_ok = (
            cost_error_p90 is None
            or cost_error_p90 <= TECH_KPI_THRESHOLDS["cost_error_p90_max_pct"]
        )

        pass_window = (
            success_rate >= target.success_rate_min
            and p95_runtime <= target.p95_runtime_max_seconds
            and resume_runtime_ok
            and retry_rate_ok
            and stuck_abort_ok
            and invalid_schema_ok
            and script_quality_rejected_ok
            and cost_error_ok
        )
        if not pass_window:
            failed_windows += 1
        window_reports.append(
            {
                "size": len(w),
                "success_rate": round(success_rate, 4),
                "p95_runtime_seconds": p95_runtime if p95_runtime != math.inf else None,
                "p95_resume_runtime_seconds": (
                    round(p95_resume_runtime, 4) if p95_resume_runtime is not None else None
                ),
                "resume_runtime_ok": resume_runtime_ok,
                "retry_rate_avg": round(retry_rate_avg, 4) if retry_rate_avg is not None else None,
                "retry_rate_ok": retry_rate_ok,
                "stuck_abort_rate": round(stuck_abort_rate, 4),
                "stuck_abort_ok": stuck_abort_ok,
                "invalid_schema_rate": round(invalid_schema_rate, 4),
                "invalid_schema_ok": invalid_schema_ok,
                "script_quality_rejected_rate": round(script_quality_rejected_rate, 4),
                "script_quality_rejected_ok": script_quality_rejected_ok,
                "cost_error_p90_pct": round(cost_error_p90, 4) if cost_error_p90 is not None else None,
                "cost_error_ok": cost_error_ok,
                "pass": pass_window,
            }
        )

    should_rollback = failed_windows >= required_failed_windows
    return {
        "has_data": True,
        "should_rollback": should_rollback,
        "failed_windows": failed_windows,
        "required_failed_windows": required_failed_windows,
        "window_reports": window_reports,
        "events_considered": len(events),
        "profile": profile,
        "component": component or "all",
    }

