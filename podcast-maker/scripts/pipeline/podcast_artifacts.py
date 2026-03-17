#!/usr/bin/env python3
from __future__ import annotations

"""Artifact contracts and persistence helpers for the redesigned script pipeline."""

import json
import os
from typing import Any, Dict, List

from .schema import canonical_json, content_hash, count_words_from_lines, validate_script_payload

EVIDENCE_MAP_FILENAME = "evidence_map.json"
EPISODE_PLAN_FILENAME = "episode_plan.json"
DRAFT_SCRIPT_FILENAME = "draft_script.json"
DRAFT_SCRIPT_RAW_FILENAME = "draft_script_raw.json"
DRAFT_SCRIPT_FACT_CHECKED_FILENAME = "draft_script_fact_checked.json"
REWRITTEN_SCRIPT_FILENAME = "rewritten_script.json"
REWRITTEN_SCRIPT_FINAL_FILENAME = "rewritten_script_final.json"
EDITORIAL_REPORT_FILENAME = "editorial_report.json"
FACT_GUARD_DRAFT_FILENAME = "fact_guard_report_draft.json"
FACT_GUARD_FINAL_FILENAME = "fact_guard_report_final.json"
STRUCTURAL_REPORT_FILENAME = "structural_report.json"
QUALITY_REPORT_FILENAME = "quality_report.json"
RESUME_COMPAT_VERSION = 1

OPENING_MODES = {
    "provocative_observation",
    "concrete_tension",
    "sharp_question",
    "contrast_frame",
    "mini_scene",
}
CLOSING_MODES = {
    "earned_synthesis",
    "practical_takeaway",
    "contrast_return",
    "resolved_question",
}
BEAT_GOALS = {
    "hook_and_frame",
    "explain_core",
    "concrete_example",
    "objection_and_tradeoff",
    "consequence",
    "practical_takeaway",
    "closing",
}
MOVE_TYPES = {
    "example",
    "objection",
    "tradeoff",
    "consequence",
    "cost",
    "awkward_question",
    "grounding",
    "counterexample",
    "decision",
}
CLAIM_KINDS = {
    "fact",
    "example",
    "tension",
    "context",
    "counterpoint",
    "quote",
}
CLAIM_SUPPORT_TYPES = {"direct", "inferred_light"}
SCRIPT_ARTIFACT_STAGES = {"draft", "rewritten", "final"}
EDITORIAL_FAILURE_TYPES = {
    "scaffold_phrase_repetition",
    "thesis_repetition",
    "dense_turns",
    "host2_not_pushing",
    "underlength",
    "undercoverage",
    "host2_missing_beat",
    "briefing_tone",
    "overlong_for_profile",
    "earned_closing_missing",
    "meta_opening",
}


def build_script_artifact_paths(*, run_dir: str) -> Dict[str, str]:
    """Return canonical artifact paths for one script run directory."""
    return {
        "evidence_map": os.path.join(run_dir, EVIDENCE_MAP_FILENAME),
        "episode_plan": os.path.join(run_dir, EPISODE_PLAN_FILENAME),
        "draft_script_raw": os.path.join(run_dir, DRAFT_SCRIPT_RAW_FILENAME),
        "draft_script": os.path.join(run_dir, DRAFT_SCRIPT_FILENAME),
        "draft_script_fact_checked": os.path.join(run_dir, DRAFT_SCRIPT_FACT_CHECKED_FILENAME),
        "rewritten_script": os.path.join(run_dir, REWRITTEN_SCRIPT_FILENAME),
        "rewritten_script_final": os.path.join(run_dir, REWRITTEN_SCRIPT_FINAL_FILENAME),
        "editorial_report": os.path.join(run_dir, EDITORIAL_REPORT_FILENAME),
        "fact_guard_report_draft": os.path.join(run_dir, FACT_GUARD_DRAFT_FILENAME),
        "fact_guard_report_final": os.path.join(run_dir, FACT_GUARD_FINAL_FILENAME),
        "structural_report": os.path.join(run_dir, STRUCTURAL_REPORT_FILENAME),
        "quality_report": os.path.join(run_dir, QUALITY_REPORT_FILENAME),
    }


def rewrite_round_filename(round_idx: int) -> str:
    """Return stable filename for one intermediate rewrite round."""
    return f"rewritten_script_round_{max(1, int(round_idx)):02d}.json"


def rewrite_round_path(*, run_dir: str, round_idx: int) -> str:
    """Return path for one intermediate rewrite round snapshot."""
    return os.path.join(run_dir, rewrite_round_filename(round_idx))


def write_json_artifact(*, path: str, payload: Dict[str, Any]) -> None:
    """Write one JSON artifact atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def read_json_artifact(path: str) -> Dict[str, Any]:
    """Read one JSON artifact and require object payload."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact at {path} must be a JSON object")
    return payload


def _require_string(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


def _require_float(value: Any, field_name: str, *, low: float = 0.0, high: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if parsed < low or parsed > high:
        raise ValueError(f"{field_name} must be between {low} and {high}")
    return parsed


def _require_list(value: Any, field_name: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return list(value)


def _normalize_string_list(value: Any, field_name: str) -> List[str]:
    items = _require_list(value, field_name)
    out: List[str] = []
    for idx, item in enumerate(items):
        text = str(item or "").strip()
        if not text:
            raise ValueError(f"{field_name}[{idx}] must be non-empty")
        out.append(text)
    return out


def _optional_string(value: Any) -> str:
    return str(value or "").strip()


def _identity_digest(payload: Dict[str, Any]) -> str:
    return content_hash(canonical_json(payload))


def _line_payload_from_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "speaker": str(turn.get("speaker", "")).strip(),
        "role": str(turn.get("role", "")).strip(),
        "instructions": str(turn.get("instructions", "")).strip(),
        "text": str(turn.get("text", "")).strip(),
    }
    pace_hint = str(turn.get("pace_hint", "") or "").strip()
    if pace_hint:
        payload["pace_hint"] = pace_hint
    return payload


def public_lines_from_script_artifact(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Project public `lines` payload from enriched script artifact."""
    turns = payload.get("turns")
    if isinstance(turns, list) and turns:
        return validate_script_payload({"lines": [_line_payload_from_turn(turn) for turn in turns]}).get("lines", [])
    return validate_script_payload({"lines": payload.get("lines", [])}).get("lines", [])


def _plan_beats(episode_plan: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(episode_plan, dict):
        return []
    beats = episode_plan.get("beats", [])
    if not isinstance(beats, list):
        return []
    out: List[Dict[str, Any]] = []
    for beat in beats:
        if isinstance(beat, dict):
            out.append(dict(beat))
    return out


def _target_word_count_from_plan(episode_plan: Dict[str, Any] | None) -> int:
    total = 0
    for beat in _plan_beats(episode_plan):
        total += max(0, int(beat.get("target_words", 0) or 0))
    return total


def _allocate_turn_counts(*, line_count: int, beats: List[Dict[str, Any]]) -> List[int]:
    if line_count <= 0 or not beats:
        return []
    target_words = [max(1, int(beat.get("target_words", 1) or 1)) for beat in beats]
    counts = [0] * len(beats)
    remaining = int(line_count)
    if remaining >= len(beats):
        counts = [1] * len(beats)
        remaining -= len(beats)
    for _ in range(remaining):
        ratios = [counts[idx] / float(max(1, target_words[idx])) for idx in range(len(beats))]
        chosen = min(range(len(beats)), key=lambda idx: (ratios[idx], counts[idx], idx))
        counts[chosen] += 1
    if sum(counts) < line_count:
        counts[-1] += line_count - sum(counts)
    return counts


def _beat_ids_for_lines(*, lines: List[Dict[str, Any]], episode_plan: Dict[str, Any] | None) -> List[str]:
    beats = _plan_beats(episode_plan)
    if not beats or not lines:
        return ["" for _ in lines]
    counts = _allocate_turn_counts(line_count=len(lines), beats=beats)
    out: List[str] = []
    for beat, count in zip(beats, counts):
        beat_id = _optional_string(beat.get("beat_id"))
        for _ in range(count):
            if len(out) < len(lines):
                out.append(beat_id)
    while len(out) < len(lines):
        out.append(_optional_string(beats[-1].get("beat_id")))
    return out[: len(lines)]


def _beat_cut_map(episode_plan: Dict[str, Any] | None) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for beat in _plan_beats(episode_plan):
        beat_id = _optional_string(beat.get("beat_id"))
        if beat_id:
            out[beat_id] = bool(beat.get("can_cut", False))
    return out


def _make_turn_id(*, stage: str, idx: int, line: Dict[str, Any], run_token: str) -> str:
    seed = canonical_json(
        {
            "stage": stage,
            "idx": idx,
            "run_token": run_token,
            "role": line.get("role", ""),
            "speaker": line.get("speaker", ""),
            "text": line.get("text", ""),
        }
    )
    return f"line_{content_hash(seed)[:12]}"


def _normalize_turns(
    *,
    stage: str,
    lines: List[Dict[str, Any]],
    run_token: str,
    episode_plan: Dict[str, Any] | None,
    raw_turns: Any = None,
    prior_turns: Any = None,
) -> List[Dict[str, Any]]:
    normalized_lines = validate_script_payload({"lines": lines}).get("lines", [])
    prior_by_idx = [dict(item) for item in list(prior_turns or []) if isinstance(item, dict)]
    preserve_prior_beat_ids = len(prior_by_idx) == len(normalized_lines)
    raw_items = list(raw_turns or []) if isinstance(raw_turns, list) else []
    beat_ids = _beat_ids_for_lines(lines=normalized_lines, episode_plan=episode_plan)
    beat_cut_map = _beat_cut_map(episode_plan)
    turns: List[Dict[str, Any]] = []
    for idx, line in enumerate(normalized_lines):
        raw_turn = raw_items[idx] if idx < len(raw_items) and isinstance(raw_items[idx], dict) else {}
        prior_turn = prior_by_idx[idx] if idx < len(prior_by_idx) else {}
        line_id = _optional_string(raw_turn.get("line_id")) or _optional_string(prior_turn.get("line_id"))
        if not line_id:
            line_id = _make_turn_id(stage=stage, idx=idx, line=line, run_token=run_token)
        beat_id = _optional_string(raw_turn.get("beat_id"))
        if not beat_id and preserve_prior_beat_ids:
            beat_id = _optional_string(prior_turn.get("beat_id"))
        if not beat_id:
            beat_id = beat_ids[idx]
        turn = {
            "line_id": line_id,
            "beat_id": beat_id,
            "can_cut": bool(beat_cut_map.get(beat_id, False)),
            **line,
        }
        turns.append(turn)
    return turns


def _coverage_from_turns(*, turns: List[Dict[str, Any]], episode_plan: Dict[str, Any] | None) -> Dict[str, Any]:
    beats = _plan_beats(episode_plan)
    beat_ids = [_optional_string(item.get("beat_id")) for item in beats if _optional_string(item.get("beat_id"))]
    non_cuttable = [
        _optional_string(item.get("beat_id"))
        for item in beats
        if _optional_string(item.get("beat_id")) and not bool(item.get("can_cut", False))
    ]
    covered = [_optional_string(turn.get("beat_id")) for turn in turns if _optional_string(turn.get("beat_id"))]
    covered_unique: List[str] = []
    for beat_id in covered:
        if beat_id not in covered_unique:
            covered_unique.append(beat_id)
    missing = [beat_id for beat_id in beat_ids if beat_id not in covered_unique]
    missing_non_cuttable = [beat_id for beat_id in non_cuttable if beat_id not in covered_unique]
    host2_missing_non_cuttable: List[str] = []
    line_counts_by_beat: Dict[str, int] = {}
    host2_turn_counts_by_beat: Dict[str, int] = {}
    for turn in turns:
        beat_id = _optional_string(turn.get("beat_id"))
        if not beat_id:
            continue
        line_counts_by_beat[beat_id] = line_counts_by_beat.get(beat_id, 0) + 1
        if str(turn.get("role", "")).strip() == "Host2":
            host2_turn_counts_by_beat[beat_id] = host2_turn_counts_by_beat.get(beat_id, 0) + 1
    for beat_id in non_cuttable:
        beat_turns = [turn for turn in turns if _optional_string(turn.get("beat_id")) == beat_id]
        if beat_turns and not any(str(turn.get("role", "")).strip() == "Host2" for turn in beat_turns):
            host2_missing_non_cuttable.append(beat_id)
    line_count = len(turns)
    word_count = count_words_from_lines([_line_payload_from_turn(turn) for turn in turns])
    host2_turn_count = sum(1 for turn in turns if str(turn.get("role", "")).strip() == "Host2")
    return {
        "beat_ids": beat_ids,
        "covered_beat_ids": covered_unique,
        "missing_beat_ids": missing,
        "non_cuttable_beat_ids": non_cuttable,
        "covered_non_cuttable_beat_ids": [beat_id for beat_id in non_cuttable if beat_id in covered_unique],
        "missing_non_cuttable_beat_ids": missing_non_cuttable,
        "host2_missing_non_cuttable_beat_ids": host2_missing_non_cuttable,
        "line_count": line_count,
        "word_count": word_count,
        "line_counts_by_beat": line_counts_by_beat,
        "host2_turn_counts_by_beat": host2_turn_counts_by_beat,
        "host2_turn_count": host2_turn_count,
        "host2_turn_ratio": round(float(host2_turn_count) / float(max(1, line_count)), 4),
    }


def build_script_artifact(
    *,
    stage: str,
    episode_id: str,
    run_token: str,
    source_digest: str,
    plan_ref: str,
    plan_digest: str,
    lines: List[Dict[str, Any]],
    episode_plan: Dict[str, Any] | None = None,
    raw_turns: Any = None,
    prior_artifact: Dict[str, Any] | None = None,
    target_word_count: int | None = None,
    public_payload_digest: str | None = None,
) -> Dict[str, Any]:
    """Build enriched internal artifact while preserving public `lines` payload."""
    turns = _normalize_turns(
        stage=stage,
        lines=lines,
        run_token=run_token,
        episode_plan=episode_plan,
        raw_turns=raw_turns,
        prior_turns=(prior_artifact or {}).get("turns", []),
    )
    public_lines = validate_script_payload({"lines": [_line_payload_from_turn(turn) for turn in turns]}).get("lines", [])
    public_payload = {"lines": public_lines}
    coverage = _coverage_from_turns(turns=turns, episode_plan=episode_plan)
    planned_beat_count = len(_plan_beats(episode_plan))
    resolved_target_word_count = max(
        0,
        int(target_word_count or _target_word_count_from_plan(episode_plan)),
    )
    resolved_public_payload_digest = _optional_string(public_payload_digest) or _identity_digest(public_payload)
    artifact = {
        "artifact_version": 1,
        "resume_compat_version": RESUME_COMPAT_VERSION,
        "stage": stage,
        "episode_id": episode_id,
        "run_token": run_token,
        "source_digest": source_digest,
        "plan_ref": plan_ref,
        "plan_digest": plan_digest,
        "planned_beat_count": planned_beat_count,
        "target_word_count": resolved_target_word_count,
        "coverage": coverage,
        "turns": turns,
        "lines": public_lines,
        "public_payload_digest": resolved_public_payload_digest,
    }
    artifact["internal_artifact_digest"] = _identity_digest(
        {
            "stage": stage,
            "episode_id": episode_id,
            "run_token": run_token,
            "source_digest": source_digest,
            "plan_digest": plan_digest,
            "turns": turns,
            "public_payload_digest": resolved_public_payload_digest,
        }
    )
    return artifact


def validate_evidence_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate normalized evidence map artifact."""
    artifact_version = int(payload.get("artifact_version", 1))
    episode_id = _require_string(payload.get("episode_id"), "episode_id")
    run_token = _require_string(payload.get("run_token"), "run_token")
    source_digest = _require_string(payload.get("source_digest"), "source_digest")
    global_thesis = _require_string(payload.get("global_thesis"), "global_thesis")
    source_segments_raw = _require_list(payload.get("source_segments", []), "source_segments")
    claims_raw = _require_list(payload.get("claims", []), "claims")
    topics_raw = _require_list(payload.get("topics", []), "topics")
    if not source_segments_raw:
        raise ValueError("source_segments must not be empty")
    if not claims_raw:
        raise ValueError("claims must not be empty")
    if not topics_raw:
        raise ValueError("topics must not be empty")
    source_segments: List[Dict[str, Any]] = []
    for idx, item in enumerate(source_segments_raw):
        if not isinstance(item, dict):
            raise ValueError(f"source_segments[{idx}] must be object")
        source_ref = _require_string(item.get("source_ref"), f"source_segments[{idx}].source_ref")
        if ":" not in source_ref:
            raise ValueError("source_ref must use stable doc_id:seg_id format")
        start_char = int(item.get("start_char", 0))
        end_char = int(item.get("end_char", 0))
        sha256 = _require_string(item.get("sha256"), f"source_segments[{idx}].sha256")
        source_segments.append(
            {
                "source_ref": source_ref,
                "start_char": max(0, start_char),
                "end_char": max(start_char, end_char),
                "sha256": sha256,
            }
        )
    claims: List[Dict[str, Any]] = []
    for idx, item in enumerate(claims_raw):
        if not isinstance(item, dict):
            raise ValueError(f"claims[{idx}] must be object")
        claim_id = _require_string(item.get("claim_id"), f"claims[{idx}].claim_id")
        statement = _require_string(item.get("statement"), f"claims[{idx}].statement")
        kind = _require_string(item.get("kind"), f"claims[{idx}].kind")
        if kind not in CLAIM_KINDS:
            raise ValueError(f"claims[{idx}].kind invalid: {kind}")
        topic_ids = _normalize_string_list(item.get("topic_ids", []), f"claims[{idx}].topic_ids")
        source_refs = _normalize_string_list(item.get("source_refs", []), f"claims[{idx}].source_refs")
        support = _require_string(item.get("support"), f"claims[{idx}].support")
        if support not in CLAIM_SUPPORT_TYPES:
            raise ValueError(f"claims[{idx}].support invalid: {support}")
        confidence = _require_float(item.get("confidence"), f"claims[{idx}].confidence")
        if support == "inferred_light" and kind in {"fact", "quote"}:
            raise ValueError("inferred_light support not allowed for fact or quote claims")
        claims.append(
            {
                "claim_id": claim_id,
                "statement": statement,
                "kind": kind,
                "topic_ids": topic_ids,
                "source_refs": source_refs,
                "support": support,
                "confidence": confidence,
            }
        )
    topics: List[Dict[str, Any]] = []
    for idx, item in enumerate(topics_raw):
        if not isinstance(item, dict):
            raise ValueError(f"topics[{idx}] must be object")
        topic_id = _require_string(item.get("topic_id"), f"topics[{idx}].topic_id")
        title = _require_string(item.get("title"), f"topics[{idx}].title")
        topics.append(
            {
                "topic_id": topic_id,
                "title": title,
                "core_claim_ids": _normalize_string_list(
                    item.get("core_claim_ids", []),
                    f"topics[{idx}].core_claim_ids",
                ),
                "example_claim_ids": _normalize_string_list(
                    item.get("example_claim_ids", []),
                    f"topics[{idx}].example_claim_ids",
                ),
                "tension_claim_ids": _normalize_string_list(
                    item.get("tension_claim_ids", []),
                    f"topics[{idx}].tension_claim_ids",
                ),
                "priority": _require_float(item.get("priority"), f"topics[{idx}].priority"),
                "discardable": _require_bool(item.get("discardable", False), f"topics[{idx}].discardable"),
            }
        )
    return {
        "artifact_version": artifact_version,
        "episode_id": episode_id,
        "run_token": run_token,
        "source_digest": source_digest,
        "source_segments": source_segments,
        "global_thesis": global_thesis,
        "claims": claims,
        "topics": topics,
    }


def validate_episode_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate normalized episode plan artifact."""
    artifact_version = int(payload.get("artifact_version", 1))
    episode_id = _require_string(payload.get("episode_id"), "episode_id")
    run_token = _require_string(payload.get("run_token"), "run_token")
    opening_mode = _require_string(payload.get("opening_mode"), "opening_mode")
    closing_mode = _require_string(payload.get("closing_mode"), "closing_mode")
    if opening_mode not in OPENING_MODES:
        raise ValueError(f"opening_mode invalid: {opening_mode}")
    if closing_mode not in CLOSING_MODES:
        raise ValueError(f"closing_mode invalid: {closing_mode}")
    host_roles_raw = payload.get("host_roles")
    if not isinstance(host_roles_raw, dict):
        raise ValueError("host_roles must be object")
    host_roles = {
        "Host1": _require_string(host_roles_raw.get("Host1"), "host_roles.Host1"),
        "Host2": _require_string(host_roles_raw.get("Host2"), "host_roles.Host2"),
    }
    beats_raw = _require_list(payload.get("beats", []), "beats")
    if not beats_raw:
        raise ValueError("beats must not be empty")
    beats: List[Dict[str, Any]] = []
    for idx, item in enumerate(beats_raw):
        if not isinstance(item, dict):
            raise ValueError(f"beats[{idx}] must be object")
        goal = _require_string(item.get("goal"), f"beats[{idx}].goal")
        if goal not in BEAT_GOALS:
            raise ValueError(f"beats[{idx}].goal invalid: {goal}")
        required_move = _require_string(item.get("required_move"), f"beats[{idx}].required_move")
        if required_move not in MOVE_TYPES:
            raise ValueError(f"beats[{idx}].required_move invalid: {required_move}")
        optional_moves = _normalize_string_list(item.get("optional_moves", []), f"beats[{idx}].optional_moves")
        for move in optional_moves:
            if move not in MOVE_TYPES:
                raise ValueError(f"beats[{idx}].optional_moves invalid: {move}")
        beats.append(
            {
                "beat_id": _require_string(item.get("beat_id"), f"beats[{idx}].beat_id"),
                "goal": goal,
                "topic_ids": _normalize_string_list(item.get("topic_ids", []), f"beats[{idx}].topic_ids"),
                "claim_ids": _normalize_string_list(item.get("claim_ids", []), f"beats[{idx}].claim_ids"),
                "required_move": required_move,
                "optional_moves": optional_moves,
                "must_cover": _normalize_string_list(item.get("must_cover", []), f"beats[{idx}].must_cover"),
                "can_cut": _require_bool(item.get("can_cut", False), f"beats[{idx}].can_cut"),
                "target_words": max(1, int(item.get("target_words", 1))),
            }
        )
    return {
        "artifact_version": artifact_version,
        "resume_compat_version": int(payload.get("resume_compat_version", RESUME_COMPAT_VERSION)),
        "episode_id": episode_id,
        "run_token": run_token,
        "opening_mode": opening_mode,
        "closing_mode": closing_mode,
        "host_roles": host_roles,
        "beats": beats,
    }


def validate_script_artifact(
    payload: Dict[str, Any],
    *,
    expected_stage: str | None = None,
    episode_plan: Dict[str, Any] | None = None,
    prior_artifact: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Validate staged internal artifact and preserve stable public projection."""
    artifact_version = int(payload.get("artifact_version", 1))
    stage = _require_string(payload.get("stage"), "stage")
    if stage not in SCRIPT_ARTIFACT_STAGES:
        raise ValueError(f"stage invalid: {stage}")
    if expected_stage and stage != expected_stage:
        raise ValueError(f"expected stage={expected_stage}, got {stage}")
    episode_id = _require_string(payload.get("episode_id"), "episode_id")
    run_token = _require_string(payload.get("run_token"), "run_token")
    source_digest = _require_string(payload.get("source_digest"), "source_digest")
    plan_ref = _require_string(payload.get("plan_ref"), "plan_ref")
    plan_digest = _optional_string(payload.get("plan_digest")) or content_hash(plan_ref)
    lines = public_lines_from_script_artifact(payload)
    return build_script_artifact(
        stage=stage,
        episode_id=episode_id,
        run_token=run_token,
        source_digest=source_digest,
        plan_ref=plan_ref,
        plan_digest=plan_digest,
        lines=lines,
        episode_plan=episode_plan,
        raw_turns=payload.get("turns"),
        prior_artifact=prior_artifact,
        target_word_count=payload.get("target_word_count"),
        public_payload_digest=_optional_string(payload.get("public_payload_digest")),
    ) | {
        "artifact_version": artifact_version,
        "resume_compat_version": int(payload.get("resume_compat_version", RESUME_COMPAT_VERSION)),
    }


def validate_editorial_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate editorial gate report artifact."""
    artifact_version = int(payload.get("artifact_version", 1))
    scores_raw = payload.get("scores")
    if not isinstance(scores_raw, dict):
        raise ValueError("scores must be object")
    required_scores = (
        "orality",
        "host_distinction",
        "progression",
        "freshness",
        "listener_engagement",
        "density_control",
    )
    scores: Dict[str, float] = {}
    for key in required_scores:
        scores[key] = _require_float(scores_raw.get(key), f"scores.{key}", low=1.0, high=5.0)
    metrics_raw = payload.get("deterministic_metrics")
    if not isinstance(metrics_raw, dict):
        raise ValueError("deterministic_metrics must be object")
    failures_raw = _require_list(payload.get("failures", []), "failures")
    failures: List[Dict[str, Any]] = []
    for idx, item in enumerate(failures_raw):
        if not isinstance(item, dict):
            raise ValueError(f"failures[{idx}] must be object")
        failure_type = _require_string(item.get("failure_type"), f"failures[{idx}].failure_type")
        if failure_type not in EDITORIAL_FAILURE_TYPES:
            raise ValueError(f"failures[{idx}].failure_type invalid: {failure_type}")
        failures.append(
            {
                "failure_type": failure_type,
                "severity": _require_string(item.get("severity"), f"failures[{idx}].severity"),
                "line_indexes": [max(0, int(v)) for v in _require_list(item.get("line_indexes", []), f"failures[{idx}].line_indexes")],
                "beat_ids": _normalize_string_list(item.get("beat_ids", []), f"failures[{idx}].beat_ids"),
                "reason": _require_string(item.get("reason"), f"failures[{idx}].reason"),
                "recommended_action": _require_string(
                    item.get("recommended_action"),
                    f"failures[{idx}].recommended_action",
                ),
            }
        )
    rewrite_budget_raw = payload.get("rewrite_budget")
    if not isinstance(rewrite_budget_raw, dict):
        raise ValueError("rewrite_budget must be object")
    return {
        "artifact_version": artifact_version,
        "resume_compat_version": int(payload.get("resume_compat_version", RESUME_COMPAT_VERSION)),
        "stage": _require_string(payload.get("stage"), "stage"),
        "run_token": _optional_string(payload.get("run_token")),
        "source_digest": _optional_string(payload.get("source_digest")),
        "plan_digest": _optional_string(payload.get("plan_digest")),
        "internal_artifact_digest": _optional_string(payload.get("internal_artifact_digest")),
        "public_payload_digest": _optional_string(payload.get("public_payload_digest")),
        "profile": _require_string(payload.get("profile"), "profile"),
        "pass": _require_bool(payload.get("pass", False), "pass"),
        "scores": scores,
        "deterministic_metrics": {
            key: metrics_raw.get(key) for key in (
                "scaffold_phrase_hits",
                "stock_opener_cluster_hits",
                "long_turn_count",
                "abrupt_transition_count",
                "question_ratio",
                "host2_push_ratio",
                "host2_turn_ratio",
                "template_reuse_hits",
                "max_question_streak",
                "overlong_for_profile",
                "word_count",
                "line_count",
                "missing_non_cuttable_beat_count",
                "host2_missing_non_cuttable_beat_count",
                "compressed_beat_count",
            )
        },
        "failures": failures,
        "rewrite_budget": {
            "max_rounds": max(0, int(rewrite_budget_raw.get("max_rounds", 0))),
            "round_used": max(0, int(rewrite_budget_raw.get("round_used", 0))),
            "stop_reason": str(rewrite_budget_raw.get("stop_reason", "") or "").strip(),
        },
    }


def validate_fact_guard_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate fact guard report artifact."""
    artifact_version = int(payload.get("artifact_version", 1))
    issues_raw = _require_list(payload.get("issues", []), "issues")
    issues: List[Dict[str, Any]] = []
    for idx, item in enumerate(issues_raw):
        if not isinstance(item, dict):
            raise ValueError(f"issues[{idx}] must be object")
        issues.append(
            {
                "issue_id": _require_string(item.get("issue_id"), f"issues[{idx}].issue_id"),
                "issue_type": _require_string(item.get("issue_type"), f"issues[{idx}].issue_type"),
                "severity": _require_string(item.get("severity"), f"issues[{idx}].severity"),
                "claim_id": _optional_string(item.get("claim_id")),
                "line_indexes": [max(0, int(v)) for v in _require_list(item.get("line_indexes", []), f"issues[{idx}].line_indexes")],
                "source_refs": _normalize_string_list(item.get("source_refs", []), f"issues[{idx}].source_refs"),
                "origin_stage": _require_string(item.get("origin_stage"), f"issues[{idx}].origin_stage"),
                "action": _require_string(item.get("action"), f"issues[{idx}].action"),
            }
        )
    return {
        "artifact_version": artifact_version,
        "resume_compat_version": int(payload.get("resume_compat_version", RESUME_COMPAT_VERSION)),
        "stage": _require_string(payload.get("stage"), "stage"),
        "run_token": _optional_string(payload.get("run_token")),
        "source_digest": _optional_string(payload.get("source_digest")),
        "plan_digest": _optional_string(payload.get("plan_digest")),
        "internal_artifact_digest": _optional_string(payload.get("internal_artifact_digest")),
        "public_payload_digest": _optional_string(payload.get("public_payload_digest")),
        "pass": _require_bool(payload.get("pass", False), "pass"),
        "issues": issues,
    }


def build_public_script_payload(
    *,
    lines: List[Dict[str, Any]] | None = None,
    artifact: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build stable public payload from lines or enriched internal artifact."""
    if artifact is not None:
        return validate_script_payload({"lines": public_lines_from_script_artifact(artifact)})
    return validate_script_payload({"lines": list(lines or [])})


def write_script_payload(*, path: str, lines: List[Dict[str, Any]]) -> None:
    """Persist final public script payload atomically using canonical formatting."""
    payload = build_public_script_payload(lines=lines)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(canonical_json(payload))
    os.replace(tmp, path)


def validate_script_patch_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate patch batch for localized line-id edits."""
    if not isinstance(payload, dict):
        raise ValueError("patch batch must be object")
    patches_raw = _require_list(payload.get("patches", []), "patches")
    patches: List[Dict[str, Any]] = []
    for idx, item in enumerate(patches_raw):
        if not isinstance(item, dict):
            raise ValueError(f"patches[{idx}] must be object")
        op = _require_string(item.get("op"), f"patches[{idx}].op")
        if op not in {"replace_line", "insert_after", "delete_line"}:
            raise ValueError(f"patches[{idx}].op invalid: {op}")
        if op == "replace_line":
            line_id = _require_string(item.get("line_id"), f"patches[{idx}].line_id")
            line = validate_script_payload({"lines": [_require_dict(item.get("line"), f"patches[{idx}].line")]}).get("lines", [])[0]
            patches.append({"op": op, "line_id": line_id, "line": line})
            continue
        if op == "insert_after":
            anchor_line_id = _require_string(item.get("anchor_line_id"), f"patches[{idx}].anchor_line_id")
            line = validate_script_payload({"lines": [_require_dict(item.get("line"), f"patches[{idx}].line")]}).get("lines", [])[0]
            patches.append({"op": op, "anchor_line_id": anchor_line_id, "line": line})
            continue
        patches.append({"op": op, "line_id": _require_string(item.get("line_id"), f"patches[{idx}].line_id")})
    return {
        "resume_compat_version": int(payload.get("resume_compat_version", RESUME_COMPAT_VERSION)),
        "patches": patches,
    }


def _require_dict(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be object")
    return dict(value)


def apply_script_patch_batch(
    *,
    script_artifact: Dict[str, Any],
    patch_batch: Dict[str, Any],
    episode_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Apply localized patch batch to enriched artifact deterministically."""
    expected_stage = _optional_string(script_artifact.get("stage"))
    validated_artifact = validate_script_artifact(
        script_artifact,
        expected_stage=expected_stage or None,
        episode_plan=episode_plan,
    )
    validated_batch = validate_script_patch_batch(patch_batch)
    turns = [dict(turn) for turn in list(validated_artifact.get("turns", []) or [])]
    if not turns:
        raise ValueError("script artifact has no turns")
    consumed_targets: set[str] = set()
    insert_after_tail: Dict[str, int] = {}
    for patch in list(validated_batch.get("patches", []) or []):
        op = str(patch.get("op", "")).strip()
        if op == "insert_after":
            anchor_line_id = str(patch.get("anchor_line_id", "")).strip()
            anchor_indexes = [idx for idx, turn in enumerate(turns) if str(turn.get("line_id", "")).strip() == anchor_line_id]
            if not anchor_indexes:
                raise ValueError(f"insert_after anchor not found: {anchor_line_id}")
            insert_idx = insert_after_tail.get(anchor_line_id, anchor_indexes[-1]) + 1
            for key, value in list(insert_after_tail.items()):
                if value >= insert_idx:
                    insert_after_tail[key] = value + 1
            turns.insert(insert_idx, dict(patch.get("line", {})))
            insert_after_tail[anchor_line_id] = insert_idx
            continue
        target_line_id = str(patch.get("line_id", "")).strip()
        if target_line_id in consumed_targets:
            raise ValueError(f"incompatible batch: repeated target line_id {target_line_id}")
        target_indexes = [idx for idx, turn in enumerate(turns) if str(turn.get("line_id", "")).strip() == target_line_id]
        if not target_indexes:
            raise ValueError(f"target line_id not found: {target_line_id}")
        consumed_targets.add(target_line_id)
        target_idx = target_indexes[0]
        if op == "replace_line":
            replacement = dict(patch.get("line", {}))
            replacement["line_id"] = target_line_id
            replacement["beat_id"] = str(turns[target_idx].get("beat_id", "")).strip()
            replacement["can_cut"] = bool(turns[target_idx].get("can_cut", False))
            turns[target_idx] = replacement
        elif op == "delete_line":
            del turns[target_idx]
            for key, value in list(insert_after_tail.items()):
                if value > target_idx:
                    insert_after_tail[key] = value - 1
    return build_script_artifact(
        stage=str(validated_artifact.get("stage")),
        episode_id=str(validated_artifact.get("episode_id")),
        run_token=str(validated_artifact.get("run_token")),
        source_digest=str(validated_artifact.get("source_digest")),
        plan_ref=str(validated_artifact.get("plan_ref")),
        plan_digest=str(validated_artifact.get("plan_digest")),
        lines=[_line_payload_from_turn(turn) for turn in turns],
        episode_plan=episode_plan,
        raw_turns=turns,
        prior_artifact=validated_artifact,
        target_word_count=validated_artifact.get("target_word_count"),
    )
