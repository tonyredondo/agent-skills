#!/usr/bin/env python3
from __future__ import annotations

"""Factual validation for draft and rewritten podcast scripts."""

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict

from .logging_utils import Logger
from .podcast_artifacts import (
    apply_script_patch_batch,
    validate_fact_guard_report,
    validate_script_artifact,
    validate_script_patch_batch,
)

_FACT_GUARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "issue_type": {"type": "string"},
                    "severity": {"type": "string"},
                    "claim_id": {"type": "string"},
                    "line_indexes": {"type": "array", "items": {"type": "integer"}},
                    "source_refs": {"type": "array", "items": {"type": "string"}},
                    "origin_stage": {"type": "string"},
                    "action": {"type": "string"},
                },
                "required": [
                    "issue_id",
                    "issue_type",
                    "severity",
                    "claim_id",
                    "line_indexes",
                    "source_refs",
                    "origin_stage",
                    "action",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["pass", "issues"],
    "additionalProperties": False,
}

_FACT_GUARD_PATCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string"},
                    "line_id": {"type": ["string", "null"]},
                    "anchor_line_id": {"type": ["string", "null"]},
                    "line": {
                        "type": ["object", "null"],
                        "properties": {
                            "speaker": {"type": "string"},
                            "role": {"type": "string"},
                            "instructions": {"type": "string"},
                            "pace_hint": {"type": ["string", "null"]},
                            "text": {"type": "string"},
                        },
                        "required": ["speaker", "role", "instructions", "pace_hint", "text"],
                        "additionalProperties": False,
                    },
                },
                "required": ["op", "line_id", "anchor_line_id", "line"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["patches"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class FactGuard:
    """Detect unsupported claims and optionally request localized fixes."""

    client: Any
    logger: Logger

    def validate(
        self,
        *,
        script_artifact: Dict[str, Any],
        evidence_map: Dict[str, Any],
        episode_plan: Dict[str, Any] | None = None,
        stage_name: str,
    ) -> Dict[str, Any]:
        validated_script = validate_script_artifact(
            script_artifact,
            expected_stage=stage_name,
            episode_plan=episode_plan,
        )
        final_stage_rule = ""
        if str(stage_name).strip().lower() == "final":
            final_stage_rule = (
                "- Because this is the final-stage script, any unresolved factual issue should default to `block` unless it is clearly fixable with a local rewrite.\n"
                "- Unresolved `overstated_causality` in the final stage must not pass.\n"
            )
        prompt = textwrap.dedent(
            f"""
            Verify this podcast script against the evidence map.
            Return ONLY JSON.

            Rules:
            - Flag invented numbers, unsupported names/dates, or overstated causality.
            - Prefer no issue over speculative issue creation.
            - Use the matching evidence-map `claim_id` when there is one; otherwise return an empty string.
            - `origin_stage` must be `{stage_name}`.
            - `action` should be `rewrite_local` or `block`.
            {final_stage_rule.rstrip()}
            - If no issues are found, return pass=true and empty issues list.

            EVIDENCE MAP:
            {json.dumps(evidence_map, ensure_ascii=False, indent=2)}

            SCRIPT:
            {json.dumps({"lines": validated_script.get("lines", [])}, ensure_ascii=False, indent=2)}
            """
        ).strip()
        payload = self.client.generate_script_json(
            prompt=prompt,
            schema=_FACT_GUARD_SCHEMA,
            max_output_tokens=2200,
            stage=f"fact_guard_{stage_name}",
        )
        if not isinstance(payload, dict) or "pass" not in payload or "issues" not in payload:
            raise ValueError(f"fact_guard_{stage_name} returned malformed payload")
        report = validate_fact_guard_report(
            {
                "artifact_version": 1,
                "resume_compat_version": int(validated_script.get("resume_compat_version", 1)),
                "stage": stage_name,
                "run_token": validated_script.get("run_token", ""),
                "source_digest": validated_script.get("source_digest", ""),
                "plan_digest": validated_script.get("plan_digest", ""),
                "internal_artifact_digest": validated_script.get("internal_artifact_digest", ""),
                "public_payload_digest": validated_script.get("public_payload_digest", ""),
                "pass": payload.get("pass", False),
                "issues": payload.get("issues", []),
            }
        )
        self.logger.info(
            "fact_guard_evaluated",
            stage=stage_name,
            passed=bool(report.get("pass", False)),
            issues=len(report.get("issues", [])),
        )
        return report

    def repair(
        self,
        *,
        script_artifact: Dict[str, Any],
        evidence_map: Dict[str, Any],
        episode_plan: Dict[str, Any] | None = None,
        report: Dict[str, Any],
        stage_name: str,
    ) -> Dict[str, Any]:
        """Request localized factual cleanup when issues are repairable."""
        validated_script = validate_script_artifact(
            script_artifact,
            expected_stage=stage_name,
            episode_plan=episode_plan,
        )
        issues = list(report.get("issues", []) or [])
        if not issues:
            return validated_script
        repairable_issues = [
            dict(item)
            for item in issues
            if isinstance(item, dict) and str(item.get("action", "")).strip().lower() == "rewrite_local"
        ]
        if not repairable_issues:
            return validated_script
        blocking_issue_ids = [
            str(item.get("issue_id", "")).strip()
            for item in issues
            if isinstance(item, dict) and str(item.get("action", "")).strip().lower() == "block"
        ]
        turn_projection = [
            {
                "line_id": str(turn.get("line_id", "")).strip(),
                "beat_id": str(turn.get("beat_id", "")).strip(),
                "role": str(turn.get("role", "")).strip(),
                "speaker": str(turn.get("speaker", "")).strip(),
                "text": str(turn.get("text", "")).strip(),
            }
            for turn in list(validated_script.get("turns", []) or [])
        ]
        prompt = textwrap.dedent(
            f"""
            Patch ONLY the spans needed to fix factual issues in this podcast script.
            Return ONLY JSON with key `patches`.

            Rules:
            - Preserve structure, voices, and valid facts.
            - Do not introduce new unsupported claims.
            - Fix only the problematic lines called out in the repairable fact-guard report.
            - If the original report also contains `block` items elsewhere, leave those lines untouched.
            - Allowed operations only: `replace_line`, `insert_after`, `delete_line`.
            - Use the provided `line_id` values exactly.
            - Do not emit a full rewritten script.
            - Every patch object must always include `line_id`, `anchor_line_id`, and `line`.
            - Use `null` for the fields that do not apply to that operation.
            - Keep spoken text in Spanish and instructions in short English.
            - Rewrite toward source-safe phrasing such as "sirve para", "indica", "se usa cuando", or "marca el umbral" when the issue is unsupported effect language.
            - Remove unsupported causal, cost, trust, staffing, or urgency claims instead of sharpening them.

            REPAIRABLE FACT REPORT:
            {json.dumps({**dict(report), "issues": repairable_issues, "pass": False}, ensure_ascii=False, indent=2)}

            BLOCKED ISSUE IDS TO LEAVE UNCHANGED:
            {json.dumps(blocking_issue_ids, ensure_ascii=False, indent=2)}

            EVIDENCE MAP:
            {json.dumps(evidence_map, ensure_ascii=False, indent=2)}

            SCRIPT:
            {json.dumps({"turns": turn_projection}, ensure_ascii=False, indent=2)}
            """
        ).strip()
        payload = self.client.generate_script_json(
            prompt=prompt,
            schema=_FACT_GUARD_PATCH_SCHEMA,
            max_output_tokens=2800,
            stage=f"fact_guard_repair_{stage_name}",
        )
        patch_batch = validate_script_patch_batch(payload)
        return apply_script_patch_batch(
            script_artifact=validated_script,
            patch_batch=patch_batch,
            episode_plan=episode_plan,
        )
