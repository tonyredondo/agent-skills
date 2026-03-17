#!/usr/bin/env python3
from __future__ import annotations

"""Evidence-map extraction for the redesigned podcast pipeline."""

import hashlib
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .logging_utils import Logger
from .podcast_artifacts import validate_evidence_map
from .script_chunker import split_source_chunks

_SEGMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "segment_summary": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "kind": {"type": "string"},
                    "topic_hint": {"type": "string"},
                    "support": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["statement", "kind", "topic_hint", "support", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["segment_summary", "claims"],
    "additionalProperties": False,
}

_THESIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "global_thesis": {"type": "string"},
    },
    "required": ["global_thesis"],
    "additionalProperties": False,
}

_CLAIM_PROMPT_MAX_CHARS = 14000
_PROCEDURAL_SEGMENT_MARKERS = (
    "=",
    "--",
    "./",
    ".json",
    ".jsonl",
    ".zip",
    "export ",
    "python3 ",
    "warn",
    "rollback",
    "bundle",
    "debug",
    "path",
    "preset",
    "gate",
    "window",
    "threshold",
    "monitor",
)
_PROCEDURAL_CLAIM_MARKERS = (
    "sirve para",
    "se usa",
    "indica",
    "marca",
    "trigger",
    "triggers",
    "activa",
    "cuando",
    "helps",
    "ayuda",
    "gathers",
    "bundle",
    "agrupa",
    "reune",
    "monitor",
    "store",
    "stores",
    "persist",
)
_EFFECT_LANGUAGE_MARKERS = (
    "cost",
    "coste",
    "consequence",
    "consecuencia",
    "impact",
    "impacto",
    "encarece",
    "trust",
    "confianza",
    "tickets",
    "excepc",
    "exceptions",
    "on-call",
    "causa",
    "causes",
    "rompe",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slugify_topic(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "topic"


@dataclass(frozen=True)
class EvidenceMapBuilder:
    """Convert source text into a structured evidence map with provenance."""

    client: Any
    logger: Logger

    def build(
        self,
        *,
        source_text: str,
        source_digest: str,
        episode_id: str,
        run_token: str,
        target_minutes: float,
        chunk_target_minutes: float,
        words_per_min: float,
    ) -> Dict[str, Any]:
        segments = self._build_source_segments(
            source_text=source_text,
            target_minutes=target_minutes,
            chunk_target_minutes=chunk_target_minutes,
            words_per_min=words_per_min,
        )
        segment_maps: List[Dict[str, Any]] = []
        for idx, segment in enumerate(segments, start=1):
            segment_maps.append(
                self._extract_segment_evidence(
                    segment_idx=idx,
                    total_segments=len(segments),
                    segment=segment,
                )
            )
        global_thesis = self._synthesize_global_thesis(segment_maps=segment_maps)
        payload = self._merge_segment_maps(
            source_digest=source_digest,
            episode_id=episode_id,
            run_token=run_token,
            segments=segments,
            segment_maps=segment_maps,
            global_thesis=global_thesis,
        )
        validated = validate_evidence_map(payload)
        self.logger.info(
            "evidence_map_built",
            episode_id=episode_id,
            source_segments=len(validated.get("source_segments", [])),
            claims=len(validated.get("claims", [])),
            topics=len(validated.get("topics", [])),
        )
        return validated

    def _build_source_segments(
        self,
        *,
        source_text: str,
        target_minutes: float,
        chunk_target_minutes: float,
        words_per_min: float,
    ) -> List[Dict[str, Any]]:
        raw_segments = split_source_chunks(
            source_text,
            target_minutes=max(3.0, float(target_minutes)),
            chunk_target_minutes=max(2.5, float(chunk_target_minutes)),
            words_per_min=max(80.0, float(words_per_min)),
        )
        if not raw_segments:
            raw_segments = [source_text.strip()]
        segments: List[Dict[str, Any]] = []
        cursor = 0
        for idx, raw in enumerate(raw_segments, start=1):
            text = str(raw or "").strip()
            if not text:
                continue
            location = source_text.find(text, cursor)
            if location < 0:
                location = source_text.find(text)
            if location < 0:
                location = cursor
            start_char = max(0, int(location))
            end_char = start_char + len(text)
            cursor = end_char
            source_ref = f"source:seg_{idx:03d}"
            segments.append(
                {
                    "source_ref": source_ref,
                    "start_char": start_char,
                    "end_char": end_char,
                    "sha256": _sha256_text(text),
                    "text": text,
                }
            )
        return segments

    def _extract_segment_evidence(
        self,
        *,
        segment_idx: int,
        total_segments: int,
        segment: Dict[str, Any],
    ) -> Dict[str, Any]:
        segment_text = str(segment.get("text", "") or "").strip()
        prompt_body = segment_text[:_CLAIM_PROMPT_MAX_CHARS]
        prompt = textwrap.dedent(
            f"""
            Build a compact evidence extraction for one podcast source segment.
            Return ONLY JSON.

            Rules:
            - Extract only claims supported by the segment.
            - Use short, concrete statements.
            - `kind` must be one of: fact, example, tension, context, counterpoint, quote.
            - `support` must be `direct` or `inferred_light`.
            - Use `inferred_light` only for light contextual synthesis, not for numbers or names.
            - `topic_hint` should be a short reusable topic label.
            - Prefer compact but complete coverage of distinct operational meanings over exhaustive paraphrase.
            - When the segment describes defaults, thresholds, paths, presets, commands, reports, or artifacts, extract purpose / trigger / usage claims if the segment supports them directly.
            - Good claim shapes include: "X sirve para Y", "X indica Y", "X se usa cuando Z", "X dispara Y cuando Z", or "X reune A/B/C para soporte o triage".
            - Do not invent downstream cost, staffing, trust, or urgency effects unless the segment says those effects directly.

            Segment {segment_idx}/{total_segments}
            source_ref={segment.get("source_ref")}

            SEGMENT:
            {prompt_body}
            """
        ).strip()
        payload = self.client.generate_script_json(
            prompt=prompt,
            schema=_SEGMENT_SCHEMA,
            max_output_tokens=2200,
            stage=f"evidence_map_segment_{segment_idx}",
        )
        if not isinstance(payload, dict):
            raise ValueError("segment evidence payload must be object")
        summary = str(payload.get("segment_summary", "") or "").strip()
        claims_out: List[Dict[str, Any]] = []
        for item in list(payload.get("claims", []) or []):
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement", "") or "").strip()
            kind = str(item.get("kind", "") or "").strip()
            topic_hint = str(item.get("topic_hint", "") or "").strip()
            support = str(item.get("support", "") or "").strip()
            if not statement or not kind or not topic_hint or not support:
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            claims_out.append(
                {
                    "statement": statement,
                    "kind": kind,
                    "topic_hint": topic_hint,
                    "support": support,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
        if not claims_out:
            claims_out.append(
                {
                    "statement": summary or segment_text[:220],
                    "kind": "context",
                    "topic_hint": "context",
                    "support": "direct",
                    "confidence": 0.75,
                }
            )
        claims_out = self._prioritize_claims(segment_text=segment_text, claims=claims_out)
        return {
            "segment_summary": summary or claims_out[0]["statement"],
            "claims": claims_out,
        }

    def _prioritize_claims(
        self,
        *,
        segment_text: str,
        claims: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        procedural_segment = self._looks_procedural(segment_text)
        segment_has_effect_language = self._has_effect_language(segment_text)

        def score(claim: Dict[str, Any]) -> tuple[int, float, str]:
            statement = str(claim.get("statement", "") or "")
            support = str(claim.get("support", "") or "").strip().lower()
            value = 0
            if support == "direct":
                value += 3
            if procedural_segment and self._looks_procedural(statement):
                value += 3
            if procedural_segment and self._has_effect_language(statement) and not segment_has_effect_language:
                value -= 4
            if procedural_segment and any(marker in statement.lower() for marker in _PROCEDURAL_CLAIM_MARKERS):
                value += 2
            return (value, float(claim.get("confidence", 0.0) or 0.0), statement)

        ranked = sorted(list(claims), key=score, reverse=True)
        return ranked

    def _looks_procedural(self, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        return any(marker in lowered for marker in _PROCEDURAL_SEGMENT_MARKERS)

    def _has_effect_language(self, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        return any(marker in lowered for marker in _EFFECT_LANGUAGE_MARKERS)

    def _synthesize_global_thesis(self, *, segment_maps: List[Dict[str, Any]]) -> str:
        summaries = []
        for idx, segment_map in enumerate(segment_maps, start=1):
            summary = str(segment_map.get("segment_summary", "") or "").strip()
            if summary:
                summaries.append(f"- Segment {idx}: {summary}")
        prompt = textwrap.dedent(
            f"""
            Infer one global thesis for a podcast episode from these segment summaries.
            Return ONLY JSON with key `global_thesis`.
            Keep it concise and factual.

            SUMMARIES:
            {chr(10).join(summaries[:16])}
            """
        ).strip()
        payload = self.client.generate_script_json(
            prompt=prompt,
            schema=_THESIS_SCHEMA,
            max_output_tokens=220,
            stage="evidence_map_global_thesis",
        )
        thesis = str(payload.get("global_thesis", "") or "").strip()
        return thesis or "Tema central del episodio"

    def _merge_segment_maps(
        self,
        *,
        source_digest: str,
        episode_id: str,
        run_token: str,
        segments: List[Dict[str, Any]],
        segment_maps: List[Dict[str, Any]],
        global_thesis: str,
    ) -> Dict[str, Any]:
        topic_accumulator: Dict[str, Dict[str, Any]] = {}
        claims: List[Dict[str, Any]] = []
        for segment, segment_map in zip(segments, segment_maps):
            source_ref = str(segment.get("source_ref"))
            for item in list(segment_map.get("claims", []) or []):
                topic_hint = str(item.get("topic_hint", "") or "").strip() or "context"
                topic_id = _slugify_topic(topic_hint)
                accumulator = topic_accumulator.setdefault(
                    topic_id,
                    {
                        "topic_id": topic_id,
                        "title": topic_hint,
                        "core_claim_ids": [],
                        "example_claim_ids": [],
                        "tension_claim_ids": [],
                        "priority_sum": 0.0,
                        "priority_count": 0,
                    },
                )
                claim_id = f"claim_{len(claims) + 1:03d}"
                kind = str(item.get("kind", "") or "").strip() or "context"
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
                claim_payload = {
                    "claim_id": claim_id,
                    "statement": str(item.get("statement", "") or "").strip(),
                    "kind": kind,
                    "topic_ids": [topic_id],
                    "source_refs": [source_ref],
                    "support": str(item.get("support", "") or "").strip() or "direct",
                    "confidence": confidence,
                }
                claims.append(claim_payload)
                if kind == "example":
                    accumulator["example_claim_ids"].append(claim_id)
                elif kind in {"tension", "counterpoint"}:
                    accumulator["tension_claim_ids"].append(claim_id)
                else:
                    accumulator["core_claim_ids"].append(claim_id)
                accumulator["priority_sum"] += confidence
                accumulator["priority_count"] += 1
        topics: List[Dict[str, Any]] = []
        for topic_id, topic in sorted(topic_accumulator.items()):
            claim_total = (
                len(topic["core_claim_ids"])
                + len(topic["example_claim_ids"])
                + len(topic["tension_claim_ids"])
            )
            priority = 0.6
            if topic["priority_count"] > 0:
                priority = max(0.2, min(0.99, topic["priority_sum"] / float(topic["priority_count"])))
            discardable = claim_total <= 1 and priority < 0.7
            topics.append(
                {
                    "topic_id": topic_id,
                    "title": topic["title"],
                    "core_claim_ids": list(topic["core_claim_ids"]),
                    "example_claim_ids": list(topic["example_claim_ids"]),
                    "tension_claim_ids": list(topic["tension_claim_ids"]),
                    "priority": round(priority, 4),
                    "discardable": bool(discardable),
                }
            )
        source_segments = [
            {
                "source_ref": str(item.get("source_ref")),
                "start_char": int(item.get("start_char", 0)),
                "end_char": int(item.get("end_char", 0)),
                "sha256": str(item.get("sha256")),
            }
            for item in segments
        ]
        return {
            "artifact_version": 1,
            "episode_id": episode_id,
            "run_token": run_token,
            "source_digest": source_digest,
            "source_segments": source_segments,
            "global_thesis": global_thesis,
            "claims": claims,
            "topics": topics,
        }
