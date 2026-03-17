#!/usr/bin/env python3
from __future__ import annotations

"""Episode planning for the redesigned podcast generator."""

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List

from .logging_utils import Logger
from .podcast_artifacts import validate_episode_plan

_EPISODE_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "opening_mode": {"type": "string"},
        "closing_mode": {"type": "string"},
        "host_roles": {
            "type": "object",
            "properties": {
                "Host1": {"type": "string"},
                "Host2": {"type": "string"},
            },
            "required": ["Host1", "Host2"],
            "additionalProperties": False,
        },
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "beat_id": {"type": "string"},
                    "goal": {"type": "string"},
                    "topic_ids": {"type": "array", "items": {"type": "string"}},
                    "claim_ids": {"type": "array", "items": {"type": "string"}},
                    "required_move": {"type": "string"},
                    "optional_moves": {"type": "array", "items": {"type": "string"}},
                    "must_cover": {"type": "array", "items": {"type": "string"}},
                    "can_cut": {"type": "boolean"},
                    "target_words": {"type": "integer"},
                },
                "required": [
                    "beat_id",
                    "goal",
                    "topic_ids",
                    "claim_ids",
                    "required_move",
                    "optional_moves",
                    "must_cover",
                    "can_cut",
                    "target_words",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["opening_mode", "closing_mode", "host_roles", "beats"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class EpisodePlanner:
    """Plan conversational beats before drafting dialogue."""

    client: Any
    logger: Logger

    def build(
        self,
        *,
        evidence_map: Dict[str, Any],
        episode_id: str,
        run_token: str,
        profile_name: str,
        min_words: int,
        max_words: int,
    ) -> Dict[str, Any]:
        prompt = textwrap.dedent(
            f"""
            Design a podcast episode plan from this evidence map.
            Return ONLY JSON.

            Rules:
            - `opening_mode` must be one of: provocative_observation, concrete_tension, sharp_question, contrast_frame, mini_scene.
            - `closing_mode` must be one of: earned_synthesis, practical_takeaway, contrast_return, resolved_question.
            - Host1 role: sintetiza_y_ordena.
            - Host2 role: desafia_y_aterriza.
            - Every beat must include one required_move chosen from:
              example, objection, tradeoff, consequence, cost, awkward_question, grounding, counterexample, decision.
            - Avoid roadmap openings and explicit index narration.
            - Avoid two consecutive purely explanatory beats.
            - For `short`, prefer 3-4 beats; for `standard`, 4-6 beats; for `long`, 5-7 beats.
            - `can_cut=true` only for lower-priority beats.
            - `target_words` across beats should fit roughly within {min_words}-{max_words} total words.

            EVIDENCE MAP:
            {json.dumps(evidence_map, ensure_ascii=False, indent=2)}
            """
        ).strip()
        payload = self.client.generate_script_json(
            prompt=prompt,
            schema=_EPISODE_PLAN_SCHEMA,
            max_output_tokens=2600,
            stage="episode_planner",
        )
        payload = self._normalize_payload(dict(payload) if isinstance(payload, dict) else {})
        if not isinstance(payload, dict) or not payload.get("opening_mode") or not payload.get("beats"):
            payload = self._fallback_plan(
                evidence_map=evidence_map,
                profile_name=profile_name,
                min_words=min_words,
                max_words=max_words,
            )
        candidate = {
            "artifact_version": 1,
            "episode_id": episode_id,
            "run_token": run_token,
            "opening_mode": payload.get("opening_mode"),
            "closing_mode": payload.get("closing_mode"),
            "host_roles": payload.get("host_roles"),
            "beats": payload.get("beats"),
        }
        validated = validate_episode_plan(candidate)
        self._validate_functional_diversity(validated)
        self._rebalance_target_words(
            validated,
            profile_name=profile_name,
            min_words=min_words,
            max_words=max_words,
        )
        self.logger.info(
            "episode_plan_built",
            episode_id=episode_id,
            beats=len(validated.get("beats", [])),
            opening_mode=validated.get("opening_mode"),
            closing_mode=validated.get("closing_mode"),
        )
        return validated

    def _validate_functional_diversity(self, plan: Dict[str, Any]) -> None:
        beats = list(plan.get("beats", []))
        previous_goal = ""
        previous_explanatory = False
        for beat in beats:
            goal = str(beat.get("goal", "") or "").strip()
            moves = {str(beat.get("required_move", "") or "").strip()}
            moves.update(str(item or "").strip() for item in list(beat.get("optional_moves", []) or []))
            explanatory = goal == "explain_core" and not moves.intersection(
                {"example", "objection", "tradeoff", "cost", "awkward_question"}
            )
            if explanatory and previous_explanatory and goal == previous_goal:
                raise ValueError("episode_plan contains consecutive purely explanatory beats")
            previous_goal = goal
            previous_explanatory = explanatory

    def _rebalance_target_words(
        self,
        plan: Dict[str, Any],
        *,
        profile_name: str,
        min_words: int,
        max_words: int,
    ) -> None:
        beats = list(plan.get("beats", []))
        if not beats:
            return
        if str(profile_name or "").strip().lower() == "short" and len(beats) > 4:
            trimmed: List[Dict[str, Any]] = []
            tail_closing = beats[-1]
            middle = list(beats[1:-1])
            kept_middle: List[Dict[str, Any]] = []
            for beat in middle:
                if len(kept_middle) >= 2:
                    break
                if not bool(beat.get("can_cut", False)) or len(middle) <= 2:
                    kept_middle.append(beat)
            while len(kept_middle) < min(2, len(middle)):
                candidate = middle[len(kept_middle)]
                if candidate not in kept_middle:
                    kept_middle.append(candidate)
            trimmed.append(beats[0])
            trimmed.extend(kept_middle[:2])
            trimmed.append(tail_closing)
            plan["beats"] = trimmed[:4]
            beats = list(plan.get("beats", []))
        target_total = max(min_words, int(round((min_words + max_words) / 2.0)))
        if str(profile_name or "").strip().lower() == "short":
            target_total = min(target_total, max_words)
        current_total = sum(max(1, int(beat.get("target_words", 1))) for beat in beats)
        if current_total <= 0:
            current_total = len(beats)
        scale = float(target_total) / float(current_total)
        for beat in beats:
            beat["target_words"] = max(45, int(round(max(1, int(beat.get("target_words", 1))) * scale)))

    def _fallback_plan(
        self,
        *,
        evidence_map: Dict[str, Any],
        profile_name: str,
        min_words: int,
        max_words: int,
    ) -> Dict[str, Any]:
        topics = list(evidence_map.get("topics", []) or [])
        claims = list(evidence_map.get("claims", []) or [])
        max_beats = {"short": 4, "standard": 5, "long": 6}.get(str(profile_name or "").strip().lower(), 5)
        selected_topics = topics[:max_beats] or [{"topic_id": "topic_001", "title": "tema central"}]
        beats: List[Dict[str, Any]] = []
        goal_cycle = [
            "hook_and_frame",
            "explain_core",
            "concrete_example",
            "objection_and_tradeoff",
            "closing",
        ]
        move_cycle = ["objection", "grounding", "example", "tradeoff", "decision"]
        words_total = max(min_words, int(round((min_words + max_words) / 2.0)))
        words_per_beat = max(50, int(round(words_total / float(max(1, len(selected_topics))))))
        for idx, topic in enumerate(selected_topics, start=1):
            topic_id = str(topic.get("topic_id", f"topic_{idx:03d}") or f"topic_{idx:03d}")
            claim_ids = [
                str(item.get("claim_id", "")).strip()
                for item in claims
                if topic_id in list(item.get("topic_ids", []) or [])
            ]
            claim_ids = [item for item in claim_ids if item]
            beats.append(
                {
                    "beat_id": f"beat_{idx:02d}",
                    "goal": goal_cycle[min(idx - 1, len(goal_cycle) - 1)],
                    "topic_ids": [topic_id],
                    "claim_ids": claim_ids[:3],
                    "required_move": move_cycle[min(idx - 1, len(move_cycle) - 1)],
                    "optional_moves": ["example"] if idx < len(selected_topics) else ["decision"],
                    "must_cover": claim_ids[:2],
                    "can_cut": idx not in {1, len(selected_topics)},
                    "target_words": words_per_beat,
                }
            )
        return {
            "opening_mode": "concrete_tension",
            "closing_mode": "earned_synthesis",
            "host_roles": {
                "Host1": "sintetiza_y_ordena",
                "Host2": "desafia_y_aterriza",
            },
            "beats": beats,
        }

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        opening_mode = self._normalize_opening_mode(payload.get("opening_mode"))
        closing_mode = self._normalize_closing_mode(payload.get("closing_mode"))
        host_roles = payload.get("host_roles")
        if not isinstance(host_roles, dict):
            host_roles = {
                "Host1": "sintetiza_y_ordena",
                "Host2": "desafia_y_aterriza",
            }
        beats_out: List[Dict[str, Any]] = []
        for idx, item in enumerate(list(payload.get("beats", []) or []), start=1):
            if not isinstance(item, dict):
                continue
            beats_out.append(
                {
                    "beat_id": str(item.get("beat_id", f"beat_{idx:02d}") or f"beat_{idx:02d}"),
                    "goal": self._normalize_goal(item.get("goal")),
                    "topic_ids": list(item.get("topic_ids", []) or []),
                    "claim_ids": list(item.get("claim_ids", []) or []),
                    "required_move": self._normalize_move(item.get("required_move")),
                    "optional_moves": [self._normalize_move(move) for move in list(item.get("optional_moves", []) or []) if self._normalize_move(move)],
                    "must_cover": list(item.get("must_cover", []) or []),
                    "can_cut": bool(item.get("can_cut", False)),
                    "target_words": max(45, int(item.get("target_words", 60) or 60)),
                }
            )
        payload["opening_mode"] = opening_mode
        payload["closing_mode"] = closing_mode
        payload["host_roles"] = host_roles
        payload["beats"] = beats_out
        return payload

    def _normalize_opening_mode(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if "scene" in text or "escena" in text:
            return "mini_scene"
        if "question" in text or "pregunta" in text:
            return "sharp_question"
        if "contrast" in text or "contraste" in text:
            return "contrast_frame"
        if "provoc" in text:
            return "provocative_observation"
        return "concrete_tension"

    def _normalize_closing_mode(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if "practic" in text or "accion" in text:
            return "practical_takeaway"
        if "question" in text or "pregunta" in text:
            return "resolved_question"
        if "contrast" in text or "contraste" in text:
            return "contrast_return"
        return "earned_synthesis"

    def _normalize_goal(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if "hook" in text or "abr" in text or "tension" in text:
            return "hook_and_frame"
        if "example" in text or "ejemplo" in text:
            return "concrete_example"
        if "objec" in text or "tradeoff" in text or "cost" in text or "coste" in text:
            return "objection_and_tradeoff"
        if "consequence" in text or "impact" in text or "consecuencia" in text:
            return "consequence"
        if "takeaway" in text or "practic" in text or "accion" in text:
            return "practical_takeaway"
        if "close" in text or "cier" in text or "synth" in text:
            return "closing"
        return "explain_core"

    def _normalize_move(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if "example" in text or "ejemplo" in text:
            return "example"
        if "objec" in text:
            return "objection"
        if "tradeoff" in text:
            return "tradeoff"
        if "cost" in text or "coste" in text:
            return "cost"
        if "awkward" in text or "incomoda" in text:
            return "awkward_question"
        if "ground" in text or "tierra" in text or "aterr" in text:
            return "grounding"
        if "counter" in text:
            return "counterexample"
        if "decision" in text or "decis" in text:
            return "decision"
        if "consequence" in text or "consecuencia" in text:
            return "consequence"
        return "grounding"
