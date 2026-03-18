#!/usr/bin/env python3
from __future__ import annotations

"""Editorial rewrite stage for the redesigned podcast pipeline."""

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict

from .logging_utils import Logger
from .podcast_artifacts import build_script_artifact, validate_script_artifact

_REWRITE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
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
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class EditorialRewriter:
    """Turn a dry draft into an oral, listener-facing script."""

    client: Any
    logger: Logger

    def rewrite(
        self,
        *,
        script_artifact: Dict[str, Any],
        evidence_map: Dict[str, Any],
        episode_plan: Dict[str, Any],
        editorial_report: Dict[str, Any] | None = None,
        round_idx: int = 1,
    ) -> Dict[str, Any]:
        validated = validate_script_artifact(script_artifact, episode_plan=episode_plan)
        report_text = json.dumps(editorial_report or {}, ensure_ascii=False, indent=2)
        prompt = textwrap.dedent(
            f"""
            Rewrite this podcast script so it sounds oral, sharp, and worth listening to.
            Return ONLY JSON with key `lines`.

            Editorial goals:
            - Preserve beat coverage and do not compress the episode into fewer beats or fewer hosts.
            - Reduce density and mini-essay turns.
            - Remove scaffold phrases and visible stitching.
            - Do not keep repeating stock turn openers such as "Por otro lado", "Ahora bien", "Exacto", "Tal cual", "Claro", or "Totalmente".
            - Keep Host1 as synthesizer/organizer.
            - Make Host2 push harder with objections, examples, tradeoffs, decisions, and grounding.
            - In profile `short`, compress aggressively: keep most beats to 3-5 turns, merge adjacent Host1 fact-dumps, and do not let Host1 stack multiple list turns when one tighter turn will do.
            - Prefer direct entries into the idea, objection, or example instead of assent-only openings.
            - In the last beat and last 4-6 turns, allow only one earned close and make each turn add a new move.
            - Do not rebuild a second abstract recap after the closing beat has already landed.
            - Do not keep assent-only turns near the close unless they immediately open an objection, example, cost, or decision.
            - Avoid repeating the thesis unless the narrative function changes.
            - Improve listener engagement and freshness.
            - Preserve facts and do not invent anything.
            - Keep rules, defaults, thresholds, paths, and presets grounded as procedures or conditions; do not upgrade them into unsupported effects.
            - Use cost, urgency, trust, staffing, or consequence language only when the evidence map states that effect directly.
            - Prefer paraphrases such as "sirve para", "indica", "te deja", "se usa cuando", or "marca el umbral".
            - Avoid unsupported effect phrasing such as "causa", "encarece", "sale mas caro", "siempre hay alguien", or "te rompe".
            - Do not turn "strict" or production presets into superlatives such as "el mas exigente" unless the evidence says that verbatim.
            - Split lines if one clause is factual and another clause drifts into interpretation.
            - If support tooling or a debug bundle appears near the end, bridge it explicitly from rollout / rollback / health-window logic instead of opening a fresh topic.
            - Let the debug-bundle example be the last operational proof, then make the final line return to the thesis rather than staying inside support detail.
            - In every beat, Host2 must contribute friction or a decision, not just ask for the next batch of facts.
            - In the closing beat, rewrite Host2 so it draws a contrast, triage order, escalation trigger, or operator decision; a plain exposition line is not enough.
            - Keep spoken text in Spanish.
            - Keep instructions in short actionable English.
            - Preserve the same topic order unless the editorial report explicitly flags a transition problem.

            Concrete actions allowed:
            - split dense turns
            - shorten abstract explanations
            - replace abstraction with example
            - remove scaffold phrase entirely
            - sharpen opening
            - make closing shorter and earned

            Rewrite round: {round_idx}

            EPISODE PLAN:
            {json.dumps(episode_plan, ensure_ascii=False, indent=2)}

            EVIDENCE MAP:
            {json.dumps(evidence_map, ensure_ascii=False, indent=2)}

            EDITORIAL REPORT:
            {report_text}

            SCRIPT:
            {json.dumps({"lines": validated.get("lines", [])}, ensure_ascii=False, indent=2)}
            """
        ).strip()
        payload = self.client.generate_script_json(
            prompt=prompt,
            schema=_REWRITE_SCHEMA,
            max_output_tokens=9000,
            stage=f"editorial_rewriter_{round_idx}",
        )
        rewritten = build_script_artifact(
            stage="rewritten",
            episode_id=str(validated.get("episode_id")),
            run_token=str(validated.get("run_token")),
            source_digest=str(validated.get("source_digest")),
            plan_ref=str(validated.get("plan_ref")),
            plan_digest=str(validated.get("plan_digest")),
            lines=payload.get("lines", []),
            episode_plan=episode_plan,
            prior_artifact=validated,
            target_word_count=validated.get("target_word_count"),
        )
        self.logger.info(
            "editorial_rewrite_done",
            round=round_idx,
            line_count=len(rewritten.get("lines", [])),
        )
        return rewritten
