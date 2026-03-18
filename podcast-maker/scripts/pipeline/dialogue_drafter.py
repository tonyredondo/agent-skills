#!/usr/bin/env python3
from __future__ import annotations

"""Beat-oriented dialogue drafting for the redesigned generator."""

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict

from .logging_utils import Logger
from .podcast_artifacts import build_script_artifact

_SCAFFOLD_PHRASES = (
    "Por otro lado",
    "Ahora bien",
    "Dicho esto",
    "A partir de ahi",
    "Pasando a",
    "En paralelo",
    "Si lo conectamos",
    "En ese sentido",
    "Por cierto",
    "Exacto",
    "Tal cual",
    "Claro",
    "Totalmente",
)


@dataclass(frozen=True)
class DialogueDrafter:
    """Generate a compact, factual draft from evidence + episode plan."""

    client: Any
    logger: Logger

    def draft(
        self,
        *,
        evidence_map: Dict[str, Any],
        episode_plan: Dict[str, Any],
        episode_id: str,
        run_token: str,
        source_digest: str,
        plan_digest: str,
        profile_name: str,
        min_words: int,
        max_words: int,
    ) -> Dict[str, Any]:
        scaffold_block = "\n".join(f"- {item}" for item in _SCAFFOLD_PHRASES)
        prompt = textwrap.dedent(
            f"""
            Draft a podcast dialogue in Spanish from this plan and evidence map.
            Return ONLY JSON with key `lines`.

            Drafting philosophy:
            - The draft must be shorter, drier, and more functional than the final version.
            - Optimize for clarity, progression, host contrast, and factual fidelity.
            - Do NOT optimize for polished final orality yet.

            Hard rules:
            - Keep spoken text in Spanish.
            - Keep `instructions` in short actionable English (1-2 sentences).
            - Keep `pace_hint` to calm, steady, or brisk when used.
            - Host1 synthesizes and orders.
            - Host2 challenges, grounds, asks for examples, and pushes on tradeoffs.
            - Use cost, urgency, trust, staffing, or operational consequence language ONLY when the source states that consequence directly.
            - Cover every non-cuttable beat in order. Do not collapse the whole episode into a short summary.
            - Keep at least one Host2 intervention in every non-cuttable beat.
            - Keep at least 2 turns per beat whenever the target_words budget for that beat is >= 45.
            - In profile `short`, keep most beats in the 3-5 turn range unless the plan absolutely requires one extra turn to land the decision.
            - Do not open with an agenda or roadmap.
            - State the core thesis once, then advance by example, consequence, objection, tradeoff, or incident.
            - If the source only gives a rule, threshold, default, path, or preset, describe what it does or when it is used; do not upgrade it into downstream impact.
            - Host2 must do more than request another list: force a tradeoff, objection, concrete implication, or operational decision in every beat.
            - In the closing beat, Host2 must force a distinction, escalation rule, or operator choice; exposition alone is not enough.
            - Avoid recap/farewell templates as mandatory ending form.
            - Earn only one close in the final beat; do not append a second recap after the close lands.
            - In the last 4-6 turns, every turn must add a new move: example, objection, cost, consequence, grounding, or decision.
            - Do not use assent-only turns near the close unless they immediately add new information.
            - Avoid visible scaffold phrases such as:
            {scaffold_block}
            - Do not chain stock openers across nearby turns. If one line starts with a filler, the next turns must enter directly.
            - Avoid validation-only turn openings such as "Exacto", "Tal cual", "Claro", or "Totalmente" unless they are exceptional and immediately add new information.
            - Avoid wording that sounds like a memo, article, or briefing.
            - Prefer factual paraphrases such as "sirve para", "indica", "te deja", "se usa cuando", or "marca el umbral".
            - Avoid unsupported outcome phrasing such as "causa", "encarece", "sale mas caro", "siempre hay alguien", or "te rompe".
            - Do not turn "strict" or production presets into superlatives such as "el mas exigente" unless the evidence says that verbatim.
            - Split multi-claim lines when one clause is directly grounded and the other is rhetorical or inferential.
            - In profile `{profile_name}`, do not over-expand just to fill quota.
            - Keep the draft roughly within {min_words}-{max_words} words.

            EPISODE PLAN:
            {json.dumps(episode_plan, ensure_ascii=False, indent=2)}

            EVIDENCE MAP:
            {json.dumps(evidence_map, ensure_ascii=False, indent=2)}
            """
        ).strip()
        payload = self.client.generate_script_json(
            prompt=prompt,
            schema={
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
            },
            max_output_tokens=8200,
            stage="dialogue_drafter",
        )
        artifact = build_script_artifact(
            stage="draft",
            episode_id=episode_id,
            run_token=run_token,
            source_digest=source_digest,
            plan_ref="episode_plan.json",
            plan_digest=plan_digest,
            lines=payload.get("lines", []),
            episode_plan=episode_plan,
            target_word_count=sum(int(beat.get("target_words", 0) or 0) for beat in list(episode_plan.get("beats", []) or [])),
        )
        self.logger.info(
            "dialogue_draft_built",
            episode_id=episode_id,
            line_count=len(artifact.get("lines", [])),
        )
        return artifact
