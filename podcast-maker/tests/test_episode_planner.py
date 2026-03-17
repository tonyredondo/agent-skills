import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import LoggingConfig  # noqa: E402
from pipeline.episode_planner import EpisodePlanner  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402


class _PlannerClient:
    def __init__(self, payload):
        self.payload = payload

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001, ARG002
        if stage != "episode_planner":
            raise AssertionError(stage)
        return self.payload


class EpisodePlannerTests(unittest.TestCase):
    def _planner(self, payload: dict) -> EpisodePlanner:
        return EpisodePlanner(client=_PlannerClient(payload), logger=Logger.create(LoggingConfig.from_env()))

    def test_build_downgrades_cost_and_consequence_moves_for_procedural_claims(self) -> None:
        evidence_map = {
            "claims": [
                {
                    "claim_id": "claim_001",
                    "statement": "SCRIPT_QUALITY_GATE_ACTION=warn deja el gate en modo advertencia antes del audio.",
                    "support": "direct",
                },
                {
                    "claim_id": "claim_002",
                    "statement": "El historial SLO se persiste en ./.podcast_slo_history.jsonl.",
                    "support": "direct",
                },
                {
                    "claim_id": "claim_003",
                    "statement": "Rollback trigger: si fallan dos ventanas consecutivas, vuelve a la ultima release conocida como buena.",
                    "support": "direct",
                },
                {
                    "claim_id": "claim_004",
                    "statement": "El debug bundle reune diagnosticos para soporte u on-call.",
                    "support": "direct",
                },
            ],
            "topics": [
                {"topic_id": "warn_default", "title": "warn default"},
                {"topic_id": "slo_history_path", "title": "slo history path"},
                {"topic_id": "rollback_trigger", "title": "rollback trigger"},
                {"topic_id": "debug_bundle", "title": "debug bundle"},
            ],
        }
        payload = {
            "opening_mode": "concrete_tension",
            "closing_mode": "earned_synthesis",
            "host_roles": {"Host1": "sintetiza_y_ordena", "Host2": "desafia_y_aterriza"},
            "beats": [
                {
                    "beat_id": "beat_01",
                    "goal": "explain_core",
                    "topic_ids": ["warn_default"],
                    "claim_ids": ["claim_001"],
                    "required_move": "consequence",
                    "optional_moves": ["cost"],
                    "must_cover": ["warn default"],
                    "can_cut": False,
                    "target_words": 60,
                },
                {
                    "beat_id": "beat_02",
                    "goal": "consequence",
                    "topic_ids": ["debug_bundle"],
                    "claim_ids": ["claim_004"],
                    "required_move": "cost",
                    "optional_moves": ["consequence"],
                    "must_cover": ["debug bundle"],
                    "can_cut": False,
                    "target_words": 60,
                },
            ],
        }
        plan = self._planner(payload).build(
            evidence_map=evidence_map,
            episode_id="episode",
            run_token="run_123",
            profile_name="short",
            min_words=120,
            max_words=160,
        )
        required_moves = [beat["required_move"] for beat in plan["beats"]]
        optional_moves = [move for beat in plan["beats"] for move in beat["optional_moves"]]
        self.assertNotIn("cost", required_moves)
        self.assertNotIn("consequence", required_moves)
        self.assertNotIn("cost", optional_moves)
        self.assertNotIn("consequence", optional_moves)
        self.assertEqual(plan["beats"][1]["goal"], "explain_core")

    def test_build_keeps_effect_move_when_claim_states_cost_directly(self) -> None:
        evidence_map = {
            "claims": [
                {
                    "claim_id": "claim_001",
                    "statement": "Las reglas opacas generan tickets y excepciones.",
                    "support": "direct",
                }
            ],
            "topics": [{"topic_id": "coste_oculto", "title": "coste oculto"}],
        }
        payload = {
            "opening_mode": "concrete_tension",
            "closing_mode": "earned_synthesis",
            "host_roles": {"Host1": "sintetiza_y_ordena", "Host2": "desafia_y_aterriza"},
            "beats": [
                {
                    "beat_id": "beat_01",
                    "goal": "objection_and_tradeoff",
                    "topic_ids": ["coste_oculto"],
                    "claim_ids": ["claim_001"],
                    "required_move": "cost",
                    "optional_moves": ["consequence"],
                    "must_cover": ["coste oculto"],
                    "can_cut": False,
                    "target_words": 70,
                }
            ],
        }
        plan = self._planner(payload).build(
            evidence_map=evidence_map,
            episode_id="episode",
            run_token="run_123",
            profile_name="short",
            min_words=70,
            max_words=90,
        )
        self.assertEqual(plan["beats"][0]["required_move"], "cost")
        self.assertIn("consequence", plan["beats"][0]["optional_moves"])

    def test_validate_functional_diversity_accepts_grounding_and_decision_beats(self) -> None:
        planner = self._planner({"opening_mode": "concrete_tension", "closing_mode": "earned_synthesis", "host_roles": {}, "beats": []})
        planner._validate_functional_diversity(
            {
                "beats": [
                    {"goal": "explain_core", "required_move": "grounding", "optional_moves": []},
                    {"goal": "explain_core", "required_move": "decision", "optional_moves": []},
                ]
            }
        )


if __name__ == "__main__":
    unittest.main()
