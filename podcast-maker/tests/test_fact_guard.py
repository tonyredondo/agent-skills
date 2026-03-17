import os
import re
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import LoggingConfig  # noqa: E402
from pipeline.fact_guard import FactGuard  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.podcast_artifacts import build_script_artifact  # noqa: E402


class _FactGuardRepairClient:
    def __init__(self) -> None:
        self.repair_calls = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001, ARG002
        if stage != "fact_guard_repair_final":
            raise AssertionError(stage)
        self.repair_calls += 1
        line_ids = re.findall(r'"line_id":\s*"([^"]+)"', prompt)
        target_line_id = line_ids[1]
        return {
            "patches": [
                {
                    "op": "replace_line",
                    "line_id": target_line_id,
                    "anchor_line_id": None,
                    "line": {
                        "speaker": "Luis",
                        "role": "Host2",
                        "instructions": "Curious, grounded tone.",
                        "pace_hint": None,
                        "text": "Mejor decirlo asi: eso indica cuando conviene mirar el bundle, no que siempre escale soporte.",
                    },
                }
            ]
        }


class FactGuardTests(unittest.TestCase):
    def test_repair_consumes_rewrite_local_even_when_report_also_has_blocks(self) -> None:
        episode_plan = {
            "beats": [
                {"beat_id": "beat_01", "can_cut": False, "target_words": 40},
                {"beat_id": "beat_02", "can_cut": False, "target_words": 40},
            ]
        }
        artifact = build_script_artifact(
            stage="final",
            episode_id="episode",
            run_token="run_123",
            source_digest="src_digest",
            plan_ref="episode_plan.json",
            plan_digest="plan_digest",
            lines=[
                {
                    "speaker": "Ana",
                    "role": "Host1",
                    "instructions": "Warm, clear tone.",
                    "text": "La regla marca cuando revertir.",
                },
                {
                    "speaker": "Luis",
                    "role": "Host2",
                    "instructions": "Curious, grounded tone.",
                    "text": "Eso significa que siempre hay alguien de soporte esperando el bundle.",
                },
            ],
            episode_plan=episode_plan,
            target_word_count=80,
        )
        report = {
            "artifact_version": 1,
            "resume_compat_version": 1,
            "stage": "final",
            "run_token": "run_123",
            "source_digest": "src_digest",
            "plan_digest": "plan_digest",
            "internal_artifact_digest": artifact["internal_artifact_digest"],
            "public_payload_digest": artifact["public_payload_digest"],
            "pass": False,
            "issues": [
                {
                    "issue_id": "issue_001",
                    "issue_type": "overstated_causality",
                    "severity": "high",
                    "claim_id": "claim_001",
                    "line_indexes": [0],
                    "source_refs": ["source:seg_001"],
                    "origin_stage": "final",
                    "action": "block",
                },
                {
                    "issue_id": "issue_002",
                    "issue_type": "unsupported_claim",
                    "severity": "medium",
                    "claim_id": "claim_002",
                    "line_indexes": [1],
                    "source_refs": ["source:seg_001"],
                    "origin_stage": "final",
                    "action": "rewrite_local",
                },
            ],
        }
        repaired = FactGuard(
            client=_FactGuardRepairClient(),
            logger=Logger.create(LoggingConfig.from_env()),
        ).repair(
            script_artifact=artifact,
            evidence_map={"claims": []},
            episode_plan=episode_plan,
            report=report,
            stage_name="final",
        )
        texts = [turn["text"] for turn in repaired["turns"]]
        self.assertEqual(texts[0], "La regla marca cuando revertir.")
        self.assertIn("indica cuando conviene mirar el bundle", texts[1].lower())


if __name__ == "__main__":
    unittest.main()
