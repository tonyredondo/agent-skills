import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.podcast_artifacts import (  # noqa: E402
    apply_script_patch_batch,
    build_script_artifact,
    validate_fact_guard_report,
    validate_script_patch_batch,
)


class PodcastArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episode_plan = {
            "beats": [
                {"beat_id": "beat_01", "can_cut": False, "target_words": 50},
                {"beat_id": "beat_02", "can_cut": True, "target_words": 40},
                {"beat_id": "beat_03", "can_cut": False, "target_words": 50},
            ]
        }
        self.lines = [
            {
                "speaker": "Ana",
                "role": "Host1",
                "instructions": "Warm, clear tone.",
                "text": "Abrimos una idea fuerte.",
            },
            {
                "speaker": "Luis",
                "role": "Host2",
                "instructions": "Curious, grounded tone.",
                "text": "Vale, aterrizalo con un ejemplo.",
            },
            {
                "speaker": "Ana",
                "role": "Host1",
                "instructions": "Warm, clear tone.",
                "text": "El ejemplo muestra el coste oculto.",
            },
            {
                "speaker": "Luis",
                "role": "Host2",
                "instructions": "Curious, grounded tone.",
                "text": "Y eso cambia como decides automatizar.",
            },
        ]

    def _artifact(self) -> dict:
        return build_script_artifact(
            stage="draft",
            episode_id="episode",
            run_token="run_123",
            source_digest="src_digest",
            plan_ref="episode_plan.json",
            plan_digest="plan_digest",
            lines=self.lines,
            episode_plan=self.episode_plan,
            target_word_count=140,
        )

    def test_build_script_artifact_adds_turns_and_coverage(self) -> None:
        artifact = self._artifact()
        self.assertEqual(artifact["resume_compat_version"], 1)
        self.assertEqual(len(artifact["turns"]), len(self.lines))
        self.assertTrue(all(turn.get("line_id") for turn in artifact["turns"]))
        self.assertIn("coverage", artifact)
        self.assertIn("public_payload_digest", artifact)
        self.assertIn("internal_artifact_digest", artifact)
        self.assertEqual(artifact["coverage"]["line_count"], len(self.lines))

    def test_patch_batch_applies_multi_insert_in_batch_order(self) -> None:
        artifact = self._artifact()
        anchor = artifact["turns"][0]["line_id"]
        patched = apply_script_patch_batch(
            script_artifact=artifact,
            patch_batch={
                "patches": [
                    {
                        "op": "insert_after",
                        "anchor_line_id": anchor,
                        "line": {
                            "speaker": "Luis",
                            "role": "Host2",
                            "instructions": "Curious, grounded tone.",
                            "text": "Primera insercion.",
                        },
                    },
                    {
                        "op": "insert_after",
                        "anchor_line_id": anchor,
                        "line": {
                            "speaker": "Ana",
                            "role": "Host1",
                            "instructions": "Warm, clear tone.",
                            "text": "Segunda insercion.",
                        },
                    },
                ]
            },
            episode_plan=self.episode_plan,
        )
        texts = [turn["text"] for turn in patched["turns"][:3]]
        self.assertEqual(texts, ["Abrimos una idea fuerte.", "Primera insercion.", "Segunda insercion."])

    def test_patch_batch_rejects_conflicting_same_target(self) -> None:
        artifact = self._artifact()
        line_id = artifact["turns"][1]["line_id"]
        with self.assertRaises(ValueError):
            apply_script_patch_batch(
                script_artifact=artifact,
                patch_batch={
                    "patches": [
                        {"op": "delete_line", "line_id": line_id},
                        {
                            "op": "replace_line",
                            "line_id": line_id,
                            "line": {
                                "speaker": "Luis",
                                "role": "Host2",
                                "instructions": "Curious, grounded tone.",
                                "text": "Conflicto.",
                            },
                        },
                    ]
                },
                episode_plan=self.episode_plan,
            )

    def test_validate_script_patch_batch_rejects_unknown_op(self) -> None:
        with self.assertRaises(ValueError):
            validate_script_patch_batch({"patches": [{"op": "merge_lines"}]})

    def test_validate_fact_guard_report_allows_empty_claim_id(self) -> None:
        report = validate_fact_guard_report(
            {
                "artifact_version": 1,
                "resume_compat_version": 1,
                "stage": "rewritten",
                "run_token": "run_123",
                "source_digest": "src_digest",
                "plan_digest": "plan_digest",
                "internal_artifact_digest": "artifact_digest",
                "public_payload_digest": "public_digest",
                "pass": False,
                "issues": [
                    {
                        "issue_id": "issue_001",
                        "issue_type": "unsupported_claim",
                        "severity": "medium",
                        "claim_id": "",
                        "line_indexes": [2],
                        "source_refs": ["doc:seg_001"],
                        "origin_stage": "rewritten",
                        "action": "rewrite_local",
                    }
                ],
            }
        )
        self.assertEqual(report["issues"][0]["claim_id"], "")


if __name__ == "__main__":
    unittest.main()
