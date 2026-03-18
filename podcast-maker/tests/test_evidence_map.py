import os
import sys
import unittest


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import LoggingConfig  # noqa: E402
from pipeline.evidence_map import EvidenceMapBuilder  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.podcast_artifacts import validate_evidence_map  # noqa: E402


class EvidenceMapTests(unittest.TestCase):
    def test_build_source_segments_splits_operations_reference_with_regular_chunk_target(self) -> None:
        builder = EvidenceMapBuilder(client=None, logger=Logger.create(LoggingConfig.from_env()))
        with open(os.path.join(ROOT_DIR, "references", "operations.md"), "r", encoding="utf-8") as f:
            source_text = f.read()
        segments = builder._build_source_segments(
            source_text=source_text,
            target_minutes=6.0,
            chunk_target_minutes=2.0,
            words_per_min=110.0,
        )
        self.assertGreater(len(segments), 1)

    def test_validate_evidence_map_normalizes_inferred_light_when_kind_and_support_are_swapped(self) -> None:
        payload = {
            "artifact_version": 1,
            "episode_id": "ep_test",
            "run_token": "run_test",
            "source_digest": "src_digest",
            "global_thesis": "Una tesis breve.",
            "source_segments": [
                {
                    "source_ref": "source:seg_001",
                    "start_char": 0,
                    "end_char": 120,
                    "sha256": "abc123",
                }
            ],
            "claims": [
                {
                    "claim_id": "claim_001",
                    "statement": "El bundle agrupa contexto para soporte.",
                    "kind": "inferred_light",
                    "topic_ids": ["bundle"],
                    "source_refs": ["source:seg_001"],
                    "support": "context",
                    "confidence": 0.7,
                }
            ],
            "topics": [
                {
                    "topic_id": "bundle",
                    "title": "bundle",
                    "core_claim_ids": ["claim_001"],
                    "example_claim_ids": [],
                    "tension_claim_ids": [],
                    "priority": 0.7,
                    "discardable": False,
                }
            ],
        }
        validated = validate_evidence_map(payload)
        claim = validated["claims"][0]
        self.assertEqual(claim["kind"], "context")
        self.assertEqual(claim["support"], "inferred_light")


if __name__ == "__main__":
    unittest.main()
