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


if __name__ == "__main__":
    unittest.main()
