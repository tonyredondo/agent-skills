import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.script_chunker import context_tail, split_source_chunks  # noqa: E402


class ScriptChunkerTests(unittest.TestCase):
    def test_empty_source_returns_no_chunks(self) -> None:
        chunks = split_source_chunks(
            "   \n\n \n",
            target_minutes=15.0,
            chunk_target_minutes=2.5,
            words_per_min=130.0,
        )
        self.assertEqual(chunks, [])

    def test_split_source_chunks(self) -> None:
        source = "\n\n".join([f"Parrafo {i} " + ("texto " * 40) for i in range(1, 9)])
        chunks = split_source_chunks(
            source,
            target_minutes=15.0,
            chunk_target_minutes=2.5,
            words_per_min=130.0,
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(c.strip() for c in chunks))

    def test_context_tail(self) -> None:
        lines = [{"speaker": "A", "role": "Host1", "instructions": "x", "text": str(i)} for i in range(10)]
        tail = context_tail(lines, 3)
        self.assertEqual(len(tail), 3)
        self.assertEqual(tail[0]["text"], "7")

    def test_monolithic_input_is_split(self) -> None:
        source = ("frase muy larga con contenido util " * 1200).strip()
        chunks = split_source_chunks(
            source,
            target_minutes=30.0,
            chunk_target_minutes=2.5,
            words_per_min=130.0,
        )
        self.assertGreater(len(chunks), 1)

    def test_tail_merge_uses_adaptive_threshold(self) -> None:
        # Crafted to create a tiny tail chunk that should be merged.
        p1 = ("uno " * 210).strip()
        p2 = ("dos " * 210).strip()
        p3 = ("cola " * 20).strip()
        source = "\n\n".join([p1, p2, p3])
        chunks = split_source_chunks(
            source,
            target_minutes=5.0,
            chunk_target_minutes=2.5,
            words_per_min=130.0,
        )
        self.assertEqual(len(chunks), 2)
        self.assertGreaterEqual(len(chunks[-1].split()), 40)

    def test_does_not_over_merge_below_desired_chunks_when_source_is_large(self) -> None:
        source = "\n\n".join([("bloque " * 120).strip() for _ in range(24)])
        chunks = split_source_chunks(
            source,
            target_minutes=15.0,
            chunk_target_minutes=2.5,  # desired ~= 6
            words_per_min=130.0,
        )
        self.assertGreaterEqual(len(chunks), 6)

    def test_tiny_middle_chunk_is_merged_when_above_desired_count(self) -> None:
        p1 = ("uno " * 120).strip()
        p2 = ("medio " * 10).strip()
        p3 = ("tres " * 120).strip()
        source = "\n\n".join([p1, p2, p3])
        chunks = split_source_chunks(
            source,
            target_minutes=5.0,  # desired ~= 2
            chunk_target_minutes=2.5,
            words_per_min=130.0,
        )
        self.assertEqual(len(chunks), 2)
        self.assertGreaterEqual(len(chunks[0].split()), 130)

    def test_tiny_tail_is_not_merged_when_at_desired_count(self) -> None:
        p1 = ("uno " * 120).strip()
        p2 = ("cola " * 10).strip()
        source = "\n\n".join([p1, p2])
        chunks = split_source_chunks(
            source,
            target_minutes=5.0,  # desired ~= 2
            chunk_target_minutes=2.5,
            words_per_min=130.0,
        )
        self.assertEqual(len(chunks), 2)
        self.assertLess(len(chunks[-1].split()), 40)

    def test_soft_cap_limits_chunk_count_to_desired_plus_one(self) -> None:
        source = "\n\n".join([("bloque " * 90).strip() for _ in range(40)])
        chunks = split_source_chunks(
            source,
            target_minutes=15.0,  # desired ~= 6
            chunk_target_minutes=2.5,
            words_per_min=130.0,
        )
        self.assertLessEqual(len(chunks), 7)


if __name__ == "__main__":
    unittest.main()

