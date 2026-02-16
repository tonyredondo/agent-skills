import os
import random
import string
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.script_chunker import (  # noqa: E402
    _split_paragraph_safely,
    _split_very_long_sentence,
    split_source_chunks,
    target_chunk_count,
)


def _random_word(rng: random.Random) -> str:
    n = rng.randint(3, 10)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


class ScriptChunkerFuzzTests(unittest.TestCase):
    def test_fuzz_split_source_chunks_non_empty_and_word_preserving(self) -> None:
        rng = random.Random(12345)
        for _ in range(25):
            paragraphs = []
            paragraph_count = rng.randint(1, 5)
            for _p in range(paragraph_count):
                sentence_count = rng.randint(1, 8)
                sentences = []
                for _s in range(sentence_count):
                    words = [_random_word(rng) for _ in range(rng.randint(8, 40))]
                    sentences.append(" ".join(words) + rng.choice([".", "!", "?"]))
                paragraphs.append(" ".join(sentences))
            source = "\n\n".join(paragraphs)
            chunks = split_source_chunks(
                source,
                target_minutes=15.0,
                chunk_target_minutes=2.5,
                words_per_min=130.0,
            )
            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(all(c.strip() for c in chunks))
            src_words = len(source.split())
            chunk_words = sum(len(c.split()) for c in chunks)
            self.assertEqual(src_words, chunk_words)

    def test_split_very_long_sentence_respects_step(self) -> None:
        sentence = " ".join(f"w{i}" for i in range(500))
        pieces = _split_very_long_sentence(sentence, 120)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p.split()) <= 120 for p in pieces))

    def test_split_paragraph_safely_handles_long_sentence(self) -> None:
        paragraph = " ".join(f"w{i}" for i in range(350))
        pieces = _split_paragraph_safely(paragraph, 120)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p.split()) <= 120 for p in pieces))

    def test_target_chunk_count_monotonic(self) -> None:
        c1 = target_chunk_count(target_minutes=5.0, chunk_target_minutes=2.0)
        c2 = target_chunk_count(target_minutes=15.0, chunk_target_minutes=2.0)
        c3 = target_chunk_count(target_minutes=30.0, chunk_target_minutes=2.0)
        self.assertLessEqual(c1, c2)
        self.assertLessEqual(c2, c3)

    def test_target_chunk_count_minimum_is_one(self) -> None:
        # Internal guards floor target_minutes to 1.0 and chunk_target_minutes to 0.8.
        self.assertEqual(target_chunk_count(target_minutes=0.1, chunk_target_minutes=0.1), 2)


if __name__ == "__main__":
    unittest.main()

