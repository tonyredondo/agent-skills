import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.tts_expressiveness import build_provider_instructions  # noqa: E402


class TTSExpressivenessTests(unittest.TestCase):
    def test_openai_render_keeps_structured_fields(self) -> None:
        rendered = build_provider_instructions(
            provider="openai",
            role="Host1",
            phase="body",
            raw_instructions="Tone: Formal | Pacing: Lento | Emotion: Curiosidad",
        )
        self.assertIn("Voice Affect:", rendered)
        self.assertIn("Tone: Formal", rendered)
        self.assertIn("Pacing: Lento", rendered)
        self.assertIn("Emotion: Curiosidad", rendered)

    def test_alibaba_render_includes_spanish_spain_directive(self) -> None:
        rendered = build_provider_instructions(
            provider="alibaba",
            role="Host2",
            phase="intro",
            raw_instructions="Tono: Conversacional | Emotion: Entusiasmo",
        )
        self.assertIn("Spanish (Spain) accent", rendered)
        self.assertIn("high enthusiasm", rendered)
        self.assertIn("natural prosody", rendered)

    def test_alibaba_render_normalizes_mixed_es_en_values(self) -> None:
        rendered = build_provider_instructions(
            provider="alibaba",
            role="Host1",
            phase="closing",
            raw_instructions="Pacing: pausado | Emotion: gratitud | Pronunciation: clara",
        )
        self.assertIn("Pacing: measured", rendered)
        self.assertIn("Emotion: gratitude", rendered)
        self.assertIn("Pronunciation: clear", rendered)


if __name__ == "__main__":
    unittest.main()
