import os
import sys
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.tts_synthesizer import split_text_for_tts, voice_for  # noqa: E402


class TTSSegmentationTests(unittest.TestCase):
    def test_split_text_respects_max_chars(self) -> None:
        text = (
            "Este es un texto bastante largo. " * 20
            + "Tambien incluye otra frase larga para forzar cortes. " * 10
        )
        parts = split_text_for_tts(text, max_chars=120)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 120 for p in parts))

    def test_split_text_prefers_secondary_punctuation_for_long_sentence(self) -> None:
        text = (
            "En este bloque vamos a cubrir tres ideas, primero contexto del problema, "
            "despues decisiones tecnicas clave; finalmente cerramos con riesgos y mitigaciones "
            "para que el equipo tenga una guia clara."
        )
        parts = split_text_for_tts(text, max_chars=95)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 95 for p in parts))
        self.assertTrue(parts[0].endswith(",") or parts[0].endswith(";") or parts[0].endswith("."))

    def test_split_text_handles_non_positive_max_chars(self) -> None:
        parts = split_text_for_tts("abc", max_chars=0)
        self.assertEqual(parts, ["a", "b", "c"])

    def test_voice_mapping(self) -> None:
        self.assertEqual(voice_for("Host1"), "cedar")
        self.assertEqual(voice_for("Host2"), "marin")
        self.assertEqual(voice_for("Otro"), "cedar")

    def test_voice_mapping_uses_speaker_gender_hints_in_auto_mode(self) -> None:
        with mock.patch.dict(os.environ, {"TTS_VOICE_ASSIGNMENT_MODE": "auto"}, clear=False):
            self.assertEqual(voice_for("Host1", speaker_name="Laura Martinez"), "marin")
            self.assertEqual(voice_for("Host2", speaker_name="Diego Herrera"), "cedar")

    def test_voice_mapping_role_mode_ignores_speaker_name_hints(self) -> None:
        with mock.patch.dict(os.environ, {"TTS_VOICE_ASSIGNMENT_MODE": "role"}, clear=False):
            self.assertEqual(voice_for("Host1", speaker_name="Laura Martinez"), "cedar")
            self.assertEqual(voice_for("Host2", speaker_name="Diego Herrera"), "marin")

    def test_voice_mapping_alibaba_provider_uses_provider_specific_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TTS_ALIBABA_MALE_VOICE": "Ethan",
                "TTS_ALIBABA_FEMALE_VOICE": "Cherry",
                "TTS_ALIBABA_DEFAULT_VOICE": "Cherry",
            },
            clear=False,
        ):
            self.assertEqual(voice_for("Host1", provider="alibaba"), "Ethan")
            self.assertEqual(voice_for("Host2", provider="alibaba"), "Cherry")

    def test_voice_mapping_alibaba_provider_gender_hints(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TTS_VOICE_ASSIGNMENT_MODE": "auto",
                "TTS_ALIBABA_MALE_VOICE": "Ethan",
                "TTS_ALIBABA_FEMALE_VOICE": "Cherry",
                "TTS_ALIBABA_DEFAULT_VOICE": "Cherry",
            },
            clear=False,
        ):
            self.assertEqual(voice_for("Host1", provider="alibaba", speaker_name="Laura Martinez"), "Cherry")
            self.assertEqual(voice_for("Host2", provider="alibaba", speaker_name="Diego Herrera"), "Ethan")


if __name__ == "__main__":
    unittest.main()

