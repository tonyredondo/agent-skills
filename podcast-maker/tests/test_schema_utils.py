import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.schema import (  # noqa: E402
    canonical_json,
    content_hash,
    count_words_from_payload,
    dedupe_key,
    load_json_text,
    normalize_line,
    salvage_script_payload,
    validate_script_payload,
)


class SchemaUtilsTests(unittest.TestCase):
    def test_load_json_text_accepts_object(self) -> None:
        payload = load_json_text('{"lines":[{"speaker":"A","role":"Host1","instructions":"x","text":"hola"}]}')
        self.assertIn("lines", payload)

    def test_load_json_text_rejects_non_object_root(self) -> None:
        with self.assertRaises(ValueError):
            load_json_text('["a", "b"]')

    def test_normalize_line_invalid_role_is_auto_fixed(self) -> None:
        line0 = normalize_line({"speaker": "Ana", "role": "Guest", "instructions": "x", "text": "hola"}, 0)
        line1 = normalize_line({"speaker": "Ana", "role": "Guest", "instructions": "x", "text": "hola"}, 1)
        self.assertEqual(line0["role"], "Host1")
        self.assertEqual(line1["role"], "Host2")

    def test_normalize_line_missing_instructions_gets_default_openai_style(self) -> None:
        line = normalize_line({"speaker": "Ana", "role": "Host1", "text": "hola"}, 0)
        self.assertIn("Speak in a warm, confident, conversational tone.", line["instructions"])
        self.assertNotIn("\n", line["instructions"])
        self.assertNotIn("|", line["instructions"])

    def test_normalize_line_preserves_structured_instruction_template(self) -> None:
        line = normalize_line(
            {
                "speaker": "Ana",
                "role": "Host1",
                "instructions": "Voice Affect: Warm | Tone: Conversational",
                "text": "hola",
            },
            0,
        )
        self.assertIn("Voice Affect: Warm", line["instructions"])

    def test_normalize_line_falls_back_when_instruction_is_ambiguous(self) -> None:
        line = normalize_line(
            {
                "speaker": "Ana",
                "role": "Host1",
                "instructions": "Speak naturally.",
                "text": "hola",
            },
            0,
        )
        self.assertEqual(
            line["instructions"],
            "Speak in a warm, confident, conversational tone. Keep pacing measured and clear with brief pauses.",
        )

    def test_validate_script_payload_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            validate_script_payload({"lines": []})

    def test_validate_script_payload_rejects_non_object_line(self) -> None:
        with self.assertRaises(ValueError):
            validate_script_payload({"lines": ["bad"]})

    def test_count_words_from_payload_counts_expected_words(self) -> None:
        payload = {
            "lines": [
                {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "uno dos tres"},
                {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "cuatro cinco"},
            ]
        }
        self.assertEqual(count_words_from_payload(payload), 5)

    def test_dedupe_key_is_case_insensitive_for_text(self) -> None:
        a = {"speaker": "Ana", "role": "Host1", "text": "Hola Mundo"}
        b = {"speaker": "Ana", "role": "Host1", "text": "hola mundo"}
        self.assertEqual(dedupe_key(a), dedupe_key(b))

    def test_canonical_json_is_compact(self) -> None:
        compact = canonical_json({"a": 1, "b": {"c": 2}})
        self.assertEqual(compact, '{"a":1,"b":{"c":2}}')

    def test_content_hash_is_stable(self) -> None:
        value = "mismo texto"
        self.assertEqual(content_hash(value), content_hash(value))

    def test_salvage_script_payload_recovers_legacy_line_fields(self) -> None:
        payload = {
            "lines": [
                {"name": "Ana", "content": "Primera linea recuperada."},
                {"name": "Luis", "dialogue": "Segunda linea recuperada."},
            ]
        }
        salvaged = salvage_script_payload(payload)
        validated = validate_script_payload(salvaged)
        self.assertEqual(len(validated["lines"]), 2)
        self.assertEqual(validated["lines"][0]["role"], "Host1")
        self.assertEqual(validated["lines"][1]["role"], "Host2")
        self.assertIn("recuperada", validated["lines"][0]["text"].lower())

    def test_salvage_script_payload_recovers_nested_script_dialogue(self) -> None:
        payload = {
            "script": {
                "dialogue": [
                    {"speaker_name": "Ana", "message": "Linea uno estable."},
                    {"speaker_name": "Luis", "message": "Linea dos estable."},
                ]
            }
        }
        salvaged = salvage_script_payload(payload)
        self.assertEqual(len(salvaged["lines"]), 2)
        self.assertEqual(salvaged["lines"][0]["speaker"], "Ana")
        self.assertEqual(salvaged["lines"][1]["speaker"], "Luis")

    def test_salvage_script_payload_deduplicates_repeated_candidates(self) -> None:
        payload = {
            "lines": [
                {"name": "Ana", "content": "Linea estable repetida."},
                {"name": "Luis", "content": "Linea unica."},
            ],
            "script": {
                "lines": [
                    {"name": "Ana", "content": "Linea estable repetida."},
                ]
            },
        }
        salvaged = salvage_script_payload(payload)
        self.assertEqual(len(salvaged["lines"]), 2)
        texts = [line.get("text", "") for line in salvaged["lines"]]
        self.assertEqual(texts.count("Linea estable repetida."), 1)

    def test_salvage_script_payload_recovers_result_wrapper(self) -> None:
        payload = {
            "result": {
                "lines": [
                    {"name": "Ana", "content": "Linea desde wrapper result."},
                    {"name": "Luis", "content": "Otra linea desde result."},
                ]
            }
        }
        salvaged = salvage_script_payload(payload)
        self.assertEqual(len(salvaged["lines"]), 2)
        self.assertEqual(salvaged["lines"][0]["speaker"], "Ana")
        self.assertEqual(salvaged["lines"][1]["speaker"], "Luis")


if __name__ == "__main__":
    unittest.main()

