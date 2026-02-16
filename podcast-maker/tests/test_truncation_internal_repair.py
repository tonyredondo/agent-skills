import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.script_postprocess import (  # noqa: E402
    detect_truncation_indices,
    evaluate_script_completeness,
    repair_script_completeness,
)


class TruncationInternalRepairTests(unittest.TestCase):
    def test_internal_truncation_is_detected_before_tail(self) -> None:
        lines = [
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Bloque 1 con contexto y"},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "Bloque 2 con desarrollo completo."},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Bloque 4 con cierre..."},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "Gracias por escuchar."},
        ]
        indices = detect_truncation_indices(lines)
        self.assertEqual(indices, [0, 2])
        report = evaluate_script_completeness(lines)
        self.assertFalse(bool(report.get("pass", True)))
        reasons = list(report.get("reasons", []))
        self.assertIn("script_contains_truncated_segments", reasons)
        self.assertIn("block_numbering_not_sequential", reasons)

    def test_repair_script_completeness_fixes_internal_and_sequence_gaps(self) -> None:
        lines = [
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Bloque 1 con contexto y"},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "Bloque 2 con desarrollo completo."},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Bloque 4 con cierre..."},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "Despedida final estable."},
        ]
        repaired = repair_script_completeness(lines)
        self.assertNotIn("...", repaired[2]["text"])
        self.assertIn("Bloque 3", repaired[2]["text"])
        self.assertFalse(repaired[0]["text"].lower().endswith(" y"))
        report = evaluate_script_completeness(repaired)
        self.assertTrue(bool(report.get("pass", False)))

    def test_repair_script_completeness_fixes_non_monotonic_block_sequence(self) -> None:
        lines = [
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Bloque 1 con contexto estable."},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "Bloque 3 con desarrollo tecnico."},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Bloque 2 con decisiones practicas."},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "Bloque 5 con cierre final."},
        ]
        report_before = evaluate_script_completeness(lines)
        self.assertFalse(bool(report_before.get("pass", True)))
        self.assertIn("block_numbering_not_sequential", list(report_before.get("reasons", [])))
        repaired = repair_script_completeness(lines)
        self.assertIn("Bloque 1", repaired[0]["text"])
        self.assertIn("Bloque 2", repaired[1]["text"])
        self.assertIn("Bloque 3", repaired[2]["text"])
        self.assertIn("Bloque 4", repaired[3]["text"])
        report_after = evaluate_script_completeness(repaired)
        self.assertTrue(bool(report_after.get("pass", False)))

    def test_repair_script_completeness_rebases_blocks_when_first_is_not_one(self) -> None:
        lines = [
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Bloque 2 con contexto base."},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "Bloque 4 con desarrollo final."},
        ]
        report_before = evaluate_script_completeness(lines)
        self.assertFalse(bool(report_before.get("pass", True)))
        self.assertIn("block_numbering_not_sequential", list(report_before.get("reasons", [])))
        repaired = repair_script_completeness(lines)
        self.assertIn("Bloque 1", repaired[0]["text"])
        self.assertIn("Bloque 2", repaired[1]["text"])
        report_after = evaluate_script_completeness(repaired)
        self.assertTrue(bool(report_after.get("pass", False)))


if __name__ == "__main__":
    unittest.main()

