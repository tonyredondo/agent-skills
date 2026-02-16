import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.script_postprocess import (  # noqa: E402
    dedupe_append,
    detect_truncation_indices,
    evaluate_script_completeness,
    ensure_en_resumen,
    ensure_farewell_last,
    fix_mid_farewells,
    harden_script_structure,
    normalize_block_numbering,
    normalize_speaker_turns,
    repair_script_completeness,
    sanitize_abrupt_tail,
)


class ScriptPostprocessTests(unittest.TestCase):
    def test_fix_mid_farewell_replaces_middle(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Hola."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Gracias por escucharnos, adios."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Seguimos."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Hasta la proxima."},
        ]
        out = fix_mid_farewells(lines)
        self.assertNotIn("adios", out[1]["text"].lower())
        self.assertIn("hasta la proxima", out[-1]["text"].lower())

    def test_fix_mid_farewell_does_not_replace_generic_gracias_phrase(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Gracias por la aclaracion, sigamos con el analisis."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Seguimos con evidencias y contexto tecnico."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Cierre con recomendaciones practicas."},
        ]
        out = fix_mid_farewells(lines)
        self.assertEqual(out[0]["text"], lines[0]["text"])

    def test_fix_mid_farewell_keeps_language_for_english_scripts(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Thanks for listening everyone."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "We continue with practical evidence and context."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Final takeaway for next episode."},
        ]
        out = fix_mid_farewells(lines)
        self.assertNotIn("Sigamos con el siguiente punto", out[0]["text"])
        self.assertIn("continue with the next point", out[0]["text"].lower())

    def test_ensure_summary_added(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Punto clave uno."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Punto clave dos."},
        ]
        out = ensure_en_resumen(lines)
        self.assertTrue(any("en resumen" in l["text"].lower() for l in out))

    def test_ensure_summary_does_not_duplicate_when_other_language_marker_exists(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Core context for decisions."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "In summary, rollout should be gradual and measurable."},
        ]
        out = ensure_en_resumen(lines)
        self.assertEqual(len(out), len(lines))

    def test_ensure_summary_existing_marker_returns_detached_list(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Core context for decisions."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "In summary, rollout should be gradual and measurable."},
        ]
        out = ensure_en_resumen(lines)
        self.assertIsNot(out, lines)
        out[0]["text"] = "mutated copy"
        self.assertEqual(lines[0]["text"], "Core context for decisions.")

    def test_ensure_summary_added_uses_english_template_when_detected(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Core context for decisions and tradeoffs."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "We connect latency, cost and reliability implications."},
        ]
        out = ensure_en_resumen(lines)
        self.assertIn("In summary", out[-1]["text"])

    def test_dedupe_append(self) -> None:
        base = [{"speaker": "A", "role": "Host1", "instructions": "x", "text": "Hola"}]
        new = [
            {"speaker": "A", "role": "Host1", "instructions": "x", "text": "Hola"},
            {"speaker": "B", "role": "Host2", "instructions": "x", "text": "Que tal"},
        ]
        merged, added = dedupe_append(base, new)
        self.assertEqual(added, 1)
        self.assertEqual(len(merged), 2)

    def test_dedupe_append_keeps_lines_with_same_text_but_different_instructions(self) -> None:
        base = [
            {
                "speaker": "A",
                "role": "Host1",
                "instructions": "Voice Affect: Warm and confident | Tone: Conversational",
                "text": "Hola",
            }
        ]
        new = [
            {
                "speaker": "A",
                "role": "Host1",
                "instructions": "Voice Affect: Bright and friendly | Tone: Conversational",
                "text": "Hola",
            }
        ]
        merged, added = dedupe_append(base, new)
        self.assertEqual(added, 1)
        self.assertEqual(len(merged), 2)

    def test_summary_and_farewell_helpers_handle_none_text(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": None},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Tema principal."},
        ]
        out = ensure_en_resumen(lines)
        out = ensure_farewell_last(out)
        self.assertTrue(any("en resumen" in (line.get("text") or "").lower() for line in out))

    def test_ensure_farewell_last_uses_english_template_when_detected(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Today we reviewed practical architecture decisions."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "We closed with rollout and observability guidance."},
        ]
        out = ensure_farewell_last(lines)
        self.assertIn("Thank you for listening", out[-1]["text"])

    def test_ensure_summary_adds_near_end_when_only_early_marker_exists(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "En resumen, abrimos con contexto general."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Ahora pasamos a detalles operativos por fase."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Sumamos criterios de validacion y riesgos."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Cerramos con acciones concretas para esta semana."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Se listan responsables y metricas minimas."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Ajustamos segun retrabajo y calidad percibida."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Ultimo bloque con recomendaciones de adopcion."},
        ]
        out = ensure_en_resumen(lines)
        self.assertTrue(any("en resumen" in (line.get("text") or "").lower() for line in out[-3:]))
        self.assertGreaterEqual(len(out), len(lines) + 1)

    def test_ensure_farewell_last_repairs_truncated_farewell_tail(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "En resumen, repasamos los puntos principales."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Gracias por escucharnos, soy Lucia y nos vemos en la pro"},
        ]
        out = ensure_farewell_last(lines)
        self.assertIn("gracias por escuch", out[-1]["text"].lower())
        self.assertTrue(out[-1]["text"].strip().endswith("."))

    def test_ensure_summary_non_latin_defaults_to_english_template(self) -> None:
        lines = [
            {"speaker": "Aoi", "role": "Host1", "instructions": "x", "text": "これは番組の主要なポイントです。"},
            {"speaker": "Kenji", "role": "Host2", "instructions": "x", "text": "次に実装時の注意点を確認します。"},
        ]
        out = ensure_en_resumen(lines)
        self.assertIn("In summary", out[-1]["text"])

    def test_ensure_farewell_non_latin_defaults_to_english_template(self) -> None:
        lines = [
            {"speaker": "Aoi", "role": "Host1", "instructions": "x", "text": "これは番組の要点です。"},
            {"speaker": "Kenji", "role": "Host2", "instructions": "x", "text": "次に運用上の注意点を整理します。"},
        ]
        out = ensure_farewell_last(lines)
        self.assertIn("Thank you for listening", out[-1]["text"])

    def test_normalize_block_numbering_fixes_monotonic_gaps(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Entramos al Bloque 1 con contexto."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Seguimos con Bloque 2 y detalles."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Ahora Bloque 4 con ejemplos."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Cerramos Bloque 6 con recomendaciones."},
        ]
        out = normalize_block_numbering(lines)
        self.assertIn("Bloque 1", out[0]["text"])
        self.assertIn("Bloque 2", out[1]["text"])
        self.assertIn("Bloque 3", out[2]["text"])
        self.assertIn("Bloque 4", out[3]["text"])

    def test_normalize_block_numbering_fixes_non_monotonic_sequence(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Arrancamos en Bloque 1 con introduccion."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Profundizamos en Bloque 3 con contexto."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Aterrizamos en Bloque 2 con decisiones."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Cierre en Bloque 5 con resumen."},
        ]
        out = normalize_block_numbering(lines)
        self.assertIn("Bloque 1", out[0]["text"])
        self.assertIn("Bloque 2", out[1]["text"])
        self.assertIn("Bloque 3", out[2]["text"])
        self.assertIn("Bloque 4", out[3]["text"])

    def test_normalize_block_numbering_rebases_sequences_not_starting_at_one(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Arrancamos en Bloque 2 con introduccion."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Seguimos en Bloque 4 con desarrollo."},
        ]
        out = normalize_block_numbering(lines)
        self.assertIn("Bloque 1", out[0]["text"])
        self.assertIn("Bloque 2", out[1]["text"])

    def test_normalize_block_numbering_supports_english_markers(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "We start in Block 1 with context."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "We continue in Block 3 with details."},
        ]
        out = normalize_block_numbering(lines)
        self.assertIn("Block 1", out[0]["text"])
        self.assertIn("Block 2", out[1]["text"])

    def test_sanitize_abrupt_tail_repairs_ellipsis_and_connector(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Punto previo con contexto estable."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Operamos con telemetria y..."},
        ]
        out = sanitize_abrupt_tail(lines, tail_window=2)
        self.assertTrue(out[-1]["text"].endswith("."))
        self.assertNotIn("...", out[-1]["text"])
        self.assertFalse(out[-1]["text"].lower().endswith(" y."))

    def test_harden_script_structure_applies_both_normalizers(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Bloque 1 con objetivos y contexto."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Bloque 2 con datos de soporte."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Bloque 4 con cierre parcial y..."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Mensaje final con despedida."},
        ]
        out = harden_script_structure(lines)
        self.assertIn("Bloque 3", out[2]["text"])
        self.assertNotIn("...", out[2]["text"])

    def test_harden_script_structure_normalizes_known_anglicism(self) -> None:
        lines = [
            {
                "speaker": "Laura",
                "role": "Host1",
                "instructions": "x",
                "text": "En este diseno anadimos un donor extra para estabilizar el catalizador.",
            },
            {
                "speaker": "Diego",
                "role": "Host2",
                "instructions": "x",
                "text": "Con eso reducimos degradacion y mantenemos rendimiento.",
            },
        ]
        out = harden_script_structure(lines)
        self.assertIn("donante adicional", out[0]["text"].lower())
        self.assertNotIn("donor extra", out[0]["text"].lower())

    def test_harden_script_structure_diversifies_repeated_spanish_openers(self) -> None:
        lines = [
            {"speaker": "Laura", "role": "Host1", "instructions": "x", "text": "Y abrimos con el contexto tecnico."},
            {"speaker": "Diego", "role": "Host2", "instructions": "x", "text": "Y seguimos con un ejemplo practico."},
            {"speaker": "Laura", "role": "Host1", "instructions": "x", "text": "Y cerramos con una decision operativa."},
        ]
        out = harden_script_structure(lines)
        self.assertTrue(out[0]["text"].lower().startswith("y "))
        self.assertFalse(out[1]["text"].lower().startswith("y "))
        self.assertNotEqual(out[1]["text"], lines[1]["text"])

    def test_normalize_speaker_turns_limits_consecutive_run(self) -> None:
        lines = [
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Linea 1"},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Linea 2"},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Linea 3"},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "Linea 4"},
        ]
        out = normalize_speaker_turns(lines, max_consecutive_same_speaker=2)
        run = 1
        max_run = 1
        for idx in range(1, len(out)):
            if out[idx]["speaker"] == out[idx - 1]["speaker"]:
                run += 1
            else:
                run = 1
            max_run = max(max_run, run)
        self.assertLessEqual(max_run, 2)

    def test_detect_truncation_indices_finds_internal_and_tail_segments(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Bloque 1 con contexto y"},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Bloque 2 con detalle estable."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Cierre operativo..."},
        ]
        indices = detect_truncation_indices(lines)
        self.assertEqual(indices, [0, 2])

    def test_detect_truncation_indices_supports_non_spanish_connectors(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Block transition and"},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Stable sentence that should pass."},
        ]
        indices = detect_truncation_indices(lines)
        self.assertEqual(indices, [0])

    def test_detect_truncation_indices_flags_final_line_without_terminal_punctuation(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Contexto completo y recomendaciones concretas."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Cierre parcial con despedida natural pero incompleta"},
        ]
        indices = detect_truncation_indices(lines)
        self.assertEqual(indices, [1])

    def test_detect_truncation_indices_ignores_mid_sentence_ellipsis(self) -> None:
        lines = [
            {
                "speaker": "Carlos",
                "role": "Host1",
                "instructions": "x",
                "text": "Explicamos riesgos... y luego cerramos con una recomendacion completa.",
            },
            {
                "speaker": "Lucia",
                "role": "Host2",
                "instructions": "x",
                "text": "Cierre estable y completo para el episodio.",
            },
        ]
        indices = detect_truncation_indices(lines)
        self.assertEqual(indices, [])

    def test_evaluate_script_completeness_allows_internal_ellipsis(self) -> None:
        lines = [
            {
                "speaker": "Carlos",
                "role": "Host1",
                "instructions": "x",
                "text": "Bloque 1 con contexto... pero cerrando la idea con una frase completa.",
            },
            {
                "speaker": "Lucia",
                "role": "Host2",
                "instructions": "x",
                "text": "Bloque 2 con desarrollo completo y despedida final.",
            },
        ]
        report = evaluate_script_completeness(lines)
        self.assertTrue(bool(report.get("pass", False)))
        self.assertNotIn("script_contains_truncated_segments", list(report.get("reasons", [])))

    def test_evaluate_script_completeness_flags_two_block_gap(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Bloque 1 con contexto principal."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Bloque 3 con cierre operativo."},
        ]
        report = evaluate_script_completeness(lines)
        self.assertFalse(bool(report.get("pass", True)))
        self.assertIn("block_numbering_not_sequential", list(report.get("reasons", [])))

    def test_evaluate_script_completeness_flags_tail_truncation_without_block_markers(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Contexto completo con conclusion estable."},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Transicion con detalle y"},
        ]
        report = evaluate_script_completeness(lines)
        self.assertFalse(bool(report.get("pass", True)))
        self.assertIn("script_contains_truncated_segments", list(report.get("reasons", [])))

    def test_repair_script_completeness_fixes_numbering_and_truncation(self) -> None:
        lines = [
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Bloque 1 con contexto y"},
            {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Bloque 2 con datos utiles."},
            {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Bloque 4 con recomendaciones..."},
        ]
        repaired = repair_script_completeness(lines)
        self.assertIn("Bloque 3", repaired[-1]["text"])
        self.assertFalse(repaired[0]["text"].lower().endswith(" y"))
        report = evaluate_script_completeness(repaired)
        self.assertTrue(bool(report.get("pass", False)))
        self.assertEqual(list(report.get("reasons", [])), [])


if __name__ == "__main__":
    unittest.main()

