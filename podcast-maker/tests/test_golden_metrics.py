import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.golden_metrics import compare_against_baseline, compute_script_metrics  # noqa: E402


def _payload(lines):
    return {"lines": lines}


class GoldenMetricsTests(unittest.TestCase):
    def test_metrics_presence(self) -> None:
        payload = _payload(
            [
                {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Hola y bienvenidos."},
                {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "En Resumen: ideas clave."},
                {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Gracias por escucharnos."},
            ]
        )
        metrics = compute_script_metrics(payload)
        self.assertTrue(metrics.has_recap_signal)
        self.assertTrue(metrics.farewell_in_last_3)
        self.assertTrue(metrics.meta_language_ok)
        self.assertEqual(metrics.unique_speakers, 2)

    def test_compare_against_baseline(self) -> None:
        baseline = compute_script_metrics(
            _payload(
                [
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "uno dos tres cuatro"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "cinco seis siete ocho"},
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "En Resumen: nueve diez"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "Gracias por escucharnos."},
                ]
            )
        )
        current = compute_script_metrics(
            _payload(
                [
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "uno dos tres cuatro"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "cinco seis siete ocho"},
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "En Resumen: nueve diez once"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "Gracias por escucharnos."},
                ]
            )
        )
        cmp = compare_against_baseline(current, baseline)
        self.assertTrue(cmp["word_ratio_ok"])
        self.assertTrue(cmp["line_ratio_ok"])
        self.assertTrue(cmp["recap_ok"])
        self.assertTrue(cmp["farewell_ok"])
        self.assertTrue(cmp["meta_language_ok"])

    def test_compare_fails_when_word_ratio_too_low(self) -> None:
        baseline = compute_script_metrics(
            _payload(
                [
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "uno dos tres cuatro cinco seis"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "siete ocho nueve diez once doce"},
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "En Resumen: trece catorce"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "Gracias por escucharnos."},
                ]
            )
        )
        current = compute_script_metrics(
            _payload(
                [
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "uno dos"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "En Resumen: tres"},
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "Gracias por escucharnos."},
                ]
            )
        )
        cmp = compare_against_baseline(current, baseline)
        self.assertFalse(cmp["word_ratio_ok"])

    def test_compare_fails_when_summary_missing(self) -> None:
        baseline = compute_script_metrics(
            _payload(
                [
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "uno dos tres cuatro"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "En Resumen: cinco seis"},
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "Gracias por escucharnos."},
                ]
            )
        )
        current = compute_script_metrics(
            _payload(
                [
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "uno dos tres cuatro"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "cinco seis siete ocho"},
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "Gracias por escucharnos."},
                ]
            )
        )
        cmp = compare_against_baseline(current, baseline)
        self.assertFalse(cmp["recap_ok"])

    def test_compare_fails_when_farewell_missing(self) -> None:
        baseline = compute_script_metrics(
            _payload(
                [
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "uno dos tres cuatro"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "En Resumen: cinco seis"},
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "Gracias por escucharnos."},
                ]
            )
        )
        current = compute_script_metrics(
            _payload(
                [
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "uno dos tres cuatro"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "En Resumen: cinco seis"},
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "cerramos sin despedida final"},
                ]
            )
        )
        cmp = compare_against_baseline(current, baseline)
        self.assertFalse(cmp["farewell_ok"])

    def test_compare_fails_when_meta_language_present(self) -> None:
        baseline = compute_script_metrics(
            _payload(
                [
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "uno dos tres cuatro"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "En Resumen: cinco seis"},
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "Gracias por escucharnos."},
                ]
            )
        )
        current = compute_script_metrics(
            _payload(
                [
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "segun el indice, vamos al siguiente tramo"},
                    {"speaker": "B", "role": "Host2", "instructions": "x", "text": "En Resumen: cinco seis"},
                    {"speaker": "A", "role": "Host1", "instructions": "x", "text": "Gracias por escucharnos."},
                ]
            )
        )
        cmp = compare_against_baseline(current, baseline)
        self.assertFalse(cmp["meta_language_ok"])


if __name__ == "__main__":
    unittest.main()

