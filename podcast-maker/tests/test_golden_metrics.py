import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.golden_metrics import compare_against_baseline, compute_script_metrics  # noqa: E402


def _payload(lines):
    return {'lines': lines}


def _line(speaker: str, role: str, text: str) -> dict[str, str]:
    return {
        'speaker': speaker,
        'role': role,
        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
        'pace_hint': 'steady',
        'text': text,
    }


class GoldenMetricsTests(unittest.TestCase):
    def test_metrics_presence(self) -> None:
        payload = _payload(
            [
                _line('Carlos', 'Host1', 'Abrimos con un problema concreto y por que importa hoy.'),
                _line('Lucia', 'Host2', 'Vale, bajemos eso a tierra: donde aparece primero el coste real.'),
                _line('Carlos', 'Host1', 'Si la regla oculta criterio, el equipo pierde tiempo interpretando excepciones.'),
                _line('Lucia', 'Host2', 'Entonces la mejora buena es la que simplifica sin esconder el por que.'),
            ]
        )
        metrics = compute_script_metrics(payload)
        self.assertEqual(metrics.unique_speakers, 2)
        self.assertGreaterEqual(metrics.alternating_ratio, 0.99)
        self.assertGreaterEqual(metrics.host2_turn_ratio, 0.4)
        self.assertGreaterEqual(metrics.host2_push_ratio, 0.5)
        self.assertEqual(metrics.scaffold_phrase_hits, 0)
        self.assertTrue(metrics.meta_language_ok)

    def test_compare_against_baseline(self) -> None:
        baseline = compute_script_metrics(
            _payload(
                [
                    _line('A', 'Host1', 'Abrimos con el problema y el contexto operativo.'),
                    _line('B', 'Host2', 'Vale, aterrizalo con un ejemplo practico.'),
                    _line('A', 'Host1', 'El ahorro existe cuando se elimina repeticion visible.'),
                    _line('B', 'Host2', 'Y el riesgo aparece cuando nadie entiende la excepcion.'),
                ]
            )
        )
        current = compute_script_metrics(
            _payload(
                [
                    _line('A', 'Host1', 'Abrimos con el mismo problema y un poco mas de detalle concreto.'),
                    _line('B', 'Host2', 'Vale, aterrizalo: donde se ahorra tiempo y donde se complica la operacion.'),
                    _line('A', 'Host1', 'El beneficio real llega cuando el equipo sigue viendo el criterio.'),
                    _line('B', 'Host2', 'Entonces la automatizacion buena no tapa el juicio operativo.'),
                ]
            )
        )
        cmp = compare_against_baseline(current, baseline)
        self.assertTrue(cmp['word_ratio_ok'])
        self.assertTrue(cmp['line_ratio_ok'])
        self.assertTrue(cmp['speaker_count_ok'])
        self.assertTrue(cmp['alternation_ok'])
        self.assertTrue(cmp['host2_presence_ok'])
        self.assertTrue(cmp['host2_push_ok'])
        self.assertTrue(cmp['scaffold_control_ok'])
        self.assertTrue(cmp['meta_language_ok'])

    def test_compare_fails_when_word_ratio_too_low(self) -> None:
        baseline = compute_script_metrics(
            _payload(
                [
                    _line('A', 'Host1', 'uno dos tres cuatro cinco seis'),
                    _line('B', 'Host2', 'vale aterrizalo con un ejemplo y un coste concreto'),
                    _line('A', 'Host1', 'siete ocho nueve diez once doce'),
                    _line('B', 'Host2', 'entonces la regla buena deja claro el criterio'),
                ]
            )
        )
        current = compute_script_metrics(
            _payload(
                [
                    _line('A', 'Host1', 'uno dos'),
                    _line('B', 'Host2', 'vale?'),
                ]
            )
        )
        cmp = compare_against_baseline(current, baseline)
        self.assertFalse(cmp['word_ratio_ok'])

    def test_compare_fails_when_host2_presence_is_lost(self) -> None:
        baseline = compute_script_metrics(
            _payload(
                [
                    _line('A', 'Host1', 'contexto operativo suficiente para arrancar.'),
                    _line('B', 'Host2', 'aterrizalo con un ejemplo y un coste.'),
                    _line('A', 'Host1', 'respuesta con tradeoff concreto.'),
                    _line('B', 'Host2', 'cierro con la consecuencia practica.'),
                ]
            )
        )
        current = compute_script_metrics(
            _payload(
                [
                    _line('A', 'Host1', 'contexto operativo suficiente para arrancar.'),
                    _line('A', 'Host1', 'respuesta larga que absorbe la conversacion.'),
                    _line('A', 'Host1', 'otra respuesta que deja fuera al segundo host.'),
                ]
            )
        )
        cmp = compare_against_baseline(current, baseline)
        self.assertFalse(cmp['host2_presence_ok'])

    def test_compare_requires_host2_push_floor_even_when_baseline_is_zero(self) -> None:
        baseline = compute_script_metrics(
            _payload(
                [
                    _line('A', 'Host1', 'Abrimos con un contexto corto y practico.'),
                    _line('B', 'Host2', 'Seguimos con una descripcion neutra del caso.'),
                    _line('A', 'Host1', 'La pieza importante es que el proceso sea legible.'),
                    _line('B', 'Host2', 'Cierro con una observacion amable pero plana.'),
                ]
            )
        )
        current = compute_script_metrics(
            _payload(
                [
                    _line('A', 'Host1', 'Abrimos con el mismo contexto corto y practico.'),
                    _line('B', 'Host2', 'Seguimos con otra descripcion neutra del caso.'),
                    _line('A', 'Host1', 'La pieza importante sigue siendo la legibilidad operativa.'),
                    _line('B', 'Host2', 'Cierro con una observacion amable y todavia plana.'),
                ]
            )
        )
        cmp = compare_against_baseline(current, baseline)
        self.assertFalse(cmp['host2_push_ok'])

    def test_compare_fails_when_meta_language_present(self) -> None:
        baseline = compute_script_metrics(
            _payload(
                [
                    _line('A', 'Host1', 'abrimos con contexto operativo claro.'),
                    _line('B', 'Host2', 'aterrizalo con ejemplo y coste.'),
                ]
            )
        )
        current = compute_script_metrics(
            _payload(
                [
                    _line('A', 'Host1', 'segun el indice, ahora vamos al siguiente tramo del episodio.'),
                    _line('B', 'Host2', 'aterrizalo con ejemplo y coste.'),
                ]
            )
        )
        cmp = compare_against_baseline(current, baseline)
        self.assertFalse(cmp['meta_language_ok'])


if __name__ == '__main__':
    unittest.main()
