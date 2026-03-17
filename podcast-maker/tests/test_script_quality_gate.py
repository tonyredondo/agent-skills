import dataclasses
import json
import os
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import ScriptConfig  # noqa: E402
from pipeline.errors import ERROR_KIND_SCRIPT_QUALITY  # noqa: E402
from pipeline.script_quality_gate import (  # noqa: E402
    ScriptQualityGateConfig,
    evaluate_script_quality,
    write_quality_report,
)


class _FakeEditorialClient:
    def __init__(self, *, scores=None) -> None:
        self.scores = scores or {
            'orality': 4.2,
            'host_distinction': 4.1,
            'progression': 4.0,
            'freshness': 4.0,
            'listener_engagement': 4.0,
            'density_control': 4.0,
        }
        self.requests_made = 0
        self.script_retries_total = 0
        self.script_json_parse_failures = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001, ARG002
        self.requests_made += 1
        if stage != 'editorial_gate_eval':
            raise AssertionError(f'unexpected stage: {stage}')
        return {'scores': self.scores, 'reasons': []}


def _line(speaker: str, role: str, text: str) -> dict[str, str]:
    return {
        'speaker': speaker,
        'role': role,
        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
        'pace_hint': 'steady',
        'text': text,
    }


def _base_script_cfg() -> ScriptConfig:
    cfg = ScriptConfig.from_env(profile_name='short')
    return dataclasses.replace(cfg, min_words=40, max_words=140, profile_name='short')


def _good_lines() -> list[dict[str, str]]:
    return [
        _line('Ana', 'Host1', 'Hoy vamos a una idea concreta: por que algunas automatizaciones ahorran tiempo y otras anaden friccion.'),
        _line('Luis', 'Host2', 'Vale, bajemos eso a tierra: dame un ejemplo donde parezca una mejora y luego complique el trabajo real.'),
        _line('Ana', 'Host1', 'Un equipo puede automatizar aprobaciones simples y ganar foco, pero si mete demasiadas reglas oculta el criterio importante.'),
        _line('Luis', 'Host2', 'Y el coste cual es: mas excepciones, mas tickets raros y mas tiempo intentando entender por que una regla salto.'),
        _line('Ana', 'Host1', 'Por eso la decision buena no es automatizar mas, sino automatizar lo repetitivo y dejar visibles los casos limite.'),
        _line('Luis', 'Host2', 'Ese cierre me sirve: menos magia, mas contexto operativo y una automatizacion que no tape el juicio del equipo.'),
    ]


def _scaffolded_lines() -> list[dict[str, str]]:
    return [
        _line('Ana', 'Host1', 'Hoy revisamos un tema de producto y operaciones.'),
        _line('Luis', 'Host2', 'Por otro lado, podemos mirar el impacto en equipos pequenos.'),
        _line('Ana', 'Host1', 'Dicho esto, tambien aparece una tension entre velocidad y control.'),
        _line('Luis', 'Host2', 'En ese sentido, todo se resume en encontrar un equilibrio.'),
    ]


def _mechanical_openers_lines() -> list[dict[str, str]]:
    return [
        _line('Ana', 'Host1', 'Abrimos con una regla concreta: si el criterio no se ve, la automatizacion acaba generando mas friccion.'),
        _line('Luis', 'Host2', 'Exacto, pero aterrizalo: donde se nota ese coste en el trabajo real del equipo.'),
        _line('Ana', 'Host1', 'Claro, aparece cuando la regla parece ahorrar tiempo arriba y luego multiplica excepciones abajo.'),
        _line('Luis', 'Host2', 'Tal cual, y ahi el problema no es la idea sino esconder el criterio que decide cada caso.'),
    ]


class ScriptQualityGateTests(unittest.TestCase):
    def test_config_defaults_by_profile_still_load(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            short_cfg = ScriptQualityGateConfig.from_env(profile_name='short')
            standard_cfg = ScriptQualityGateConfig.from_env(profile_name='standard')
            long_cfg = ScriptQualityGateConfig.from_env(profile_name='long')
        self.assertEqual(short_cfg.action, 'warn')
        self.assertEqual(standard_cfg.evaluator, 'hybrid')
        self.assertEqual(long_cfg.llm_sample_rate, 1.0)

    def test_evaluate_quality_passes_for_new_oral_script(self) -> None:
        report = evaluate_script_quality(
            validated_payload={'lines': _good_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=ScriptQualityGateConfig.from_env(profile_name='short'),
            script_path='/tmp/oral_script.json',
            client=_FakeEditorialClient(),
        )
        self.assertTrue(report['pass'])
        self.assertIsNone(report['failure_kind'])
        self.assertTrue(report['rules']['alternating_roles'])
        self.assertGreaterEqual(report['scores']['editorial_scores']['orality'], 3.8)

    def test_evaluate_quality_fails_for_scaffolded_script(self) -> None:
        report = evaluate_script_quality(
            validated_payload={'lines': _scaffolded_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=ScriptQualityGateConfig.from_env(profile_name='short'),
            script_path='/tmp/scaffolded_script.json',
            client=_FakeEditorialClient(),
        )
        self.assertFalse(report['pass'])
        self.assertEqual(report['failure_kind'], ERROR_KIND_SCRIPT_QUALITY)
        self.assertIn('scaffold_phrase_repetition', set(report['reasons']))

    def test_evaluate_quality_fails_for_repeated_stock_openers(self) -> None:
        report = evaluate_script_quality(
            validated_payload={'lines': _mechanical_openers_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=ScriptQualityGateConfig.from_env(profile_name='short'),
            script_path='/tmp/mechanical_openers.json',
            client=_FakeEditorialClient(),
        )
        self.assertFalse(report['pass'])
        self.assertEqual(report['failure_kind'], ERROR_KIND_SCRIPT_QUALITY)
        self.assertIn('scaffold_phrase_repetition', set(report['reasons']))

    def test_evaluate_quality_fails_when_host2_does_not_push(self) -> None:
        lines = [
            _line('Ana', 'Host1', 'La tesis es simple y la voy a explicar varias veces.'),
            _line('Luis', 'Host2', 'Suena bien.'),
            _line('Ana', 'Host1', 'La tesis es simple y la voy a explicar otra vez con mas palabras.'),
            _line('Luis', 'Host2', 'Si, claro.'),
        ]
        report = evaluate_script_quality(
            validated_payload={'lines': lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=ScriptQualityGateConfig.from_env(profile_name='short'),
            script_path='/tmp/host2_flat.json',
            client=_FakeEditorialClient(),
        )
        self.assertFalse(report['pass'])
        self.assertIn('host2_not_pushing', set(report['reasons']))

    def test_write_quality_report_round_trips(self) -> None:
        report = evaluate_script_quality(
            validated_payload={'lines': _good_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=ScriptQualityGateConfig.from_env(profile_name='short'),
            script_path='/tmp/report.json',
            client=_FakeEditorialClient(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'quality_report.json')
            write_quality_report(path, report)
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
        self.assertEqual(loaded['pass'], report['pass'])
        self.assertEqual(loaded['component'], 'script_quality_gate')


if __name__ == '__main__':
    unittest.main()
