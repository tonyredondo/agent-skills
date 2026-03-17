import dataclasses
import os
import sys
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
)


class _MatrixClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001, ARG002
        self.calls += 1
        if stage != 'editorial_gate_eval':
            raise AssertionError(f'unexpected stage: {stage}')
        return {
            'scores': {
                'orality': 4.4,
                'host_distinction': 4.3,
                'progression': 4.2,
                'freshness': 4.1,
                'listener_engagement': 4.2,
                'density_control': 4.1,
            },
            'reasons': [],
        }


def _script_cfg(profile: str) -> ScriptConfig:
    cfg = ScriptConfig.from_env(profile_name=profile)
    return dataclasses.replace(cfg, min_words=20, max_words=400, profile_name=profile)


def _line(speaker: str, role: str, text: str) -> dict[str, str]:
    return {
        'speaker': speaker,
        'role': role,
        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
        'pace_hint': 'steady',
        'text': text,
    }


GOOD_LINES = [
    _line('Ana', 'Host1', 'Arrancamos con una duda util: cuando una automatizacion ahorra tiempo y cuando solo desplaza el problema.'),
    _line('Luis', 'Host2', 'Vale, aterrizalo: donde se nota el beneficio y donde aparece el coste oculto en la operacion diaria.'),
    _line('Ana', 'Host1', 'Se nota cuando desaparece repeticion visible, pero empeora si nadie entiende por que saltan las excepciones.'),
    _line('Luis', 'Host2', 'Entonces el criterio es simple: menos magia y mas contexto para leer cada decision.'),
]

BAD_LINES = [
    _line('Ana', 'Host1', 'repetido repetido repetido'),
    _line('Ana', 'Host1', 'repetido repetido repetido'),
    _line('Ana', 'Host1', 'repetido repetido repetido'),
    _line('Ana', 'Host1', 'repetido repetido repetido'),
]


class QualityGateRegressionMatrixTests(unittest.TestCase):
    def test_good_script_matrix_profiles_actions_evaluators(self) -> None:
        for profile in ('short', 'standard', 'long'):
            for action in ('off', 'warn', 'enforce'):
                for evaluator in ('rules', 'hybrid', 'llm'):
                    with self.subTest(profile=profile, action=action, evaluator=evaluator):
                        env = {
                            'SCRIPT_QUALITY_GATE_ACTION': action,
                            'SCRIPT_QUALITY_GATE_EVALUATOR': evaluator,
                            'SCRIPT_QUALITY_GATE_LLM_SAMPLE': '1.0',
                            'SCRIPT_QUALITY_MIN_WORDS_RATIO': '0.1',
                            'SCRIPT_QUALITY_MAX_WORDS_RATIO': '4.0',
                        }
                        with mock.patch.dict(os.environ, env, clear=True):
                            cfg = ScriptQualityGateConfig.from_env(profile_name=profile)
                        client = _MatrixClient()
                        report = evaluate_script_quality(
                            validated_payload={'lines': GOOD_LINES},
                            script_cfg=_script_cfg(profile),
                            quality_cfg=cfg,
                            script_path=f'/tmp/{profile}.json',
                            client=client,
                        )
                        self.assertTrue(report['pass'])
                        if evaluator == 'rules':
                            self.assertFalse(report['llm_called'])
                            self.assertEqual(client.calls, 0)
                        else:
                            self.assertTrue(report['llm_called'])
                            self.assertEqual(client.calls, 1)

    def test_bad_script_matrix_reports_rejection(self) -> None:
        for profile in ('short', 'standard', 'long'):
            for evaluator in ('rules', 'hybrid', 'llm'):
                with self.subTest(profile=profile, evaluator=evaluator):
                    env = {
                        'SCRIPT_QUALITY_GATE_ACTION': 'enforce',
                        'SCRIPT_QUALITY_GATE_EVALUATOR': evaluator,
                        'SCRIPT_QUALITY_GATE_LLM_SAMPLE': '1.0',
                        'SCRIPT_QUALITY_MIN_WORDS_RATIO': '0.0',
                        'SCRIPT_QUALITY_MAX_WORDS_RATIO': '10.0',
                    }
                    with mock.patch.dict(os.environ, env, clear=True):
                        cfg = ScriptQualityGateConfig.from_env(profile_name=profile)
                    client = _MatrixClient()
                    report = evaluate_script_quality(
                        validated_payload={'lines': BAD_LINES},
                        script_cfg=_script_cfg(profile),
                        quality_cfg=cfg,
                        script_path=f'/tmp/{profile}.json',
                        client=client,
                    )
                    self.assertFalse(report['pass'])
                    self.assertEqual(report['failure_kind'], ERROR_KIND_SCRIPT_QUALITY)
                    if evaluator == 'rules':
                        self.assertFalse(report['llm_called'])
                    else:
                        self.assertTrue(report['llm_called'])

    def test_env_fuzz_for_finite_numeric_parsing(self) -> None:
        bad_values = ['nan', 'inf', '-inf', 'abc', '', ' ']
        for bad in bad_values:
            with self.subTest(value=bad):
                with mock.patch.dict(
                    os.environ,
                    {
                        'SCRIPT_QUALITY_GATE_LLM_SAMPLE': bad,
                        'SCRIPT_QUALITY_MIN_OVERALL_SCORE': bad,
                        'SCRIPT_QUALITY_MIN_CADENCE_SCORE': bad,
                        'SCRIPT_QUALITY_MIN_LOGIC_SCORE': bad,
                        'SCRIPT_QUALITY_MIN_CLARITY_SCORE': bad,
                    },
                    clear=True,
                ):
                    cfg = ScriptQualityGateConfig.from_env(profile_name='short')
                self.assertGreaterEqual(cfg.llm_sample_rate, 0.0)
                self.assertLessEqual(cfg.llm_sample_rate, 1.0)
                self.assertGreaterEqual(cfg.min_overall_score, 0.0)
                self.assertLessEqual(cfg.min_overall_score, 5.0)
                self.assertGreaterEqual(cfg.min_logic_score, 0.0)
                self.assertLessEqual(cfg.min_logic_score, 5.0)


if __name__ == '__main__':
    unittest.main()
