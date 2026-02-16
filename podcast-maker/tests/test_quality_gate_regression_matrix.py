import dataclasses
import os
import sys
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import ScriptConfig  # noqa: E402
from pipeline.errors import ERROR_KIND_SCRIPT_QUALITY  # noqa: E402
from pipeline.script_quality_gate import (  # noqa: E402
    ScriptQualityGateConfig,
    evaluate_script_quality,
)


class _MatrixClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def generate_freeform_text(self, *, prompt: str, max_output_tokens: int, stage: str) -> str:  # noqa: ARG002
        self.calls += 1
        return self.payload


def _script_cfg(profile: str) -> ScriptConfig:
    cfg = ScriptConfig.from_env(profile_name=profile)
    return dataclasses.replace(cfg, min_words=20, max_words=400, profile_name=profile)


GOOD_LINES = [
    {"speaker": "Ana", "text": "Inicio del episodio con contexto y objetivos claros para la audiencia."},
    {"speaker": "Luis", "text": "Desglosamos datos clave y explicamos por que importan hoy."},
    {"speaker": "Ana", "text": "Contrastamos ventajas y riesgos de cada opcion practicamente."},
    {"speaker": "Luis", "text": "Anadimos ejemplos operativos para bajar ideas a accion concreta."},
    {"speaker": "Ana", "text": "En resumen, conviene validar antes de escalar cambios criticos."},
    {"speaker": "Luis", "text": "Gracias por escuchar, nos vemos en el siguiente episodio."},
]

BAD_LINES = [
    {"speaker": "Ana", "text": "repetido"},
    {"speaker": "Ana", "text": "repetido"},
    {"speaker": "Ana", "text": "repetido"},
    {"speaker": "Ana", "text": "repetido"},
]


class QualityGateRegressionMatrixTests(unittest.TestCase):
    def test_good_script_matrix_profiles_actions_evaluators(self) -> None:
        pass_payload = (
            '{"overall_score":4.5,"cadence_score":4.5,"logic_score":4.5,'
            '"clarity_score":4.5,"pass":true,"reasons":[]}'
        )
        for profile in ("short", "standard", "long"):
            for action in ("off", "warn", "enforce"):
                for evaluator in ("rules", "hybrid", "llm"):
                    with self.subTest(profile=profile, action=action, evaluator=evaluator):
                        env = {
                            "SCRIPT_QUALITY_GATE_ACTION": action,
                            "SCRIPT_QUALITY_GATE_EVALUATOR": evaluator,
                            "SCRIPT_QUALITY_GATE_LLM_SAMPLE": "1.0",
                            "SCRIPT_QUALITY_MIN_WORDS_RATIO": "0.1",
                            "SCRIPT_QUALITY_MAX_WORDS_RATIO": "4.0",
                        }
                        with mock.patch.dict(os.environ, env, clear=True):
                            cfg = ScriptQualityGateConfig.from_env(profile_name=profile)
                        client = _MatrixClient(pass_payload)
                        report = evaluate_script_quality(
                            validated_payload={"lines": GOOD_LINES},
                            script_cfg=_script_cfg(profile),
                            quality_cfg=cfg,
                            script_path=f"/tmp/{profile}.json",
                            client=client,
                        )
                        self.assertTrue(report["pass"])
                        if evaluator == "rules":
                            self.assertFalse(report["llm_called"])
                            self.assertEqual(client.calls, 0)
                        else:
                            self.assertTrue(report["llm_called"])
                            self.assertEqual(client.calls, 1)

    def test_bad_script_matrix_reports_rejection(self) -> None:
        fail_payload = (
            '{"overall_score":2.2,"cadence_score":2.0,"logic_score":2.1,'
            '"clarity_score":2.3,"pass":false,"reasons":["weak"]}'
        )
        for profile in ("short", "standard", "long"):
            for evaluator in ("rules", "hybrid", "llm"):
                with self.subTest(profile=profile, evaluator=evaluator):
                    env = {
                        "SCRIPT_QUALITY_GATE_ACTION": "enforce",
                        "SCRIPT_QUALITY_GATE_EVALUATOR": evaluator,
                        "SCRIPT_QUALITY_GATE_LLM_SAMPLE": "1.0",
                        "SCRIPT_QUALITY_MIN_WORDS_RATIO": "0.0",
                        "SCRIPT_QUALITY_MAX_WORDS_RATIO": "10.0",
                    }
                    with mock.patch.dict(os.environ, env, clear=True):
                        cfg = ScriptQualityGateConfig.from_env(profile_name=profile)
                    client = _MatrixClient(fail_payload)
                    report = evaluate_script_quality(
                        validated_payload={"lines": BAD_LINES},
                        script_cfg=_script_cfg(profile),
                        quality_cfg=cfg,
                        script_path=f"/tmp/{profile}.json",
                        client=client,
                    )
                    self.assertFalse(report["pass"])
                    self.assertEqual(report["failure_kind"], ERROR_KIND_SCRIPT_QUALITY)
                    if evaluator == "llm":
                        self.assertTrue(report["llm_called"])
                    elif evaluator == "hybrid":
                        # Hybrid does not spend an extra call when deterministic rules already fail.
                        self.assertFalse(report["llm_called"])

    def test_env_fuzz_for_finite_numeric_parsing(self) -> None:
        bad_values = ["nan", "inf", "-inf", "abc", "", " "]
        for bad in bad_values:
            with self.subTest(value=bad):
                with mock.patch.dict(
                    os.environ,
                    {
                        "SCRIPT_QUALITY_GATE_LLM_SAMPLE": bad,
                        "SCRIPT_QUALITY_MIN_OVERALL_SCORE": bad,
                        "SCRIPT_QUALITY_MIN_CADENCE_SCORE": bad,
                        "SCRIPT_QUALITY_MIN_LOGIC_SCORE": bad,
                        "SCRIPT_QUALITY_MIN_CLARITY_SCORE": bad,
                    },
                    clear=True,
                ):
                    cfg = ScriptQualityGateConfig.from_env(profile_name="short")
                self.assertGreaterEqual(cfg.llm_sample_rate, 0.0)
                self.assertLessEqual(cfg.llm_sample_rate, 1.0)
                self.assertGreaterEqual(cfg.min_overall_score, 0.0)
                self.assertLessEqual(cfg.min_overall_score, 5.0)
                self.assertGreaterEqual(cfg.min_logic_score, 0.0)
                self.assertLessEqual(cfg.min_logic_score, 5.0)


if __name__ == "__main__":
    unittest.main()

