import dataclasses
import json
import os
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import ScriptConfig  # noqa: E402
from pipeline.errors import ERROR_KIND_SCRIPT_QUALITY  # noqa: E402
from pipeline.script_quality_gate import (  # noqa: E402
    ScriptQualityGateConfig,
    attempt_script_quality_repair,
    evaluate_script_quality,
    write_quality_report,
)


class _FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate_freeform_text(self, *, prompt: str, max_output_tokens: int, stage: str) -> str:  # noqa: ARG002
        self.calls += 1
        return self.response


class _FakeRepairClient(_FakeLLMClient):
    def __init__(self, *, repaired_payload: dict[str, object]) -> None:
        super().__init__(
            '{"overall_score":4.5,"cadence_score":4.4,"logic_score":4.4,"clarity_score":4.4,"pass":true,"reasons":[]}'
        )
        self.repaired_payload = repaired_payload
        self.repair_calls = 0

    def generate_script_json(self, *, prompt: str, schema: dict[str, object], max_output_tokens: int, stage: str):  # noqa: ANN201, ARG002
        self.repair_calls += 1
        return self.repaired_payload


class _CapturingRepairClient(_FakeRepairClient):
    def __init__(self, *, repaired_payload: dict[str, object]) -> None:
        super().__init__(repaired_payload=repaired_payload)
        self.max_output_tokens_seen: list[int] = []

    def generate_script_json(self, *, prompt: str, schema: dict[str, object], max_output_tokens: int, stage: str):  # noqa: ANN201, ARG002
        self.max_output_tokens_seen.append(int(max_output_tokens))
        return super().generate_script_json(
            prompt=prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
            stage=stage,
        )


class _SlowFailingRepairClient(_FakeRepairClient):
    def __init__(self, *, repaired_payload: dict[str, object], sleep_seconds: float) -> None:
        super().__init__(repaired_payload=repaired_payload)
        self.sleep_seconds = float(max(0.0, sleep_seconds))

    def generate_script_json(self, *, prompt: str, schema: dict[str, object], max_output_tokens: int, stage: str):  # noqa: ANN201, ARG002
        import time as _time

        self.repair_calls += 1
        if self.sleep_seconds > 0.0:
            _time.sleep(self.sleep_seconds)
        return self.repaired_payload


class _StageAwareLLMClient:
    def __init__(self, responses_by_stage: dict[str, str]) -> None:
        self.responses_by_stage = dict(responses_by_stage)
        self.calls: list[str] = []

    def generate_freeform_text(self, *, prompt: str, max_output_tokens: int, stage: str) -> str:  # noqa: ARG002
        self.calls.append(stage)
        return self.responses_by_stage.get(stage, "{}")


class _FailingSemanticClient:
    def __init__(self, message: str = "semantic transport error") -> None:
        self.message = message
        self.calls: list[str] = []

    def generate_freeform_text(self, *, prompt: str, max_output_tokens: int, stage: str) -> str:  # noqa: ARG002
        self.calls.append(stage)
        raise RuntimeError(self.message)


def _line(speaker: str, role: str, text: str) -> dict[str, str]:
    return {
        "speaker": speaker,
        "role": role,
        "instructions": (
            "Voice Affect: Warm and confident | Tone: Conversational | "
            "Pacing: Measured | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief"
        ),
        "text": text,
    }


def _base_script_cfg() -> ScriptConfig:
    cfg = ScriptConfig.from_env(profile_name="short")
    return dataclasses.replace(cfg, min_words=40, max_words=150, profile_name="short")


def _good_lines() -> list[dict[str, str]]:
    return [
        _line("Ana", "Host1", "Hoy repasamos titulares de tecnologia con foco practico."),
        _line("Luis", "Host2", "Arrancamos con contexto, causas y datos comparables para el oyente."),
        _line("Ana", "Host1", "Pasamos a impactos reales en producto, equipo y usuarios finales."),
        _line("Luis", "Host2", "Tambien cubrimos riesgos, tradeoffs y ejemplos para tomar decisiones."),
        _line("Ana", "Host1", "En resumen, conviene priorizar cambios graduales con medicion clara."),
        _line("Luis", "Host2", "Gracias por escuchar, nos vemos en la proxima entrega del podcast."),
    ]


def _bad_lines() -> list[dict[str, str]]:
    return [
        _line("Ana", "Host1", "Lo mismo."),
        _line("Ana", "Host1", "Lo mismo."),
        _line("Ana", "Host1", "Lo mismo."),
        _line("Ana", "Host1", "Lo mismo."),
    ]


class ScriptQualityGateTests(unittest.TestCase):
    def test_config_defaults_by_profile(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            short_cfg = ScriptQualityGateConfig.from_env(profile_name="short")
            standard_cfg = ScriptQualityGateConfig.from_env(profile_name="standard")
            long_cfg = ScriptQualityGateConfig.from_env(profile_name="long")
        self.assertEqual(short_cfg.action, "enforce")
        self.assertEqual(short_cfg.evaluator, "hybrid")
        self.assertAlmostEqual(short_cfg.llm_sample_rate, 0.5)
        self.assertEqual(short_cfg.repair_attempts, 2)
        self.assertEqual(short_cfg.repair_max_output_tokens, 5200)
        self.assertTrue(short_cfg.semantic_rule_fallback)
        self.assertGreaterEqual(short_cfg.semantic_min_confidence, 0.0)
        self.assertLessEqual(short_cfg.semantic_min_confidence, 1.0)
        self.assertAlmostEqual(standard_cfg.llm_sample_rate, 1.0)
        self.assertEqual(standard_cfg.repair_attempts, 2)
        self.assertEqual(standard_cfg.repair_max_output_tokens, 5200)
        self.assertAlmostEqual(long_cfg.llm_sample_rate, 1.0)
        self.assertEqual(long_cfg.repair_attempts, 2)
        self.assertEqual(long_cfg.repair_max_output_tokens, 5200)

    def test_config_production_strict_profile_defaults(self) -> None:
        with mock.patch.dict(os.environ, {"SCRIPT_QUALITY_GATE_PROFILE": "production_strict"}, clear=True):
            cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        self.assertEqual(cfg.action, "enforce")
        self.assertEqual(cfg.evaluator, "hybrid")
        self.assertAlmostEqual(cfg.llm_sample_rate, 1.0)
        self.assertEqual(cfg.repair_attempts, 2)
        self.assertTrue(cfg.repair_revert_on_fail)
        self.assertGreater(cfg.repair_min_word_ratio, 0.0)
        self.assertEqual(cfg.max_consecutive_same_speaker, 1)
        self.assertAlmostEqual(cfg.max_repeat_line_ratio, 0.12)
        self.assertGreaterEqual(cfg.min_overall_score, 4.0)
        self.assertGreaterEqual(cfg.min_logic_score, 4.0)

    def test_config_invalid_values_fallback_to_safe_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SCRIPT_QUALITY_GATE_ACTION": "bad",
                "SCRIPT_QUALITY_GATE_EVALUATOR": "bad",
                "SCRIPT_QUALITY_GATE_LLM_SAMPLE": "nan",
                "SCRIPT_QUALITY_MIN_WORDS_RATIO": "oops",
                "SCRIPT_QUALITY_MAX_WORDS_RATIO": "-2",
                "SCRIPT_QUALITY_MAX_REPEAT_LINE_RATIO": "5",
                "SCRIPT_QUALITY_MAX_CONSECUTIVE_SAME_SPEAKER": "x",
                "SCRIPT_QUALITY_GATE_SEMANTIC_MIN_CONFIDENCE": "not-a-number",
                "SCRIPT_QUALITY_GATE_SEMANTIC_TAIL_LINES": "999",
            },
            clear=True,
        ):
            cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        self.assertEqual(cfg.action, "enforce")
        self.assertEqual(cfg.evaluator, "hybrid")
        self.assertAlmostEqual(cfg.llm_sample_rate, 0.5)
        self.assertEqual(cfg.max_consecutive_same_speaker, 1)
        self.assertGreaterEqual(cfg.max_words_ratio, cfg.min_words_ratio)
        self.assertGreaterEqual(cfg.max_repeat_line_ratio, 0.0)
        self.assertLessEqual(cfg.max_repeat_line_ratio, 1.0)
        self.assertGreaterEqual(cfg.semantic_min_confidence, 0.0)
        self.assertLessEqual(cfg.semantic_min_confidence, 1.0)
        self.assertLessEqual(cfg.semantic_tail_lines, 24)

    def test_rules_fail_for_internal_workflow_disclosure(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.1,
            max_words_ratio=5.0,
            max_repeat_line_ratio=1.0,
        )
        lines = _good_lines()[:-1] + [
            _line(
                "Luis",
                "Host2",
                "Nota de transparencia: este DailyRead se elaboro usando el script scripts/search_combo.sh con Tavily y Serper.",
            )
        ]
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/internal_workflow_disclosure.json",
        )
        self.assertFalse(report["pass"])
        self.assertIn("no_internal_workflow_meta_ok", set(report["reasons"]))

    def test_rules_fail_for_repetition_and_missing_structure(self) -> None:
        cfg = ScriptQualityGateConfig(
            action="enforce",
            evaluator="rules",
            llm_sample_rate=1.0,
            min_words_ratio=0.1,
            max_words_ratio=5.0,
            max_consecutive_same_speaker=2,
            max_repeat_line_ratio=0.1,
            require_summary=True,
            require_closing=True,
            min_overall_score=3.8,
            min_cadence_score=3.7,
            min_logic_score=3.8,
            min_clarity_score=3.8,
            llm_max_output_tokens=512,
            llm_max_prompt_chars=8000,
            auto_repair=False,
            repair_attempts=0,
            repair_max_output_tokens=700,
            repair_max_input_chars=12000,
        )
        report = evaluate_script_quality(
            validated_payload={"lines": _bad_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/bad.json",
        )
        self.assertFalse(report["pass"])
        self.assertEqual(report["failure_kind"], ERROR_KIND_SCRIPT_QUALITY)
        reasons = set(report["reasons"])
        self.assertIn("summary_ok", reasons)
        self.assertIn("closing_ok", reasons)
        self.assertIn("repeat_line_ratio_ok", reasons)

    def test_rules_pass_for_well_formed_script(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.2,
            max_words_ratio=3.0,
            max_repeat_line_ratio=0.3,
        )
        report = evaluate_script_quality(
            validated_payload={"lines": _good_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/good.json",
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["status"], "passed")
        self.assertIsNone(report["failure_kind"])

    def test_rules_fail_when_question_is_unanswered_before_summary(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            max_repeat_line_ratio=1.0,
        )
        lines = [
            _line("Ana", "Host1", "Hoy aterrizamos riesgos de operacion y de producto con casos reales."),
            _line(
                "Luis",
                "Host2",
                "Bien, pero en terminos operativos, te parece que conviene priorizar latencia o robustez en este momento?",
            ),
            _line("Ana", "Host1", "En resumen, conviene priorizar decisiones con evidencia y medicion continua."),
            _line("Luis", "Host2", "Gracias por escuchar, nos vemos en la proxima entrega."),
        ]
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/unanswered_tail_question.json",
        )
        self.assertFalse(bool(report["pass"]))
        self.assertFalse(bool(report["rules"]["open_questions_resolved_ok"]))
        self.assertIn("open_questions_resolved_ok", set(report["reasons"]))

    def test_rules_pass_when_question_is_answered_before_summary(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            max_repeat_line_ratio=1.0,
        )
        lines = [
            _line("Ana", "Host1", "Hoy revisamos riesgos y tradeoffs para mantener calidad de audio estable."),
            _line(
                "Luis",
                "Host2",
                "Te parece que priorizar consistencia de cierre por encima de velocidad de entrega tiene sentido?",
            ),
            _line(
                "Ana",
                "Host1",
                "Si, porque un cierre coherente evita confusion y mejora la retencion de ideas clave.",
            ),
            _line("Luis", "Host2", "En resumen, priorizamos consistencia, evidencia y decisiones iterativas."),
            _line("Ana", "Host1", "Gracias por escucharnos, nos vemos en el proximo episodio."),
        ]
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/answered_tail_question.json",
        )
        self.assertTrue(bool(report["pass"]))
        self.assertTrue(bool(report["rules"]["open_questions_resolved_ok"]))

    def test_rules_accept_accented_farewell(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
        )
        lines = [
            _line("Ana", "Host1", "Hoy repasamos hallazgos y decisiones de implementacion."),
            _line("Luis", "Host2", "Incluimos contexto, riesgos y tradeoffs para el equipo."),
            _line("Ana", "Host1", "En resumen, priorizamos fiabilidad operativa y trazabilidad."),
            _line("Luis", "Host2", "Gracias por escuchar. Hasta la próxima."),
        ]
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/accented_closing.json",
        )
        self.assertTrue(report["pass"])
        self.assertTrue(bool(report["rules"]["closing_ok"]))

    def test_semantic_fallback_unblocks_unknown_closing_phrase(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="hybrid",
            llm_sample_rate=0.0,
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            semantic_rule_fallback=True,
            semantic_min_confidence=0.55,
        )
        lines = [
            _line("Ana", "Host1", "Hoy revisamos decisiones tecnicas con impacto real."),
            _line("Luis", "Host2", "Aterrizamos costos, riesgos y mitigaciones concretas."),
            _line("Ana", "Host1", "En resumen, priorizamos estabilidad y observabilidad."),
            _line("Luis", "Host2", "Seguimos esta conversacion en una futura entrega del programa."),
        ]
        report_without_client = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/semantic_without_client.json",
            client=None,
        )
        self.assertFalse(bool(report_without_client["pass"]))
        self.assertIn("closing_ok", set(report_without_client["reasons"]))
        self.assertTrue(bool(report_without_client["semantic_rule_fallback"]["called"]))

        client = _StageAwareLLMClient(
            {
                "script_quality_semantic_rules": (
                    '{"summary_semantic": true, "closing_semantic": true, '
                    '"confidence": 0.92, "evidence": ["future episode invitation"]}'
                )
            }
        )
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/semantic_with_client.json",
            client=client,
        )
        self.assertTrue(bool(report["pass"]))
        self.assertTrue(bool(report["rules"]["closing_ok"]))
        semantic = dict(report.get("semantic_rule_fallback", {}))
        self.assertTrue(bool(semantic.get("used", False)))
        self.assertEqual(client.calls, ["script_quality_semantic_rules"])

    def test_semantic_fallback_respects_confidence_threshold(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="hybrid",
            llm_sample_rate=0.0,
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            semantic_rule_fallback=True,
            semantic_min_confidence=0.9,
        )
        lines = [
            _line("Ana", "Host1", "Hoy revisamos decisiones tecnicas con impacto real."),
            _line("Luis", "Host2", "Aterrizamos costos, riesgos y mitigaciones concretas."),
            _line("Ana", "Host1", "En resumen, priorizamos estabilidad y observabilidad."),
            _line("Luis", "Host2", "Seguimos esta conversacion en una futura entrega del programa."),
        ]
        client = _StageAwareLLMClient(
            {
                "script_quality_semantic_rules": (
                    '{"summary_semantic": true, "closing_semantic": true, '
                    '"confidence": 0.35, "evidence": ["future episode invitation"]}'
                )
            }
        )
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/semantic_low_conf.json",
            client=client,
        )
        self.assertFalse(bool(report["pass"]))
        self.assertIn("closing_ok", set(report["reasons"]))
        semantic = dict(report.get("semantic_rule_fallback", {}))
        self.assertFalse(bool(semantic.get("used", False)))
        self.assertFalse(bool(semantic.get("confidence_gate_passed", True)))

    def test_semantic_fallback_client_exception_marks_error(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            llm_sample_rate=0.0,
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            semantic_rule_fallback=True,
        )
        lines = [
            _line("Ana", "Host1", "Hoy revisamos decisiones tecnicas con impacto real."),
            _line("Luis", "Host2", "Aterrizamos costos, riesgos y mitigaciones concretas."),
            _line("Ana", "Host1", "Cerramos ideas operativas para los siguientes sprints."),
            _line("Luis", "Host2", "Seguimos la conversacion en una futura entrega del programa."),
        ]
        client = _FailingSemanticClient("semantic endpoint timeout")
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/semantic_client_error.json",
            client=client,
        )
        self.assertFalse(bool(report["pass"]))
        semantic = dict(report.get("semantic_rule_fallback", {}))
        self.assertTrue(bool(semantic.get("called", False)))
        self.assertTrue(bool(semantic.get("error", False)))
        self.assertIn("semantic endpoint timeout", str(semantic.get("skipped_reason", "")))
        self.assertEqual(client.calls, ["script_quality_semantic_rules"])

    def test_semantic_fallback_disabled_never_calls_client(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            llm_sample_rate=0.0,
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            semantic_rule_fallback=False,
        )
        lines = [
            _line("Ana", "Host1", "Hoy revisamos decisiones tecnicas con impacto real."),
            _line("Luis", "Host2", "Aterrizamos costos, riesgos y mitigaciones concretas."),
            _line("Ana", "Host1", "Cerramos ideas operativas para los siguientes sprints."),
            _line("Luis", "Host2", "Seguimos la conversacion en una futura entrega del programa."),
        ]
        client = _StageAwareLLMClient({"script_quality_semantic_rules": '{"closing_semantic": true}'})
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/semantic_disabled.json",
            client=client,
        )
        self.assertFalse(bool(report["pass"]))
        semantic = dict(report.get("semantic_rule_fallback", {}))
        self.assertFalse(bool(semantic.get("called", True)))
        self.assertEqual(client.calls, [])

    def test_rules_evaluator_can_use_semantic_fallback_when_markers_are_missing(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            llm_sample_rate=0.0,
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            semantic_rule_fallback=True,
            semantic_min_confidence=0.55,
        )
        lines = [
            _line("Ana", "Host1", "Today we review architecture decisions and implementation risks."),
            _line("Luis", "Host2", "We connect latency, cost and reliability tradeoffs for delivery."),
            _line("Ana", "Host1", "Stepping back, the key thread is gradual rollout with observability."),
            _line("Luis", "Host2", "We continue this thread in a future installment of the show."),
        ]
        without_client = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/rules_semantic_no_client.json",
            client=None,
        )
        self.assertFalse(bool(without_client["pass"]))
        self.assertTrue(bool(without_client["semantic_rule_fallback"]["called"]))
        self.assertIn("summary_ok", set(without_client["reasons"]))
        self.assertIn("closing_ok", set(without_client["reasons"]))

        client = _StageAwareLLMClient(
            {
                "script_quality_semantic_rules": (
                    '{"summary_semantic": true, "closing_semantic": true, '
                    '"confidence": 0.91, "evidence": ["recap intent", "continuation invite"]}'
                )
            }
        )
        with_client = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/rules_semantic_with_client.json",
            client=client,
        )
        self.assertTrue(bool(with_client["pass"]))
        self.assertTrue(bool(with_client["semantic_rule_fallback"]["used"]))
        self.assertEqual(client.calls, ["script_quality_semantic_rules"])

    def test_rules_pass_portuguese_summary_and_closing_without_semantic_fallback(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            llm_sample_rate=0.0,
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            semantic_rule_fallback=False,
        )
        lines = [
            _line("Ana", "Host1", "Hoje revisamos decisoes tecnicas com impacto operacional."),
            _line("Luis", "Host2", "Conectamos custo, risco e observabilidade para execucao."),
            _line("Ana", "Host1", "Em resumo, priorizamos rollout gradual com sinais claros."),
            _line("Luis", "Host2", "Obrigado por ouvir, ate a proxima edicao."),
        ]
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/portuguese_rules.json",
        )
        self.assertTrue(bool(report["pass"]))
        self.assertTrue(bool(report["rules"]["summary_ok"]))
        self.assertTrue(bool(report["rules"]["closing_ok"]))

    def test_rules_detect_summary_by_recap_overlap_without_keyword(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            llm_sample_rate=0.0,
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            semantic_rule_fallback=False,
        )
        lines = [
            _line("Ana", "Host1", "Latency reliability observability metrics guide rollout planning today."),
            _line("Luis", "Host2", "Reliability latency metrics improve rollout observability in production systems."),
            _line("Ana", "Host1", "Observability and latency metrics support reliability decisions for rollout teams."),
            _line(
                "Luis",
                "Host2",
                "This discussion connected latency, reliability, observability, metrics, and rollout priorities for delivery.",
            ),
            _line("Ana", "Host1", "Thank you for listening, see you next episode."),
        ]
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/rules_recap_overlap.json",
        )
        self.assertTrue(bool(report["pass"]))
        self.assertTrue(bool(report["rules"]["summary_ok"]))
        self.assertTrue(bool(report["rules"]["closing_ok"]))

    def test_rules_detect_summary_by_recap_overlap_with_non_latin_script(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            llm_sample_rate=0.0,
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            semantic_rule_fallback=False,
            require_closing=False,
        )
        lines = [
            _line("Ana", "Host1", "Сегодня обсуждаем надежность наблюдаемость метрики релизы риски команды и процессы."),
            _line("Luis", "Host2", "Надежность и наблюдаемость через метрики помогают команде планировать релизы и снижать риски."),
            _line("Ana", "Host1", "Команда использует метрики надежности и наблюдаемости чтобы выпускать релизы без лишних рисков."),
            _line(
                "Luis",
                "Host2",
                "В этом выпуске мы связали надежность, наблюдаемость, метрики, релизы и риски для команды.",
            ),
            _line("Ana", "Host1", "Следующий шаг — внедрить проверки и ретроспективы по метрикам надежности."),
        ]
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/rules_recap_overlap_non_latin.json",
        )
        self.assertTrue(bool(report["pass"]))
        self.assertTrue(bool(report["rules"]["summary_ok"]))

    def test_evaluate_quality_applies_structural_hardening(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            require_summary=False,
            require_closing=False,
        )
        lines = [
            _line("Ana", "Host1", "Entramos al Bloque 1 con objetivos claros."),
            _line("Luis", "Host2", "Seguimos con Bloque 2 y datos relevantes."),
            _line("Ana", "Host1", "Ahora Bloque 4 para decisiones practicas."),
            _line("Luis", "Host2", "Cierre operativo con telemetria y..."),
        ]
        report = evaluate_script_quality(
            validated_payload={"lines": lines},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/hardened.json",
        )
        self.assertTrue(bool(report.get("structural_hardening_applied")))
        self.assertTrue(report["pass"])

    def test_hybrid_with_zero_sample_rate_skips_llm(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="hybrid",
            llm_sample_rate=0.0,
            min_words_ratio=0.2,
            max_words_ratio=3.0,
        )
        fake_client = _FakeLLMClient(
            '{"overall_score":5,"cadence_score":5,"logic_score":5,"clarity_score":5,"pass":true,"reasons":[]}'
        )
        report = evaluate_script_quality(
            validated_payload={"lines": _good_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/good.json",
            client=fake_client,
        )
        self.assertTrue(report["pass"])
        self.assertFalse(report["llm_sampled"])
        self.assertFalse(report["llm_called"])
        self.assertEqual(fake_client.calls, 0)

    def test_hybrid_with_sample_one_calls_llm(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="hybrid",
            llm_sample_rate=1.0,
            min_words_ratio=0.2,
            max_words_ratio=3.0,
        )
        fake_client = _FakeLLMClient(
            '{"overall_score":4.3,"cadence_score":4.2,"logic_score":4.1,"clarity_score":4.0,"pass":true,"reasons":["ok"]}'
        )
        report = evaluate_script_quality(
            validated_payload={"lines": _good_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/good.json",
            client=fake_client,
        )
        self.assertTrue(report["pass"])
        self.assertTrue(report["llm_called"])
        self.assertEqual(fake_client.calls, 1)

    def test_llm_mode_low_scores_warn_only_by_default(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="llm",
            llm_sample_rate=1.0,
            min_words_ratio=0.0,
            max_words_ratio=5.0,
        )
        fake_client = _FakeLLMClient(
            '{"overall_score":2.0,"cadence_score":2.1,"logic_score":2.2,"clarity_score":2.3,"pass":false,"reasons":["incoherent"]}'
        )
        report = evaluate_script_quality(
            validated_payload={"lines": _good_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/good.json",
            client=fake_client,
        )
        self.assertTrue(report["pass"])
        self.assertTrue(report["llm_called"])
        self.assertTrue(bool(report.get("editorial_warn_only", False)))
        self.assertFalse(bool(report.get("hard_fail_eligible", True)))
        self.assertIsNone(report.get("failure_kind"))

    def test_llm_tail_truncation_reason_filtered_when_tail_is_complete(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="llm",
            llm_sample_rate=1.0,
            min_words_ratio=0.0,
            max_words_ratio=5.0,
        )
        fake_client = _FakeLLMClient(
            '{"overall_score":4.8,"cadence_score":4.8,"logic_score":4.7,"clarity_score":4.8,'
            '"pass":false,"reasons":["El guion queda truncado al final y no cierra de forma natural."]}'
        )
        report = evaluate_script_quality(
            validated_payload={"lines": _good_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/good_tail_complete.json",
            client=fake_client,
        )
        self.assertTrue(report["pass"])
        self.assertEqual(int(report.get("llm_truncation_claims_filtered", 0)), 1)
        self.assertFalse(bool(report.get("llm_explicit_fail", True)))
        self.assertFalse(bool(report.get("llm_editorial_fail", True)))
        self.assertFalse(any("trunc" in str(reason).lower() for reason in report.get("reasons_llm", [])))

    def test_llm_mode_low_scores_can_block_in_strict_score_mode(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="llm",
            llm_sample_rate=1.0,
            min_words_ratio=0.0,
            max_words_ratio=5.0,
        )
        fake_client = _FakeLLMClient(
            '{"overall_score":2.0,"cadence_score":2.1,"logic_score":2.2,"clarity_score":2.3,"pass":false,"reasons":["incoherent"]}'
        )
        with mock.patch.dict(
            os.environ,
            {
                "SCRIPT_QUALITY_GATE_STRICT_SCORE_BLOCKING": "1",
                "SCRIPT_QUALITY_GATE_CRITICAL_SCORE_THRESHOLD": "2.5",
            },
            clear=False,
        ):
            report = evaluate_script_quality(
                validated_payload={"lines": _good_lines()},
                script_cfg=_base_script_cfg(),
                quality_cfg=cfg,
                script_path="/tmp/good_strict.json",
                client=fake_client,
            )
        self.assertFalse(report["pass"])
        self.assertTrue(bool(report.get("hard_fail_eligible", False)))
        self.assertEqual(report.get("failure_kind"), ERROR_KIND_SCRIPT_QUALITY)

    def test_structural_fail_still_blocks_even_with_good_llm_scores(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="hybrid",
            llm_sample_rate=1.0,
            min_words_ratio=0.0,
            max_words_ratio=10.0,
        )
        fake_client = _FakeLLMClient(
            '{"overall_score":4.8,"cadence_score":4.8,"logic_score":4.7,"clarity_score":4.7,"pass":true,"reasons":[]}'
        )
        report = evaluate_script_quality(
            validated_payload={"lines": _bad_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/bad_structural.json",
            client=fake_client,
        )
        self.assertFalse(report["pass"])
        self.assertTrue(bool(report.get("hard_fail_eligible", False)))
        self.assertIn("summary_ok", set(report.get("reasons_structural", [])))

    def test_llm_mode_invalid_json_degrades_to_rules(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="llm",
            llm_sample_rate=1.0,
            min_words_ratio=0.2,
            max_words_ratio=5.0,
        )
        fake_client = _FakeLLMClient("not a json payload")
        report = evaluate_script_quality(
            validated_payload={"lines": _good_lines()},
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/good.json",
            client=fake_client,
        )
        self.assertTrue(report["pass"])
        self.assertTrue(bool(report.get("llm_error")))
        self.assertTrue(bool(report.get("llm_degraded_to_rules")))
        self.assertIn("llm_evaluator_error", " ".join(report["reasons"]))

    def test_hybrid_llm_error_degrades_to_rules_without_repair(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="hybrid",
            llm_sample_rate=1.0,
            min_words_ratio=0.2,
            max_words_ratio=5.0,
            auto_repair=True,
            repair_attempts=2,
        )
        initial_payload = {"lines": _good_lines()}
        initial_report = evaluate_script_quality(
            validated_payload=initial_payload,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/good.json",
            client=_FakeLLMClient("not a json payload"),
        )
        self.assertTrue(initial_report["pass"])
        outcome = attempt_script_quality_repair(
            validated_payload=initial_payload,
            initial_report=initial_report,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/final.json",
            client=_FakeLLMClient("still invalid"),
        )
        self.assertFalse(bool(outcome["report"]["repair_attempted"]))
        self.assertFalse(bool(outcome["repaired"]))

    def test_write_quality_report_writes_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            write_quality_report(path, {"status": "passed", "pass": True})
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["pass"])

    def test_attempt_script_quality_repair_makes_script_pass(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            auto_repair=True,
            repair_attempts=1,
        )
        initial_payload = {"lines": _bad_lines()}
        initial_report = evaluate_script_quality(
            validated_payload=initial_payload,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/initial.json",
        )
        repaired_payload = {
            "lines": [
                _line("Ana", "Host1", "Introduccion del tema con objetivos claros y contexto actual."),
                _line("Luis", "Host2", "Desarrollo con ejemplos utiles para tomar decisiones practicas."),
                _line("Ana", "Host1", "En resumen, conviene priorizar cambios graduales y medibles."),
                _line("Luis", "Host2", "Gracias por escuchar, nos vemos en la proxima."),
            ]
        }
        client = _FakeRepairClient(repaired_payload=repaired_payload)
        outcome = attempt_script_quality_repair(
            validated_payload=initial_payload,
            initial_report=initial_report,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/final.json",
            client=client,
        )
        self.assertTrue(bool(outcome["report"]["pass"]))
        self.assertTrue(bool(outcome["report"]["repair_attempted"]))
        self.assertTrue(bool(outcome["report"]["repair_succeeded"]))
        self.assertTrue(bool(outcome["repaired"]))
        self.assertEqual(client.repair_calls, 1)

    def test_attempt_script_quality_repair_uses_deterministic_tail_fix_without_llm(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            auto_repair=True,
            repair_attempts=1,
            repair_revert_on_fail=True,
        )
        initial_payload = {
            "lines": [
                _line("Ana", "Host1", "Compartimos los hallazgos principales del dia."),
                _line("Luis", "Host2", "Desarrollamos impacto practico y decisiones accionables."),
                _line("Ana", "Host1", "Cerramos con recomendaciones operativas para el equipo."),
            ]
        }
        initial_report = evaluate_script_quality(
            validated_payload=initial_payload,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/needs_tail_fix.json",
        )
        self.assertFalse(initial_report["pass"])
        self.assertIn("summary_ok", set(initial_report["reasons"]))
        self.assertIn("closing_ok", set(initial_report["reasons"]))

        outcome = attempt_script_quality_repair(
            validated_payload=initial_payload,
            initial_report=initial_report,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/needs_tail_fix.json",
            client=None,
        )
        self.assertTrue(bool(outcome["report"]["pass"]))
        self.assertTrue(bool(outcome["report"]["repair_attempted"]))
        self.assertTrue(bool(outcome["report"]["repair_succeeded"]))
        self.assertEqual(int(outcome["report"]["repair_attempts_used"]), 0)
        repaired_lines = list(outcome["payload"].get("lines", []))
        repaired_text = "\n".join(line.get("text", "") for line in repaired_lines)
        self.assertIn("En Resumen", repaired_text)
        self.assertIn("Gracias por escuch", repaired_text)

    def test_attempt_script_quality_repair_deterministically_fixes_speaker_runs(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            auto_repair=True,
            repair_attempts=1,
            max_consecutive_same_speaker=2,
        )
        initial_payload = {
            "lines": [
                _line("Ana", "Host1", "Bloque 1 con contexto y decisiones tecnicas."),
                _line("Ana", "Host1", "Bloque 2 con riesgos y mitigaciones."),
                _line("Ana", "Host1", "En resumen, conviene medir antes de escalar."),
                _line("Ana", "Host1", "Gracias por escuchar, nos vemos en la proxima."),
            ]
        }
        initial_report = evaluate_script_quality(
            validated_payload=initial_payload,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/needs_speaker_fix.json",
        )
        self.assertTrue(bool(initial_report["pass"]))
        self.assertTrue(bool(initial_report["rules"]["max_consecutive_speaker_ok"]))

        outcome = attempt_script_quality_repair(
            validated_payload=initial_payload,
            initial_report=initial_report,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/needs_speaker_fix.json",
            client=None,
        )
        self.assertTrue(bool(outcome["report"]["pass"]))
        self.assertFalse(bool(outcome["report"]["repair_attempted"]))
        self.assertEqual(int(outcome["report"]["repair_attempts_used"]), 0)

    def test_attempt_script_quality_repair_handles_missing_client(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            auto_repair=True,
            repair_attempts=1,
        )
        initial_payload = {"lines": _bad_lines()}
        initial_report = evaluate_script_quality(
            validated_payload=initial_payload,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/initial.json",
        )
        outcome = attempt_script_quality_repair(
            validated_payload=initial_payload,
            initial_report=initial_report,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/final.json",
            client=None,
        )
        self.assertFalse(bool(outcome["report"]["pass"]))
        self.assertTrue(bool(outcome["report"]["repair_attempted"]))
        self.assertFalse(bool(outcome["report"]["repair_succeeded"]))

    def test_attempt_script_quality_repair_reverts_on_failed_guardrail(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            auto_repair=True,
            repair_attempts=1,
            repair_revert_on_fail=True,
            repair_min_word_ratio=0.95,
        )
        initial_payload = {"lines": _bad_lines()}
        initial_report = evaluate_script_quality(
            validated_payload=initial_payload,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/initial_guardrail.json",
        )
        repaired_payload = {
            "lines": [
                _line("Ana", "Host1", "Texto corto"),
                _line("Luis", "Host2", "Gracias"),
            ]
        }
        client = _FakeRepairClient(repaired_payload=repaired_payload)
        outcome = attempt_script_quality_repair(
            validated_payload=initial_payload,
            initial_report=initial_report,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/final_guardrail.json",
            client=client,
        )
        self.assertFalse(bool(outcome["repaired"]))
        self.assertFalse(bool(outcome["report"]["repair_succeeded"]))
        self.assertFalse(bool(outcome["report"]["repair_changed_script"]))
        self.assertEqual(outcome["payload"], initial_payload)
        history = list(outcome["report"].get("repair_history", []))
        self.assertTrue(any(str(item.get("status")) == "rejected_guardrail" for item in history))

    def test_attempt_script_quality_repair_can_keep_failed_candidate_when_revert_disabled(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            auto_repair=True,
            repair_attempts=1,
            repair_revert_on_fail=False,
            repair_min_word_ratio=0.0,
        )
        initial_payload = {"lines": _bad_lines()}
        initial_report = evaluate_script_quality(
            validated_payload=initial_payload,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/initial_nonrevert.json",
        )
        repaired_payload = {
            "lines": [
                _line("Ana", "Host1", "Linea alternativa"),
                _line("Luis", "Host2", "Cierre alternativo"),
            ]
        }
        client = _FakeRepairClient(repaired_payload=repaired_payload)
        outcome = attempt_script_quality_repair(
            validated_payload=initial_payload,
            initial_report=initial_report,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/final_nonrevert.json",
            client=client,
        )
        self.assertTrue(bool(outcome["repaired"]))
        self.assertFalse(bool(outcome["report"]["pass"]))
        self.assertFalse(bool(outcome["report"]["repair_succeeded"]))
        self.assertEqual(outcome["payload"], repaired_payload)

    def test_attempt_script_quality_repair_scales_output_budget_for_large_payload(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            max_repeat_line_ratio=0.0,
            auto_repair=True,
            repair_attempts=1,
            repair_revert_on_fail=False,
            repair_min_word_ratio=0.0,
            repair_max_output_tokens=700,
        )
        long_lines = [
            _line("Ana", "Host1", "Dato repetido para forzar reparacion y revisar presupuesto de salida.")
            for _ in range(120)
        ]
        initial_payload = {"lines": long_lines}
        initial_report = evaluate_script_quality(
            validated_payload=initial_payload,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/long_repair_budget.json",
        )
        repaired_payload = {
            "lines": [
                _line("Ana", "Host1", "Introduccion clara con objetivos y contexto."),
                _line("Luis", "Host2", "Desarrollo con evidencia y ejemplos concretos."),
                _line("Ana", "Host1", "En resumen, priorizamos decisiones medibles."),
                _line("Luis", "Host2", "Gracias por escuchar, nos vemos en la proxima."),
            ]
        }
        client = _CapturingRepairClient(repaired_payload=repaired_payload)
        outcome = attempt_script_quality_repair(
            validated_payload=initial_payload,
            initial_report=initial_report,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/long_repair_budget.json",
            client=client,
        )
        self.assertTrue(client.max_output_tokens_seen)
        self.assertGreater(max(client.max_output_tokens_seen), cfg.repair_max_output_tokens)
        self.assertLessEqual(max(client.max_output_tokens_seen), 6400)
        self.assertTrue(bool(outcome["repaired"]))

    def test_attempt_script_quality_repair_timeout_reports_actual_attempts_used(self) -> None:
        cfg = ScriptQualityGateConfig.from_env(profile_name="short")
        cfg = dataclasses.replace(
            cfg,
            evaluator="rules",
            min_words_ratio=0.0,
            max_words_ratio=10.0,
            auto_repair=True,
            repair_attempts=2,
            repair_revert_on_fail=False,
            repair_min_word_ratio=0.0,
        )
        initial_payload = {"lines": _bad_lines()}
        initial_report = evaluate_script_quality(
            validated_payload=initial_payload,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/initial_timeout_attempts.json",
        )
        still_bad_payload = {
            "lines": [
                _line("Ana", "Host1", "Texto corto sin resumen."),
                _line("Luis", "Host2", "Cierre corto."),
            ]
        }
        client = _SlowFailingRepairClient(
            repaired_payload=still_bad_payload,
            sleep_seconds=0.15,
        )
        outcome = attempt_script_quality_repair(
            validated_payload=initial_payload,
            initial_report=initial_report,
            script_cfg=_base_script_cfg(),
            quality_cfg=cfg,
            script_path="/tmp/final_timeout_attempts.json",
            client=client,
            total_timeout_seconds=0.1,
        )
        self.assertEqual(client.repair_calls, 1)
        self.assertTrue(bool(outcome["report"].get("repair_timeout_reached", False)))
        self.assertEqual(int(outcome["report"].get("repair_attempts_used", -1)), 1)


if __name__ == "__main__":
    unittest.main()

