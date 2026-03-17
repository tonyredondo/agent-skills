import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import LoggingConfig  # noqa: E402
from pipeline.editorial_gate import EditorialGate  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.podcast_artifacts import build_script_artifact  # noqa: E402


def _line(speaker: str, role: str, text: str) -> dict[str, str]:
    return {
        "speaker": speaker,
        "role": role,
        "instructions": "Warm, clear, conversational tone. Keep pacing measured.",
        "pace_hint": "steady",
        "text": text,
    }


def _plan() -> dict:
    return {
        "artifact_version": 1,
        "episode_id": "episode_test",
        "run_token": "run_test",
        "opening_mode": "concrete_tension",
        "closing_mode": "earned_synthesis",
        "host_roles": {
            "Host1": "sintetiza_y_ordena",
            "Host2": "desafia_y_aterriza",
        },
        "beats": [
            {
                "beat_id": "beat_01",
                "goal": "hook_and_frame",
                "topic_ids": ["topic_01"],
                "claim_ids": ["claim_01"],
                "required_move": "objection",
                "optional_moves": ["example"],
                "must_cover": ["claim_01"],
                "can_cut": False,
                "target_words": 60,
            },
            {
                "beat_id": "beat_02",
                "goal": "concrete_example",
                "topic_ids": ["topic_02"],
                "claim_ids": ["claim_02"],
                "required_move": "grounding",
                "optional_moves": ["tradeoff"],
                "must_cover": ["claim_02"],
                "can_cut": False,
                "target_words": 60,
            },
            {
                "beat_id": "beat_03",
                "goal": "closing",
                "topic_ids": ["topic_03"],
                "claim_ids": ["claim_03"],
                "required_move": "decision",
                "optional_moves": ["consequence"],
                "must_cover": ["claim_03"],
                "can_cut": False,
                "target_words": 55,
            },
        ],
    }


class EditorialGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = Logger.create(LoggingConfig.from_env())
        self.plan = _plan()

    def test_evaluate_fails_for_repeated_closing_tail(self) -> None:
        lines = [
            _line("Ana", "Host1", "Abrimos con una regla simple: automatizar ayuda si el criterio sigue visible para quien opera."),
            _line("Luis", "Host2", "Vale, pero aterrizalo con un caso donde la regla parezca ahorrar tiempo y luego complique el trabajo real."),
            _line("Ana", "Host1", "Ese coste aparece cuando la excepcion valida termina escondida dentro de una cola de tickets y nadie ve el criterio."),
            _line("Luis", "Host2", "Y ahi la friccion no es teorica: el equipo rehecha contexto y retrabaja decisiones que parecian cerradas."),
            _line("Ana", "Host1", "En resumen, la clave es automatizar lo repetitivo y dejar visible el criterio cuando llega una excepcion."),
            _line("Luis", "Host2", "Al final, la clave es exactamente esa: menos magia, menos friccion y mas criterio operativo visible."),
        ]
        artifact = build_script_artifact(
            stage="final",
            episode_id="episode_test",
            run_token="run_test",
            source_digest="src_digest",
            plan_ref="episode_plan.json",
            plan_digest="plan_digest",
            lines=lines,
            episode_plan=self.plan,
            target_word_count=175,
        )
        report = EditorialGate(client=None, logger=self.logger).evaluate(
            script_artifact=artifact,
            script_lines=list(artifact.get("lines", [])),
            episode_plan=self.plan,
            evidence_map={},
            profile_name="short",
            min_words=40,
            max_words=180,
        )
        failure_types = {item["failure_type"] for item in report["failures"]}
        repeated_tail = next(item for item in report["failures"] if item["failure_type"] == "repeated_closing_tail")
        self.assertIn("repeated_closing_tail", failure_types)
        self.assertIn(5, repeated_tail["line_indexes"])
        self.assertIn("beat_03", repeated_tail["beat_ids"])

    def test_evaluate_fails_for_dense_turns_with_specific_span(self) -> None:
        lines = [
            _line("Ana", "Host1", "Abrimos con una idea concreta: automatizar solo sirve si no vuelve opaco el criterio que decide cada caso."),
            _line(
                "Luis",
                "Host2",
                "Vale, pero bajemos eso a tierra con un ejemplo, con el coste operativo, con la excepcion que rompe la regla, con la persona que pierde contexto, y con la decision que nadie sabe explicar cuando el sistema responde algo inesperado.",
            ),
            _line(
                "Ana",
                "Host1",
                "Cuando eso pasa, la automatizacion parece ordenada por fuera, pero por dentro multiplica tickets, rehace contexto, reparte dudas entre personas distintas, y convierte una decision corta en una explicacion larga que nadie queria dar varias veces por semana.",
            ),
            _line("Luis", "Host2", "Entonces la decision sana es hacer visible el criterio y medir si la regla realmente reduce friccion con un ejemplo revisable."),
            _line("Ana", "Host1", "Nos quedamos con una idea util: automatizar si, pero sin esconder el juicio operativo."),
            _line("Luis", "Host2", "Y con eso cerramos dejando un criterio verificable para el equipo."),
        ]
        artifact = build_script_artifact(
            stage="final",
            episode_id="episode_test",
            run_token="run_test",
            source_digest="src_digest",
            plan_ref="episode_plan.json",
            plan_digest="plan_digest",
            lines=lines,
            episode_plan=self.plan,
            target_word_count=175,
        )
        report = EditorialGate(client=None, logger=self.logger).evaluate(
            script_artifact=artifact,
            script_lines=list(artifact.get("lines", [])),
            episode_plan=self.plan,
            evidence_map={},
            profile_name="short",
            min_words=40,
            max_words=180,
        )
        dense_failure = next(item for item in report["failures"] if item["failure_type"] == "dense_turns")
        self.assertIn(1, dense_failure["line_indexes"])
        self.assertIn(2, dense_failure["line_indexes"])


if __name__ == "__main__":
    unittest.main()
