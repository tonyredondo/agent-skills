import dataclasses
import json
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import LoggingConfig, ReliabilityConfig, ScriptConfig  # noqa: E402
from pipeline.errors import (  # noqa: E402
    ERROR_KIND_OPENAI_EMPTY_OUTPUT,
    ERROR_KIND_RESUME_BLOCKED,
    ERROR_KIND_SCRIPT_COMPLETENESS,
    ERROR_KIND_SOURCE_TOO_SHORT,
    ScriptOperationError,
)
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.schema import count_words_from_payload  # noqa: E402
from pipeline.script_generator import ScriptGenerator  # noqa: E402


class FakeScriptClient:
    def __init__(self) -> None:
        self.requests_made = 0
        self.estimated_cost_usd = 0.0
        self._seq = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self._seq += 1
        base_text = (
            f"Bloque {self._seq} con detalles utiles, ejemplos practicos y explicaciones claras "
            f"para mantener una conversacion natural y completa en el episodio."
        )
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": base_text,
                },
                {
                    "speaker": "Lucia",
                    "role": "Host2",
                    "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                    "text": base_text + " Ademas, resolvemos dudas comunes de la audiencia.",
                },
            ]
        }

    def generate_freeform_text(self, *, prompt, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        return "Resumen intermedio con hechos clave y contexto util."


class FakeRepairScriptClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self._repair_seq = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self._repair_seq += 1
        if "schema_repair" in stage:
            return {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": f"Linea reparada numero {self._repair_seq} con detalles utiles para continuar el episodio.",
                    }
                ]
            }
        if stage.startswith("continuation_"):
            return {
                "lines": [
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            f"Bloque adicional {self._repair_seq} con ejemplos concretos, contexto, "
                            "preguntas y respuestas para aumentar el detalle del episodio."
                        ),
                    }
                ]
            }
        # Invalid schema on purpose (missing required fields).
        return {"lines": [{"speaker": "Carlos"}]}


class FakePreSummaryScriptClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.freeform_calls = 0

    def generate_freeform_text(self, *, prompt, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self.freeform_calls += 1
        return f"Resumen reducido {self.freeform_calls} con puntos clave y contexto."


class LowYieldScriptClient(FakeScriptClient):
    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self._seq += 1
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": f"detalle{self._seq}",
                }
            ]
        }


class AlwaysInvalidRepairScriptClient(FakeScriptClient):
    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        return {"lines": [{"speaker": "Carlos"}]}


class CaptureTokensScriptClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_tokens: dict[str, int] = {}

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        if stage.startswith("chunk_"):
            self.chunk_tokens[stage] = int(max_output_tokens)
        text = (
            f"Detalle {stage} suficiente para avanzar con contenido claro, ejemplos, "
            "contexto y continuidad en el guion."
        )
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": text,
                }
            ]
        }


class TruncationRecoveryScriptClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.recovery_calls = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        if stage == "truncation_recovery_1":
            self.recovery_calls += 1
            return {
                "lines": [
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Cerramos con conclusiones practicas, pasos accionables y una recomendacion final "
                            "para que el equipo ejecute cambios sin perder foco en calidad, riesgo y aprendizaje. "
                            "Gracias por escuchar y nos vemos en el siguiente episodio."
                        ),
                    }
                ]
            }
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": (
                        "Hoy revisamos decisiones tecnicas importantes, su impacto en equipos de producto, "
                        "arquitectura y operaciones, con ejemplos reales y tradeoffs que ayudan a priorizar "
                        "mejor en contextos de incertidumbre..."
                    ),
                }
            ]
        }


class FailingTruncationRecoveryScriptClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.recovery_calls = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        if stage == "truncation_recovery_1":
            self.recovery_calls += 1
            import time as _time

            _time.sleep(0.02)
            raise RuntimeError("forced truncation recovery failure")
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": (
                        "Bloque 1 con contexto tecnico, decisiones de arquitectura, latencia, "
                        "coste operativo y tradeoffs que el equipo debe priorizar..."
                    ),
                }
            ]
        }


class EmptyOutputTruncationRecoveryScriptClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.recovery_calls = 0
        self.script_empty_output_events = 0
        self.script_empty_output_retries = 0
        self.script_empty_output_failures = 0
        self.script_empty_output_by_stage: dict[str, int] = {}
        self.script_json_parse_failures_by_kind: dict[str, int] = {}
        self.script_json_parse_failures_by_stage: dict[str, int] = {}
        self.script_json_parse_repair_successes_by_stage: dict[str, int] = {}
        self.script_json_parse_repair_failures_by_stage: dict[str, int] = {}
        self.script_json_parse_repair_successes_by_kind: dict[str, int] = {}
        self.script_json_parse_repair_failures_by_kind: dict[str, int] = {}
        self.script_json_parse_failures = 0
        self.script_requests_made = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self.script_requests_made += 1
        if stage == "truncation_recovery_1":
            self.recovery_calls += 1
            self.script_empty_output_events += 1
            self.script_empty_output_failures += 1
            self.script_empty_output_by_stage[stage] = int(self.script_empty_output_by_stage.get(stage, 0)) + 1
            raise RuntimeError(
                "OpenAI returned empty text for stage=truncation_recovery_1; parse_failure_kind=empty_output"
            )
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": (
                        "Hoy revisamos decisiones tecnicas importantes, su impacto en equipos de producto, "
                        "arquitectura y operaciones, con ejemplos reales y tradeoffs que ayudan a priorizar "
                        "mejor en contextos de incertidumbre..."
                    ),
                }
            ]
        }


class InvalidSchemaTruncationRecoveryScriptClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.recovery_calls = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        if stage == "truncation_recovery_1":
            self.recovery_calls += 1
            raise RuntimeError(
                "Failed to parse JSON output for stage=truncation_recovery_1; parse_failure_kind=malformed"
            )
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": (
                        "Hoy revisamos decisiones tecnicas importantes, su impacto en equipos de producto, "
                        "arquitectura y operaciones, con ejemplos reales y tradeoffs que ayudan a priorizar "
                        "mejor en contextos de incertidumbre..."
                    ),
                }
            ]
        }


class AlwaysEmptyOutputFailureClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.script_empty_output_events = 1
        self.script_empty_output_retries = 2
        self.script_empty_output_failures = 1
        self.script_empty_output_by_stage = {"chunk_1": 1}
        self.script_json_parse_failures = 1
        self.script_json_parse_failures_by_kind = {"empty_output": 1}
        self.script_json_parse_failures_by_stage = {"chunk_1": 1}
        self.script_json_parse_repair_successes_by_stage = {}
        self.script_json_parse_repair_failures_by_stage = {}
        self.script_json_parse_repair_successes_by_kind = {}
        self.script_json_parse_repair_failures_by_kind = {}
        self.script_requests_made = 1

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        raise RuntimeError("OpenAI returned empty text for stage=chunk_1; parse_failure_kind=empty_output")


class HighTruncationPressureClient(CaptureTokensScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.script_json_parse_failures_by_kind = {"truncation": 20}
        self.script_json_parse_failures_by_stage = {"chunk_1": 20}
        self.script_json_parse_repair_successes_by_stage = {}
        self.script_json_parse_repair_failures_by_stage = {}
        self.script_json_parse_repair_successes_by_kind = {}
        self.script_json_parse_repair_failures_by_kind = {}
        self.script_json_parse_failures = 20
        self.script_requests_made = 20
        self.script_empty_output_events = 0
        self.script_empty_output_retries = 0
        self.script_empty_output_failures = 0
        self.script_empty_output_by_stage = {}

class StructuralArtifactsScriptClient(FakeScriptClient):
    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": "Entramos al Bloque 1 con objetivos claros y contexto para el episodio.",
                },
                {
                    "speaker": "Lucia",
                    "role": "Host2",
                    "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                    "text": "Seguimos con Bloque 2 y datos para priorizar decisiones sin perder foco.",
                },
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": "Pasamos al Bloque 4 con una observacion operativa sobre telemetria y...",
                },
                {
                    "speaker": "Lucia",
                    "role": "Host2",
                    "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                    "text": "Gracias por escuchar, nos vemos en el siguiente episodio.",
                },
            ]
        }


class SubsplitRecoveryScriptClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.stages: list[str] = []

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self.stages.append(stage)
        if stage == "chunk_1":
            raise RuntimeError("simulated chunk failure before schema validation")
        if stage in {"chunk_1_subsplit_1", "chunk_1_subsplit_2"}:
            return {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Bloque 1 con ideas practicas sobre arquitectura, observabilidad, "
                            "tolerancia a fallos y calidad de salida."
                        ),
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Bloque 2 con resumen accionable, cierre completo y despedida para evitar "
                            "cortes abruptos en el guion final."
                        ),
                    },
                ]
            }
        return super().generate_script_json(
            prompt=prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
            stage=stage,
        )


class SubsplitPartRecoveryScriptClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.stages: list[str] = []

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self.stages.append(stage)
        if stage == "chunk_1":
            raise RuntimeError("Failed to parse JSON output in initial chunk request")
        if stage == "chunk_1_subsplit_1":
            raise RuntimeError("Failed to parse JSON output in subsplit part")
        if stage in {"chunk_1_subsplit_1_retry_1", "chunk_1_subsplit_2"}:
            return {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Bloque 1 con contexto tecnico, riesgos operativos y decisiones para "
                            "mejorar la estabilidad del pipeline."
                        ),
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Bloque 2 con mitigaciones, validaciones y cierre completo para "
                            "mantener consistencia narrativa."
                        ),
                    },
                ]
            }
        return super().generate_script_json(
            prompt=prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
            stage=stage,
        )


class SchemaSalvageScriptClient(FakeScriptClient):
    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        return {
            "lines": [
                {
                    "name": "Carlos",
                    "content": (
                        "Bloque 1 con contexto tecnico, decisiones y ejemplos aplicables para mantener "
                        "consistencia narrativa durante todo el episodio."
                    ),
                },
                {
                    "name": "Lucia",
                    "content": (
                        "Bloque 2 con recomendaciones accionables, riesgos y cierre operativo para el "
                        "equipo de implementacion."
                    ),
                },
            ]
        }


class AdaptiveSubpartRecoveryClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.stages: list[str] = []
        self.script_json_parse_failures_by_kind = {"truncation": 12}
        self.script_json_parse_failures_by_stage = {"chunk_1": 12}
        self.script_json_parse_repair_successes_by_stage = {}
        self.script_json_parse_repair_failures_by_stage = {}
        self.script_json_parse_repair_successes_by_kind = {}
        self.script_json_parse_repair_failures_by_kind = {}
        self.script_json_parse_failures = 12
        self.script_requests_made = 12
        self.script_empty_output_events = 1
        self.script_empty_output_retries = 0
        self.script_empty_output_failures = 1
        self.script_empty_output_by_stage = {"chunk_1_adaptive_2": 1}

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self.script_requests_made += 1
        self.stages.append(stage)
        if stage == "chunk_1_adaptive_2":
            raise RuntimeError(
                "OpenAI returned empty text for stage=chunk_1_adaptive_2; parse_failure_kind=empty_output"
            )
        if stage == "chunk_1_adaptive_2_retry_1":
            return {
                "lines": [
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Bloque 2 con detalles de resiliencia, recovery y observabilidad para "
                            "evitar abortos en fases tardias."
                        ),
                    }
                ]
            }
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": (
                        "Bloque 1 con contexto suficiente, decisiones tecnicas y acciones practicas "
                        "para que la salida final se mantenga estable."
                    ),
                }
            ]
        }


class PresplitSkipFallbackClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.stages: list[str] = []
        self.script_json_parse_failures_by_kind = {"truncation": 10}
        self.script_json_parse_failures_by_stage = {"chunk_1": 10}
        self.script_json_parse_repair_successes_by_stage = {}
        self.script_json_parse_repair_failures_by_stage = {}
        self.script_json_parse_repair_successes_by_kind = {}
        self.script_json_parse_repair_failures_by_kind = {}
        self.script_json_parse_failures = 10
        self.script_requests_made = 10
        self.script_empty_output_events = 0
        self.script_empty_output_retries = 0
        self.script_empty_output_failures = 0
        self.script_empty_output_by_stage = {}

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self.script_requests_made += 1
        self.stages.append(stage)
        if stage.startswith("chunk_1_adaptive_"):
            self.script_empty_output_events += 1
            self.script_empty_output_failures += 1
            self.script_empty_output_by_stage[stage] = int(self.script_empty_output_by_stage.get(stage, 0)) + 1
            raise RuntimeError(
                f"OpenAI returned empty text for stage={stage}; parse_failure_kind=empty_output"
            )
        if stage == "chunk_1":
            return {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Bloque 1 con contexto tecnico estable, decisiones de arquitectura y "
                            "recomendaciones practicas para avanzar con seguridad."
                        ),
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Bloque 2 con detalles operativos, mitigaciones de riesgo y cierre "
                            "completo del episodio para cumplir objetivos."
                        ),
                    },
                ]
            }
        return super().generate_script_json(
            prompt=prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
            stage=stage,
        )


class TripleSpeakerRunClient(FakeScriptClient):
    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": "Contexto inicial completo con objetivos, restricciones y alcance del episodio.",
                },
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": "Desarrollo de decisiones con tradeoffs tecnicos y acciones recomendadas para implementacion.",
                },
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": "Cierre del analisis con riesgos residuales, mitigaciones y seguimiento operativo.",
                },
                {
                    "speaker": "Lucia",
                    "role": "Host2",
                    "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                    "text": "Gracias por escuchar, nos vemos en el siguiente episodio.",
                },
            ]
        }


class ContinuationRecoveryClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.stages: list[str] = []

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self.stages.append(stage)
        if stage == "continuation_1":
            raise RuntimeError(
                "OpenAI returned empty text for stage=continuation_1; parse_failure_kind=empty_output"
            )
        if stage == "continuation_1_recovery_1":
            return {
                "lines": [
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Agregamos continuidad con ejemplos, preguntas y respuestas para completar "
                            "el objetivo de palabras sin repetir contenido previo."
                        ),
                    }
                ]
            }
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": "Bloque inicial con contexto breve y foco principal.",
                }
            ]
        }


class ContinuationClosureFallbackClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.stages: list[str] = []
        self.script_empty_output_events = 2
        self.script_empty_output_retries = 0
        self.script_empty_output_failures = 2
        self.script_empty_output_by_stage = {"continuation_1": 1, "continuation_1_recovery_1": 1}
        self.script_json_parse_failures_by_kind = {"empty_output": 2}
        self.script_json_parse_failures_by_stage = {
            "continuation_1": 1,
            "continuation_1_recovery_1": 1,
        }
        self.script_json_parse_repair_successes_by_stage = {}
        self.script_json_parse_repair_failures_by_stage = {}
        self.script_json_parse_repair_successes_by_kind = {}
        self.script_json_parse_repair_failures_by_kind = {}
        self.script_json_parse_failures = 2
        self.script_requests_made = 2

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self.script_requests_made += 1
        self.stages.append(stage)
        if stage.startswith("continuation_"):
            raise RuntimeError(
                f"OpenAI returned empty text for stage={stage}; parse_failure_kind=empty_output"
            )
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": (
                        "Bloque inicial con suficiente contexto tecnico y operativo para quedar cerca "
                        "del minimo de palabras requerido por la configuracion."
                    ),
                },
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": (
                        "Profundizamos en decisiones de arquitectura, riesgo y observabilidad para "
                        "permitir cierre deterministico sin LLM adicional."
                    ),
                },
            ]
        }


class ContinuationExtensionFallbackClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.stages: list[str] = []

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self.stages.append(stage)
        if stage.startswith("continuation_"):
            raise RuntimeError(
                f"OpenAI returned empty text for stage={stage}; parse_failure_kind=empty_output"
            )
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": (
                        "Bloque inicial con contexto tecnico detallado, criterios de arquitectura, "
                        "riesgos operativos, estrategia de observabilidad, decisiones de rollout y "
                        "priorizacion de mitigaciones para orientar al equipo sin perder trazabilidad."
                    ),
                },
                {
                    "speaker": "Lucia",
                    "role": "Host2",
                    "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                    "text": (
                        "En resumen, priorizamos cambios incrementales, validacion continua, medicion "
                        "de impacto y aprendizaje iterativo en produccion."
                    ),
                },
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": "Gracias por escuchar, nos vemos en el siguiente episodio.",
                },
            ]
        }


class WholeChunkRetryClient(FakeScriptClient):
    def __init__(self) -> None:
        super().__init__()
        self.stages: list[str] = []

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self.stages.append(stage)
        if stage == "chunk_1":
            raise RuntimeError("OpenAI returned empty text for stage=chunk_1; parse_failure_kind=empty_output")
        if stage == "chunk_1_whole_retry":
            return {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Bloque 1 con analisis de incidentes, causas raiz y mitigaciones para "
                            "mejorar resiliencia operativa."
                        ),
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": (
                            "Bloque 2 con plan de validacion, monitoreo y acciones concretas para "
                            "sostener calidad del pipeline."
                        ),
                    },
                ]
            }
        return super().generate_script_json(
            prompt=prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
            stage=stage,
        )


class ScriptGeneratorIntegrationTests(unittest.TestCase):
    def test_chunk_prompt_uses_different_closing_rules_for_final_chunk(self) -> None:
        cfg = ScriptConfig.from_env(profile_name="standard", target_minutes=15, words_per_min=130, min_words=600, max_words=900)
        reliability = ReliabilityConfig.from_env()
        logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        client = FakeScriptClient()
        gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]

        section_plan = {
            "objective": "Expand details",
            "topic": "seguridad de LLMs",
            "target_words": 250,
        }
        non_final_prompt = gen._build_chunk_prompt(  # noqa: SLF001
            source_chunk="contenido fuente base",
            chunk_idx=1,
            chunk_total=3,
            section_plan=section_plan,
            lines_so_far=[],
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        final_prompt = gen._build_chunk_prompt(  # noqa: SLF001
            source_chunk="contenido fuente final",
            chunk_idx=3,
            chunk_total=3,
            section_plan=section_plan,
            lines_so_far=[],
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        self.assertIn("This is not the final chunk: do not add farewell/closing yet.", non_final_prompt)
        self.assertIn("Do not use explicit section labels", non_final_prompt)
        self.assertIn("Use elegant spoken transitions", non_final_prompt)
        self.assertIn("This is the final chunk: add a coherent ending", final_prompt)
        self.assertIn("do not leave trailing ellipsis", final_prompt)

    def test_continuation_and_truncation_prompts_include_structure_guards(self) -> None:
        cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=120, min_words=80, max_words=120)
        reliability = ReliabilityConfig.from_env()
        logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        client = FakeScriptClient()
        gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
        lines = [
            {
                "speaker": "Ana",
                "role": "Host1",
                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                "text": "Bloque 1 con contexto inicial.",
            }
        ]
        continuation_prompt = gen._build_continuation_prompt(  # noqa: SLF001
            lines_so_far=lines,
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        truncation_prompt = gen._build_truncation_recovery_prompt(  # noqa: SLF001
            lines_so_far=lines,
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        self.assertIn("Do not use explicit section labels", continuation_prompt)
        self.assertIn("Use elegant spoken transitions", continuation_prompt)
        self.assertIn("no trailing ellipsis or dangling connectors", continuation_prompt)
        self.assertIn("Do not use explicit section labels", truncation_prompt)
        self.assertIn("Use elegant spoken transitions", truncation_prompt)
        self.assertIn("no trailing ellipsis or dangling connectors", truncation_prompt)

    def test_extract_source_authors_from_metadata_line(self) -> None:
        cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=120, min_words=80, max_words=120)
        reliability = ReliabilityConfig.from_env()
        logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        client = FakeScriptClient()
        gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
        source_text = (
            "Titulo: Practicas de adopcion.\n"
            "Autores: Ada Lovelace y Alan Turing\n"
            "Contenido con recomendaciones operativas."
        )
        authors = gen._extract_source_authors(source_text)  # noqa: SLF001
        self.assertEqual(authors, ["Ada Lovelace", "Alan Turing"])

    def test_prompts_include_author_reference_guidance_when_detected(self) -> None:
        cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=120, min_words=80, max_words=120)
        reliability = ReliabilityConfig.from_env()
        logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        client = FakeScriptClient()
        gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
        section_plan = {"objective": "Expand", "topic": "IA operativa", "target_words": 120}
        chunk_prompt = gen._build_chunk_prompt(  # noqa: SLF001
            source_chunk="Author: Ada Lovelace\nContenido base para episodio.",
            chunk_idx=1,
            chunk_total=2,
            section_plan=section_plan,
            lines_so_far=[],
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        self.assertIn("Source metadata includes author: Ada Lovelace.", chunk_prompt)
        gen._source_authors_detected = ["Ada Lovelace"]  # noqa: SLF001
        continuation_prompt = gen._build_continuation_prompt(  # noqa: SLF001
            lines_so_far=[],
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        truncation_prompt = gen._build_truncation_recovery_prompt(  # noqa: SLF001
            lines_so_far=[],
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        self.assertIn("Source metadata includes author: Ada Lovelace.", continuation_prompt)
        self.assertIn("Source metadata includes author: Ada Lovelace.", truncation_prompt)

    def test_extract_source_agenda_topics_from_index_header(self) -> None:
        cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=120, min_words=80, max_words=120)
        reliability = ReliabilityConfig.from_env()
        logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        client = FakeScriptClient()
        gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
        source_text = (
            "Titulo: Episodio de estrategia.\n"
            "Indice:\n"
            "1) Contexto actual de adopcion\n"
            "2) Riesgos y control de calidad\n"
            "- Metricas para decidir escala\n"
            "Contenido adicional de desarrollo."
        )
        topics = gen._extract_source_agenda_topics(source_text)  # noqa: SLF001
        self.assertEqual(
            topics,
            [
                "Contexto actual de adopcion",
                "Riesgos y control de calidad",
                "Metricas para decidir escala",
            ],
        )

    def test_source_topic_plan_builds_category_mix_for_outline(self) -> None:
        cfg = ScriptConfig.from_env(profile_name="short", target_minutes=6, words_per_min=120, min_words=180, max_words=260)
        reliability = ReliabilityConfig.from_env()
        logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        client = FakeScriptClient()
        gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
        source_text = """
        - 2026-02-16 09:30 · Science · Quantum sensor drift benchmark (src 1)
        - 2026-02-16 10:15 · Health · New trial protocol for low-dose therapy (src 2)
        - 2026-02-16 11:00 · Business · Cost controls for pilot rollout (src 3)
        - 2026-02-16 11:45 · Science · Diamond calibration update (src 4)
        - 2026-02-16 12:30 · Technology · On-device telemetry filters (src 5)
        """.strip()
        entries = gen._extract_source_index_entries(source_text)  # noqa: SLF001
        self.assertGreaterEqual(len(entries), 4)
        gen._source_index_entries = entries  # noqa: SLF001
        outline = gen._build_outline(  # noqa: SLF001
            chunks=[
                "chunk uno",
                "chunk dos",
                "chunk tres",
                "chunk cuatro",
            ],
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        categories = [str(section.get("category", "")).strip() for section in outline]
        self.assertTrue(all(bool(category) for category in categories))
        self.assertGreaterEqual(len(set(categories)), 3)

    def test_first_chunk_prompt_includes_opening_agenda_guidance(self) -> None:
        cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=120, min_words=80, max_words=120)
        reliability = ReliabilityConfig.from_env()
        logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        client = FakeScriptClient()
        gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
        gen._source_agenda_topics = ["Contexto", "Riesgos", "Metricas"]  # noqa: SLF001
        section_plan = {"objective": "Expand", "topic": "IA operativa", "target_words": 120}
        first_chunk_prompt = gen._build_chunk_prompt(  # noqa: SLF001
            source_chunk="Contenido base para episodio.",
            chunk_idx=1,
            chunk_total=3,
            section_plan=section_plan,
            lines_so_far=[],
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        later_chunk_prompt = gen._build_chunk_prompt(  # noqa: SLF001
            source_chunk="Contenido base para episodio.",
            chunk_idx=2,
            chunk_total=3,
            section_plan=section_plan,
            lines_so_far=[],
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        self.assertIn("include a brief natural roadmap of the episode", first_chunk_prompt)
        self.assertIn("comenzamos con", first_chunk_prompt)
        self.assertIn("Avoid \"two parallel monologues\"", first_chunk_prompt)
        self.assertIn("Include direct host-to-host questions regularly", first_chunk_prompt)
        self.assertIn("must answer explicitly before recap or farewell", first_chunk_prompt)
        self.assertIn("Never pre-announce tension", first_chunk_prompt)
        self.assertIn("Section category hint", first_chunk_prompt)
        self.assertIn("final 3 turns before recap/farewell, do not introduce new questions", first_chunk_prompt)
        self.assertNotIn("include a brief natural roadmap of the episode", later_chunk_prompt)

    def test_looks_truncated_detects_non_spanish_connectors(self) -> None:
        cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=120, min_words=80, max_words=120)
        reliability = ReliabilityConfig.from_env()
        logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        client = FakeScriptClient()
        gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
        truncated = [
            {
                "speaker": "Ana",
                "role": "Host1",
                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                "text": "We should prioritize reliability and",
            }
        ]
        complete = [
            {
                "speaker": "Ana",
                "role": "Host1",
                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                "text": "We should prioritize reliability and observability in each release.",
            }
        ]
        self.assertTrue(bool(gen._looks_truncated(truncated)))  # noqa: SLF001
        self.assertFalse(bool(gen._looks_truncated(complete)))  # noqa: SLF001

    def test_source_validation_matrix_by_duration_and_source_size(self) -> None:
        durations = [2, 5, 15, 30, 45, 60]
        source_ratio_cases = [
            ("very_short", 0.10, "warn"),
            ("medium", 0.35, "ok"),
            ("long", 0.80, "ok"),
        ]
        for minutes in durations:
            for label, ratio, expected_status in source_ratio_cases:
                with self.subTest(minutes=minutes, source_size=label):
                    cfg = ScriptConfig.from_env(
                        profile_name="standard",
                        target_minutes=minutes,
                        words_per_min=130,
                        min_words=max(60, int(minutes * 120)),
                        max_words=max(80, int(minutes * 140)),
                    )
                    cfg = dataclasses.replace(
                        cfg,
                        source_validation_mode="warn",
                        source_validation_warn_ratio=0.30,
                        source_validation_enforce_ratio=0.18,
                    )
                    reliability = ReliabilityConfig.from_env()
                    logger = Logger.create(
                        LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
                    )
                    client = FakeScriptClient()
                    gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
                    target_wc = max(cfg.min_words, int((cfg.min_words + cfg.max_words) / 2))
                    source_wc = max(1, int(round(target_wc * ratio)))
                    details = gen._validate_source_length(  # noqa: SLF001
                        source_word_count=source_wc,
                        min_words=cfg.min_words,
                        max_words=cfg.max_words,
                    )
                    self.assertEqual(details.get("source_validation_status"), expected_status)
                    self.assertEqual(details.get("target_word_range"), [cfg.min_words, cfg.max_words])

    def test_canary_durations_5_15_30_generate_script_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for minutes in (5, 15, 30):
                with self.subTest(target_minutes=minutes):
                    cfg = ScriptConfig.from_env(
                        profile_name="standard",
                        target_minutes=minutes,
                        words_per_min=130,
                        min_words=max(120, int(minutes * 40)),
                        max_words=max(180, int(minutes * 55)),
                    )
                    cfg = dataclasses.replace(
                        cfg,
                        checkpoint_dir=os.path.join(tmp, f"ckpt_{minutes}"),
                        max_continuations_per_chunk=2,
                        no_progress_rounds=10,
                        min_word_delta=1,
                    )
                    reliability = ReliabilityConfig.from_env()
                    logger = Logger.create(
                        LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
                    )
                    client = FakeScriptClient()
                    gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
                    source = ("tema clave con detalles utiles para canary " * max(100, int(minutes * 25))).strip()
                    out_path = os.path.join(tmp, f"episode_canary_{minutes}.json")
                    result = gen.generate(source_text=source, output_path=out_path)
                    self.assertTrue(os.path.exists(result.output_path))
                    with open(result.run_summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    self.assertEqual(summary.get("status"), "completed")
                    self.assertIn("generation", summary.get("phase_seconds", {}))
                    self.assertGreaterEqual(int(summary.get("word_count", 0)), cfg.min_words)

    def test_source_validation_enforce_threshold_boundary_behavior(self) -> None:
        cfg = ScriptConfig.from_env(
            profile_name="standard",
            target_minutes=15,
            words_per_min=130,
            min_words=1800,
            max_words=2100,
        )
        cfg = dataclasses.replace(
            cfg,
            source_validation_mode="enforce",
            source_validation_warn_ratio=0.30,
            source_validation_enforce_ratio=0.18,
        )
        reliability = ReliabilityConfig.from_env()
        logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        client = FakeScriptClient()
        gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
        target_wc = max(cfg.min_words, int((cfg.min_words + cfg.max_words) / 2))

        at_enforce = int(round(target_wc * cfg.source_validation_enforce_ratio))
        details_at = gen._validate_source_length(  # noqa: SLF001
            source_word_count=at_enforce,
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        # Boundary is strict (< enforce), so exact boundary is warning but not blocked.
        self.assertEqual(details_at.get("source_validation_status"), "warn")
        self.assertFalse(bool(details_at.get("source_validation_blocked", False)))

        below_enforce = max(1, at_enforce - 1)
        details_below = gen._validate_source_length(  # noqa: SLF001
            source_word_count=below_enforce,
            min_words=cfg.min_words,
            max_words=cfg.max_words,
        )
        self.assertEqual(details_below.get("source_validation_status"), "warn")
        self.assertTrue(bool(details_below.get("source_validation_blocked", False)))

    def test_generator_with_mock_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=120,
                max_words=180,
            )
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False))
            client = FakeScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]

            # One huge paragraph to ensure chunker can still split it.
            source = ("Tema principal con muchos detalles " * 400).strip()
            out_path = os.path.join(tmp, "episode.json")
            result = gen.generate(source_text=source, output_path=out_path)

            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(result.word_count, 0)
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.loads(f.read())
            self.assertGreaterEqual(count_words_from_payload(payload), 120)
            self.assertGreaterEqual(client.requests_made, 1)

    def test_schema_repair_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                min_words=5,
                max_words=20,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                repair_max_attempts=1,
                min_word_delta=1,
                no_progress_rounds=10,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False))
            client = FakeRepairScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            source = ("tema " * 200).strip()
            out_path = os.path.join(tmp, "episode_repair.json")
            result = gen.generate(source_text=source, output_path=out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(result.line_count, 0)
            self.assertGreaterEqual(client.requests_made, 2)

    def test_resume_force_recovers_from_corrupt_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=80,
                max_words=140,
            )
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]

            source = ("Tema con contenido suficiente para reiniciar y completar el guion " * 80).strip()
            out_path = os.path.join(tmp, "episode_resume.json")
            run_dir = os.path.join(cfg.checkpoint_dir, "episode_resume")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "script_checkpoint.json"), "w", encoding="utf-8") as f:
                f.write("{corrupt")

            result = gen.generate(
                source_text=source,
                output_path=out_path,
                resume=True,
                resume_force=True,
            )
            self.assertTrue(os.path.exists(result.output_path))
            leftovers = [p for p in os.listdir(run_dir) if ".corrupt." in p]
            self.assertTrue(leftovers)

    def test_resume_uses_cached_generation_source_after_pre_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=90,
                max_words=130,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                pre_summary_trigger_words=200,
                pre_summary_chunk_target_minutes=30.0,
                pre_summary_target_words=120,
                pre_summary_max_rounds=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakePreSummaryScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]

            source = ("Tema amplio con detalles " * 600).strip()
            out_path = os.path.join(tmp, "episode_cached_source.json")

            with self.assertRaises(InterruptedError):
                gen.generate(
                    source_text=source,
                    output_path=out_path,
                    cancel_check=lambda: client.freeform_calls > 0,
                )
            first_calls = client.freeform_calls
            self.assertGreater(first_calls, 0)

            result = gen.generate(
                source_text=source,
                output_path=out_path,
                resume=True,
            )
            self.assertTrue(os.path.exists(result.output_path))
            self.assertEqual(client.freeform_calls, first_calls)

    def test_pre_summary_parallel_uses_thread_pool_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=80,
                max_words=140,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                pre_summary_trigger_words=200,
                pre_summary_chunk_target_minutes=2.0,
                pre_summary_target_words=220,
                pre_summary_max_rounds=1,
                pre_summary_parallel=True,
                pre_summary_parallel_workers=3,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakePreSummaryScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]

            source = "\n\n".join([("Tema amplio con muchos detalles " * 80).strip() for _ in range(10)])
            out_path = os.path.join(tmp, "episode_presummary_parallel.json")
            with mock.patch(
                "pipeline.script_generator.ThreadPoolExecutor",
                wraps=ThreadPoolExecutor,
            ) as pool_spy:
                result = gen.generate(source_text=source, output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertTrue(pool_spy.called)
            self.assertGreater(client.freeform_calls, 1)

    def test_pre_summary_sequential_does_not_use_thread_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=80,
                max_words=140,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                pre_summary_trigger_words=200,
                pre_summary_chunk_target_minutes=2.0,
                pre_summary_target_words=220,
                pre_summary_max_rounds=1,
                pre_summary_parallel=False,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakePreSummaryScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]

            source = "\n\n".join([("Tema amplio con muchos detalles " * 80).strip() for _ in range(10)])
            out_path = os.path.join(tmp, "episode_presummary_sequential.json")
            with mock.patch(
                "pipeline.script_generator.ThreadPoolExecutor",
                wraps=ThreadPoolExecutor,
            ) as pool_spy:
                result = gen.generate(source_text=source, output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertFalse(pool_spy.called)
            self.assertGreater(client.freeform_calls, 1)

    def test_resume_blocks_when_generation_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=80,
                max_words=120,
            )
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]

            source = ("Tema principal para pruebas de resume " * 200).strip()
            out_path = os.path.join(tmp, "episode_missing_generation_source.json")
            first = gen.generate(source_text=source, output_path=out_path)
            self.assertTrue(os.path.exists(first.output_path))

            run_dir = os.path.join(cfg.checkpoint_dir, "episode_missing_generation_source")
            ckpt = os.path.join(run_dir, "script_checkpoint.json")
            with open(ckpt, "r", encoding="utf-8") as f:
                state = json.load(f)
            state.pop("generation_source", None)
            state["chunks_done"] = max(1, int(state.get("chunks_done", 1)))
            state["status"] = "running"
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")

            with self.assertRaises(ScriptOperationError) as ctx:
                gen.generate(source_text=source, output_path=out_path, resume=True)
            self.assertEqual(ctx.exception.error_kind, ERROR_KIND_RESUME_BLOCKED)
            run_summary = os.path.join(run_dir, "run_summary.json")
            self.assertTrue(os.path.exists(run_summary))
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_RESUME_BLOCKED)

    def test_resume_migrates_legacy_checkpoint_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=70,
                max_words=110,
            )
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]

            source = ("Contenido base para migrar lineas legacy " * 150).strip()
            out_path = os.path.join(tmp, "episode_legacy_lines.json")
            gen.generate(source_text=source, output_path=out_path)

            run_dir = os.path.join(cfg.checkpoint_dir, "episode_legacy_lines")
            ckpt = os.path.join(run_dir, "script_checkpoint.json")
            with open(ckpt, "r", encoding="utf-8") as f:
                state = json.load(f)
            state["lines"] = [
                {"name": "Ana", "content": "Linea legacy uno"},
                {"name": "Luis", "content": "Linea legacy dos"},
            ]
            state["chunks_done"] = 1
            state["status"] = "running"
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")

            result = gen.generate(source_text=source, output_path=out_path, resume=True)
            self.assertTrue(os.path.exists(result.output_path))
            with open(ckpt, "r", encoding="utf-8") as f:
                migrated = json.load(f)
            self.assertTrue(all("role" in line and "instructions" in line for line in migrated["lines"]))

    def test_generator_fails_when_min_words_not_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                min_words=200,
                max_words=220,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=1,
                no_progress_rounds=10,
                min_word_delta=1,
                min_words=40,
                max_words=120,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = LowYieldScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            source = ("tema " * 200).strip()
            out_path = os.path.join(tmp, "episode_min_words.json")
            with self.assertRaises(ScriptOperationError) as ctx:
                gen.generate(source_text=source, output_path=out_path)
            self.assertIn("below minimum words target", str(ctx.exception).lower())
            self.assertEqual(ctx.exception.error_kind, ERROR_KIND_SCRIPT_COMPLETENESS)
            self.assertFalse(os.path.exists(out_path))
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_min_words", "run_summary.json")
            self.assertTrue(os.path.exists(run_summary))
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_SCRIPT_COMPLETENESS)
            self.assertIn("generation", payload.get("phase_seconds", {}))
            self.assertIn("source_validation_status", payload)
            self.assertIn("target_word_range", payload)

    def test_schema_repair_success_counter_only_increments_on_valid_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                min_words=60,
                max_words=90,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                repair_max_attempts=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = AlwaysInvalidRepairScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            source = ("tema " * 200).strip()
            out_path = os.path.join(tmp, "episode_schema_metrics.json")
            with self.assertRaises(RuntimeError):
                gen.generate(source_text=source, output_path=out_path)
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_schema_metrics", "run_summary.json")
            self.assertTrue(os.path.exists(run_summary))
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(int(payload.get("schema_repair_successes", -1)), 0)
            self.assertGreaterEqual(int(payload.get("schema_repair_failures", 0)), 1)

    def test_initial_chunk_uses_initial_max_output_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=10,
                max_words=80,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                chunk_target_minutes=1.0,
                max_output_tokens_initial=2222,
                max_output_tokens_chunk=1111,
                max_continuations_per_chunk=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = CaptureTokensScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            source = (
                ("uno " * 130).strip()
                + "\n\n"
                + ("dos " * 130).strip()
                + "\n\n"
                + ("tres " * 130).strip()
            )
            out_path = os.path.join(tmp, "episode_tokens.json")
            result = gen.generate(source_text=source, output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertIn("chunk_1", client.chunk_tokens)
            self.assertEqual(client.chunk_tokens["chunk_1"], 2222)
            self.assertIn("chunk_2", client.chunk_tokens)
            self.assertEqual(client.chunk_tokens["chunk_2"], 1111)

    def test_source_validation_enforce_blocks_insufficient_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="standard",
                target_minutes=30,
                words_per_min=130,
                min_words=3000,
                max_words=3400,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                source_validation_mode="enforce",
                source_validation_warn_ratio=0.5,
                source_validation_enforce_ratio=0.4,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_source_validation.json")
            with self.assertRaises(ScriptOperationError) as raised:
                gen.generate(source_text="texto demasiado corto", output_path=out_path)
            self.assertEqual(getattr(raised.exception, "error_kind", ""), ERROR_KIND_SOURCE_TOO_SHORT)
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_source_validation", "run_summary.json")
            self.assertTrue(os.path.exists(run_summary))
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("source_validation_status"), "warn")
            self.assertIn("source_ratio_below_enforce_threshold", str(payload.get("source_validation_reason", "")))
            self.assertEqual(payload.get("target_word_range"), [3000, 3400])
            self.assertTrue(bool(payload.get("source_validation_blocked", False)))
            blocked_message = str(payload.get("source_validation_blocked_message", ""))
            self.assertIn("Provide at least", blocked_message)
            self.assertIn("recommended", blocked_message)
            self.assertLessEqual(
                int(payload.get("source_required_min_words", 0)),
                int(payload.get("source_recommended_min_words", 0)),
            )

    def test_source_validation_warn_mode_continues_and_marks_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=60,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                source_validation_mode="warn",
                source_validation_warn_ratio=0.8,
                source_validation_enforce_ratio=0.2,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_source_warn.json")
            result = gen.generate(source_text="texto base corto " * 8, output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_source_warn", "run_summary.json")
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("source_validation_status"), "warn")
            self.assertIn("source_ratio_below_warn_threshold", str(payload.get("source_validation_reason", "")))
            self.assertIn("generation", payload.get("phase_seconds", {}))
            self.assertGreaterEqual(int(payload.get("source_recommended_min_words", 0)), 120)

    def test_source_validation_off_mode_keeps_status_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=60,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                source_validation_mode="off",
                source_validation_warn_ratio=0.9,
                source_validation_enforce_ratio=0.8,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_source_off.json")
            result = gen.generate(source_text="texto minimo " * 4, output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_source_off", "run_summary.json")
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("source_validation_status"), "ok")
            self.assertEqual(str(payload.get("source_validation_reason", "")), "")

    def test_target_minutes_outside_recommended_range_runs_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="standard",
                target_minutes=70,
                words_per_min=120,
                min_words=20,
                max_words=60,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                source_validation_mode="warn",
                source_validation_warn_ratio=0.2,
                source_validation_enforce_ratio=0.1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_out_of_range.json")
            result = gen.generate(source_text="fuente amplia " * 80, output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_out_of_range", "run_summary.json")
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "completed")
            self.assertEqual(payload.get("source_validation_status"), "ok")
            self.assertGreaterEqual(int(payload.get("source_recommended_min_words", 0)), 120)

    def test_target_minutes_below_recommended_range_runs_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=1.5,
                words_per_min=120,
                min_words=20,
                max_words=60,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                source_validation_mode="warn",
                source_validation_warn_ratio=0.2,
                source_validation_enforce_ratio=0.1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_below_range.json")
            result = gen.generate(source_text="fuente amplia " * 60, output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_below_range", "run_summary.json")
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "completed")
            self.assertEqual(payload.get("source_validation_status"), "ok")

    def test_truncation_recovery_adds_closing_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=10,
                max_words=120,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=1,
                no_progress_rounds=10,
                min_word_delta=1,
                min_words=40,
                max_words=120,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = TruncationRecoveryScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            source = ("tema clave " * 200).strip()
            out_path = os.path.join(tmp, "episode_truncation.json")
            result = gen.generate(source_text=source, output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertEqual(client.recovery_calls, 1)
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_truncation", "run_summary.json")
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertTrue(bool(payload.get("truncation_recovery_triggered", False)))
            self.assertGreaterEqual(int(payload.get("truncation_recovery_added_words", 0)), 1)
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("generation", phase_seconds)
            self.assertGreaterEqual(float(phase_seconds.get("generation", 0.0)), 0.0)
            components_sum = round(
                float(phase_seconds.get("pre_summary", 0.0))
                + float(phase_seconds.get("chunk_generation", 0.0))
                + float(phase_seconds.get("continuations", 0.0))
                + float(phase_seconds.get("truncation_recovery", 0.0))
                + float(phase_seconds.get("postprocess", 0.0)),
                3,
            )
            self.assertEqual(float(phase_seconds.get("generation", 0.0)), components_sum)

    def test_truncation_recovery_failure_tracks_stage_and_partial_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=10,
                max_words=120,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=1,
                no_progress_rounds=10,
                min_word_delta=1,
                min_words=40,
                max_words=120,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FailingTruncationRecoveryScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_truncation_fail.json")
            with self.assertRaises(RuntimeError):
                gen.generate(source_text=("tema clave " * 200).strip(), output_path=out_path)
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_truncation_fail", "run_summary.json")
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "failed")
            self.assertIn("truncation_recovery", payload.get("failed_stage", ""))
            phase_seconds = payload.get("phase_seconds", {})
            self.assertGreater(float(phase_seconds.get("truncation_recovery", 0.0)), 0.0)

    def test_truncation_recovery_empty_output_uses_fallback_and_keeps_script_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=10,
                max_words=120,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=1,
                no_progress_rounds=10,
                min_word_delta=1,
                min_words=40,
                max_words=120,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = EmptyOutputTruncationRecoveryScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_truncation_empty_output.json")
            result = gen.generate(source_text=("tema clave " * 200).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertEqual(client.recovery_calls, 1)
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertTrue(bool(payload.get("truncation_recovery_triggered", False)))
            self.assertTrue(bool(payload.get("truncation_recovery_fallback_used", False)))
            self.assertGreaterEqual(int(payload.get("script_empty_output_events", 0)), 1)

    def test_truncation_recovery_invalid_schema_uses_fallback_and_keeps_script_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=10,
                max_words=120,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=1,
                no_progress_rounds=10,
                min_word_delta=1,
                min_words=40,
                max_words=120,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = InvalidSchemaTruncationRecoveryScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_truncation_invalid_schema.json")
            result = gen.generate(source_text=("tema clave " * 200).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertEqual(client.recovery_calls, 1)
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertTrue(bool(payload.get("truncation_recovery_triggered", False)))
            self.assertTrue(bool(payload.get("truncation_recovery_fallback_used", False)))
            fallback_modes = dict(payload.get("fallback_modes_by_stage", {}))
            stage_modes = list(fallback_modes.get("truncation_recovery_1", []))
            self.assertIn("invalid_schema_preserve_partial", stage_modes)

    def test_unrecoverable_empty_output_maps_specific_failure_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=100,
            )
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = AlwaysEmptyOutputFailureClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_empty_output_failure.json")
            with self.assertRaises(RuntimeError):
                gen.generate(source_text=("tema base " * 120).strip(), output_path=out_path)
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_empty_output_failure", "run_summary.json")
            self.assertTrue(os.path.exists(run_summary))
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_OPENAI_EMPTY_OUTPUT)

    def test_high_truncation_pressure_triggers_adaptive_policy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=80,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                chunk_target_minutes=10.0,
                max_output_tokens_initial=2000,
                max_output_tokens_chunk=1500,
                min_words=5,
                max_words=80,
                max_continuations_per_chunk=0,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = HighTruncationPressureClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_adaptive_pressure.json")
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_TRUNCATION_PRESSURE_ADAPTIVE": "1",
                    "SCRIPT_TRUNCATION_PRESSURE_THRESHOLD": "0.1",
                    "SCRIPT_TRUNCATION_PRESSURE_PRESPLIT_THRESHOLD": "0.2",
                },
                clear=False,
            ):
                result = gen.generate(source_text=("tema " * 400).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertGreaterEqual(int(payload.get("truncation_pressure_adaptive_events", 0)), 1)
            self.assertGreaterEqual(int(payload.get("truncation_pressure_presplit_events", 0)), 1)
            self.assertGreater(float(payload.get("parse_truncation_pressure_peak", 0.0)), 0.0)

    def test_postprocess_hardens_block_numbering_and_tail_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=160,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=0,
                min_word_delta=1,
                min_words=40,
                max_words=160,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = StructuralArtifactsScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_structural_hardening.json")
            result = gen.generate(source_text=("fuente amplia " * 200).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            with open(result.output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            lines = list(payload.get("lines", []))
            text_blob = "\n".join(str(line.get("text", "")) for line in lines)
            self.assertIn("Bloque 3", text_blob)
            self.assertNotIn("Bloque 4", text_blob)
            self.assertNotIn("...", text_blob)

    def test_postprocess_completeness_failure_uses_specific_error_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=160,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=0,
                min_word_delta=1,
                min_words=40,
                max_words=160,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = StructuralArtifactsScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_structural_fail.json")
            with mock.patch(
                "pipeline.script_generator.evaluate_script_completeness",
                side_effect=[
                    {
                        "pass": True,
                        "reasons": [],
                        "truncation_indices": [],
                        "block_sequence": [1, 2, 3],
                    },
                    {
                        "pass": False,
                        "reasons": ["script_contains_truncated_segments"],
                        "truncation_indices": [2],
                        "block_sequence": [1, 2, 3],
                    },
                ],
            ):
                with self.assertRaises(ScriptOperationError) as ctx:
                    gen.generate(source_text=("fuente amplia " * 200).strip(), output_path=out_path)
            self.assertEqual(ctx.exception.error_kind, ERROR_KIND_SCRIPT_COMPLETENESS)
            run_summary = os.path.join(cfg.checkpoint_dir, "episode_structural_fail", "run_summary.json")
            self.assertTrue(os.path.exists(run_summary))
            with open(run_summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_SCRIPT_COMPLETENESS)
            self.assertFalse(bool(payload.get("script_completeness_pass", True)))
            self.assertIn("script_completeness_before_repair", payload)
            self.assertIn("script_completeness_after_repair", payload)
            self.assertIn(
                "script_contains_truncated_segments",
                list(payload.get("script_completeness_reasons", [])),
            )

    def test_generate_accepts_explicit_episode_id_for_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=80,
            )
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "script.json")
            result = gen.generate(
                source_text=("fuente amplia " * 120).strip(),
                output_path=out_path,
                episode_id="episode_explicit",
                run_token="run_test_123",
            )
            self.assertEqual(result.episode_id, "episode_explicit")
            self.assertIn(os.path.join("episode_explicit", "script_checkpoint.json"), result.checkpoint_path)
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("episode_id"), "episode_explicit")
            self.assertEqual(summary.get("run_token"), "run_test_123")

    def test_chunk_subsplit_recovery_unblocks_initial_chunk_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=100,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                no_progress_rounds=5,
                min_word_delta=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = SubsplitRecoveryScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_subsplit.json")
            result = gen.generate(source_text=("tema " * 300).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertIn("chunk_1_subsplit_1", client.stages)
            self.assertIn("chunk_1_subsplit_2", client.stages)
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertGreaterEqual(int(summary.get("chunk_subsplit_recoveries", 0)), 1)
            self.assertIn("script_json_parse_failures_by_stage", summary)

    def test_chunk_whole_retry_handles_unsplittable_chunk_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=80,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                chunk_target_minutes=15.0,
                max_continuations_per_chunk=0,
                no_progress_rounds=5,
                min_word_delta=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = WholeChunkRetryClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_whole_retry.json")
            result = gen.generate(source_text=("tema " * 80).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertIn("chunk_1_whole_retry", client.stages)
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            chunk_modes = list(dict(summary.get("fallback_modes_by_stage", {})).get("chunk_1", []))
            self.assertIn("whole_chunk_retry", chunk_modes)

    def test_chunk_subsplit_uses_adaptive_subpart_recovery_for_part_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=100,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                no_progress_rounds=5,
                min_word_delta=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = SubsplitPartRecoveryScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_subsplit_part_recovery.json")
            result = gen.generate(source_text=("tema " * 300).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertIn("chunk_1_subsplit_1", client.stages)
            self.assertIn("chunk_1_subsplit_1_retry_1", client.stages)
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertGreaterEqual(int(summary.get("chunk_subsplit_recoveries", 0)), 1)
            self.assertGreaterEqual(int(summary.get("adaptive_subpart_failures", 0)), 1)
            self.assertGreaterEqual(int(summary.get("adaptive_subpart_recoveries", 0)), 1)

    def test_schema_salvage_recovers_partial_payload_without_schema_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=20,
                max_words=90,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=0,
                min_word_delta=1,
                min_words=40,
                max_words=90,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = SchemaSalvageScriptClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_schema_salvage.json")
            result = gen.generate(source_text=("fuente amplia " * 160).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertGreaterEqual(int(summary.get("schema_salvage_attempts", 0)), 1)
            self.assertGreaterEqual(int(summary.get("schema_salvage_successes", 0)), 1)

    def test_adaptive_subpart_recovery_avoids_abort_on_presplit_part_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=15,
                max_words=90,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                chunk_target_minutes=12.0,
                max_continuations_per_chunk=0,
                min_word_delta=1,
                min_words=40,
                max_words=90,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = AdaptiveSubpartRecoveryClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_adaptive_subpart.json")
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_TRUNCATION_PRESSURE_ADAPTIVE": "1",
                    "SCRIPT_TRUNCATION_PRESSURE_THRESHOLD": "0.1",
                    "SCRIPT_TRUNCATION_PRESSURE_PRESPLIT_THRESHOLD": "0.2",
                },
                clear=False,
            ):
                result = gen.generate(source_text=("tema " * 420).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertIn("chunk_1_adaptive_2_retry_1", client.stages)
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertGreaterEqual(int(summary.get("adaptive_subpart_failures", 0)), 1)
            self.assertGreaterEqual(int(summary.get("adaptive_subpart_recoveries", 0)), 1)

    def test_adaptive_presplit_empty_uses_regular_chunk_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=30,
                max_words=90,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                chunk_target_minutes=12.0,
                max_continuations_per_chunk=0,
                min_word_delta=1,
                min_words=30,
                max_words=90,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = PresplitSkipFallbackClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_adaptive_presplit_fallback.json")
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_TRUNCATION_PRESSURE_ADAPTIVE": "1",
                    "SCRIPT_TRUNCATION_PRESSURE_THRESHOLD": "0.1",
                    "SCRIPT_TRUNCATION_PRESSURE_PRESPLIT_THRESHOLD": "0.2",
                    "SCRIPT_ADAPTIVE_SUBPART_ALLOW_SKIP": "1",
                },
                clear=False,
            ):
                result = gen.generate(source_text=("tema " * 420).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertIn("chunk_1_adaptive_1", client.stages)
            self.assertIn("chunk_1_adaptive_2", client.stages)
            self.assertIn("chunk_1", client.stages)
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            chunk_modes = list(dict(summary.get("fallback_modes_by_stage", {})).get("chunk_1", []))
            self.assertIn("adaptive_presplit_empty_fallback", chunk_modes)

    def test_generator_postprocess_respects_configured_max_consecutive_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=30,
                max_words=90,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=0,
                min_word_delta=1,
                min_words=30,
                max_words=90,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = TripleSpeakerRunClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_speaker_run_limit.json")
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_MAX_CONSECUTIVE_SAME_SPEAKER": "3",
                    "SCRIPT_QUALITY_GATE_PROFILE": "default",
                },
                clear=False,
            ):
                result = gen.generate(source_text=("tema base " * 180).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            with open(result.output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            lines = list(payload.get("lines", []))
            self.assertGreaterEqual(len(lines), 3)
            self.assertEqual(lines[0].get("role"), "Host1")
            self.assertEqual(lines[1].get("role"), "Host1")
            self.assertEqual(lines[2].get("role"), "Host1")

    def test_continuation_recovery_succeeds_after_empty_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=45,
                max_words=120,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=2,
                no_progress_rounds=6,
                min_word_delta=1,
                min_words=40,
                max_words=120,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = ContinuationRecoveryClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_continuation_recovery.json")
            result = gen.generate(source_text=("tema base " * 180).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertIn("continuation_1_recovery_1", client.stages)
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertGreaterEqual(int(summary.get("continuation_recovery_attempts", 0)), 1)
            self.assertGreaterEqual(int(summary.get("continuation_recovery_successes", 0)), 1)

    def test_continuation_closure_fallback_preserves_run_near_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=55,
                max_words=120,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=2,
                no_progress_rounds=6,
                min_word_delta=1,
                min_words=45,
                max_words=120,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = ContinuationClosureFallbackClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_continuation_closure.json")
            with mock.patch.dict(
                os.environ,
                {"SCRIPT_CONTINUATION_FALLBACK_MIN_RATIO": "0.5"},
                clear=False,
            ):
                result = gen.generate(source_text=("tema base " * 200).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertGreaterEqual(int(summary.get("continuation_recovery_attempts", 0)), 1)
            self.assertGreaterEqual(int(summary.get("continuation_fallback_closures", 0)), 1)

    def test_continuation_extension_fallback_handles_already_closed_near_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=120,
                min_words=70,
                max_words=140,
            )
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "ckpt"),
                max_continuations_per_chunk=2,
                no_progress_rounds=6,
                min_word_delta=1,
                min_words=70,
                max_words=140,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = ContinuationExtensionFallbackClient()
            gen = ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            out_path = os.path.join(tmp, "episode_continuation_extension.json")
            result = gen.generate(source_text=("tema base " * 220).strip(), output_path=out_path)
            self.assertTrue(os.path.exists(result.output_path))
            with open(result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertGreaterEqual(int(summary.get("continuation_recovery_attempts", 0)), 1)
            self.assertGreaterEqual(int(summary.get("continuation_fallback_extensions", 0)), 1)
            stage_modes = list(dict(summary.get("fallback_modes_by_stage", {})).get("continuation_1", []))
            self.assertIn("continuation_extension_fallback", stage_modes)


if __name__ == "__main__":
    unittest.main()

