import dataclasses
import json
import os
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import LoggingConfig, ReliabilityConfig, ScriptConfig  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.script_generator import ScriptGenerator  # noqa: E402


class _RedesignClient:
    def __init__(self) -> None:
        self.requests_made = 0
        self.estimated_cost_usd = 0.0
        self.script_retries_total = 0
        self.script_json_parse_failures = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001, ARG002
        self.requests_made += 1
        if stage.startswith('evidence_map_segment_'):
            return {
                'segment_summary': 'El material muestra que automatizar sin criterio puede generar friccion operativa.',
                'claims': [
                    {
                        'statement': 'Automatizar procesos repetitivos ahorra tiempo.',
                        'kind': 'fact',
                        'topic_hint': 'automatizacion util',
                        'support': 'direct',
                        'confidence': 0.93,
                    },
                    {
                        'statement': 'Reglas opacas pueden generar tickets y excepciones.',
                        'kind': 'tension',
                        'topic_hint': 'coste oculto',
                        'support': 'direct',
                        'confidence': 0.9,
                    },
                ],
            }
        if stage == 'evidence_map_global_thesis':
            return {
                'global_thesis': 'La automatizacion buena reduce trabajo repetitivo sin esconder criterio ni costes.'
            }
        if stage == 'episode_planner':
            return {
                'opening_mode': 'concrete_tension',
                'closing_mode': 'earned_synthesis',
                'host_roles': {
                    'Host1': 'sintetiza_y_ordena',
                    'Host2': 'desafia_y_aterriza',
                },
                'beats': [
                    {
                        'beat_id': 'beat_01',
                        'goal': 'hook_and_frame',
                        'topic_ids': ['automatizacion_util'],
                        'claim_ids': ['claim_001'],
                        'required_move': 'objection',
                        'optional_moves': ['example'],
                        'must_cover': ['claim_001'],
                        'can_cut': False,
                        'target_words': 70,
                    },
                    {
                        'beat_id': 'beat_02',
                        'goal': 'concrete_example',
                        'topic_ids': ['coste_oculto'],
                        'claim_ids': ['claim_002'],
                        'required_move': 'grounding',
                        'optional_moves': ['tradeoff'],
                        'must_cover': ['claim_002'],
                        'can_cut': True,
                        'target_words': 65,
                    },
                    {
                        'beat_id': 'beat_03',
                        'goal': 'closing',
                        'topic_ids': ['coste_oculto'],
                        'claim_ids': ['claim_002'],
                        'required_move': 'decision',
                        'optional_moves': ['consequence'],
                        'must_cover': ['claim_002'],
                        'can_cut': False,
                        'target_words': 55,
                    },
                ],
            }
        if stage == 'dialogue_drafter':
            return {
                'lines': [
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'Hoy vamos a una tension simple: automatizar puede ahorrar tiempo o esconder problemas.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'steady',
                        'text': 'Vale, bajemos eso a tierra: cuando ayuda de verdad y cuando solo tapa el trabajo que sigue ahi.',
                    },
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'Ayuda cuando quita repeticion visible, pero empeora si convierte excepciones normales en tickets dificiles de entender.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'steady',
                        'text': 'Ese es el coste oculto: menos trabajo manual arriba y mas friccion rara abajo.',
                    },
                ]
            }
        if stage.startswith('fact_guard_'):
            return {'pass': True, 'issues': []}
        if stage == 'editorial_gate_eval':
            return {
                'scores': {
                    'orality': 4.2,
                    'host_distinction': 4.1,
                    'progression': 4.1,
                    'freshness': 4.0,
                    'listener_engagement': 4.0,
                    'density_control': 4.1,
                },
                'reasons': [],
            }
        if stage.startswith('editorial_rewriter_'):
            return {
                'lines': [
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'Automatizar bien no es meter mas reglas, es quitar trabajo repetitivo sin volver opaco el criterio para quien opera.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'steady',
                        'text': 'Vale, pero dame el borde incomodo: que pasa cuando un caso raro rompe la regla y nadie entiende por que salto.',
                    },
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'Ahi aparece el coste oculto: ahorras minutos arriba, pero si el criterio queda escondido multiplicas tickets y excepciones abajo.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'brisk',
                        'text': 'Y eso en la practica se traduce en mas friccion, mas contexto perdido y mas tiempo persiguiendo una decision que parecia automatica.',
                    },
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'La conclusion util no es automatizar menos, sino automatizar solo lo repetitivo y dejar visible el criterio cuando llega una excepcion.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'calm',
                        'text': 'Claro, porque si el equipo puede ver el coste y el por que de la regla, la automatizacion deja de sonar magica y vuelve a ser una herramienta util.',
                    },
                ]
            }
        raise AssertionError(f'unexpected stage: {stage}')


class RedesignedScriptGeneratorIntegrationTests(unittest.TestCase):
    def _generator(self, *, checkpoint_dir: str, client: _RedesignClient, profile_name: str = 'short') -> ScriptGenerator:
        cfg = ScriptConfig.from_env(profile_name=profile_name)
        cfg = dataclasses.replace(cfg, checkpoint_dir=checkpoint_dir, min_words=40, max_words=180)
        reliability = ReliabilityConfig.from_env()
        logger = Logger.create(LoggingConfig.from_env())
        return ScriptGenerator(config=cfg, reliability=reliability, logger=logger, client=client)

    def test_generator_writes_redesigned_artifacts_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _RedesignClient()
            generator = self._generator(checkpoint_dir=tmpdir, client=client)
            output_path = os.path.join(tmpdir, 'episode.json')
            result = generator.generate(source_text=('tema base ' * 180).strip(), output_path=output_path)
            self.assertTrue(os.path.exists(result.output_path))
            self.assertTrue(os.path.exists(result.quality_report_path))
            self.assertTrue(os.path.exists(result.artifact_paths['evidence_map']))
            self.assertTrue(os.path.exists(result.artifact_paths['episode_plan']))
            self.assertTrue(os.path.exists(result.artifact_paths['draft_script_raw']))
            self.assertTrue(os.path.exists(result.artifact_paths['draft_script']))
            self.assertTrue(os.path.exists(result.artifact_paths['draft_script_fact_checked']))
            self.assertTrue(os.path.exists(result.artifact_paths['rewritten_script']))
            self.assertTrue(os.path.exists(result.artifact_paths['rewritten_script_final']))
            with open(result.output_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            self.assertIn('lines', payload)
            self.assertGreaterEqual(len(payload['lines']), 2)

    def test_generator_persists_episode_id_and_run_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _RedesignClient()
            generator = self._generator(checkpoint_dir=tmpdir, client=client)
            output_path = os.path.join(tmpdir, 'custom_episode.json')
            result = generator.generate(
                source_text=('tema base ' * 180).strip(),
                output_path=output_path,
                episode_id='episode_custom',
                run_token='run_123',
            )
            with open(result.run_summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            self.assertEqual(summary['episode_id'], 'episode_custom')
            self.assertEqual(summary['run_token'], 'run_123')

    def test_resume_reuses_completed_artifacts_without_new_llm_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _RedesignClient()
            generator = self._generator(checkpoint_dir=tmpdir, client=client)
            output_path = os.path.join(tmpdir, 'resume_episode.json')
            generator.generate(
                source_text=('tema base ' * 180).strip(),
                output_path=output_path,
                episode_id='resume_episode',
            )
            first_call_count = client.requests_made
            generator.generate(
                source_text=('tema base ' * 180).strip(),
                output_path=output_path,
                episode_id='resume_episode',
                resume=True,
            )
            self.assertEqual(client.requests_made, first_call_count)

    def test_run_summary_exposes_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _RedesignClient()
            generator = self._generator(checkpoint_dir=tmpdir, client=client)
            output_path = os.path.join(tmpdir, 'summary_episode.json')
            result = generator.generate(source_text=('tema base ' * 180).strip(), output_path=output_path)
            with open(result.run_summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            self.assertIn('artifact_paths', summary)
            self.assertEqual(summary['quality_report_path'], result.quality_report_path)
            self.assertEqual(summary['script_quality_report_path'], result.quality_report_path)


if __name__ == '__main__':
    unittest.main()
