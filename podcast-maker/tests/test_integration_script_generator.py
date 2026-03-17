import dataclasses
import json
import os
import re
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import LoggingConfig, ReliabilityConfig, ScriptConfig  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.podcast_artifacts import build_script_artifact  # noqa: E402
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


class _UnderlengthExpansionClient(_RedesignClient):
    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001, ARG002
        if stage.startswith('editorial_rewriter_'):
            self.requests_made += 1
            return {
                'lines': [
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'Automatizar sirve cuando el criterio sigue visible para quien opera.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'steady',
                        'text': 'Vale, dame el borde donde eso deja de cumplirse.',
                    },
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'Se rompe cuando la regla tapa excepciones.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'steady',
                        'text': 'Y eso dispara friccion operativa.',
                    },
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'La salida es automatizar lo repetitivo y dejar criterio visible.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'calm',
                        'text': 'Asi la regla ayuda sin vender magia.',
                    },
                ]
            }
        if stage == 'editorial_underlength_contextual_beat_expand':
            self.requests_made += 1
            return {
                'lines': [
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'En un equipo real eso se nota cuando una excepcion legitima acaba en una cola de tickets porque nadie ve el criterio que la regla estaba usando.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'brisk',
                        'text': 'Y ahi el coste no es filosofico: son minutos perdidos, contexto rehecho y mas gente preguntando a quien le toca decidir ese caso raro.',
                    },
                ]
            }
        return super().generate_script_json(
            prompt=prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
            stage=stage,
        )


class _FinalFactIssueClient(_RedesignClient):
    def __init__(self) -> None:
        super().__init__()
        self.final_fact_calls = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001, ARG002
        if stage == 'fact_guard_final':
            self.requests_made += 1
            self.final_fact_calls += 1
            return {
                'pass': False,
                'issues': [
                    {
                        'issue_id': 'issue_001',
                        'issue_type': 'overstated_causality',
                        'severity': 'medium',
                        'claim_id': 'claim_002',
                        'line_indexes': [4],
                        'source_refs': ['segment_001'],
                        'origin_stage': 'final',
                        'action': 'rewrite_local',
                    }
                ],
            }
        if stage == 'fact_guard_repair_final':
            self.requests_made += 1
            return {'patches': []}
        return super().generate_script_json(
            prompt=prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
            stage=stage,
        )


class _EscalatingFinalRepairClient(_RedesignClient):
    def __init__(self) -> None:
        super().__init__()
        self.final_fact_calls = 0
        self.repair_calls = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001, ARG002
        if stage == 'fact_guard_final':
            self.requests_made += 1
            self.final_fact_calls += 1
            if self.final_fact_calls == 1:
                return {
                    'pass': False,
                    'issues': [
                        {
                            'issue_id': 'issue_001',
                            'issue_type': 'unsupported_claim',
                            'severity': 'medium',
                            'claim_id': 'claim_002',
                            'line_indexes': [0],
                            'source_refs': ['segment_001'],
                            'origin_stage': 'final',
                            'action': 'rewrite_local',
                        }
                    ],
                }
            return {
                'pass': False,
                'issues': [
                    {
                        'issue_id': 'issue_002',
                        'issue_type': 'overstated_causality',
                        'severity': 'high',
                        'claim_id': 'claim_002',
                        'line_indexes': [0],
                        'source_refs': ['segment_001'],
                        'origin_stage': 'final',
                        'action': 'block',
                    },
                    {
                        'issue_id': 'issue_003',
                        'issue_type': 'unsupported_claim',
                        'severity': 'medium',
                        'claim_id': 'claim_002',
                        'line_indexes': [1],
                        'source_refs': ['segment_001'],
                        'origin_stage': 'final',
                        'action': 'rewrite_local',
                    },
                ],
            }
        if stage == 'fact_guard_repair_final':
            self.requests_made += 1
            self.repair_calls += 1
            match = re.search(r'"line_id":\s*"([^"]+)"', prompt)
            line_id = match.group(1) if match else ''
            return {
                'patches': [
                    {
                        'op': 'replace_line',
                        'line_id': line_id,
                        'anchor_line_id': None,
                        'line': {
                            'speaker': 'Ana',
                            'role': 'Host1',
                            'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                            'pace_hint': 'steady',
                            'text': 'Automatizar bien quita repeticion visible sin esconder el criterio operativo.',
                        },
                    }
                ]
            }
        return super().generate_script_json(
            prompt=prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
            stage=stage,
        )


class RedesignedScriptGeneratorIntegrationTests(unittest.TestCase):
    def _generator(
        self,
        *,
        checkpoint_dir: str,
        client: _RedesignClient,
        profile_name: str = 'short',
        min_words: int = 40,
        max_words: int = 180,
    ) -> ScriptGenerator:
        cfg = ScriptConfig.from_env(profile_name=profile_name)
        cfg = dataclasses.replace(cfg, checkpoint_dir=checkpoint_dir, min_words=min_words, max_words=max_words)
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
            self.assertTrue(os.path.exists(result.artifact_paths['final_evaluated_pre_fact_repair']))
            self.assertTrue(os.path.exists(result.artifact_paths['final_evaluated_post_fact_repair']))
            with open(result.output_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            with open(result.artifact_paths['editorial_report'], 'r', encoding='utf-8') as f:
                editorial_report = json.load(f)
            with open(result.artifact_paths['fact_guard_report_final'], 'r', encoding='utf-8') as f:
                fact_guard_report = json.load(f)
            with open(result.artifact_paths['final_evaluated_post_fact_repair'], 'r', encoding='utf-8') as f:
                final_snapshot = json.load(f)
            with open(result.quality_report_path, 'r', encoding='utf-8') as f:
                quality_report = json.load(f)
            self.assertIn('lines', payload)
            self.assertGreaterEqual(len(payload['lines']), 2)
            self.assertEqual(editorial_report['stage'], 'final')
            self.assertEqual(editorial_report['internal_artifact_digest'], quality_report['internal_artifact_digest'])
            self.assertEqual(quality_report['editorial_report_path'], result.artifact_paths['editorial_report'])
            self.assertEqual(quality_report['final_evaluated_pre_fact_repair_path'], result.artifact_paths['final_evaluated_pre_fact_repair'])
            self.assertEqual(quality_report['final_evaluated_post_fact_repair_path'], result.artifact_paths['final_evaluated_post_fact_repair'])
            self.assertEqual(fact_guard_report['internal_artifact_digest'], final_snapshot['internal_artifact_digest'])
            self.assertEqual(quality_report['internal_artifact_digest'], final_snapshot['internal_artifact_digest'])

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

    def test_underlength_recovery_targets_and_inserts_inside_beat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _RedesignClient()
            generator = self._generator(checkpoint_dir=tmpdir, client=client)
            episode_plan = client.generate_script_json(
                prompt='',
                schema={},
                max_output_tokens=0,
                stage='episode_planner',
            )
            rewritten_artifact = build_script_artifact(
                stage='rewritten',
                episode_id='underlength_episode',
                run_token='run_test',
                source_digest='src_digest',
                plan_ref='episode_plan.json',
                plan_digest='plan_digest',
                lines=[
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'Automatizar sirve cuando el criterio sigue visible para quien opera y nadie pierde contexto.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'steady',
                        'text': 'Vale, dame el borde donde eso deja de cumplirse con un caso concreto.',
                    },
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'Se rompe cuando la regla tapa excepciones.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'steady',
                        'text': 'Y eso dispara friccion operativa.',
                    },
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'La salida es automatizar lo repetitivo y dejar criterio visible.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'calm',
                        'text': 'Asi la regla ayuda sin vender magia.',
                    },
                ],
                episode_plan=episode_plan,
                target_word_count=190,
            )
            target_beat_id = generator._select_underlength_expansion_beat(
                script_artifact=rewritten_artifact,
                episode_plan=episode_plan,
            )
            self.assertEqual(target_beat_id, 'beat_02')
            expanded_artifact = generator._insert_lines_into_beat(
                script_artifact=rewritten_artifact,
                episode_plan=episode_plan,
                beat_id=target_beat_id,
                new_lines=[
                    {
                        'speaker': 'Ana',
                        'role': 'Host1',
                        'instructions': 'Warm, clear, conversational tone. Keep pacing measured.',
                        'pace_hint': 'steady',
                        'text': 'En un equipo real eso se nota cuando una excepcion legitima acaba en una cola de tickets porque nadie ve el criterio que la regla estaba usando.',
                    },
                    {
                        'speaker': 'Luis',
                        'role': 'Host2',
                        'instructions': 'Curious, grounded tone. Ask for concrete examples.',
                        'pace_hint': 'brisk',
                        'text': 'Y ahi el coste no es filosofico: son minutos perdidos, contexto rehecho y mas gente preguntando a quien le toca decidir ese caso raro.',
                    },
                ],
            )
            turns = list(expanded_artifact.get('turns', []))
            inserted_idx = next(
                idx for idx, turn in enumerate(turns)
                if 'equipo real' in str(turn.get('text', '')).lower()
            )
            first_closing_idx = next(
                idx for idx, turn in enumerate(turns)
                if str(turn.get('beat_id', '')) == 'beat_03'
            )
            self.assertEqual(turns[inserted_idx]['beat_id'], 'beat_02')
            self.assertLess(inserted_idx, first_closing_idx)

    def test_final_unresolved_factual_issue_blocks_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _FinalFactIssueClient()
            generator = self._generator(checkpoint_dir=tmpdir, client=client)
            output_path = os.path.join(tmpdir, 'fact_blocked_episode.json')
            with self.assertRaises(Exception):
                generator.generate(
                    source_text=('tema base ' * 180).strip(),
                    output_path=output_path,
                    episode_id='fact_blocked_episode',
                )
            run_summary_path = os.path.join(tmpdir, 'fact_blocked_episode', 'run_summary.json')
            quality_report_path = os.path.join(tmpdir, 'fact_blocked_episode', 'quality_report.json')
            with open(run_summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            with open(quality_report_path, 'r', encoding='utf-8') as f:
                quality_report = json.load(f)
            self.assertEqual(summary['status'], 'failed')
            self.assertFalse(quality_report['pass'])
            self.assertTrue(quality_report['fact_guard_blocking'])
            self.assertEqual(client.final_fact_calls, 1)

    def test_final_fact_repair_rejects_escalation_and_keeps_original_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _EscalatingFinalRepairClient()
            generator = self._generator(checkpoint_dir=tmpdir, client=client)
            output_path = os.path.join(tmpdir, 'escalating_fact_episode.json')
            with self.assertRaises(Exception):
                generator.generate(
                    source_text=('tema base ' * 180).strip(),
                    output_path=output_path,
                    episode_id='escalating_fact_episode',
                )
            run_dir = os.path.join(tmpdir, 'escalating_fact_episode')
            with open(os.path.join(run_dir, 'fact_guard_report_final.json'), 'r', encoding='utf-8') as f:
                fact_guard_report = json.load(f)
            with open(os.path.join(run_dir, 'quality_report.json'), 'r', encoding='utf-8') as f:
                quality_report = json.load(f)
            with open(os.path.join(run_dir, 'final_evaluated_pre_fact_repair.json'), 'r', encoding='utf-8') as f:
                pre_snapshot = json.load(f)
            with open(os.path.join(run_dir, 'final_evaluated_post_fact_repair.json'), 'r', encoding='utf-8') as f:
                post_snapshot = json.load(f)
            self.assertEqual(client.repair_calls, 1)
            self.assertEqual(client.final_fact_calls, 2)
            self.assertEqual(len(fact_guard_report['issues']), 1)
            self.assertEqual(fact_guard_report['issues'][0]['action'], 'rewrite_local')
            self.assertEqual(pre_snapshot['internal_artifact_digest'], post_snapshot['internal_artifact_digest'])
            self.assertEqual(quality_report['internal_artifact_digest'], post_snapshot['internal_artifact_digest'])


if __name__ == '__main__':
    unittest.main()
