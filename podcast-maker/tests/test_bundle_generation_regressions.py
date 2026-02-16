import argparse
import dataclasses
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import make_script  # noqa: E402


class _FakeClient:
    requests_made = 0
    script_requests_made = 0
    script_retries_total = 0
    estimated_cost_usd = 0.0
    script_json_parse_failures = 0
    script_empty_output_failures = 0


class BundleGenerationRegressionTests(unittest.TestCase):
    def _base_args(self, tmp: str) -> argparse.Namespace:
        return argparse.Namespace(
            input_path=os.path.join(tmp, "source.txt"),
            output_path=os.path.join(tmp, "episode.json"),
            profile="standard",
            target_minutes=None,
            words_per_min=None,
            min_words=None,
            max_words=None,
            episode_id=None,
            run_token=None,
            resume=False,
            resume_force=False,
            force_unlock=False,
            verbose=False,
            debug=False,
            dry_run_cleanup=False,
            force_clean=False,
        )

    def _prepare_success_result(self, *, tmp: str, script_cfg) -> SimpleNamespace:  # noqa: ANN001
        output_path = os.path.join(tmp, "episode.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "lines": [
                        {
                            "speaker": "Ana",
                            "role": "Host1",
                            "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Measured | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                            "text": "Bloque 1 con analisis del incidente, causas y mitigaciones.",
                        },
                        {
                            "speaker": "Luis",
                            "role": "Host2",
                            "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                            "text": "En resumen, cerramos con plan accionable y despedida final.",
                        },
                    ]
                },
                f,
                ensure_ascii=False,
            )
        checkpoint_path = os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json")
        run_summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"status": "completed", "lines": [], "current_word_count": 0}, f)
        with open(run_summary_path, "w", encoding="utf-8") as f:
            json.dump({"status": "completed", "phase_seconds": {}}, f)
        return SimpleNamespace(
            output_path=output_path,
            line_count=2,
            word_count=22,
            checkpoint_path=checkpoint_path,
            run_summary_path=run_summary_path,
            script_retry_rate=0.0,
            invalid_schema_rate=0.0,
        )

    def _run_incident_case(
        self,
        *,
        failure_exc: Exception,
    ) -> tuple[int, list[dict[str, object]], dict[str, object]]:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            with open(args.input_path, "w", encoding="utf-8") as f:
                f.write("contenido base suficiente para reproducir incidentes " * 120)
            script_cfg = make_script.ScriptConfig.from_env(profile_name="standard")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="standard")
            fake_result = self._prepare_success_result(tmp=tmp, script_cfg=script_cfg)
            calls: list[dict[str, object]] = []

            def _generate_side_effect(**kwargs):  # noqa: ANN003, ANN202
                calls.append(dict(kwargs))
                if len(calls) == 1:
                    raise failure_exc
                return fake_result

            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "off",
                    "SCRIPT_ORCHESTRATED_RETRY_ENABLED": "1",
                    "SCRIPT_ORCHESTRATED_MAX_ATTEMPTS": "2",
                    "SCRIPT_ORCHESTRATED_RETRY_BACKOFF_MS": "0",
                },
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para la prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                            with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                                with mock.patch.object(make_script, "ensure_min_free_disk"):
                                    with mock.patch.object(
                                        make_script,
                                        "cleanup_dir",
                                        return_value=SimpleNamespace(
                                            deleted_files=0,
                                            deleted_bytes=0,
                                            kept_files=0,
                                        ),
                                    ):
                                        with mock.patch.object(
                                            make_script.OpenAIClient,
                                            "from_configs",
                                            return_value=_FakeClient(),
                                        ):
                                            fake_generator = mock.Mock()
                                            fake_generator.generate.side_effect = _generate_side_effect
                                            with mock.patch.object(
                                                make_script,
                                                "ScriptGenerator",
                                                return_value=fake_generator,
                                            ):
                                                with mock.patch.object(make_script, "append_slo_event"):
                                                    with mock.patch.object(
                                                        make_script,
                                                        "evaluate_slo_windows",
                                                        return_value={"should_rollback": False},
                                                    ):
                                                        rc = make_script.main()
            with open(fake_result.run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            return rc, calls, summary

    def test_18_incident_patterns_recover_with_orchestrated_retry(self) -> None:
        incident_matrix = [
            ("2015", lambda: RuntimeError("OpenAI returned empty text for stage=chunk_1; parse_failure_kind=empty_output")),
            ("2038", lambda: RuntimeError("OpenAI returned empty text for stage=chunk_6_adaptive_2; parse_failure_kind=empty_output")),
            ("2046", lambda: RuntimeError("OpenAI returned empty text for stage=continuation_1; parse_failure_kind=empty_output")),
            ("2106", lambda: RuntimeError("OpenAI returned empty text for stage=continuation_3; parse_failure_kind=empty_output")),
            (
                "2112",
                lambda: make_script.ScriptOperationError(
                    "schema mismatch in chunk_2",
                    error_kind=make_script.ERROR_KIND_INVALID_SCHEMA,
                ),
            ),
            (
                "2120",
                lambda: make_script.ScriptOperationError(
                    "schema mismatch in adaptive subpart",
                    error_kind=make_script.ERROR_KIND_INVALID_SCHEMA,
                ),
            ),
            (
                "2140",
                lambda: make_script.ScriptOperationError(
                    "schema mismatch in continuation",
                    error_kind=make_script.ERROR_KIND_INVALID_SCHEMA,
                ),
            ),
            (
                "2203",
                lambda: make_script.ScriptOperationError(
                    "schema mismatch in truncation recovery",
                    error_kind=make_script.ERROR_KIND_INVALID_SCHEMA,
                ),
            ),
            (
                "2208",
                lambda: make_script.ScriptOperationError(
                    "quality gate rejected max_consecutive",
                    error_kind=make_script.ERROR_KIND_SCRIPT_QUALITY,
                ),
            ),
            (
                "2239",
                lambda: make_script.ScriptOperationError(
                    "quality gate rejected closing",
                    error_kind=make_script.ERROR_KIND_SCRIPT_QUALITY,
                ),
            ),
            (
                "2242",
                lambda: make_script.ScriptOperationError(
                    "quality gate rejected summary",
                    error_kind=make_script.ERROR_KIND_SCRIPT_QUALITY,
                ),
            ),
            ("2243", lambda: RuntimeError("Failed to parse JSON output for stage=chunk_3: wrapper artifact")),
            ("2244", lambda: RuntimeError("Failed to parse JSON output for stage=chunk_4: truncation artifact")),
            ("2245", lambda: RuntimeError("Failed to parse JSON output for stage=chunk_5: malformed artifact")),
            ("2246", lambda: RuntimeError("OpenAI returned empty text for stage=truncation_recovery_1; parse_failure_kind=empty_output")),
            (
                "2247",
                lambda: make_script.ScriptOperationError(
                    "schema mismatch while repairing",
                    error_kind=make_script.ERROR_KIND_INVALID_SCHEMA,
                ),
            ),
            ("2248", lambda: RuntimeError("OpenAI returned empty text for stage=chunk_3_subsplit_2; parse_failure_kind=empty_output")),
            (
                "2249",
                lambda: make_script.ScriptOperationError(
                    "quality gate rejected repeat_line_ratio",
                    error_kind=make_script.ERROR_KIND_SCRIPT_QUALITY,
                ),
            ),
        ]
        self.assertEqual(len(incident_matrix), 18)
        for incident_id, exc_factory in incident_matrix:
            with self.subTest(incident=incident_id):
                rc, calls, summary = self._run_incident_case(failure_exc=exc_factory())
                self.assertEqual(rc, 0)
                self.assertEqual(len(calls), 2)
                self.assertFalse(bool(calls[0].get("resume", False)))
                self.assertTrue(bool(calls[1].get("resume", False)))
                self.assertTrue(bool(calls[1].get("resume_force", False)))
                self.assertEqual(int(summary.get("script_orchestrated_retry_recoveries", 0)), 1)
                self.assertEqual(int(summary.get("script_orchestrated_retry_attempts_used", 0)), 2)


if __name__ == "__main__":
    unittest.main()
