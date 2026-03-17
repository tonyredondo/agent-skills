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


class MakeScriptCliIntegrationTests(unittest.TestCase):
    def _base_args(self, tmp: str) -> argparse.Namespace:
        return argparse.Namespace(
            input_path=os.path.join(tmp, "source.txt"),
            output_path=os.path.join(tmp, "episode.json"),
            profile="short",
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

    def _prepare_generator_result(self, *, tmp: str, script_cfg) -> SimpleNamespace:  # noqa: ANN001
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
                            "text": "Bloque 1 con contexto util para revisar riesgos y decisiones.",
                        },
                        {
                            "speaker": "Luis",
                            "role": "Host2",
                            "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                            "text": "Cerramos con recomendaciones practicas y acciones siguientes.",
                        },
                    ]
                },
                f,
                ensure_ascii=False,
            )
        checkpoint_path = os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json")
        run_summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
        quality_report_path = os.path.join(script_cfg.checkpoint_dir, "episode", "quality_report.json")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"status": "completed", "lines": [], "current_word_count": 0}, f)
        with open(run_summary_path, "w", encoding="utf-8") as f:
            json.dump({"status": "completed", "phase_seconds": {}}, f)
        with open(quality_report_path, "w", encoding="utf-8") as f:
            json.dump({"status": "passed", "pass": True, "reasons": []}, f)
        return SimpleNamespace(
            output_path=output_path,
            line_count=2,
            word_count=24,
            checkpoint_path=checkpoint_path,
            run_summary_path=run_summary_path,
            script_retry_rate=0.0,
            invalid_schema_rate=0.0,
            quality_report_path=quality_report_path,
        )





    def test_manifest_final_update_runs_when_manifest_init_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            args.run_token = "run_manifest_after_init_failure"
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            fake_result = self._prepare_generator_result(tmp=tmp, script_cfg=script_cfg)
            with mock.patch.dict(
                os.environ,
                {
                    "RUN_MANIFEST_V2": "1",
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "off",
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
                                            fake_generator.generate.return_value = fake_result
                                            with mock.patch.object(
                                                make_script,
                                                "ScriptGenerator",
                                                return_value=fake_generator,
                                            ):
                                                with mock.patch.object(
                                                    make_script,
                                                    "init_manifest",
                                                    side_effect=RuntimeError("manifest init write failed"),
                                                ):
                                                    with mock.patch.object(make_script, "append_slo_event"):
                                                        with mock.patch.object(
                                                            make_script,
                                                            "evaluate_slo_windows",
                                                            return_value={"should_rollback": False},
                                                        ):
                                                            rc = make_script.main()
            self.assertEqual(rc, 0)
            manifest_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_manifest.json")
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(str(manifest.get("run_token", "")), args.run_token)
            self.assertEqual(str(manifest.get("script_output_path", "")), fake_result.output_path)
            status_by_stage = dict(manifest.get("status_by_stage", {}))
            self.assertEqual(str(status_by_stage.get("script", "")), "completed")
            self.assertEqual(str(status_by_stage.get("audio", "")), "not_started")
            self.assertEqual(str(status_by_stage.get("bundle", "")), "not_started")
            script_stage = dict(manifest.get("script", {}))
            self.assertEqual(str(script_stage.get("status", "")), "completed")
            self.assertTrue(int(script_stage.get("started_at", 0)) > 0)
            self.assertTrue(int(script_stage.get("completed_at", 0)) > 0)
            self.assertEqual(script_stage.get("failure_kind"), None)
            self.assertEqual(str(script_stage.get("run_summary_path", "")), fake_result.run_summary_path)

    def test_fallback_run_summary_merges_existing_same_run_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            args.run_token = "run_merge_fallback_summary"
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            run_summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
            os.makedirs(os.path.dirname(run_summary_path), exist_ok=True)
            with open(run_summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "status": "failed",
                        "schema_validation_failures_by_stage": {"chunk_1": 2},
                    },
                    f,
                )

            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "off",
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
                                            fake_generator.generate.side_effect = RuntimeError(
                                                "simulated generation failure"
                                            )
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
            self.assertEqual(rc, 1)
            with open(run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(str(summary.get("run_token", "")), args.run_token)
            self.assertEqual(str(summary.get("status", "")), "failed")
            self.assertEqual(
                dict(summary.get("schema_validation_failures_by_stage", {})).get("chunk_1"),
                2,
            )

    def test_failed_run_preserves_existing_quality_report_path_for_same_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            args.run_token = "run_preserve_quality_path"
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            run_summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
            quality_report_path = os.path.join(script_cfg.checkpoint_dir, "episode", "quality_report.json")
            os.makedirs(os.path.dirname(run_summary_path), exist_ok=True)
            with open(quality_report_path, "w", encoding="utf-8") as f:
                json.dump({"status": "failed", "pass": False}, f)
            with open(run_summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_token": args.run_token,
                        "status": "failed",
                        "quality_report_path": quality_report_path,
                        "script_quality_report_path": quality_report_path,
                    },
                    f,
                )

            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "off",
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
                                            fake_generator.generate.side_effect = make_script.ScriptOperationError(
                                                "final quality failed",
                                                error_kind=make_script.ERROR_KIND_SCRIPT_QUALITY,
                                            )
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
            self.assertEqual(rc, 1)
            with open(run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("quality_report_path"), quality_report_path)
            self.assertEqual(summary.get("script_quality_report_path"), quality_report_path)

    def test_quality_gate_hardening_respects_configured_max_consecutive_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            fake_result = self._prepare_generator_result(tmp=tmp, script_cfg=script_cfg)
            with open(fake_result.output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Measured | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                                "text": "Bloque 1 con contexto practico y decisiones de producto.",
                            },
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Measured | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                                "text": "Bloque 2 con evidencia operativa y aprendizajes aplicables.",
                            },
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Measured | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                                "text": "Bloque 3 con riesgos, mitigaciones y resumen de decisiones.",
                            },
                            {
                                "speaker": "Luis",
                                "role": "Host2",
                                "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                                "text": "Gracias por escucharnos, nos vemos en el siguiente episodio.",
                            },
                        ]
                    },
                    f,
                    ensure_ascii=False,
                )
            with open(fake_result.quality_report_path, "w", encoding="utf-8") as f:
                json.dump({"status": "passed", "pass": True, "reasons": []}, f)

            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "warn",
                    "SCRIPT_QUALITY_MAX_CONSECUTIVE_SAME_SPEAKER": "3",
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
                                            fake_generator.generate.return_value = fake_result
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
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(fake_result.quality_report_path))

    def test_orchestrated_retry_recovers_invalid_schema_generation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            fake_result = self._prepare_generator_result(tmp=tmp, script_cfg=script_cfg)
            call_kwargs: list[dict[str, object]] = []

            def _generate_side_effect(**kwargs):  # noqa: ANN003, ANN202
                call_kwargs.append(dict(kwargs))
                if len(call_kwargs) == 1:
                    raise make_script.ScriptOperationError(
                        "schema mismatch during generation",
                        error_kind=make_script.ERROR_KIND_INVALID_SCHEMA,
                    )
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
            self.assertEqual(rc, 0)
            self.assertEqual(len(call_kwargs), 2)
            self.assertFalse(bool(call_kwargs[0].get("resume", False)))
            self.assertTrue(bool(call_kwargs[1].get("resume", False)))
            self.assertTrue(bool(call_kwargs[1].get("resume_force", False)))
            run_summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
            with open(run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(int(summary.get("script_orchestrated_retry_recoveries", 0)), 1)
            self.assertEqual(int(summary.get("script_orchestrated_retry_attempts_used", 0)), 2)

    def test_orchestrated_retry_recovers_script_completeness_generation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            fake_result = self._prepare_generator_result(tmp=tmp, script_cfg=script_cfg)
            call_kwargs: list[dict[str, object]] = []

            def _generate_side_effect(**kwargs):  # noqa: ANN003, ANN202
                call_kwargs.append(dict(kwargs))
                if len(call_kwargs) == 1:
                    raise make_script.ScriptOperationError(
                        "Generated script below minimum words target (90 < 100). Increase source detail or continuation limits.",
                        error_kind=make_script.ERROR_KIND_SCRIPT_COMPLETENESS,
                    )
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
            self.assertEqual(rc, 0)
            self.assertEqual(len(call_kwargs), 2)
            self.assertFalse(bool(call_kwargs[0].get("resume", False)))
            self.assertTrue(bool(call_kwargs[1].get("resume", False)))
            self.assertTrue(bool(call_kwargs[1].get("resume_force", False)))
            run_summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
            with open(run_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(int(summary.get("script_orchestrated_retry_recoveries", 0)), 1)
            self.assertEqual(int(summary.get("script_orchestrated_retry_attempts_used", 0)), 2)

    def test_orchestrated_retry_skips_nonrecoverable_failure_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            call_kwargs: list[dict[str, object]] = []

            def _generate_side_effect(**kwargs):  # noqa: ANN003, ANN202
                call_kwargs.append(dict(kwargs))
                raise make_script.ScriptOperationError(
                    "source is too short",
                    error_kind="source_too_short",
                )

            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "off",
                    "SCRIPT_ORCHESTRATED_RETRY_ENABLED": "1",
                    "SCRIPT_ORCHESTRATED_MAX_ATTEMPTS": "3",
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
            self.assertEqual(rc, 1)
            self.assertEqual(len(call_kwargs), 1)

    def test_orchestrated_retry_ignores_stale_client_failure_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            call_kwargs: list[dict[str, object]] = []

            class _StaleCountersClient(_FakeClient):
                script_json_parse_failures = 7
                script_empty_output_failures = 3

            def _generate_side_effect(**kwargs):  # noqa: ANN003, ANN202
                call_kwargs.append(dict(kwargs))
                raise RuntimeError("upstream gateway temporarily unavailable")

            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "off",
                    "SCRIPT_ORCHESTRATED_RETRY_ENABLED": "1",
                    "SCRIPT_ORCHESTRATED_MAX_ATTEMPTS": "3",
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
                                            return_value=_StaleCountersClient(),
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
            self.assertEqual(rc, 1)
            self.assertEqual(len(call_kwargs), 1)

    def test_orchestrated_retry_prefers_exception_kind_over_stale_signal_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            args.run_token = "run_signal_precedence"
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            call_kwargs: list[dict[str, object]] = []
            episode_id = "episode"
            os.makedirs(os.path.join(script_cfg.checkpoint_dir, episode_id), exist_ok=True)
            stale_summary_path = os.path.join(script_cfg.checkpoint_dir, episode_id, "run_summary.json")
            with open(stale_summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_token": args.run_token,
                        "status": "failed",
                        "failure_kind": make_script.ERROR_KIND_INVALID_SCHEMA,
                    },
                    f,
                )

            def _generate_side_effect(**kwargs):  # noqa: ANN003, ANN202
                call_kwargs.append(dict(kwargs))
                raise make_script.ScriptOperationError(
                    "source is too short",
                    error_kind="source_too_short",
                )

            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "off",
                    "SCRIPT_ORCHESTRATED_RETRY_ENABLED": "1",
                    "SCRIPT_ORCHESTRATED_MAX_ATTEMPTS": "3",
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
            self.assertEqual(rc, 1)
            self.assertEqual(len(call_kwargs), 1)

    def test_orchestrated_retry_never_retries_stuck_even_if_env_includes_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._base_args(tmp)
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            call_kwargs: list[dict[str, object]] = []

            def _generate_side_effect(**kwargs):  # noqa: ANN003, ANN202
                call_kwargs.append(dict(kwargs))
                raise make_script.ScriptOperationError(
                    "no progress while expanding script",
                    error_kind=make_script.ERROR_KIND_STUCK,
                )

            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "off",
                    "SCRIPT_ORCHESTRATED_RETRY_ENABLED": "1",
                    "SCRIPT_ORCHESTRATED_MAX_ATTEMPTS": "3",
                    "SCRIPT_ORCHESTRATED_RETRY_BACKOFF_MS": "0",
                    "SCRIPT_ORCHESTRATED_RETRY_FAILURE_KINDS": f"{make_script.ERROR_KIND_STUCK},{make_script.ERROR_KIND_INVALID_SCHEMA}",
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
            self.assertEqual(rc, 1)
            self.assertEqual(len(call_kwargs), 1)


if __name__ == "__main__":
    unittest.main()
