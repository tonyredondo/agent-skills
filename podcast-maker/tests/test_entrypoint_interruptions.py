"""Regression tests for interruption, fallback, and quality-gate entrypoint behavior.

These tests intentionally exercise failure paths that are hard to reproduce in
end-to-end runs (interruptions, stale summaries, resume mismatches, and gate
enforcement). The goal is to keep run-summary/manifest semantics stable for
operators and orchestrated retry logic.
"""

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

import make_podcast  # noqa: E402
import make_script  # noqa: E402
from pipeline.errors import (  # noqa: E402
    ERROR_KIND_INTERRUPTED,
    ERROR_KIND_INVALID_SCHEMA,
    ERROR_KIND_RUN_MISMATCH,
    ERROR_KIND_RESUME_BLOCKED,
    ERROR_KIND_SOURCE_TOO_SHORT,
    ERROR_KIND_SCRIPT_QUALITY,
    ERROR_KIND_STUCK,
    ERROR_KIND_TIMEOUT,
    ERROR_KIND_UNKNOWN,
    TTSBatchError,
    TTSOperationError,
)


class _FakeClient:
    """Minimal client double exposing counters consumed by entrypoints."""

    requests_made = 0
    estimated_cost_usd = 0.0
    script_retries_total = 0
    tts_retries_total = 0
    script_json_parse_failures = 0


class _RepairingScriptClient(_FakeClient):
    """Returns a complete recap+farewell script for successful auto-repair paths."""

    def generate_script_json(self, **kwargs):  # noqa: ANN003, ANN201
        return {
            "lines": [
                {
                    "speaker": "Ana",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Measured | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": "En resumen, este episodio organiza ideas con pasos claros.",
                },
                {
                    "speaker": "Luis",
                    "role": "Host2",
                    "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                    "text": "Gracias por escuchar, nos vemos en la proxima.",
                },
            ]
        }


class _RepairStillFailClient(_FakeClient):
    """Returns an intentionally incomplete correction to keep gate failing."""

    def generate_script_json(self, **kwargs):  # noqa: ANN003, ANN201
        return {
            "lines": [
                {
                    "speaker": "Ana",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Measured | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": "Version corregida parcial sin cierre todavia.",
                }
            ]
        }


class _RepairThrowsClient(_FakeClient):
    """Raises during repair to validate revert-to-original behavior."""

    def generate_script_json(self, **kwargs):  # noqa: ANN003, ANN201
        raise RuntimeError("repair failed")


class EntryPointInterruptionTests(unittest.TestCase):
    """Coverage for entrypoint-level resilience and fallback contracts."""

    def test_make_script_default_gate_action_aligns_with_profile(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                make_script._default_script_gate_action(script_profile_name="short"),  # noqa: SLF001
                "warn",
            )
            self.assertEqual(
                make_script._default_script_gate_action(script_profile_name="standard"),  # noqa: SLF001
                "enforce",
            )
            self.assertEqual(
                make_script._default_script_gate_action(script_profile_name="long"),  # noqa: SLF001
                "enforce",
            )

    def test_make_script_default_gate_action_production_strict_forces_enforce(self) -> None:
        with mock.patch.dict(os.environ, {"SCRIPT_QUALITY_GATE_PROFILE": "production_strict"}, clear=True):
            self.assertEqual(
                make_script._default_script_gate_action(script_profile_name="short"),  # noqa: SLF001
                "enforce",
            )

    def test_make_script_keyboard_interrupt_exits_130(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "out.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            with mock.patch.object(make_script, "parse_args", return_value=args):
                with mock.patch.object(
                    make_script,
                    "read_text_file_with_fallback",
                    return_value=("contenido base suficiente para una prueba", "utf-8"),
                ):
                    with mock.patch.object(make_script, "ensure_min_free_disk"):
                        with mock.patch.object(
                            make_script,
                            "cleanup_dir",
                            return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                        ):
                            with mock.patch.object(
                                make_script.OpenAIClient,
                                "from_configs",
                                return_value=_FakeClient(),
                            ):
                                fake_generator = mock.Mock()
                                fake_generator.generate.side_effect = KeyboardInterrupt("ctrl-c")
                                with mock.patch.object(make_script, "ScriptGenerator", return_value=fake_generator):
                                    with mock.patch.object(make_script, "append_slo_event"):
                                        with mock.patch.object(
                                            make_script,
                                            "evaluate_slo_windows",
                                            return_value={"should_rollback": False},
                                        ):
                                            rc = make_script.main()
            self.assertEqual(rc, 130)

    def test_make_podcast_keyboard_interrupt_exits_130(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            script_ckpt = os.path.join(tmp, "script_ckpt")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                                "text": "Hola mundo",
                            }
                        ]
                    },
                    f,
                )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=os.path.join(tmp, "out"),
                basename="episode",
                profile="short",
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast, "ensure_min_free_disk"):
                    with mock.patch.object(
                        make_podcast,
                        "cleanup_dir",
                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                    ):
                        with mock.patch.object(
                            make_podcast.OpenAIClient,
                            "from_configs",
                            return_value=_FakeClient(),
                        ):
                            fake_mixer = mock.Mock()
                            fake_mixer.check_dependencies.return_value = None
                            with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                                fake_synth = mock.Mock()
                                fake_synth.synthesize.side_effect = KeyboardInterrupt("ctrl-c")
                                with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                                    with mock.patch.object(make_podcast, "append_slo_event"):
                                        with mock.patch.object(
                                            make_podcast,
                                            "evaluate_slo_windows",
                                            return_value={"should_rollback": False},
                                        ):
                                            with mock.patch.dict(
                                                os.environ,
                                                {
                                                    "SCRIPT_QUALITY_GATE_ACTION": "off",
                                                    "RUN_MANIFEST_V2": "1",
                                                    "SCRIPT_CHECKPOINT_DIR": script_ckpt,
                                                },
                                                clear=False,
                                            ):
                                                rc = make_podcast.main()
            self.assertEqual(rc, 130)
            summary_path = os.path.join(args.outdir, ".audio_checkpoints", "episode", "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "interrupted")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_INTERRUPTED)
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("tts", phase_seconds)
            self.assertIn("mix", phase_seconds)
            manifest_path = os.path.join(script_ckpt, "episode", "run_manifest.json")
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_payload = json.load(f)
            self.assertEqual(manifest_payload.get("status_by_stage", {}).get("audio"), "interrupted")
            self.assertEqual(manifest_payload.get("audio", {}).get("status"), "interrupted")
            self.assertEqual(manifest_payload.get("audio", {}).get("failure_kind"), ERROR_KIND_INTERRUPTED)

    def test_make_script_interrupt_uses_structured_run_summary_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            client = _FakeClient()
            client.requests_made = 4
            client.script_retries_total = 1

            def _interrupt_with_summary(**kwargs):  # noqa: ANN003, ANN201
                # Simulate generator writing a current-token summary before an
                # interruption, so make_script.main can source structured signals.
                output_path = kwargs["output_path"]
                run_token = kwargs.get("run_token")
                episode_id = os.path.splitext(os.path.basename(output_path))[0] or "episode"
                run_dir = os.path.join(script_cfg.checkpoint_dir, episode_id)
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "status": "interrupted",
                            "run_token": run_token,
                            "script_retry_rate": 0.25,
                            "invalid_schema_rate": 0.4,
                            "stuck_abort": True,
                            "failure_kind": ERROR_KIND_INTERRUPTED,
                        },
                        f,
                    )
                raise InterruptedError("ctrl-c")

            with mock.patch.object(make_script, "parse_args", return_value=args):
                with mock.patch.object(
                    make_script,
                    "read_text_file_with_fallback",
                    return_value=("contenido base suficiente para una prueba", "utf-8"),
                ):
                    with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                        with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                            with mock.patch.object(make_script, "ensure_min_free_disk"):
                                with mock.patch.object(
                                    make_script,
                                    "cleanup_dir",
                                    return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                ):
                                    with mock.patch.object(
                                        make_script.OpenAIClient,
                                        "from_configs",
                                        return_value=client,
                                    ):
                                        fake_generator = mock.Mock()
                                        fake_generator.generate.side_effect = _interrupt_with_summary
                                        with mock.patch.object(
                                            make_script, "ScriptGenerator", return_value=fake_generator
                                        ):
                                            with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                                with mock.patch.object(
                                                    make_script,
                                                    "evaluate_slo_windows",
                                                    return_value={"should_rollback": False},
                                                ):
                                                    rc = make_script.main()
            self.assertEqual(rc, 130)
            # SLO emission should reflect structured summary fields, not only the
            # generic InterruptedError type.
            self.assertEqual(float(append_slo.call_args.kwargs.get("retry_rate")), 0.25)
            self.assertTrue(bool(append_slo.call_args.kwargs.get("invalid_schema")))
            self.assertTrue(bool(append_slo.call_args.kwargs.get("stuck_abort")))
            self.assertEqual(
                append_slo.call_args.kwargs.get("failure_kind"),
                ERROR_KIND_INTERRUPTED,
            )

    def test_make_script_uses_structured_run_summary_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")

            def _raise_with_summary(**kwargs):  # noqa: ANN003, ANN201
                output_path = kwargs["output_path"]
                run_token = kwargs.get("run_token")
                episode_id = os.path.splitext(os.path.basename(output_path))[0] or "episode"
                run_dir = os.path.join(script_cfg.checkpoint_dir, episode_id)
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "status": "failed",
                            "stuck_abort": True,
                            "invalid_schema_rate": 0.3,
                            "failure_kind": ERROR_KIND_STUCK,
                            "marker": "from_generator",
                            "run_token": run_token,
                        },
                        f,
                    )
                raise RuntimeError("boom without schema wording")

            with mock.patch.object(make_script, "parse_args", return_value=args):
                with mock.patch.object(
                    make_script,
                    "read_text_file_with_fallback",
                    return_value=("contenido base suficiente para una prueba", "utf-8"),
                ):
                    with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                        with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                            with mock.patch.object(make_script, "ensure_min_free_disk"):
                                with mock.patch.object(
                                    make_script,
                                    "cleanup_dir",
                                    return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                ):
                                    with mock.patch.object(
                                        make_script.OpenAIClient,
                                        "from_configs",
                                        return_value=_FakeClient(),
                                    ):
                                        fake_generator = mock.Mock()
                                        fake_generator.generate.side_effect = _raise_with_summary
                                        with mock.patch.object(
                                            make_script, "ScriptGenerator", return_value=fake_generator
                                        ):
                                            with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                                with mock.patch.object(
                                                    make_script,
                                                    "evaluate_slo_windows",
                                                    return_value={"should_rollback": False},
                                                ):
                                                    rc = make_script.main()
            self.assertEqual(rc, 1)
            self.assertTrue(bool(append_slo.call_args.kwargs.get("stuck_abort")))
            self.assertTrue(bool(append_slo.call_args.kwargs.get("invalid_schema")))
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_STUCK)
            self.assertNotEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_INVALID_SCHEMA)
            summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("marker"), "from_generator")

    def test_make_script_ignores_stale_run_summary_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "missing.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            run_dir = os.path.join(script_cfg.checkpoint_dir, "episode")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
                # Seed a stale token: runtime must ignore these signals and
                # recompute failure classification for the current run.
                json.dump(
                    {
                        "status": "failed",
                        "stuck_abort": True,
                        "invalid_schema_rate": 1.0,
                        "failure_kind": ERROR_KIND_STUCK,
                        "run_token": "stale-token",
                    },
                    f,
                )

            with mock.patch.object(make_script, "parse_args", return_value=args):
                with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                    with mock.patch.object(make_script, "append_slo_event") as append_slo:
                        with mock.patch.object(
                            make_script,
                            "evaluate_slo_windows",
                            return_value={"should_rollback": False},
                        ):
                            rc = make_script.main()
            self.assertEqual(rc, 1)
            # Structured flags should not leak from stale-token summaries.
            self.assertFalse(bool(append_slo.call_args.kwargs.get("stuck_abort")))
            self.assertFalse(bool(append_slo.call_args.kwargs.get("invalid_schema")))
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_UNKNOWN)
            refreshed_summary_path = os.path.join(run_dir, "run_summary.json")
            self.assertTrue(os.path.exists(refreshed_summary_path))
            with open(refreshed_summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_UNKNOWN)
            self.assertNotEqual(str(payload.get("run_token", "")).strip(), "stale-token")
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("repair", phase_seconds)
            self.assertIn("quality_repair", phase_seconds)

    def test_make_script_missing_source_writes_fallback_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "missing_source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            with mock.patch.object(make_script, "parse_args", return_value=args):
                with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                    with mock.patch.object(make_script, "append_slo_event"):
                        with mock.patch.object(
                            make_script,
                            "evaluate_slo_windows",
                            return_value={"should_rollback": False},
                        ):
                            rc = make_script.main()
            self.assertEqual(rc, 1)
            summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            # Even early I/O failures must emit a fallback run summary artifact.
            self.assertEqual(payload.get("component"), "make_script")
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_UNKNOWN)
            self.assertEqual(payload.get("input_path"), args.input_path)
            self.assertEqual(payload.get("output_path"), args.output_path)
            self.assertEqual(int(payload.get("source_word_count", -1)), 0)
            self.assertTrue(str(payload.get("run_token", "")).strip())
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("repair", phase_seconds)
            self.assertIn("quality_repair", phase_seconds)

    def test_make_script_operation_error_without_summary_writes_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            with mock.patch.object(make_script, "parse_args", return_value=args):
                with mock.patch.object(
                    make_script,
                    "read_text_file_with_fallback",
                    return_value=("contenido base suficiente para una prueba", "utf-8"),
                ):
                    with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                        with mock.patch.object(make_script, "ensure_min_free_disk"):
                            with mock.patch.object(
                                make_script,
                                "cleanup_dir",
                                return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                            ):
                                with mock.patch.object(
                                    make_script.OpenAIClient,
                                    "from_configs",
                                    return_value=_FakeClient(),
                                ):
                                    fake_generator = mock.Mock()
                                    # Simulate generator failing before writing
                                    # run_summary so make_script must synthesize fallback.
                                    fake_generator.generate.side_effect = make_script.ScriptOperationError(
                                        "Resume blocked in generator",
                                        error_kind=ERROR_KIND_RESUME_BLOCKED,
                                    )
                                    with mock.patch.object(
                                        make_script,
                                        "ScriptGenerator",
                                        return_value=fake_generator,
                                    ):
                                        with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                            with mock.patch.object(
                                                make_script,
                                                "evaluate_slo_windows",
                                                return_value={"should_rollback": False},
                                            ):
                                                rc = make_script.main()
            self.assertEqual(rc, 1)
            self.assertEqual(
                append_slo.call_args.kwargs.get("failure_kind"),
                ERROR_KIND_RESUME_BLOCKED,
            )
            summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            # Fallback summary still keeps the precise operation failure kind.
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_RESUME_BLOCKED)
            self.assertTrue(str(payload.get("run_token", "")).strip())
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("repair", phase_seconds)
            self.assertIn("quality_repair", phase_seconds)

    def test_make_script_source_too_short_propagates_failure_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            with mock.patch.object(make_script, "parse_args", return_value=args):
                with mock.patch.object(
                    make_script,
                    "read_text_file_with_fallback",
                    return_value=("contenido base insuficiente", "utf-8"),
                ):
                    with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                        with mock.patch.object(make_script, "ensure_min_free_disk"):
                            with mock.patch.object(
                                make_script,
                                "cleanup_dir",
                                return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                            ):
                                with mock.patch.object(
                                    make_script.OpenAIClient,
                                    "from_configs",
                                    return_value=_FakeClient(),
                                ):
                                    fake_generator = mock.Mock()
                                    fake_generator.generate.side_effect = make_script.ScriptOperationError(
                                        "Source is too short for requested target length",
                                        error_kind=ERROR_KIND_SOURCE_TOO_SHORT,
                                    )
                                    with mock.patch.object(
                                        make_script,
                                        "ScriptGenerator",
                                        return_value=fake_generator,
                                    ):
                                        with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                            with mock.patch.object(
                                                make_script,
                                                "evaluate_slo_windows",
                                                return_value={"should_rollback": False},
                                            ):
                                                rc = make_script.main()
            self.assertEqual(rc, 1)
            self.assertEqual(
                append_slo.call_args.kwargs.get("failure_kind"),
                ERROR_KIND_SOURCE_TOO_SHORT,
            )
            summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_SOURCE_TOO_SHORT)

    def test_make_script_replaces_corrupt_run_summary_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "missing.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            run_dir = os.path.join(script_cfg.checkpoint_dir, "episode")
            os.makedirs(run_dir, exist_ok=True)
            corrupt_summary_path = os.path.join(run_dir, "run_summary.json")
            with open(corrupt_summary_path, "w", encoding="utf-8") as f:
                f.write("{bad")
            with mock.patch.object(make_script, "parse_args", return_value=args):
                with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                    with mock.patch.object(make_script, "append_slo_event") as append_slo:
                        with mock.patch.object(
                            make_script,
                            "evaluate_slo_windows",
                            return_value={"should_rollback": False},
                        ):
                            rc = make_script.main()
            self.assertEqual(rc, 1)
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_UNKNOWN)
            with open(corrupt_summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_UNKNOWN)
            self.assertTrue(str(payload.get("run_token", "")).strip())

    def test_make_script_quality_gate_script_side_enforce_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Texto corto",
                            }
                        ]
                    },
                    f,
                )
            fake_result = SimpleNamespace(
                output_path=args.output_path,
                line_count=1,
                word_count=2,
                checkpoint_path=os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json"),
                run_summary_path=os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json"),
                script_retry_rate=0.0,
                invalid_schema_rate=0.0,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "enforce",
                    "SCRIPT_QUALITY_GATE_EVALUATOR": "rules",
                },
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para una prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                            with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                                with mock.patch.object(make_script, "ensure_min_free_disk"):
                                    with mock.patch.object(
                                        make_script,
                                        "cleanup_dir",
                                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                    ):
                                        with mock.patch.object(
                                            make_script.OpenAIClient,
                                            "from_configs",
                                            return_value=_FakeClient(),
                                        ):
                                            fake_generator = mock.Mock()
                                            fake_generator.generate.return_value = fake_result
                                            with mock.patch.object(
                                                make_script, "ScriptGenerator", return_value=fake_generator
                                            ):
                                                with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                                    with mock.patch.object(
                                                        make_script,
                                                        "evaluate_slo_windows",
                                                        return_value={"should_rollback": False},
                                                    ):
                                                        rc = make_script.main()
            self.assertEqual(rc, 1)
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_SCRIPT_QUALITY)
            report_path = os.path.join(script_cfg.checkpoint_dir, "episode", "quality_report.json")
            self.assertTrue(os.path.exists(report_path))

    def test_make_script_quality_gate_script_side_warn_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Texto corto",
                            }
                        ]
                    },
                    f,
                )
            fake_result = SimpleNamespace(
                output_path=args.output_path,
                line_count=1,
                word_count=2,
                checkpoint_path=os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json"),
                run_summary_path=os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json"),
                script_retry_rate=0.0,
                invalid_schema_rate=0.0,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "warn",
                    "SCRIPT_QUALITY_GATE_EVALUATOR": "rules",
                },
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para una prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                            with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                                with mock.patch.object(make_script, "ensure_min_free_disk"):
                                    with mock.patch.object(
                                        make_script,
                                        "cleanup_dir",
                                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                    ):
                                        with mock.patch.object(
                                            make_script.OpenAIClient,
                                            "from_configs",
                                            return_value=_FakeClient(),
                                        ):
                                            fake_generator = mock.Mock()
                                            fake_generator.generate.return_value = fake_result
                                            with mock.patch.object(
                                                make_script, "ScriptGenerator", return_value=fake_generator
                                            ):
                                                with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                                    with mock.patch.object(
                                                        make_script,
                                                        "evaluate_slo_windows",
                                                        return_value={"should_rollback": False},
                                                    ):
                                                        rc = make_script.main()
            self.assertEqual(rc, 0)
            self.assertIsNone(append_slo.call_args.kwargs.get("failure_kind"))
            report_path = os.path.join(script_cfg.checkpoint_dir, "episode", "quality_report.json")
            self.assertTrue(os.path.exists(report_path))

    def test_make_script_quality_gate_warn_applies_last_correction_even_if_still_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # Seed an original script artifact so we can verify warn-mode repair
            # does not corrupt successful existing content on failed corrections.
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Texto corto original",
                            }
                        ]
                    },
                    f,
                )
            checkpoint_path = os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json")
            run_summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"lines": [], "current_word_count": 0, "status": "completed"}, f)
            with open(run_summary_path, "w", encoding="utf-8") as f:
                json.dump({"line_count": 1, "word_count": 3, "status": "completed"}, f)
            fake_result = SimpleNamespace(
                output_path=args.output_path,
                line_count=1,
                word_count=3,
                checkpoint_path=checkpoint_path,
                run_summary_path=run_summary_path,
                script_retry_rate=0.0,
                invalid_schema_rate=0.0,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "warn",
                    "SCRIPT_QUALITY_GATE_EVALUATOR": "rules",
                    "SCRIPT_QUALITY_GATE_AUTO_REPAIR": "1",
                    "SCRIPT_QUALITY_GATE_REPAIR_ATTEMPTS": "1",
                },
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para una prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                            with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                                with mock.patch.object(make_script, "ensure_min_free_disk"):
                                    with mock.patch.object(
                                        make_script,
                                        "cleanup_dir",
                                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                    ):
                                        with mock.patch.object(
                                            make_script.OpenAIClient,
                                            "from_configs",
                                            return_value=_RepairStillFailClient(),
                                        ):
                                            # Repair client returns still-failing
                                            # payloads to force warn-path fallback.
                                            fake_generator = mock.Mock()
                                            fake_generator.generate.return_value = fake_result
                                            with mock.patch.object(
                                                make_script, "ScriptGenerator", return_value=fake_generator
                                            ):
                                                with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                                    with mock.patch.object(
                                                        make_script,
                                                        "evaluate_slo_windows",
                                                        return_value={"should_rollback": False},
                                                    ):
                                                        rc = make_script.main()
            self.assertEqual(rc, 0)
            self.assertIsNone(append_slo.call_args.kwargs.get("failure_kind"))
            with open(args.output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            text = payload["lines"][0]["text"]
            # Original content remains unchanged when warn-mode repair cannot
            # produce a valid improved script.
            self.assertEqual(text, "Texto corto original")
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_payload = json.load(f)
            self.assertEqual(checkpoint_payload.get("status"), "completed")
            with open(run_summary_path, "r", encoding="utf-8") as f:
                summary_payload = json.load(f)
            self.assertEqual(summary_payload.get("line_count"), 1)
            self.assertGreaterEqual(int(summary_payload.get("word_count", 0)), 1)
            self.assertEqual(summary_payload.get("status"), "completed")
            self.assertEqual(summary_payload.get("script_gate_action_effective"), "warn")
            self.assertTrue(bool(summary_payload.get("quality_gate_executed")))
            # Script-only entrypoint should never mark audio handoff fields.
            self.assertFalse(bool(summary_payload.get("handoff_to_audio_started")))
            self.assertFalse(bool(summary_payload.get("handoff_to_audio_completed")))

    def test_make_script_quality_gate_enforce_marks_artifacts_failed_with_last_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Texto corto original",
                            }
                        ]
                    },
                    f,
                )
            checkpoint_path = os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json")
            run_summary_path = os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"lines": [], "current_word_count": 0, "status": "completed"}, f)
            with open(run_summary_path, "w", encoding="utf-8") as f:
                json.dump({"line_count": 1, "word_count": 3, "status": "completed"}, f)
            fake_result = SimpleNamespace(
                output_path=args.output_path,
                line_count=1,
                word_count=3,
                checkpoint_path=checkpoint_path,
                run_summary_path=run_summary_path,
                script_retry_rate=0.0,
                invalid_schema_rate=0.0,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "enforce",
                    "SCRIPT_QUALITY_GATE_EVALUATOR": "rules",
                    "SCRIPT_QUALITY_GATE_AUTO_REPAIR": "1",
                    "SCRIPT_QUALITY_GATE_REPAIR_ATTEMPTS": "1",
                },
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para una prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                            with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                                with mock.patch.object(make_script, "ensure_min_free_disk"):
                                    with mock.patch.object(
                                        make_script,
                                        "cleanup_dir",
                                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                    ):
                                        with mock.patch.object(
                                            make_script.OpenAIClient,
                                            "from_configs",
                                            return_value=_RepairStillFailClient(),
                                        ):
                                            fake_generator = mock.Mock()
                                            fake_generator.generate.return_value = fake_result
                                            with mock.patch.object(
                                                make_script, "ScriptGenerator", return_value=fake_generator
                                            ):
                                                with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                                    with mock.patch.object(
                                                        make_script,
                                                        "evaluate_slo_windows",
                                                        return_value={"should_rollback": False},
                                                    ):
                                                        rc = make_script.main()
            self.assertEqual(rc, 1)
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_SCRIPT_QUALITY)
            with open(args.output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            text = payload["lines"][0]["text"]
            self.assertEqual(text, "Texto corto original")
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_payload = json.load(f)
            self.assertEqual(checkpoint_payload.get("status"), "failed")
            self.assertEqual(checkpoint_payload.get("lines"), [])
            with open(run_summary_path, "r", encoding="utf-8") as f:
                summary_payload = json.load(f)
            self.assertEqual(summary_payload.get("status"), "failed")
            self.assertEqual(summary_payload.get("failure_kind"), ERROR_KIND_SCRIPT_QUALITY)
            phase_seconds = summary_payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("repair", phase_seconds)
            self.assertIn("quality_repair", phase_seconds)

    def test_make_script_quality_gate_warn_keeps_original_when_repair_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            original_payload = {
                "lines": [
                    {
                        "speaker": "Ana",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                        "text": "Texto corto original",
                    }
                ]
            }
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(original_payload, f)
            fake_result = SimpleNamespace(
                output_path=args.output_path,
                line_count=1,
                word_count=3,
                checkpoint_path=os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json"),
                run_summary_path=os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json"),
                script_retry_rate=0.0,
                invalid_schema_rate=0.0,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "warn",
                    "SCRIPT_QUALITY_GATE_EVALUATOR": "rules",
                    "SCRIPT_QUALITY_GATE_AUTO_REPAIR": "1",
                    "SCRIPT_QUALITY_GATE_REPAIR_ATTEMPTS": "1",
                },
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para una prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                            with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                                with mock.patch.object(make_script, "ensure_min_free_disk"):
                                    with mock.patch.object(
                                        make_script,
                                        "cleanup_dir",
                                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                    ):
                                        with mock.patch.object(
                                            make_script.OpenAIClient,
                                            "from_configs",
                                            return_value=_RepairThrowsClient(),
                                        ):
                                            fake_generator = mock.Mock()
                                            fake_generator.generate.return_value = fake_result
                                            with mock.patch.object(
                                                make_script, "ScriptGenerator", return_value=fake_generator
                                            ):
                                                with mock.patch.object(make_script, "append_slo_event"):
                                                    with mock.patch.object(
                                                        make_script,
                                                        "evaluate_slo_windows",
                                                        return_value={"should_rollback": False},
                                                    ):
                                                        rc = make_script.main()
            self.assertEqual(rc, 0)
            with open(args.output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["lines"][0]["text"], original_payload["lines"][0]["text"])

    def test_make_script_success_retry_rate_uses_client_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "En resumen, este texto habilita una salida valida para test.",
                            },
                            {
                                "speaker": "Luis",
                                "role": "Host2",
                                "instructions": "Voice Affect: Bright | Tone: Conversational | Pacing: Brisk",
                                "text": "Gracias por escuchar este episodio de prueba.",
                            },
                        ]
                    },
                    f,
                )
            fake_result = SimpleNamespace(
                output_path=args.output_path,
                line_count=2,
                word_count=25,
                checkpoint_path=os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json"),
                run_summary_path=os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json"),
                script_retry_rate=0.0,
                invalid_schema_rate=0.0,
            )
            client = _FakeClient()
            client.requests_made = 10
            client.script_retries_total = 4
            with mock.patch.dict(
                os.environ,
                {"SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "off"},
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para una prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                            with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                                with mock.patch.object(make_script, "ensure_min_free_disk"):
                                    with mock.patch.object(
                                        make_script,
                                        "cleanup_dir",
                                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                    ):
                                        with mock.patch.object(
                                            make_script.OpenAIClient,
                                            "from_configs",
                                            return_value=client,
                                        ):
                                            fake_generator = mock.Mock()
                                            fake_generator.generate.return_value = fake_result
                                            with mock.patch.object(
                                                make_script, "ScriptGenerator", return_value=fake_generator
                                            ):
                                                with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                                    with mock.patch.object(
                                                        make_script,
                                                        "evaluate_slo_windows",
                                                        return_value={"should_rollback": False},
                                                    ):
                                                        rc = make_script.main()
            self.assertEqual(rc, 0)
            self.assertEqual(float(append_slo.call_args.kwargs.get("retry_rate")), 0.4)

    def test_make_script_quality_gate_missing_output_fails_even_in_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            fake_result = SimpleNamespace(
                output_path=args.output_path,
                line_count=1,
                word_count=2,
                checkpoint_path=os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json"),
                run_summary_path=os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json"),
                script_retry_rate=0.0,
                invalid_schema_rate=0.0,
            )
            with mock.patch.dict(
                os.environ,
                {"SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "warn"},
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para una prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                            with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                                with mock.patch.object(make_script, "ensure_min_free_disk"):
                                    with mock.patch.object(
                                        make_script,
                                        "cleanup_dir",
                                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                    ):
                                        with mock.patch.object(
                                            make_script.OpenAIClient,
                                            "from_configs",
                                            return_value=_FakeClient(),
                                        ):
                                            fake_generator = mock.Mock()
                                            fake_generator.generate.return_value = fake_result
                                            with mock.patch.object(
                                                make_script, "ScriptGenerator", return_value=fake_generator
                                            ):
                                                with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                                    with mock.patch.object(
                                                        make_script,
                                                        "evaluate_slo_windows",
                                                        return_value={"should_rollback": False},
                                                    ):
                                                        rc = make_script.main()
            self.assertEqual(rc, 1)
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_SCRIPT_QUALITY)

    def test_make_script_quality_gate_production_strict_defaults_to_enforce(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Texto corto",
                            }
                        ]
                    },
                    f,
                )
            fake_result = SimpleNamespace(
                output_path=args.output_path,
                line_count=1,
                word_count=2,
                checkpoint_path=os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json"),
                run_summary_path=os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json"),
                script_retry_rate=0.0,
                invalid_schema_rate=0.0,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_PROFILE": "production_strict",
                    "SCRIPT_QUALITY_GATE_EVALUATOR": "rules",
                },
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para una prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                            with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                                with mock.patch.object(make_script, "ensure_min_free_disk"):
                                    with mock.patch.object(
                                        make_script,
                                        "cleanup_dir",
                                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                    ):
                                        with mock.patch.object(
                                            make_script.OpenAIClient,
                                            "from_configs",
                                            return_value=_FakeClient(),
                                        ):
                                            fake_generator = mock.Mock()
                                            fake_generator.generate.return_value = fake_result
                                            with mock.patch.object(
                                                make_script, "ScriptGenerator", return_value=fake_generator
                                            ):
                                                with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                                    with mock.patch.object(
                                                        make_script,
                                                        "evaluate_slo_windows",
                                                        return_value={"should_rollback": False},
                                                    ):
                                                        rc = make_script.main()
            self.assertEqual(rc, 1)
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_SCRIPT_QUALITY)

    def test_make_script_quality_gate_enforce_repairs_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "episode.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            script_cfg = make_script.ScriptConfig.from_env(profile_name="short")
            script_cfg = dataclasses.replace(script_cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            audio_cfg = make_script.AudioConfig.from_env(profile_name="short")
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Texto corto",
                            }
                        ]
                    },
                    f,
                )
            fake_result = SimpleNamespace(
                output_path=args.output_path,
                line_count=1,
                word_count=2,
                checkpoint_path=os.path.join(script_cfg.checkpoint_dir, "episode", "script_checkpoint.json"),
                run_summary_path=os.path.join(script_cfg.checkpoint_dir, "episode", "run_summary.json"),
                script_retry_rate=0.0,
                invalid_schema_rate=0.0,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "enforce",
                    "SCRIPT_QUALITY_GATE_EVALUATOR": "rules",
                    "SCRIPT_QUALITY_MIN_WORDS_RATIO": "0.0",
                    "SCRIPT_QUALITY_GATE_AUTO_REPAIR": "1",
                    "SCRIPT_QUALITY_GATE_REPAIR_ATTEMPTS": "1",
                },
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para una prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script.ScriptConfig, "from_env", return_value=script_cfg):
                            with mock.patch.object(make_script.AudioConfig, "from_env", return_value=audio_cfg):
                                with mock.patch.object(make_script, "ensure_min_free_disk"):
                                    with mock.patch.object(
                                        make_script,
                                        "cleanup_dir",
                                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                                    ):
                                        with mock.patch.object(
                                            make_script.OpenAIClient,
                                            "from_configs",
                                            return_value=_RepairingScriptClient(),
                                        ):
                                            fake_generator = mock.Mock()
                                            fake_generator.generate.return_value = fake_result
                                            with mock.patch.object(
                                                make_script, "ScriptGenerator", return_value=fake_generator
                                            ):
                                                with mock.patch.object(make_script, "append_slo_event") as append_slo:
                                                    with mock.patch.object(
                                                        make_script,
                                                        "evaluate_slo_windows",
                                                        return_value={"should_rollback": False},
                                                    ):
                                                        rc = make_script.main()
            self.assertEqual(rc, 0)
            self.assertIsNone(append_slo.call_args.kwargs.get("failure_kind"))
            with open(args.output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            joined = " ".join(str(line.get("text", "")) for line in payload.get("lines", []))
            self.assertIn("nos quedamos con", joined.lower())
            self.assertIn("Gracias por escuchar", joined)

    def test_make_podcast_marks_stuck_abort_from_structured_batch_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                                "text": "Hola mundo",
                            }
                        ]
                    },
                    f,
                )
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile="short",
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast, "ensure_min_free_disk"):
                    with mock.patch.object(
                        make_podcast,
                        "cleanup_dir",
                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                    ):
                        with mock.patch.object(
                            make_podcast.OpenAIClient,
                            "from_configs",
                            return_value=_FakeClient(),
                        ):
                            fake_mixer = mock.Mock()
                            fake_mixer.check_dependencies.return_value = None
                            with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                                fake_synth = mock.Mock()
                                # Batch error with timeout kind should propagate
                                # as structured failure_kind/stuck_abort telemetry.
                                fake_synth.synthesize.side_effect = TTSBatchError(
                                    manifest_path=os.path.join(outdir, ".audio_checkpoints", "episode", "audio_manifest.json"),
                                    failed_segments=[{"segment_id": "0001"}],
                                    failed_kinds=[ERROR_KIND_TIMEOUT],
                                )
                                with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                                    with mock.patch.object(make_podcast, "append_slo_event") as append_slo:
                                        with mock.patch.object(
                                            make_podcast,
                                            "evaluate_slo_windows",
                                            return_value={"should_rollback": False},
                                        ):
                                            with mock.patch.dict(
                                                os.environ,
                                                {"SCRIPT_QUALITY_GATE_ACTION": "off"},
                                                clear=False,
                                            ):
                                                rc = make_podcast.main()
            self.assertEqual(rc, 1)
            self.assertTrue(append_slo.called)
            self.assertTrue(bool(append_slo.call_args.kwargs.get("stuck_abort")))
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_TIMEOUT)
            summary_path = os.path.join(outdir, ".audio_checkpoints", "episode", "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_TIMEOUT)
            self.assertEqual(payload.get("audio_stage"), "failed_during_tts")
            self.assertIn("audio_manifest.json", str(payload.get("manifest_path", "")))
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("tts", phase_seconds)
            self.assertIn("mix", phase_seconds)

    def test_make_podcast_invalid_script_sets_unknown_failure_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "bad_script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write("{bad")
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile="short",
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast, "append_slo_event") as append_slo:
                    with mock.patch.object(
                        make_podcast,
                        "evaluate_slo_windows",
                        return_value={"should_rollback": False},
                    ):
                        rc = make_podcast.main()
            self.assertEqual(rc, 1)
            self.assertFalse(bool(append_slo.call_args.kwargs.get("stuck_abort")))
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_UNKNOWN)
            summary_path = os.path.join(outdir, ".audio_checkpoints", "episode", "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_UNKNOWN)
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("tts", phase_seconds)
            self.assertIn("mix", phase_seconds)

    def test_make_podcast_ffmpeg_missing_without_raw_only_writes_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Script suficiente para intentar audio",
                            }
                        ]
                    },
                    f,
                )
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile="short",
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast, "ensure_min_free_disk"):
                    with mock.patch.object(
                        make_podcast,
                        "cleanup_dir",
                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                    ):
                        with mock.patch.object(
                            make_podcast.OpenAIClient,
                            "from_configs",
                            return_value=_FakeClient(),
                        ):
                            with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                                with mock.patch.object(make_podcast, "TTSSynthesizer") as synth_cls:
                                    with mock.patch.object(make_podcast, "append_slo_event") as append_slo:
                                        with mock.patch.object(
                                            make_podcast,
                                            "evaluate_slo_windows",
                                            return_value={"should_rollback": False},
                                        ):
                                            with mock.patch.dict(
                                                os.environ,
                                                {"SCRIPT_QUALITY_GATE_ACTION": "off"},
                                                clear=False,
                                            ):
                                                rc = make_podcast.main()
            self.assertEqual(rc, 1)
            synth_cls.assert_not_called()
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_UNKNOWN)
            summary_path = os.path.join(outdir, ".audio_checkpoints", "episode", "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_UNKNOWN)
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("tts", phase_seconds)
            self.assertIn("mix", phase_seconds)

    def test_make_podcast_resume_blocked_operation_error_is_not_stuck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                                "text": "Hola mundo",
                            }
                        ]
                    },
                    f,
                )
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile="short",
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast, "ensure_min_free_disk"):
                    with mock.patch.object(
                        make_podcast,
                        "cleanup_dir",
                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                    ):
                        with mock.patch.object(
                            make_podcast.OpenAIClient,
                            "from_configs",
                            return_value=_FakeClient(),
                        ):
                            fake_mixer = mock.Mock()
                            fake_mixer.check_dependencies.return_value = None
                            with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                                fake_synth = mock.Mock()
                                # Resume-blocked failures are operationally
                                # actionable but must not be labeled as "stuck".
                                fake_synth.synthesize.side_effect = TTSOperationError(
                                    "Resume blocked: audio config fingerprint changed",
                                    error_kind=ERROR_KIND_RESUME_BLOCKED,
                                )
                                with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                                    with mock.patch.object(make_podcast, "append_slo_event") as append_slo:
                                        with mock.patch.object(
                                            make_podcast,
                                            "evaluate_slo_windows",
                                            return_value={"should_rollback": False},
                                        ):
                                            with mock.patch.dict(
                                                os.environ,
                                                {"SCRIPT_QUALITY_GATE_ACTION": "off"},
                                                clear=False,
                                            ):
                                                rc = make_podcast.main()
            self.assertEqual(rc, 1)
            self.assertFalse(bool(append_slo.call_args.kwargs.get("stuck_abort")))
            self.assertEqual(
                append_slo.call_args.kwargs.get("failure_kind"),
                ERROR_KIND_RESUME_BLOCKED,
            )
            summary_path = os.path.join(outdir, ".audio_checkpoints", "episode", "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_RESUME_BLOCKED)
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("tts", phase_seconds)
            self.assertIn("mix", phase_seconds)

    def test_make_podcast_manifest_script_path_mismatch_sets_run_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Hola mundo",
                            }
                        ]
                    },
                    f,
                )
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            script_ckpt = os.path.join(tmp, "script_ckpt")
            os.makedirs(os.path.join(script_ckpt, "episode"), exist_ok=True)
            manifest_script_path = os.path.join(tmp, "different_script.json")
            with open(manifest_script_path, "w", encoding="utf-8") as f:
                json.dump({"lines": []}, f)
            manifest_path = os.path.join(script_ckpt, "episode", "run_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "manifest_version": 2,
                        "episode_id": "episode",
                        "script_output_path": manifest_script_path,
                        "script_checkpoint_dir": script_ckpt,
                        "audio_checkpoint_dir": os.path.join(outdir, ".audio_checkpoints"),
                    },
                    f,
                )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                episode_id="episode",
                profile="short",
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "SCRIPT_CHECKPOINT_DIR": script_ckpt,
                    "RUN_MANIFEST_V2": "1",
                },
                clear=False,
            ):
                # Manifest path mismatch should fail before any expensive audio
                # stage work begins.
                with mock.patch.object(make_podcast, "parse_args", return_value=args):
                    with mock.patch.object(make_podcast, "append_slo_event") as append_slo:
                        with mock.patch.object(
                            make_podcast,
                            "evaluate_slo_windows",
                            return_value={"should_rollback": False},
                        ):
                            rc = make_podcast.main()
            self.assertEqual(rc, 1)
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_RUN_MISMATCH)
            summary_path = os.path.join(outdir, ".audio_checkpoints", "episode", "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_RUN_MISMATCH)
            self.assertEqual(payload.get("audio_stage"), "failed_before_tts")

    def test_make_podcast_quality_gate_enforce_blocks_tts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Script corto",
                            }
                        ]
                    },
                    f,
                )
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile="short",
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast, "ensure_min_free_disk"):
                    with mock.patch.object(
                        make_podcast,
                        "cleanup_dir",
                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                    ):
                        with mock.patch.object(
                            make_podcast.OpenAIClient,
                            "from_configs",
                            return_value=_FakeClient(),
                        ):
                            with mock.patch.object(make_podcast, "AudioMixer") as mixer_cls:
                                with mock.patch.object(make_podcast, "TTSSynthesizer") as synth_cls:
                                    with mock.patch.object(make_podcast, "append_slo_event") as append_slo:
                                        with mock.patch.object(
                                            make_podcast,
                                            "evaluate_slo_windows",
                                            return_value={"should_rollback": False},
                                        ):
                                            with mock.patch.dict(
                                                os.environ,
                                                {
                                                    "SCRIPT_QUALITY_GATE_ACTION": "enforce",
                                                    "SCRIPT_QUALITY_GATE_EVALUATOR": "rules",
                                                },
                                                clear=False,
                                            ):
                                                rc = make_podcast.main()
            self.assertEqual(rc, 4)
            # Enforce mode blocks all downstream audio stages when quality fails.
            mixer_cls.assert_not_called()
            synth_cls.assert_not_called()
            self.assertEqual(append_slo.call_args.kwargs.get("failure_kind"), ERROR_KIND_SCRIPT_QUALITY)
            self.assertFalse(bool(append_slo.call_args.kwargs.get("stuck_abort")))
            report_path = os.path.join(outdir, ".audio_checkpoints", "episode", "quality_report.json")
            self.assertTrue(os.path.exists(report_path))
            summary_path = os.path.join(outdir, ".audio_checkpoints", "episode", "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "failed")
            self.assertEqual(payload.get("failure_kind"), ERROR_KIND_SCRIPT_QUALITY)
            self.assertFalse(bool(payload.get("quality_gate_pass", True)))
            self.assertEqual(payload.get("quality_report_path"), report_path)
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("tts", phase_seconds)
            self.assertIn("mix", phase_seconds)

    def test_make_podcast_quality_gate_warn_allows_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Script corto",
                            }
                        ]
                    },
                    f,
                )
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            checkpoint_dir = os.path.join(outdir, ".audio_checkpoints", "episode")
            os.makedirs(checkpoint_dir, exist_ok=True)
            seg_file = os.path.join(tmp, "seg_0001.mp3")
            with open(seg_file, "wb") as f:
                f.write(b"A")
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile="short",
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.return_value = None
            fake_mixer.mix.return_value = SimpleNamespace(
                final_path=os.path.join(outdir, "episode_norm_eq.mp3"),
                raw_path=os.path.join(outdir, "episode.mp3"),
                norm_path=os.path.join(outdir, "episode_norm.mp3"),
            )
            fake_synth = mock.Mock()
            fake_synth.synthesize.return_value = SimpleNamespace(
                segment_files=[seg_file],
                manifest_path=os.path.join(checkpoint_dir, "audio_manifest.json"),
                summary_path=os.path.join(checkpoint_dir, "run_summary.json"),
                checkpoint_dir=checkpoint_dir,
            )
            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast, "ensure_min_free_disk"):
                    with mock.patch.object(
                        make_podcast,
                        "cleanup_dir",
                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                    ):
                        with mock.patch.object(
                            make_podcast.OpenAIClient,
                            "from_configs",
                            return_value=_FakeClient(),
                        ):
                            with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                                with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                                    with mock.patch.object(make_podcast, "append_slo_event") as append_slo:
                                        with mock.patch.object(
                                            make_podcast,
                                            "evaluate_slo_windows",
                                            return_value={"should_rollback": False},
                                        ):
                                            with mock.patch.dict(
                                                os.environ,
                                                {
                                                    "SCRIPT_QUALITY_GATE_ACTION": "warn",
                                                    "SCRIPT_QUALITY_GATE_EVALUATOR": "rules",
                                                },
                                                clear=False,
                                            ):
                                                rc = make_podcast.main()
            self.assertEqual(rc, 0)
            # Warn mode keeps pipeline moving even when quality report fails.
            self.assertTrue(fake_synth.synthesize.called)
            self.assertIsNone(append_slo.call_args.kwargs.get("failure_kind"))
            summary_path = os.path.join(checkpoint_dir, "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("quality_gate_pass", payload)
            self.assertFalse(bool(payload.get("quality_gate_pass")))
            self.assertTrue(str(payload.get("quality_report_path", "")).endswith("quality_report.json"))
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("tts", phase_seconds)
            self.assertIn("mix", phase_seconds)

    def test_make_podcast_quality_gate_off_skips_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm | Tone: Conversational | Pacing: Brisk",
                                "text": "Script minimo",
                            }
                        ]
                    },
                    f,
                )
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            checkpoint_dir = os.path.join(outdir, ".audio_checkpoints", "episode")
            os.makedirs(checkpoint_dir, exist_ok=True)
            seg_file = os.path.join(tmp, "seg_0001.mp3")
            with open(seg_file, "wb") as f:
                f.write(b"A")
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile="short",
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.return_value = None
            fake_mixer.mix.return_value = SimpleNamespace(
                final_path=os.path.join(outdir, "episode_norm_eq.mp3"),
                raw_path=os.path.join(outdir, "episode.mp3"),
                norm_path=os.path.join(outdir, "episode_norm.mp3"),
            )
            fake_synth = mock.Mock()
            fake_synth.synthesize.return_value = SimpleNamespace(
                segment_files=[seg_file],
                manifest_path=os.path.join(checkpoint_dir, "audio_manifest.json"),
                summary_path=os.path.join(checkpoint_dir, "run_summary.json"),
                checkpoint_dir=checkpoint_dir,
            )
            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast, "ensure_min_free_disk"):
                    with mock.patch.object(
                        make_podcast,
                        "cleanup_dir",
                        return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                    ):
                        with mock.patch.object(
                            make_podcast.OpenAIClient,
                            "from_configs",
                            return_value=_FakeClient(),
                        ):
                            with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                                with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                                    with mock.patch.object(
                                        make_podcast,
                                        "evaluate_script_quality",
                                    ) as eval_quality:
                                        with mock.patch.object(make_podcast, "append_slo_event"):
                                            with mock.patch.object(
                                                make_podcast,
                                                "evaluate_slo_windows",
                                                return_value={"should_rollback": False},
                                            ):
                                                with mock.patch.dict(
                                                    os.environ,
                                                    {"SCRIPT_QUALITY_GATE_ACTION": "off"},
                                                    clear=False,
                                                ):
                                                    rc = make_podcast.main()
            self.assertEqual(rc, 0)
            # Gate-off mode should never invoke quality evaluator.
            eval_quality.assert_not_called()
            summary_path = os.path.join(checkpoint_dir, "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            phase_seconds = payload.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("tts", phase_seconds)
            self.assertIn("mix", phase_seconds)
            self.assertEqual(float(phase_seconds.get("quality_eval", -1.0)), 0.0)

    def test_make_script_invalid_slo_env_uses_default_window_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                input_path=os.path.join(tmp, "source.txt"),
                output_path=os.path.join(tmp, "out.json"),
                profile="short",
                target_minutes=None,
                words_per_min=None,
                min_words=None,
                max_words=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Measured | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                                "text": "En resumen, esta prueba mantiene salida valida para gate.",
                            },
                            {
                                "speaker": "Luis",
                                "role": "Host2",
                                "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                                "text": "Gracias por escuchar, nos vemos en la proxima.",
                            },
                        ]
                    },
                    f,
                )
            with mock.patch.dict(
                os.environ,
                {"SLO_GATE_MODE": "warn", "SLO_WINDOW_SIZE": "bad", "SLO_REQUIRED_FAILED_WINDOWS": "oops"},
                clear=False,
            ):
                with mock.patch.object(make_script, "parse_args", return_value=args):
                    with mock.patch.object(
                        make_script,
                        "read_text_file_with_fallback",
                        return_value=("contenido base suficiente para una prueba", "utf-8"),
                    ):
                        with mock.patch.object(make_script, "ensure_min_free_disk"):
                            with mock.patch.object(
                                make_script,
                                "cleanup_dir",
                                return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                            ):
                                with mock.patch.object(
                                    make_script.OpenAIClient,
                                    "from_configs",
                                    return_value=_FakeClient(),
                                ):
                                    fake_generator = mock.Mock()
                                    fake_generator.generate.return_value = SimpleNamespace(
                                        output_path=args.output_path,
                                        line_count=2,
                                        word_count=120,
                                        checkpoint_path=os.path.join(tmp, "ckpt", "ep", "script_checkpoint.json"),
                                        run_summary_path=os.path.join(tmp, "ckpt", "ep", "run_summary.json"),
                                        script_retry_rate=0.0,
                                        invalid_schema_rate=0.0,
                                    )
                                    with mock.patch.object(
                                        make_script, "ScriptGenerator", return_value=fake_generator
                                    ):
                                        with mock.patch.object(make_script, "append_slo_event"):
                                            with mock.patch.object(
                                                make_script,
                                                "evaluate_slo_windows",
                                                return_value={"should_rollback": False},
                                            ) as eval_slo:
                                                rc = make_script.main()
            self.assertEqual(rc, 0)
            self.assertEqual(eval_slo.call_args.kwargs.get("window_size"), 20)
            self.assertEqual(eval_slo.call_args.kwargs.get("required_failed_windows"), 2)

    def test_make_podcast_invalid_slo_env_uses_default_window_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lines": [
                            {
                                "speaker": "Ana",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                                "text": "Hola mundo",
                            }
                        ]
                    },
                    f,
                )
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            checkpoint_dir = os.path.join(outdir, ".audio_checkpoints", "episode")
            os.makedirs(checkpoint_dir, exist_ok=True)
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile="short",
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=False,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )
            with mock.patch.dict(
                os.environ,
                {"SLO_GATE_MODE": "warn", "SLO_WINDOW_SIZE": "bad", "SLO_REQUIRED_FAILED_WINDOWS": "oops"},
                clear=False,
            ):
                with mock.patch.object(make_podcast, "parse_args", return_value=args):
                    with mock.patch.object(make_podcast, "ensure_min_free_disk"):
                        with mock.patch.object(
                            make_podcast,
                            "cleanup_dir",
                            return_value=SimpleNamespace(deleted_files=0, deleted_bytes=0, kept_files=0),
                        ):
                            with mock.patch.object(
                                make_podcast.OpenAIClient,
                                "from_configs",
                                return_value=_FakeClient(),
                            ):
                                fake_mixer = mock.Mock()
                                fake_mixer.check_dependencies.return_value = None
                                fake_mixer.mix.return_value = SimpleNamespace(
                                    final_path=os.path.join(outdir, "episode_norm_eq.mp3"),
                                    raw_path=os.path.join(outdir, "episode.mp3"),
                                    norm_path=os.path.join(outdir, "episode_norm.mp3"),
                                )
                                with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                                    fake_synth = mock.Mock()
                                    fake_synth.synthesize.return_value = SimpleNamespace(
                                        segment_files=[os.path.join(tmp, "seg_0001.mp3")],
                                        manifest_path=os.path.join(checkpoint_dir, "audio_manifest.json"),
                                        summary_path=os.path.join(checkpoint_dir, "run_summary.json"),
                                        checkpoint_dir=checkpoint_dir,
                                    )
                                    with mock.patch.object(
                                        make_podcast, "TTSSynthesizer", return_value=fake_synth
                                    ):
                                        with mock.patch.object(make_podcast, "append_slo_event"):
                                            with mock.patch.object(
                                                make_podcast,
                                                "evaluate_slo_windows",
                                                return_value={"should_rollback": False},
                                            ) as eval_slo:
                                                with mock.patch.dict(
                                                    os.environ,
                                                    {"SCRIPT_QUALITY_GATE_ACTION": "off"},
                                                    clear=False,
                                                ):
                                                    rc = make_podcast.main()
            self.assertEqual(rc, 0)
            self.assertEqual(eval_slo.call_args.kwargs.get("window_size"), 20)
            self.assertEqual(eval_slo.call_args.kwargs.get("required_failed_windows"), 2)


if __name__ == "__main__":
    unittest.main()

