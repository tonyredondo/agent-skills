import dataclasses
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import AudioConfig, LoggingConfig, ReliabilityConfig, ScriptConfig  # noqa: E402
from pipeline.errors import ERROR_KIND_STUCK, TTSOperationError  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.script_generator import ScriptGenerator  # noqa: E402
from pipeline.tts_synthesizer import TTSSynthesizer  # noqa: E402


class _InterruptScriptClient:
    def __init__(self) -> None:
        self.requests_made = 0
        self.estimated_cost_usd = 0.0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        return {
            "lines": [
                {
                    "speaker": "Ana",
                    "role": "Host1",
                    "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                    "text": "texto breve para avanzar poco",
                }
            ]
        }

    def generate_freeform_text(self, *, prompt, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        return "resumen"


class _InterruptTTSClient:
    def __init__(self) -> None:
        self.requests_made = 0
        self.estimated_cost_usd = 0.0

    def synthesize_speech(self, **kwargs):  # noqa: ANN003, ANN201
        self.requests_made += 1
        return b"ID3"


class _SlowInterruptTTSClient(_InterruptTTSClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def synthesize_speech(self, **kwargs):  # noqa: ANN003, ANN201
        self.requests_made += 1
        self.started.set()
        time.sleep(0.1)
        return b"ID3"


class _VerySlowInterruptTTSClient(_InterruptTTSClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def synthesize_speech(self, **kwargs):  # noqa: ANN003, ANN201
        self.requests_made += 1
        self.started.set()
        time.sleep(3.0)
        return b"ID3"


class _StuckTTSClient(_InterruptTTSClient):
    def synthesize_speech(self, **kwargs):  # noqa: ANN003, ANN201
        self.requests_made += 1
        time.sleep(2.0)
        return b"ID3"


class InterruptionTests(unittest.TestCase):
    def _logger(self):  # noqa: ANN202
        return Logger.create(LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False))

    def test_script_generator_interrupts_before_first_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, min_words=80, max_words=120)
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"))
            gen = ScriptGenerator(
                config=cfg,
                reliability=ReliabilityConfig.from_env(),
                logger=self._logger(),
                client=_InterruptScriptClient(),  # type: ignore[arg-type]
            )
            out_path = os.path.join(tmp, "episode.json")
            with self.assertRaises(InterruptedError):
                gen.generate(
                    source_text=("tema " * 200).strip(),
                    output_path=out_path,
                    cancel_check=lambda: True,
                )
            ckpt = os.path.join(cfg.checkpoint_dir, "episode", "script_checkpoint.json")
            self.assertTrue(os.path.exists(ckpt))
            with open(ckpt, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.assertEqual(state.get("status"), "interrupted")

    def test_script_generator_interrupts_during_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, min_words=200, max_words=240)
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "ckpt"), no_progress_rounds=10)
            client = _InterruptScriptClient()
            gen = ScriptGenerator(
                config=cfg,
                reliability=ReliabilityConfig.from_env(),
                logger=self._logger(),
                client=client,  # type: ignore[arg-type]
            )
            out_path = os.path.join(tmp, "episode2.json")
            calls = {"n": 0}

            def cancel_check() -> bool:
                calls["n"] += 1
                return calls["n"] >= 2

            with self.assertRaises(InterruptedError):
                gen.generate(
                    source_text=("tema " * 200).strip(),
                    output_path=out_path,
                    cancel_check=cancel_check,
                )
            self.assertGreaterEqual(client.requests_made, 1)

    def test_tts_synthesizer_interrupts_before_chunk_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "audio"), chunk_lines=1, max_concurrent=1)
            synth = TTSSynthesizer(
                config=cfg,
                reliability=ReliabilityConfig.from_env(),
                logger=self._logger(),
                client=_InterruptTTSClient(),  # type: ignore[arg-type]
            )
            lines = [
                {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "hola"},
                {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "mundo"},
            ]
            with self.assertRaises(InterruptedError):
                synth.synthesize(
                    lines=lines,
                    episode_id="ep_i",
                    cancel_check=lambda: True,
                )
            summary = os.path.join(cfg.checkpoint_dir, "ep_i", "run_summary.json")
            self.assertTrue(os.path.exists(summary))
            with open(summary, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "interrupted")

    def test_tts_synthesizer_interrupt_does_not_leave_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "audio"), chunk_lines=1, max_concurrent=1)
            synth = TTSSynthesizer(
                config=cfg,
                reliability=ReliabilityConfig.from_env(),
                logger=self._logger(),
                client=_InterruptTTSClient(),  # type: ignore[arg-type]
            )
            lines = [{"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "hola"}]
            with self.assertRaises(InterruptedError):
                synth.synthesize(lines=lines, episode_id="ep_lock", cancel_check=lambda: True)
            lock_path = os.path.join(cfg.checkpoint_dir, "ep_lock", ".lock")
            self.assertFalse(os.path.exists(lock_path))

    def test_tts_synthesizer_interrupt_during_chunk_keeps_manifest_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(cfg, checkpoint_dir=os.path.join(tmp, "audio"), chunk_lines=1, max_concurrent=1)
            client = _SlowInterruptTTSClient()
            synth = TTSSynthesizer(
                config=cfg,
                reliability=ReliabilityConfig.from_env(),
                logger=self._logger(),
                client=client,  # type: ignore[arg-type]
            )
            lines = [
                {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "hola uno"},
                {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "hola dos"},
            ]
            with self.assertRaises(InterruptedError):
                synth.synthesize(
                    lines=lines,
                    episode_id="ep_interrupt_chunk",
                    cancel_check=lambda: client.started.is_set(),
                )
            manifest = os.path.join(cfg.checkpoint_dir, "ep_interrupt_chunk", "audio_manifest.json")
            self.assertTrue(os.path.exists(manifest))
            with open(manifest, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "interrupted")

    def test_tts_interrupt_during_chunk_returns_without_waiting_worker_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "audio"),
                chunk_lines=1,
                max_concurrent=1,
                timeout_seconds=60,
            )
            client = _VerySlowInterruptTTSClient()
            synth = TTSSynthesizer(
                config=cfg,
                reliability=ReliabilityConfig.from_env(),
                logger=self._logger(),
                client=client,  # type: ignore[arg-type]
            )
            lines = [
                {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "hola uno"},
                {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "hola dos"},
            ]
            start = time.time()
            with self.assertRaises(InterruptedError):
                synth.synthesize(
                    lines=lines,
                    episode_id="ep_interrupt_fast_return",
                    cancel_check=lambda: client.started.is_set(),
                )
            elapsed = time.time() - start
            self.assertLess(elapsed, 2.8)
            # Give background worker a chance to finish before tempdir cleanup.
            time.sleep(2.3)

    def test_tts_stuck_operation_error_writes_failed_manifest_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "audio"),
                chunk_lines=1,
                max_concurrent=1,
                global_timeout_seconds=0,
                retries=1,
                timeout_seconds=5,
            )
            synth = TTSSynthesizer(
                config=cfg,
                reliability=ReliabilityConfig.from_env(),
                logger=self._logger(),
                client=_StuckTTSClient(),  # type: ignore[arg-type]
            )
            lines = [{"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "hola"}]
            fake_clock = {"t": 1000.0}

            def _fast_time() -> float:
                fake_clock["t"] += 20.0
                return fake_clock["t"]

            with mock.patch("pipeline.tts_synthesizer.time.time", side_effect=_fast_time):
                with self.assertRaises(TTSOperationError) as ctx:
                    synth.synthesize(lines=lines, episode_id="ep_stuck")
            self.assertEqual(ctx.exception.error_kind, ERROR_KIND_STUCK)

            manifest = os.path.join(cfg.checkpoint_dir, "ep_stuck", "audio_manifest.json")
            summary = os.path.join(cfg.checkpoint_dir, "ep_stuck", "run_summary.json")
            self.assertTrue(os.path.exists(manifest))
            self.assertTrue(os.path.exists(summary))

            with open(manifest, "r", encoding="utf-8") as f:
                manifest_payload = json.load(f)
            self.assertEqual(manifest_payload.get("status"), "failed")
            self.assertEqual(manifest_payload["segments"][0].get("status"), "failed")
            self.assertEqual(manifest_payload["segments"][0].get("error_kind"), ERROR_KIND_STUCK)

            with open(summary, "r", encoding="utf-8") as f:
                summary_payload = json.load(f)
            self.assertEqual(summary_payload.get("status"), "failed")
            self.assertEqual(summary_payload.get("failure_kind"), ERROR_KIND_STUCK)
            self.assertTrue(bool(summary_payload.get("stuck_abort")))
            self.assertIn(ERROR_KIND_STUCK, summary_payload.get("failure_kinds", []))


if __name__ == "__main__":
    unittest.main()

