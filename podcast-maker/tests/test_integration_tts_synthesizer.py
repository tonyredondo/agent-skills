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

from pipeline.config import AudioConfig, LoggingConfig, ReliabilityConfig  # noqa: E402
from pipeline.errors import (  # noqa: E402
    ERROR_KIND_RESUME_BLOCKED,
    ERROR_KIND_TIMEOUT,
    TTSBatchError,
    TTSOperationError,
)
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.tts_synthesizer import TTSSynthesizer  # noqa: E402


class FakeTTSClient:
    def __init__(self) -> None:
        self.requests_made = 0
        self.estimated_cost_usd = 0.0

    def synthesize_speech(  # noqa: ANN001
        self,
        *,
        text,
        instructions,
        voice,
        speed=None,
        stage,
        timeout_seconds_override=None,
        cancel_check=None,
    ):
        self.requests_made += 1
        return (f"ID3-{voice}-{stage}-{text[:20]}").encode("utf-8")


class FailingTTSClient(FakeTTSClient):
    def synthesize_speech(  # noqa: ANN001
        self,
        *,
        text,
        instructions,
        voice,
        speed=None,
        stage,
        timeout_seconds_override=None,
        cancel_check=None,
    ):
        self.requests_made += 1
        raise RuntimeError("TTS global timeout reached")


class TTSSynthesizerIntegrationTests(unittest.TestCase):
    def test_synthesize_with_chunk_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "audio_ckpt"),
                chunk_lines=1,
                pause_between_segments_ms=0,
                max_concurrent=2,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False))
            client = FakeTTSClient()
            synth = TTSSynthesizer(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]

            lines = [
                {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Hola, esto es una prueba."},
                {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Seguimos con otro bloque interesante."},
                {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Cerramos con mas contexto util."},
            ]
            result = synth.synthesize(lines=lines, episode_id="ep_tts", resume=False)
            self.assertGreaterEqual(len(result.segment_files), 3)
            self.assertTrue(os.path.exists(result.manifest_path))
            with open(result.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(all("chunk_id" in seg for seg in manifest["segments"]))
            self.assertTrue(all(str(seg.get("phase", "")).strip() for seg in manifest["segments"]))
            self.assertTrue(all(isinstance(seg.get("speed"), (int, float)) for seg in manifest["segments"]))
            self.assertTrue(all(seg.get("checksum_sha256") for seg in manifest["segments"]))
            self.assertTrue(bool(manifest.get("manifest_checksum_sha256")))
            self.assertIn("tts_speed_stats", manifest)
            with open(result.summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertIn("tts_speed_stats", summary)
            self.assertIn("tts_phase_counts", summary)

    def test_resume_force_recovers_from_corrupt_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "audio_ckpt"),
                chunk_lines=1,
                pause_between_segments_ms=0,
                max_concurrent=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeTTSClient()
            synth = TTSSynthesizer(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            lines = [
                {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Linea uno de prueba."},
                {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Linea dos de prueba."},
            ]
            run_dir = os.path.join(cfg.checkpoint_dir, "ep_resume")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "audio_manifest.json"), "w", encoding="utf-8") as f:
                f.write("{corrupt")

            result = synth.synthesize(
                lines=lines,
                episode_id="ep_resume",
                resume=True,
                resume_force=True,
            )
            self.assertTrue(os.path.exists(result.manifest_path))
            leftovers = [p for p in os.listdir(run_dir) if ".corrupt." in p]
            self.assertTrue(leftovers)

    def test_failed_segments_raise_structured_batch_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "audio_ckpt"),
                chunk_lines=1,
                pause_between_segments_ms=0,
                max_concurrent=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FailingTTSClient()
            synth = TTSSynthesizer(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            lines = [
                {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Linea de prueba."},
            ]
            with self.assertRaises(TTSBatchError) as ctx:
                synth.synthesize(lines=lines, episode_id="ep_fail")
            exc = ctx.exception
            self.assertTrue(exc.stuck_abort)
            self.assertIn(ERROR_KIND_TIMEOUT, exc.failed_kinds)
            manifest_path = os.path.join(cfg.checkpoint_dir, "ep_fail", "audio_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            failed = [s for s in manifest["segments"] if s.get("status") == "failed"]
            self.assertTrue(failed)
            self.assertEqual(failed[0].get("error_kind"), ERROR_KIND_TIMEOUT)

    def test_resume_blocked_does_not_mutate_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "audio_ckpt"),
                chunk_lines=1,
                pause_between_segments_ms=0,
                max_concurrent=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeTTSClient()
            synth = TTSSynthesizer(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            lines = [{"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Linea de prueba."}]

            run_dir = os.path.join(cfg.checkpoint_dir, "ep_resume_blocked")
            os.makedirs(run_dir, exist_ok=True)
            manifest_path = os.path.join(run_dir, "audio_manifest.json")
            initial_manifest = {
                "checkpoint_version": reliability.checkpoint_version,
                "episode_id": "ep_resume_blocked",
                "config_fingerprint": "stale-fingerprint",
                "script_hash": "stale-hash",
                "created_at": 1,
                "updated_at": 1,
                "status": "running",
                "segments": [
                    {
                        "segment_id": "0001",
                        "index": 1,
                        "line_index": 1,
                        "chunk_id": 1,
                        "speaker": "Carlos",
                        "role": "Host1",
                        "voice": "cedar",
                        "instructions": "x",
                        "text": "Linea de prueba.",
                        "text_len": 15,
                        "status": "pending",
                        "attempts": 0,
                        "error": "",
                        "error_kind": "",
                        "file_name": "seg_0001.mp3",
                    }
                ],
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(initial_manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")

            with self.assertRaises(TTSOperationError) as ctx:
                synth.synthesize(lines=lines, episode_id="ep_resume_blocked", resume=True, resume_force=False)
            self.assertEqual(ctx.exception.error_kind, ERROR_KIND_RESUME_BLOCKED)

            with open(manifest_path, "r", encoding="utf-8") as f:
                after_manifest = json.load(f)
            self.assertEqual(after_manifest, initial_manifest)

            summary_path = os.path.join(run_dir, "run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("status"), "failed")
            self.assertIn(ERROR_KIND_RESUME_BLOCKED, summary.get("failure_kinds", []))

    def test_resume_blocked_summary_keeps_current_error_kind_with_old_failed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "audio_ckpt"),
                chunk_lines=1,
                pause_between_segments_ms=0,
                max_concurrent=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeTTSClient()
            synth = TTSSynthesizer(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            lines = [{"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Linea de prueba."}]

            run_dir = os.path.join(cfg.checkpoint_dir, "ep_resume_blocked_old_failed")
            os.makedirs(run_dir, exist_ok=True)
            manifest_path = os.path.join(run_dir, "audio_manifest.json")
            manifest = {
                "checkpoint_version": reliability.checkpoint_version,
                "episode_id": "ep_resume_blocked_old_failed",
                "config_fingerprint": "stale-fingerprint",
                "script_hash": "stale-hash",
                "created_at": 1,
                "updated_at": 1,
                "status": "failed",
                "segments": [
                    {
                        "segment_id": "0001",
                        "index": 1,
                        "line_index": 1,
                        "chunk_id": 1,
                        "speaker": "Carlos",
                        "role": "Host1",
                        "voice": "cedar",
                        "instructions": "x",
                        "text": "Linea de prueba.",
                        "text_len": 15,
                        "status": "failed",
                        "attempts": 1,
                        "error": "TTS global timeout reached",
                        "error_kind": ERROR_KIND_TIMEOUT,
                        "file_name": "seg_0001.mp3",
                    }
                ],
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")

            with self.assertRaises(TTSOperationError) as ctx:
                synth.synthesize(
                    lines=lines,
                    episode_id="ep_resume_blocked_old_failed",
                    resume=True,
                    resume_force=False,
                )
            self.assertEqual(ctx.exception.error_kind, ERROR_KIND_RESUME_BLOCKED)

            summary_path = os.path.join(run_dir, "run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            failure_kinds = summary.get("failure_kinds", [])
            self.assertEqual(summary.get("failure_kind"), ERROR_KIND_RESUME_BLOCKED)
            self.assertIn(ERROR_KIND_RESUME_BLOCKED, failure_kinds)
            self.assertIn(ERROR_KIND_TIMEOUT, failure_kinds)
            self.assertFalse(bool(summary.get("stuck_abort")))

    def test_resume_promotes_running_segment_with_existing_file_without_new_tts_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "audio_ckpt"),
                chunk_lines=1,
                pause_between_segments_ms=0,
                max_concurrent=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeTTSClient()
            synth = TTSSynthesizer(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            lines = [{"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Linea de prueba."}]

            # First run creates a valid manifest and segment file.
            synth.synthesize(lines=lines, episode_id="ep_running_reuse", resume=False)
            run_dir = os.path.join(cfg.checkpoint_dir, "ep_running_reuse")
            manifest_path = os.path.join(run_dir, "audio_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["segments"][0]["status"], "done")

            # Simulate interrupted update: file exists, status left as running.
            manifest["segments"][0]["status"] = "running"
            manifest["segments"][0]["error"] = ""
            manifest["segments"][0]["error_kind"] = ""
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")

            # Reset request counter and resume: should not make new TTS requests.
            client.requests_made = 0
            result = synth.synthesize(lines=lines, episode_id="ep_running_reuse", resume=True, resume_force=True)
            self.assertTrue(result.segment_files)
            self.assertEqual(client.requests_made, 0)
            with open(manifest_path, "r", encoding="utf-8") as f:
                resumed_manifest = json.load(f)
            self.assertEqual(resumed_manifest["segments"][0]["status"], "done")

    def test_resume_requeues_running_segment_without_checksum_and_regenerates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "audio_ckpt"),
                chunk_lines=1,
                pause_between_segments_ms=0,
                max_concurrent=1,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeTTSClient()
            synth = TTSSynthesizer(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            lines = [{"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Linea de prueba."}]

            synth.synthesize(lines=lines, episode_id="ep_running_no_checksum", resume=False)
            run_dir = os.path.join(cfg.checkpoint_dir, "ep_running_no_checksum")
            manifest_path = os.path.join(run_dir, "audio_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            manifest["segments"][0]["status"] = "running"
            manifest["segments"][0]["error"] = ""
            manifest["segments"][0]["error_kind"] = ""
            manifest["segments"][0].pop("checksum_sha256", None)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")

            client.requests_made = 0
            result = synth.synthesize(
                lines=lines,
                episode_id="ep_running_no_checksum",
                resume=True,
                resume_force=True,
            )
            self.assertTrue(result.segment_files)
            self.assertGreater(client.requests_made, 0)
            with open(manifest_path, "r", encoding="utf-8") as f:
                resumed_manifest = json.load(f)
            self.assertEqual(resumed_manifest["segments"][0]["status"], "done")
            self.assertTrue(str(resumed_manifest["segments"][0].get("checksum_sha256", "")).strip())

    def test_cross_chunk_parallel_merges_groups_into_single_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AudioConfig.from_env(profile_name="short")
            cfg = dataclasses.replace(
                cfg,
                checkpoint_dir=os.path.join(tmp, "audio_ckpt"),
                chunk_lines=1,
                pause_between_segments_ms=0,
                max_concurrent=2,
                cross_chunk_parallel=True,
            )
            reliability = ReliabilityConfig.from_env()
            logger = Logger.create(
                LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
            )
            client = FakeTTSClient()
            synth = TTSSynthesizer(config=cfg, reliability=reliability, logger=logger, client=client)  # type: ignore[arg-type]
            lines = [
                {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Bloque uno de prueba."},
                {"speaker": "Lucia", "role": "Host2", "instructions": "x", "text": "Bloque dos de prueba."},
                {"speaker": "Carlos", "role": "Host1", "instructions": "x", "text": "Bloque tres de prueba."},
            ]
            with mock.patch.object(synth.logger, "info", wraps=synth.logger.info) as info_spy:
                result = synth.synthesize(lines=lines, episode_id="ep_parallel", resume=False)
            self.assertTrue(result.segment_files)
            chunk_start_calls = [
                call
                for call in info_spy.call_args_list
                if call.args and call.args[0] == "tts_chunk_start"
            ]
            self.assertEqual(len(chunk_start_calls), 1)
            self.assertEqual(chunk_start_calls[0].kwargs.get("chunk_id"), "parallel_all")


if __name__ == "__main__":
    unittest.main()

