import argparse
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


class _FakeClient:
    requests_made = 2
    estimated_cost_usd = 0.11
    tts_retries_total = 0


class IntegrationRawOnlyFallbackTests(unittest.TestCase):
    def test_write_podcast_run_summary_is_atomic_on_dump_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = os.path.join(tmp, "podcast_run_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"status": "old", "value": 1}, f)
            with mock.patch.object(make_podcast.json, "dump", side_effect=RuntimeError("dump failed")):
                with self.assertRaises(RuntimeError):
                    make_podcast._write_podcast_run_summary(summary_path, {"status": "new"})  # noqa: SLF001
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "old")
            self.assertEqual(payload.get("value"), 1)
            self.assertFalse(os.path.exists(f"{summary_path}.tmp"))

    def test_audio_canary_profiles_short_standard_long_succeed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for profile in ("short", "standard", "long"):
                with self.subTest(profile=profile):
                    script_path = os.path.join(tmp, f"script_{profile}.json")
                    outdir = os.path.join(tmp, f"out_{profile}")
                    os.makedirs(outdir, exist_ok=True)
                    payload = {
                        "lines": [
                            {
                                "speaker": "Carlos",
                                "role": "Host1",
                                "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                                "text": "Bloque 1 con contexto claro y objetivo del episodio.",
                            },
                            {
                                "speaker": "Lucia",
                                "role": "Host2",
                                "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                                "text": "Bloque 2 con ejemplos practicos, decisiones y recomendaciones utiles.",
                            },
                        ]
                    }
                    with open(script_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)

                    ckpt_dir = os.path.join(outdir, ".audio_checkpoints", f"episode_{profile}")
                    seg_dir = os.path.join(ckpt_dir, "segments")
                    os.makedirs(seg_dir, exist_ok=True)
                    seg_file = os.path.join(seg_dir, "seg_0001.mp3")
                    with open(seg_file, "wb") as f:
                        f.write(b"AUDIO")

                    fake_tts_result = SimpleNamespace(
                        segment_files=[seg_file],
                        manifest_path=os.path.join(ckpt_dir, "audio_manifest.json"),
                        summary_path=os.path.join(ckpt_dir, "run_summary.json"),
                        checkpoint_dir=ckpt_dir,
                    )
                    args = argparse.Namespace(
                        script_path=script_path,
                        outdir=outdir,
                        basename=f"episode_{profile}",
                        profile=profile,
                        resume=False,
                        resume_force=False,
                        force_unlock=False,
                        allow_raw_only=True,
                        verbose=False,
                        debug=False,
                        dry_run_cleanup=False,
                        force_clean=False,
                    )

                    fake_mixer = mock.Mock()
                    fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
                    fake_synth = mock.Mock()
                    fake_synth.synthesize.return_value = fake_tts_result

                    with mock.patch.object(make_podcast, "parse_args", return_value=args):
                        with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                            with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                                with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                                    with mock.patch.dict(
                                        os.environ,
                                        {"SCRIPT_QUALITY_GATE_ACTION": "off"},
                                        clear=False,
                                    ):
                                        rc = make_podcast.main()

                    self.assertEqual(rc, 0)
                    final_path = os.path.join(outdir, f"episode_{profile}_raw_only.mp3")
                    self.assertTrue(os.path.exists(final_path))
                    summary_path = os.path.join(ckpt_dir, "podcast_run_summary.json")
                    self.assertTrue(os.path.exists(summary_path))
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    self.assertEqual(summary.get("status"), "completed")
                    self.assertEqual(summary.get("output_mode"), "raw_only")

    def test_main_uses_raw_only_when_ffmpeg_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.json")
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Hola mundo.",
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": "Seguimos con otra linea.",
                    },
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            ckpt_dir = os.path.join(outdir, ".audio_checkpoints", "episode")
            seg_dir = os.path.join(ckpt_dir, "segments")
            os.makedirs(seg_dir, exist_ok=True)
            seg1 = os.path.join(seg_dir, "seg_0001.mp3")
            seg2 = os.path.join(seg_dir, "seg_0002.mp3")
            with open(seg1, "wb") as f:
                f.write(b"AAA")
            with open(seg2, "wb") as f:
                f.write(b"BBB")

            fake_tts_result = SimpleNamespace(
                segment_files=[seg1, seg2],
                manifest_path=os.path.join(ckpt_dir, "audio_manifest.json"),
                summary_path=os.path.join(ckpt_dir, "run_summary.json"),
                checkpoint_dir=ckpt_dir,
            )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")

            fake_synth = mock.Mock()
            fake_synth.synthesize.return_value = fake_tts_result

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {"SCRIPT_QUALITY_GATE_ACTION": "off"},
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 0)
            final_path = os.path.join(outdir, "episode_raw_only.mp3")
            self.assertTrue(os.path.exists(final_path))
            with open(final_path, "rb") as f:
                self.assertEqual(f.read(), b"AAABBB")
            summary_path = os.path.join(ckpt_dir, "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("output_mode"), "raw_only")
            phase_seconds = summary.get("phase_seconds", {})
            self.assertIn("quality_eval", phase_seconds)
            self.assertIn("tts", phase_seconds)
            self.assertIn("mix", phase_seconds)

    def test_main_raw_only_rejects_non_mp3_segments_without_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_non_mp3.json")
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Hola mundo.",
                    }
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            ckpt_dir = os.path.join(outdir, ".audio_checkpoints", "episode")
            seg_dir = os.path.join(ckpt_dir, "segments")
            os.makedirs(seg_dir, exist_ok=True)
            seg1 = os.path.join(seg_dir, "seg_0001.wav")
            with open(seg1, "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVE")

            fake_tts_result = SimpleNamespace(
                segment_files=[seg1],
                manifest_path=os.path.join(ckpt_dir, "audio_manifest.json"),
                summary_path=os.path.join(ckpt_dir, "run_summary.json"),
                checkpoint_dir=ckpt_dir,
            )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
            fake_synth = mock.Mock()
            fake_synth.synthesize.return_value = fake_tts_result

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {"SCRIPT_QUALITY_GATE_ACTION": "off"},
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 1)
            self.assertFalse(os.path.exists(os.path.join(outdir, "episode_raw_only.mp3")))

    def test_main_writes_normalized_script_snapshot_and_references_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_structural.json")
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Entramos al Bloque 1 con contexto inicial.",
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": "Seguimos con Bloque 2 y datos importantes.",
                    },
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Cerramos el Bloque 4 con recomendaciones y...",
                    },
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            ckpt_dir = os.path.join(outdir, ".audio_checkpoints", "episode")
            seg_dir = os.path.join(ckpt_dir, "segments")
            os.makedirs(seg_dir, exist_ok=True)
            seg1 = os.path.join(seg_dir, "seg_0001.mp3")
            with open(seg1, "wb") as f:
                f.write(b"AAA")

            fake_tts_result = SimpleNamespace(
                segment_files=[seg1],
                manifest_path=os.path.join(ckpt_dir, "audio_manifest.json"),
                summary_path=os.path.join(ckpt_dir, "run_summary.json"),
                checkpoint_dir=ckpt_dir,
            )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
            fake_synth = mock.Mock()
            fake_synth.synthesize.return_value = fake_tts_result

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {"SCRIPT_QUALITY_GATE_ACTION": "off"},
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 0)
            normalized_script_path = os.path.join(ckpt_dir, "normalized_script.json")
            self.assertTrue(os.path.exists(normalized_script_path))
            with open(normalized_script_path, "r", encoding="utf-8") as f:
                normalized_payload = json.load(f)
            text_blob = "\n".join(str(line.get("text", "")) for line in normalized_payload.get("lines", []))
            self.assertIn("Bloque 3", text_blob)
            self.assertNotIn("Bloque 4", text_blob)
            self.assertNotIn("...", text_blob)

            summary_path = os.path.join(ckpt_dir, "podcast_run_summary.json")
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("normalized_script_path"), normalized_script_path)

    def test_precheck_hardening_respects_configured_max_consecutive_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_speaker_runs.json")
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Contexto tecnico inicial completo para la audiencia.",
                    },
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Desarrollo de decisiones con tradeoffs y mitigaciones claras.",
                    },
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Cierre de hallazgos con recomendaciones para ejecucion.",
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": "Gracias por escuchar, nos vemos en el siguiente episodio.",
                    },
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            ckpt_dir = os.path.join(outdir, ".audio_checkpoints", "episode")
            seg_dir = os.path.join(ckpt_dir, "segments")
            os.makedirs(seg_dir, exist_ok=True)
            seg1 = os.path.join(seg_dir, "seg_0001.mp3")
            with open(seg1, "wb") as f:
                f.write(b"AAA")

            fake_tts_result = SimpleNamespace(
                segment_files=[seg1],
                manifest_path=os.path.join(ckpt_dir, "audio_manifest.json"),
                summary_path=os.path.join(ckpt_dir, "run_summary.json"),
                checkpoint_dir=ckpt_dir,
            )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
            fake_synth = mock.Mock()
            fake_synth.synthesize.return_value = fake_tts_result

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {
                                    "SCRIPT_QUALITY_GATE_ACTION": "off",
                                    "SCRIPT_QUALITY_MAX_CONSECUTIVE_SAME_SPEAKER": "3",
                                },
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 0)
            normalized_script_path = os.path.join(ckpt_dir, "normalized_script.json")
            self.assertTrue(os.path.exists(normalized_script_path))
            with open(normalized_script_path, "r", encoding="utf-8") as f:
                normalized_payload = json.load(f)
            normalized_lines = list(normalized_payload.get("lines", []))
            self.assertGreaterEqual(len(normalized_lines), 3)
            self.assertEqual(normalized_lines[0].get("role"), "Host1")
            self.assertEqual(normalized_lines[1].get("role"), "Host1")
            self.assertEqual(normalized_lines[2].get("role"), "Host1")

    def test_failure_summary_keeps_normalized_script_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_failure.json")
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Bloque 1 con base.",
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": "Bloque 2 con continuidad.",
                    },
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Bloque 4 con cierre incompleto y...",
                    },
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            ckpt_dir = os.path.join(outdir, ".audio_checkpoints", "episode")
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.return_value = None
            fake_synth = mock.Mock()
            fake_synth.synthesize.side_effect = make_podcast.TTSOperationError(
                "network down",
                error_kind="network",
            )

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {"SCRIPT_QUALITY_GATE_ACTION": "off"},
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 1)
            normalized_script_path = os.path.join(ckpt_dir, "normalized_script.json")
            self.assertTrue(os.path.exists(normalized_script_path))
            summary_path = os.path.join(ckpt_dir, "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("status"), "failed")
            self.assertEqual(summary.get("normalized_script_path"), normalized_script_path)

    def test_audio_orchestrated_retry_recovers_from_transient_tts_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_retry_ok.json")
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Bloque 1 con contexto suficiente.",
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": "Bloque 2 con cierre concreto y accionable.",
                    },
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            ckpt_dir = os.path.join(outdir, ".audio_checkpoints", "episode")
            seg_dir = os.path.join(ckpt_dir, "segments")
            os.makedirs(seg_dir, exist_ok=True)
            seg1 = os.path.join(seg_dir, "seg_0001.mp3")
            with open(seg1, "wb") as f:
                f.write(b"AAA")

            fake_tts_result = SimpleNamespace(
                segment_files=[seg1],
                manifest_path=os.path.join(ckpt_dir, "audio_manifest.json"),
                summary_path=os.path.join(ckpt_dir, "run_summary.json"),
                checkpoint_dir=ckpt_dir,
            )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
            fake_synth = mock.Mock()
            fake_synth.synthesize.side_effect = [
                make_podcast.TTSOperationError("network down", error_kind="network"),
                fake_tts_result,
            ]

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {
                                    "SCRIPT_QUALITY_GATE_ACTION": "off",
                                    "AUDIO_ORCHESTRATED_RETRY_ENABLED": "1",
                                    "AUDIO_ORCHESTRATED_MAX_ATTEMPTS": "2",
                                    "AUDIO_ORCHESTRATED_RETRY_BACKOFF_MS": "0",
                                },
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 0)
            self.assertEqual(fake_synth.synthesize.call_count, 2)
            first_call = fake_synth.synthesize.call_args_list[0].kwargs
            second_call = fake_synth.synthesize.call_args_list[1].kwargs
            self.assertFalse(bool(first_call.get("resume", False)))
            self.assertFalse(bool(first_call.get("resume_force", False)))
            self.assertFalse(bool(first_call.get("force_unlock", False)))
            self.assertTrue(bool(second_call.get("resume", False)))
            self.assertTrue(bool(second_call.get("resume_force", False)))
            self.assertTrue(bool(second_call.get("force_unlock", False)))
            summary_path = os.path.join(ckpt_dir, "podcast_run_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary.get("status"), "completed")
            self.assertEqual(int(summary.get("audio_orchestrated_retry_attempts_used", 0)), 2)
            self.assertEqual(int(summary.get("audio_orchestrated_retry_recoveries", 0)), 1)

    def test_audio_orchestrated_retry_does_not_retry_nonrecoverable_tts_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_retry_no.json")
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Linea de prueba para retry.",
                    }
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
            fake_synth = mock.Mock()
            fake_synth.synthesize.side_effect = make_podcast.TTSOperationError(
                "resume blocked",
                error_kind="resume_blocked",
            )

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {
                                    "SCRIPT_QUALITY_GATE_ACTION": "off",
                                    "AUDIO_ORCHESTRATED_RETRY_ENABLED": "1",
                                    "AUDIO_ORCHESTRATED_MAX_ATTEMPTS": "3",
                                    "AUDIO_ORCHESTRATED_RETRY_BACKOFF_MS": "0",
                                },
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 1)
            self.assertEqual(fake_synth.synthesize.call_count, 1)

    def test_mix_failure_uses_tts_checkpoint_run_summary_path_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_mix_failure.json")
            outdir = os.path.join(tmp, "out")
            script_checkpoint_dir = os.path.join(tmp, "script_ckpt")
            configured_audio_ckpt = os.path.join(tmp, "audio_cfg_ckpt")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Bloque 1 con contexto tecnico y decisiones para reducir incidentes.",
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": "Bloque 2 con validaciones, monitoreo y planes de mitigacion.",
                    },
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            tts_ckpt_dir = os.path.join(tmp, "tts_ckpt", "episode")
            seg_dir = os.path.join(tts_ckpt_dir, "segments")
            os.makedirs(seg_dir, exist_ok=True)
            seg1 = os.path.join(seg_dir, "seg_0001.mp3")
            with open(seg1, "wb") as f:
                f.write(b"AAA")

            fake_tts_result = SimpleNamespace(
                segment_files=[seg1],
                manifest_path=os.path.join(tts_ckpt_dir, "audio_manifest.json"),
                summary_path=os.path.join(tts_ckpt_dir, "run_summary.json"),
                checkpoint_dir=tts_ckpt_dir,
            )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile=None,
                run_token=None,
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
            fake_mixer.mix.side_effect = RuntimeError("mix failed after synthesis")
            fake_synth = mock.Mock()
            fake_synth.synthesize.return_value = fake_tts_result

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {
                                    "SCRIPT_CHECKPOINT_DIR": script_checkpoint_dir,
                                    "AUDIO_CHECKPOINT_DIR": configured_audio_ckpt,
                                    "SCRIPT_QUALITY_GATE_ACTION": "off",
                                },
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 1)
            expected_summary_path = os.path.join(tts_ckpt_dir, "podcast_run_summary.json")
            self.assertTrue(os.path.exists(expected_summary_path))
            with open(expected_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(str(summary.get("status", "")), "failed")

            manifest_path = make_podcast.run_manifest_path(
                checkpoint_dir=script_checkpoint_dir,
                episode_id="episode",
            )
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            audio_block = dict(manifest.get("audio", {}))
            self.assertEqual(
                str(audio_block.get("podcast_run_summary_path", "")),
                expected_summary_path,
            )

    def test_script_gate_action_override_takes_precedence_in_make_podcast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_quality_fail.json")
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Analizamos decisiones de arquitectura con ejemplos y contexto.",
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": "Revisamos riesgos, tradeoffs y acciones para el equipo tecnico.",
                    },
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile=None,
                run_token=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
            fake_synth = mock.Mock()

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {
                                    "SCRIPT_QUALITY_GATE_ACTION": "off",
                                    "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "enforce",
                                    "SCRIPT_QUALITY_GATE_EVALUATOR": "rules",
                                },
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 4)
            fake_synth.synthesize.assert_not_called()

    def test_make_podcast_updates_manifest_run_token_on_standalone_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_manifest_token.json")
            outdir = os.path.join(tmp, "out")
            script_checkpoint_dir = os.path.join(tmp, "script_ckpt")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Bloque 1 con contexto claro y decisiones practicas.",
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": "En resumen, priorizamos cambios graduales y medicion continua.",
                    },
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Gracias por escuchar, nos vemos en el siguiente episodio.",
                    },
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            episode_id = "episode"
            old_token = "old-run-token"
            new_token = "new-run-token"
            audio_checkpoint_dir = os.path.join(outdir, ".audio_checkpoints")
            make_podcast.init_manifest(
                checkpoint_dir=script_checkpoint_dir,
                episode_id=episode_id,
                run_token=old_token,
                script_output_path=script_path,
                script_checkpoint_dir=script_checkpoint_dir,
                audio_checkpoint_dir=audio_checkpoint_dir,
            )

            ckpt_dir = os.path.join(audio_checkpoint_dir, episode_id)
            seg_dir = os.path.join(ckpt_dir, "segments")
            os.makedirs(seg_dir, exist_ok=True)
            seg1 = os.path.join(seg_dir, "seg_0001.mp3")
            with open(seg1, "wb") as f:
                f.write(b"AAA")

            fake_tts_result = SimpleNamespace(
                segment_files=[seg1],
                manifest_path=os.path.join(ckpt_dir, "audio_manifest.json"),
                summary_path=os.path.join(ckpt_dir, "run_summary.json"),
                checkpoint_dir=ckpt_dir,
            )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename=episode_id,
                profile=None,
                run_token=new_token,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
            fake_synth = mock.Mock()
            fake_synth.synthesize.return_value = fake_tts_result

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {
                                    "SCRIPT_CHECKPOINT_DIR": script_checkpoint_dir,
                                    "SCRIPT_QUALITY_GATE_ACTION": "off",
                                },
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 0)
            manifest_path = make_podcast.run_manifest_path(
                checkpoint_dir=script_checkpoint_dir,
                episode_id=episode_id,
            )
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(str(manifest.get("run_token", "")), new_token)

    def test_make_podcast_reuses_manifest_run_token_when_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_manifest_token_reuse.json")
            outdir = os.path.join(tmp, "out")
            script_checkpoint_dir = os.path.join(tmp, "script_ckpt")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Bloque 1 con contexto util y decisiones accionables.",
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": "Gracias por escuchar, nos vemos en el siguiente episodio.",
                    },
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            episode_id = "episode"
            existing_token = "existing-manifest-token"
            audio_checkpoint_dir = os.path.join(outdir, ".audio_checkpoints")
            make_podcast.init_manifest(
                checkpoint_dir=script_checkpoint_dir,
                episode_id=episode_id,
                run_token=existing_token,
                script_output_path=script_path,
                script_checkpoint_dir=script_checkpoint_dir,
                audio_checkpoint_dir=audio_checkpoint_dir,
            )

            ckpt_dir = os.path.join(audio_checkpoint_dir, episode_id)
            seg_dir = os.path.join(ckpt_dir, "segments")
            os.makedirs(seg_dir, exist_ok=True)
            seg1 = os.path.join(seg_dir, "seg_0001.mp3")
            with open(seg1, "wb") as f:
                f.write(b"AAA")

            fake_tts_result = SimpleNamespace(
                segment_files=[seg1],
                manifest_path=os.path.join(ckpt_dir, "audio_manifest.json"),
                summary_path=os.path.join(ckpt_dir, "run_summary.json"),
                checkpoint_dir=ckpt_dir,
            )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename=episode_id,
                profile=None,
                run_token=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
            fake_synth = mock.Mock()
            fake_synth.synthesize.return_value = fake_tts_result

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.dict(
                                os.environ,
                                {
                                    "SCRIPT_CHECKPOINT_DIR": script_checkpoint_dir,
                                    "SCRIPT_QUALITY_GATE_ACTION": "off",
                                },
                                clear=False,
                            ):
                                rc = make_podcast.main()

            self.assertEqual(rc, 0)
            summary_path = os.path.join(ckpt_dir, "podcast_run_summary.json")
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(str(summary.get("run_token", "")), existing_token)

    def test_make_podcast_profile_default_gate_action_warn_for_short(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script_short_default_gate.json")
            outdir = os.path.join(tmp, "out")
            os.makedirs(outdir, exist_ok=True)
            payload = {
                "lines": [
                    {
                        "speaker": "Carlos",
                        "role": "Host1",
                        "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
                        "text": "Linea corta que provocaria fallo en quality gate estricto.",
                    },
                    {
                        "speaker": "Lucia",
                        "role": "Host2",
                        "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
                        "text": "Otra linea corta sin resumen ni cierre completo.",
                    },
                ]
            }
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            ckpt_dir = os.path.join(outdir, ".audio_checkpoints", "episode")
            seg_dir = os.path.join(ckpt_dir, "segments")
            os.makedirs(seg_dir, exist_ok=True)
            seg1 = os.path.join(seg_dir, "seg_0001.mp3")
            with open(seg1, "wb") as f:
                f.write(b"AAA")

            fake_tts_result = SimpleNamespace(
                segment_files=[seg1],
                manifest_path=os.path.join(ckpt_dir, "audio_manifest.json"),
                summary_path=os.path.join(ckpt_dir, "run_summary.json"),
                checkpoint_dir=ckpt_dir,
            )
            args = argparse.Namespace(
                script_path=script_path,
                outdir=outdir,
                basename="episode",
                profile="short",
                run_token=None,
                resume=False,
                resume_force=False,
                force_unlock=False,
                allow_raw_only=True,
                verbose=False,
                debug=False,
                dry_run_cleanup=False,
                force_clean=False,
            )

            fake_mixer = mock.Mock()
            fake_mixer.check_dependencies.side_effect = RuntimeError("ffmpeg missing")
            fake_synth = mock.Mock()
            fake_synth.synthesize.return_value = fake_tts_result

            with mock.patch.object(make_podcast, "parse_args", return_value=args):
                with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeClient()):
                    with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                        with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                            with mock.patch.object(
                                make_podcast,
                                "evaluate_script_quality",
                                return_value={
                                    "status": "failed",
                                    "pass": False,
                                    "reasons": ["summary_ok", "closing_ok"],
                                },
                            ):
                                with mock.patch.dict(
                                    os.environ,
                                    {
                                        "SCRIPT_QUALITY_GATE_ACTION": "",
                                        "SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "",
                                    },
                                    clear=False,
                                ):
                                    rc = make_podcast.main()

            self.assertEqual(rc, 0)
            fake_synth.synthesize.assert_called_once()
            summary_path = os.path.join(ckpt_dir, "podcast_run_summary.json")
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(str(summary.get("script_gate_action_effective", "")), "warn")


if __name__ == "__main__":
    unittest.main()

