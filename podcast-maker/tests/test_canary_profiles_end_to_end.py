import argparse
import dataclasses
import json
import os
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import make_podcast  # noqa: E402
from export_debug_bundle import create_debug_bundle, parse_args as parse_bundle_args  # noqa: E402
from pipeline.config import LoggingConfig, ReliabilityConfig, ScriptConfig  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.script_generator import ScriptGenerator  # noqa: E402


class _FakeScriptClient:
    def __init__(self) -> None:
        self.requests_made = 0
        self.estimated_cost_usd = 0.0
        self.script_retries_total = 0
        self.script_json_parse_failures = 0
        self._seq = 0

    def generate_script_json(self, *, prompt, schema, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        self._seq += 1
        return {
            "lines": [
                {
                    "speaker": "Carlos",
                    "role": "Host1",
                    "instructions": (
                        "Voice Affect: Warm and confident | Tone: Conversational | "
                        "Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief"
                    ),
                    "text": (
                        f"Bloque {self._seq} con contexto operativo, decisiones practicas y ejemplos claros "
                        "para que la audiencia pueda aplicar mejoras de forma inmediata."
                    ),
                },
                {
                    "speaker": "Lucia",
                    "role": "Host2",
                    "instructions": (
                        "Voice Affect: Bright and friendly | Tone: Conversational | "
                        "Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief"
                    ),
                    "text": (
                        f"Seguimos en Bloque {self._seq} con riesgos, tradeoffs y recomendaciones accionables "
                        "manteniendo una narrativa coherente de inicio a cierre."
                    ),
                },
            ]
        }

    def generate_freeform_text(self, *, prompt, max_output_tokens, stage):  # noqa: ANN001
        self.requests_made += 1
        return "Resumen breve con hechos clave y continuidad narrativa."


class _FakeAudioClient:
    requests_made = 2
    tts_requests_made = 2
    estimated_cost_usd = 0.11
    tts_retries_total = 0


class CanaryProfilesEndToEndTests(unittest.TestCase):
    def test_canary_profiles_5_15_30_script_plus_audio_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for profile, minutes in (("short", 5), ("standard", 15), ("long", 30)):
                with self.subTest(profile=profile, minutes=minutes):
                    episode_id = f"episode_{profile}_{minutes}"
                    script_cfg = ScriptConfig.from_env(
                        profile_name=profile,
                        target_minutes=minutes,
                        words_per_min=130,
                        min_words=max(80, int(minutes * 5)),
                        max_words=max(140, int(minutes * 9)),
                    )
                    script_cfg = dataclasses.replace(
                        script_cfg,
                        checkpoint_dir=os.path.join(tmp, f"script_ckpt_{profile}_{minutes}"),
                        max_continuations_per_chunk=3,
                        no_progress_rounds=10,
                        min_word_delta=1,
                    )
                    reliability = ReliabilityConfig.from_env()
                    logger = Logger.create(
                        LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
                    )
                    script_client = _FakeScriptClient()
                    generator = ScriptGenerator(
                        config=script_cfg,
                        reliability=reliability,
                        logger=logger,
                        client=script_client,  # type: ignore[arg-type]
                    )
                    source_text = (
                        ("tema canary con detalles tecnicos, contexto y decisiones aplicables " * max(120, int(minutes * 30)))
                        .strip()
                    )
                    script_path = os.path.join(tmp, f"{episode_id}.json")
                    result = generator.generate(
                        source_text=source_text,
                        output_path=script_path,
                        episode_id=episode_id,
                    )
                    self.assertTrue(os.path.exists(result.output_path))
                    self.assertGreaterEqual(result.word_count, script_cfg.min_words)
                    with open(result.run_summary_path, "r", encoding="utf-8") as f:
                        script_summary = json.load(f)
                    script_summary["quality_gate_executed"] = False
                    script_summary["script_gate_action_effective"] = "off"
                    with open(result.run_summary_path, "w", encoding="utf-8") as f:
                        json.dump(script_summary, f, ensure_ascii=False, indent=2)
                        f.write("\n")

                    outdir = os.path.join(tmp, f"audio_out_{profile}_{minutes}")
                    os.makedirs(outdir, exist_ok=True)
                    audio_ckpt_dir = os.path.join(outdir, ".audio_checkpoints", episode_id)
                    segments_dir = os.path.join(audio_ckpt_dir, "segments")
                    os.makedirs(segments_dir, exist_ok=True)
                    seg_1 = os.path.join(segments_dir, "seg_0001.mp3")
                    seg_2 = os.path.join(segments_dir, "seg_0002.mp3")
                    with open(seg_1, "wb") as f:
                        f.write(b"AAA")
                    with open(seg_2, "wb") as f:
                        f.write(b"BBB")
                    manifest_path = os.path.join(audio_ckpt_dir, "audio_manifest.json")
                    summary_path = os.path.join(audio_ckpt_dir, "run_summary.json")
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump({"status": "completed"}, f, ensure_ascii=False, indent=2)
                        f.write("\n")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump({"status": "completed"}, f, ensure_ascii=False, indent=2)
                        f.write("\n")
                    fake_tts_result = SimpleNamespace(
                        segment_files=[seg_1, seg_2],
                        manifest_path=manifest_path,
                        summary_path=summary_path,
                        checkpoint_dir=audio_ckpt_dir,
                    )
                    args = argparse.Namespace(
                        script_path=script_path,
                        outdir=outdir,
                        basename=episode_id,
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
                        with mock.patch.object(make_podcast.OpenAIClient, "from_configs", return_value=_FakeAudioClient()):
                            with mock.patch.object(make_podcast, "AudioMixer", return_value=fake_mixer):
                                with mock.patch.object(make_podcast, "TTSSynthesizer", return_value=fake_synth):
                                    with mock.patch.dict(
                                        os.environ,
                                        {
                                            "SCRIPT_QUALITY_GATE_ACTION": "off",
                                            "SCRIPT_CHECKPOINT_DIR": script_cfg.checkpoint_dir,
                                            "AUDIO_CHECKPOINT_DIR": os.path.join(outdir, ".audio_checkpoints"),
                                        },
                                        clear=False,
                                    ):
                                        rc = make_podcast.main()

                    self.assertEqual(rc, 0)
                    final_path = os.path.join(outdir, f"{episode_id}_raw_only.mp3")
                    self.assertTrue(os.path.exists(final_path))
                    run_summary_path = os.path.join(audio_ckpt_dir, "podcast_run_summary.json")
                    self.assertTrue(os.path.exists(run_summary_path))
                    with open(run_summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    self.assertEqual(summary.get("status"), "completed")
                    self.assertEqual(summary.get("output_mode"), "raw_only")
                    self.assertEqual(int(summary.get("segment_count", 0)), 2)
                    phase_seconds = dict(summary.get("phase_seconds", {}))
                    self.assertIn("tts", phase_seconds)
                    self.assertIn("mix", phase_seconds)

                    bundle_path = os.path.join(tmp, f"bundle_{profile}_{minutes}.zip")
                    bundle_args = parse_bundle_args(
                        [
                            episode_id,
                            "--script-checkpoint-dir",
                            script_cfg.checkpoint_dir,
                            "--audio-checkpoint-dir",
                            os.path.join(outdir, ".audio_checkpoints"),
                            "--script-path",
                            script_path,
                            "--output",
                            bundle_path,
                        ]
                    )
                    bundle_out = create_debug_bundle(bundle_args)
                    self.assertTrue(os.path.exists(bundle_out))
                    with zipfile.ZipFile(bundle_out, "r") as zf:
                        bundle_metadata = json.loads(
                            zf.read("debug_bundle_metadata.json").decode("utf-8")
                        )
                    self.assertTrue(bool(bundle_metadata.get("collection_complete", False)))
                    self.assertEqual(
                        int(bundle_metadata.get("collection_status_counts", {}).get("missing", 0)),
                        0,
                    )


if __name__ == "__main__":
    unittest.main()
