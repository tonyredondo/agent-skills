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
from pipeline.logging_utils import Logger  # noqa: E402
from pipeline.tts_synthesizer import TTSSynthesizer  # noqa: E402


class _StubClient:
    requests_made = 0
    estimated_cost_usd = 0.0

    def synthesize_speech(self, **kwargs):  # noqa: ANN003, ANN201
        return b"ID3"


class TTSSynthesizerHelpersExtraTests(unittest.TestCase):
    def setUp(self) -> None:
        cfg = AudioConfig.from_env(profile_name="short")
        self.cfg = dataclasses.replace(cfg, chunk_lines=2, pause_between_segments_ms=0, tts_max_chars_per_segment=20)
        self.logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )
        self.reliability = ReliabilityConfig.from_env()
        self.synth = TTSSynthesizer(
            config=self.cfg,
            reliability=self.reliability,
            logger=self.logger,
            client=_StubClient(),  # type: ignore[arg-type]
        )

    def test_build_segments_applies_default_instructions(self) -> None:
        lines = [{"speaker": "Ana", "role": "Host2", "instructions": "", "text": "hola mundo desde aqui"}]
        segments = self.synth._build_segments(lines)
        self.assertGreaterEqual(len(segments), 1)
        self.assertIn("Speak in a bright, friendly, conversational tone.", segments[0]["instructions"])
        self.assertNotIn("|", segments[0]["instructions"])
        self.assertEqual(segments[0]["voice"], "marin")

    def test_build_segments_assigns_voice_from_speaker_name_hints(self) -> None:
        lines = [
            {"speaker": "Laura Martinez", "role": "Host1", "instructions": "x", "text": "hola"},
            {"speaker": "Diego Herrera", "role": "Host2", "instructions": "x", "text": "que tal"},
        ]
        segments = self.synth._build_segments(lines)
        self.assertEqual(segments[0]["voice"], "marin")
        self.assertEqual(segments[1]["voice"], "cedar")

    def test_build_segments_skips_invalid_lines(self) -> None:
        lines = [
            {"speaker": "", "role": "Host1", "instructions": "x", "text": "hola"},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": ""},
            {"speaker": "Luis", "role": "Host1", "instructions": "x", "text": "texto valido"},
        ]
        segments = self.synth._build_segments(lines)
        self.assertGreaterEqual(len(segments), 1)
        self.assertTrue(all(seg["speaker"] == "Luis" for seg in segments))

    def test_build_segments_raises_when_no_valid_lines(self) -> None:
        with self.assertRaises(RuntimeError):
            self.synth._build_segments([{"speaker": "", "role": "Host1", "instructions": "", "text": ""}])

    def test_ensure_chunk_metadata_adds_fields(self) -> None:
        manifest = {"segments": [{"index": 3, "status": "pending"}]}
        changed = self.synth._ensure_chunk_metadata(manifest)
        self.assertTrue(changed)
        self.assertIn("line_index", manifest["segments"][0])
        self.assertIn("chunk_id", manifest["segments"][0])
        self.assertIn("error_kind", manifest["segments"][0])
        self.assertIn("phase", manifest["segments"][0])
        self.assertIn("speed", manifest["segments"][0])

    def test_ensure_chunk_metadata_noop_when_present(self) -> None:
        manifest = {
            "segments": [
                {
                    "index": 1,
                    "line_index": 1,
                    "chunk_id": 1,
                    "status": "pending",
                    "error_kind": "",
                    "phase": "closing",
                    "speed": self.cfg.tts_speed_closing,
                }
            ]
        }
        changed = self.synth._ensure_chunk_metadata(manifest)
        self.assertFalse(changed)

    def test_build_segments_assigns_phase_speed_and_refined_instructions(self) -> None:
        lines = [
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "linea 1"},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "linea 2"},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "linea 3"},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "linea 4"},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "linea 5"},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "linea 6"},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "linea 7"},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "linea 8"},
        ]
        segments = self.synth._build_segments(lines)
        self.assertEqual(segments[0]["phase"], "intro")
        self.assertEqual(segments[0]["speed"], self.cfg.tts_speed_intro)
        self.assertEqual(segments[-1]["phase"], "closing")
        self.assertEqual(segments[-1]["speed"], self.cfg.tts_speed_closing)
        self.assertIn("Speak in a warm, confident, conversational tone.", segments[0]["instructions"])
        self.assertIn("For this intro segment", segments[0]["instructions"])
        self.assertNotIn(" x ", f" {segments[0]['instructions']} ")

    def test_build_segments_replaces_legacy_structured_instruction_format(self) -> None:
        lines = [
            {
                "speaker": "Ana",
                "role": "Host1",
                "instructions": (
                    "Tone: Formal y pausado | Pacing: Lento | Emotion: Calma | "
                    "Pronunciation: Clara | Pauses: Largas"
                ),
                "text": "linea 1",
            }
        ]
        segments = self.synth._build_segments(lines)
        instructions = segments[0]["instructions"]
        self.assertNotIn("Tone: Formal y pausado", instructions)
        self.assertNotIn("|", instructions)
        self.assertIn("Speak in a warm, confident, conversational tone.", instructions)

    def test_build_segments_guarantees_intro_and_closing_for_short_valid_scripts(self) -> None:
        lines = [
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "linea 1"},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "linea 2"},
            {"speaker": "Ana", "role": "Host1", "instructions": "x", "text": "linea 3"},
            {"speaker": "Luis", "role": "Host2", "instructions": "x", "text": "linea 4"},
        ]
        segments = self.synth._build_segments(lines)
        phases = [seg["phase"] for seg in segments]
        self.assertIn("intro", phases)
        self.assertIn("closing", phases)
        self.assertEqual(segments[0]["phase"], "intro")
        self.assertEqual(segments[-1]["phase"], "closing")

    def test_ensure_chunk_metadata_normalizes_invalid_phase_and_speed(self) -> None:
        manifest = {
            "segments": [
                {
                    "index": 1,
                    "line_index": 1,
                    "chunk_id": 1,
                    "status": "pending",
                    "error_kind": "",
                    "phase": "INVALID_PHASE",
                    "speed": "bad",
                }
            ]
        }
        changed = self.synth._ensure_chunk_metadata(manifest)
        self.assertTrue(changed)
        self.assertEqual(manifest["segments"][0]["phase"], "body")
        self.assertEqual(manifest["segments"][0]["speed"], self.cfg.tts_speed_body)

    def test_ensure_chunk_metadata_assigns_missing_phases_by_segment_order(self) -> None:
        manifest = {
            "segments": [
                {"index": 1, "line_index": 10, "chunk_id": 1, "status": "pending", "error_kind": ""},
                {"index": 2, "line_index": 20, "chunk_id": 1, "status": "pending", "error_kind": ""},
                {"index": 3, "line_index": 30, "chunk_id": 2, "status": "pending", "error_kind": ""},
            ]
        }
        changed = self.synth._ensure_chunk_metadata(manifest)
        self.assertTrue(changed)
        phases = [seg["phase"] for seg in manifest["segments"]]
        self.assertEqual(phases[0], "intro")
        self.assertEqual(phases[-1], "closing")

    def test_pending_groups_chunked(self) -> None:
        manifest = {
            "segments": [
                {"chunk_id": 2, "status": "pending"},
                {"chunk_id": 1, "status": "pending"},
                {"chunk_id": 2, "status": "done"},
                {"chunk_id": 1, "status": "pending"},
            ]
        }
        groups = self.synth._pending_groups(manifest)
        self.assertEqual([g[0] for g in groups], [1, 2])
        self.assertEqual(groups[0][1], [1, 3])

    def test_pending_groups_single_when_chunk_lines_disabled(self) -> None:
        synth = dataclasses.replace(self.synth, config=dataclasses.replace(self.cfg, chunk_lines=0))
        manifest = {
            "segments": [
                {"chunk_id": 3, "status": "pending"},
                {"chunk_id": 1, "status": "done"},
                {"chunk_id": 2, "status": "pending"},
            ]
        }
        groups = synth._pending_groups(manifest)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][1], [0, 2])

    def test_manifest_checksum_changes_with_status(self) -> None:
        manifest = {
            "checkpoint_version": 2,
            "episode_id": "ep",
            "status": "running",
            "segments": [{"segment_id": "0001", "index": 1, "status": "pending", "attempts": 0}],
        }
        a = self.synth._manifest_checksum(manifest)
        manifest["segments"][0]["status"] = "done"
        b = self.synth._manifest_checksum(manifest)
        self.assertNotEqual(a, b)

    def test_manifest_checksum_changes_with_file_name(self) -> None:
        manifest = {
            "checkpoint_version": 2,
            "episode_id": "ep",
            "status": "running",
            "segments": [
                {
                    "segment_id": "0001",
                    "index": 1,
                    "file_name": "seg_0001.mp3",
                    "status": "done",
                    "attempts": 1,
                }
            ],
        }
        a = self.synth._manifest_checksum(manifest)
        manifest["segments"][0]["file_name"] = "seg_0001_alt.mp3"
        b = self.synth._manifest_checksum(manifest)
        self.assertNotEqual(a, b)

    def test_build_output_segment_files_includes_pause_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = mock.Mock()
            store.segments_dir = tmp
            manifest = {
                "segments": [
                    {"index": 2, "segment_id": "0002", "file_name": "seg_0002.mp3", "status": "done"},
                    {"index": 1, "segment_id": "0001", "file_name": "seg_0001.mp3", "status": "done"},
                ]
            }
            synth = dataclasses.replace(self.synth, config=dataclasses.replace(self.cfg, pause_between_segments_ms=200))
            with mock.patch.object(synth, "_create_pause_file", return_value=True):
                files = synth._build_output_segment_files(manifest, store)
            self.assertEqual(files[0], os.path.join(tmp, "seg_0001.mp3"))
            self.assertTrue(any("pause_" in os.path.basename(p) for p in files))

    def test_create_pause_file_short_circuits_when_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pause.mp3")
            with open(path, "wb") as f:
                f.write(b"ID3")
            self.assertTrue(self.synth._create_pause_file(path, 100))

    def test_create_pause_file_no_ffmpeg_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pause.mp3")
            with mock.patch("pipeline.tts_synthesizer.shutil.which", return_value=None):
                self.assertFalse(self.synth._create_pause_file(path, 100))

    def test_write_json_atomic_replaces_existing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = os.path.join(tmp, "run_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"status": "old", "value": 1}, f)
            self.synth._write_json_atomic(summary_path, {"status": "new", "value": 2})
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "new")
            self.assertEqual(payload.get("value"), 2)

    def test_write_json_atomic_keeps_previous_file_on_dump_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = os.path.join(tmp, "run_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"status": "old", "value": 1}, f)
            with mock.patch("pipeline.tts_synthesizer.json.dump", side_effect=RuntimeError("dump failed")):
                with self.assertRaises(RuntimeError):
                    self.synth._write_json_atomic(summary_path, {"status": "new"})
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("status"), "old")
            self.assertEqual(payload.get("value"), 1)
            tmp_files = [name for name in os.listdir(tmp) if name.startswith("run_summary.json.") and name.endswith(".tmp")]
            self.assertEqual(tmp_files, [])


if __name__ == "__main__":
    unittest.main()

