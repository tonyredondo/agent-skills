import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.audio_mixer import AudioMixer, _ffconcat_line, _parse_loudnorm_json, _run  # noqa: E402
from pipeline.config import AudioConfig, LoggingConfig  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402


class AudioMixerHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = Logger.create(
            LoggingConfig(level="ERROR", heartbeat_seconds=1, debug_events=False, include_event_ids=False)
        )

    def test_ffconcat_line_escapes_quotes_and_backslashes(self) -> None:
        line = _ffconcat_line(r"/tmp/it's\segment.mp3")
        self.assertTrue(line.startswith("file '"))
        self.assertTrue(line.endswith("'\n"))
        self.assertIn("'\\''", line)
        self.assertIn("\\\\", line)

    def test_parse_loudnorm_json_uses_last_valid_block(self) -> None:
        stderr_text = (
            "noise {\"input_i\":\"-30\",\"input_tp\":\"-2\",\"input_lra\":\"10\","
            "\"input_thresh\":\"-40\",\"target_offset\":\"1.0\"} "
            "tail {\"input_i\":\"-20\",\"input_tp\":\"-1\",\"input_lra\":\"7\","
            "\"input_thresh\":\"-33\",\"target_offset\":\"0.2\"}"
        )
        parsed = _parse_loudnorm_json(stderr_text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["input_i"], "-20")
        self.assertEqual(parsed["target_offset"], "0.2")

    def test_parse_loudnorm_json_returns_none_for_missing_keys(self) -> None:
        parsed = _parse_loudnorm_json('{"foo": 1, "bar": 2}')
        self.assertIsNone(parsed)

    def test_run_raises_when_command_fails_and_not_allowed(self) -> None:
        failed = subprocess.CompletedProcess(args=["cmd"], returncode=1, stdout="", stderr="boom")
        with mock.patch("pipeline.audio_mixer.subprocess.run", return_value=failed):
            with self.assertRaises(RuntimeError):
                _run(["cmd"], self.logger)

    def test_run_allow_failure_returns_process(self) -> None:
        failed = subprocess.CompletedProcess(args=["cmd"], returncode=1, stdout="", stderr="boom")
        with mock.patch("pipeline.audio_mixer.subprocess.run", return_value=failed):
            proc = _run(["cmd"], self.logger, allow_failure=True)
        self.assertEqual(proc.returncode, 1)

    def test_run_handles_none_stderr(self) -> None:
        failed = subprocess.CompletedProcess(args=["cmd"], returncode=1, stdout="", stderr=None)
        with mock.patch("pipeline.audio_mixer.subprocess.run", return_value=failed):
            with self.assertRaises(RuntimeError):
                _run(["cmd"], self.logger)

    def test_mix_cleans_tmp_files_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seg = os.path.join(tmp, "seg.mp3")
            with open(seg, "wb") as f:
                f.write(b"ID3")

            class _FailingMixer(AudioMixer):
                def _ensure_dependencies(self) -> None:  # type: ignore[override]
                    return None

                def _concat_with_reencode_fallback(self, files, out_path):  # type: ignore[override]  # noqa: ANN001
                    with open(out_path, "wb") as f:
                        f.write(b"raw")

                def _loudnorm_two_pass(self, raw_path, norm_path):  # type: ignore[override]  # noqa: ANN001
                    raise RuntimeError("boom")

            mixer = _FailingMixer(config=AudioConfig.from_env(profile_name="short"), logger=self.logger)
            with self.assertRaises(RuntimeError):
                mixer.mix(segment_files=[seg], outdir=tmp, basename="episode")

            self.assertFalse(os.path.exists(os.path.join(tmp, ".episode.raw.tmp.mp3")))
            self.assertFalse(os.path.exists(os.path.join(tmp, ".episode.norm.tmp.mp3")))
            self.assertFalse(os.path.exists(os.path.join(tmp, ".episode.final.tmp.mp3")))

    def test_concat_copy_writes_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seg_abs = os.path.join(tmp, "seg.mp3")
            with open(seg_abs, "wb") as f:
                f.write(b"ID3")
            seg_rel = os.path.relpath(seg_abs, start=os.getcwd())
            out_path = os.path.join(tmp, "out.mp3")
            mixer = AudioMixer(config=AudioConfig.from_env(profile_name="short"), logger=self.logger)
            captured: dict[str, str] = {}

            def fake_run(command, logger, *, allow_failure=False):  # noqa: ANN001
                del logger, allow_failure
                concat_path = command[command.index("-i") + 1]
                with open(concat_path, "r", encoding="utf-8") as fh:
                    captured["concat"] = fh.read()
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with mock.patch("pipeline.audio_mixer._run", side_effect=fake_run):
                mixer._concat_copy([seg_rel], out_path)

            self.assertIn(_ffconcat_line(os.path.abspath(seg_rel)).strip(), captured.get("concat", ""))


if __name__ == "__main__":
    unittest.main()

