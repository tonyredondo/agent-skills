#!/usr/bin/env python3
from __future__ import annotations

"""Audio post-processing utilities for podcast output.

The mixer concatenates TTS segment MP3 files, applies loudness normalization,
and adds a light EQ pass for final delivery quality.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import AudioConfig
from .logging_utils import Logger


def _run(command: List[str], logger: Logger, *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    """Execute ffmpeg command and raise on failure unless allowed."""
    logger.debug("run_command", command=" ".join(command))
    proc = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0 and not allow_failure:
        logger.error(
            "command_failed",
            command=" ".join(command),
            returncode=proc.returncode,
            stderr=(proc.stderr or "")[-1000:],
        )
        raise RuntimeError(f"Command failed: {' '.join(command)}")
    return proc


def _ffconcat_line(path: str) -> str:
    """Build one safe ffconcat input line for a file path."""
    escaped = path.replace("\\", "\\\\").replace("'", "'\\''")
    return f"file '{escaped}'\n"


def _parse_loudnorm_json(stderr_text: str) -> Optional[Dict[str, str]]:
    """Extract `loudnorm` analysis payload from ffmpeg stderr output."""
    # ffmpeg prints a JSON object in stderr when print_format=json.
    matches = re.findall(r"\{[\s\S]*?\}", stderr_text)
    for candidate in reversed(matches):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        keys = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
        if keys.issubset(payload.keys()):
            return payload
    return None


@dataclass
class AudioMixResult:
    """Final output paths produced by the mixer pipeline."""

    raw_path: str
    norm_path: str
    final_path: str


@dataclass
class AudioMixer:
    """Concatenate and post-process synthesized segment audio."""

    config: AudioConfig
    logger: Logger

    def _ensure_dependencies(self) -> None:
        """Validate required external dependencies for full mixing."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required but not found in PATH")

    def check_dependencies(self) -> None:
        """Public dependency probe used by callers."""
        self._ensure_dependencies()

    def _concat_copy(self, files: List[str], out_path: str) -> None:
        """Try lossless concatenation using ffmpeg concat demuxer."""
        with tempfile.TemporaryDirectory() as tmp:
            concat_file = os.path.join(tmp, "concat.txt")
            with open(concat_file, "w", encoding="utf-8") as f:
                for path in files:
                    f.write(_ffconcat_line(path))
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                self.config.ffmpeg_loglevel,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file,
                "-c",
                "copy",
                out_path,
            ]
            _run(cmd, self.logger)

    def _concat_with_reencode_fallback(self, files: List[str], out_path: str) -> None:
        """Concat with codec-normalization fallback when stream-copy fails."""
        try:
            self._concat_copy(files, out_path)
            return
        except Exception:
            self.logger.warn("concat_copy_failed_reencoding")

        with tempfile.TemporaryDirectory() as tmp:
            normalized_paths: List[str] = []
            for idx, in_path in enumerate(files, start=1):
                norm_seg = os.path.join(tmp, f"seg_norm_{idx:04d}.mp3")
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    self.config.ffmpeg_loglevel,
                    "-y",
                    "-i",
                    in_path,
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-b:a",
                    "128k",
                    norm_seg,
                ]
                _run(cmd, self.logger)
                normalized_paths.append(norm_seg)
            self._concat_copy(normalized_paths, out_path)

    def _loudnorm_two_pass(self, raw_path: str, norm_path: str) -> None:
        """Apply EBU loudness normalization using two-pass analysis."""
        base_filter = (
            f"loudnorm=I={self.config.loudnorm_i}:TP={self.config.loudnorm_tp}:"
            f"LRA={self.config.loudnorm_lra}:print_format=json"
        )
        analyze_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            self.config.ffmpeg_loglevel,
            "-y",
            "-i",
            raw_path,
            "-af",
            base_filter,
            "-f",
            "null",
            "-",
        ]
        analyze = _run(analyze_cmd, self.logger, allow_failure=True)
        measured = _parse_loudnorm_json(analyze.stderr or "")
        if not measured:
            # Fall back to single-pass mode when analysis payload cannot be
            # parsed (keeps pipeline moving in constrained environments).
            self.logger.warn("loudnorm_analysis_failed_fallback_to_single_pass")
            fallback_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                self.config.ffmpeg_loglevel,
                "-y",
                "-i",
                raw_path,
                "-af",
                f"loudnorm=I={self.config.loudnorm_i}:TP={self.config.loudnorm_tp}:LRA={self.config.loudnorm_lra}",
                norm_path,
            ]
            _run(fallback_cmd, self.logger)
            return

        second_filter = (
            f"loudnorm=I={self.config.loudnorm_i}:TP={self.config.loudnorm_tp}:"
            f"LRA={self.config.loudnorm_lra}:measured_I={measured['input_i']}:"
            f"measured_LRA={measured['input_lra']}:measured_TP={measured['input_tp']}:"
            f"measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:"
            "linear=true:print_format=summary"
        )
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            self.config.ffmpeg_loglevel,
            "-y",
            "-i",
            raw_path,
            "-af",
            second_filter,
            norm_path,
        ]
        _run(cmd, self.logger)

    def _apply_eq(self, norm_path: str, final_path: str) -> None:
        """Apply final EQ sweetening pass on normalized audio."""
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            self.config.ffmpeg_loglevel,
            "-y",
            "-i",
            norm_path,
            "-af",
            f"equalizer=f={self.config.bass_eq_freq}:t=q:w=1:g={self.config.bass_eq_gain}",
            final_path,
        ]
        _run(cmd, self.logger)

    def mix(self, *, segment_files: List[str], outdir: str, basename: str) -> AudioMixResult:
        """Run full mixing pipeline and return output artifact paths."""
        self._ensure_dependencies()
        if not segment_files:
            raise RuntimeError("No segment files to mix")
        os.makedirs(outdir, exist_ok=True)
        raw_path = os.path.join(outdir, f"{basename}.mp3")
        norm_path = os.path.join(outdir, f"{basename}_norm.mp3")
        final_path = os.path.join(outdir, f"{basename}_norm_eq.mp3")
        raw_tmp = os.path.join(outdir, f".{basename}.raw.tmp.mp3")
        norm_tmp = os.path.join(outdir, f".{basename}.norm.tmp.mp3")
        final_tmp = os.path.join(outdir, f".{basename}.final.tmp.mp3")
        for tmp_path in (raw_tmp, norm_tmp, final_tmp):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

        try:
            self.logger.info("audio_concat_start", segments=len(segment_files))
            self._concat_with_reencode_fallback(segment_files, raw_tmp)
            self.logger.info("audio_loudnorm_start")
            self._loudnorm_two_pass(raw_tmp, norm_tmp)
            self.logger.info("audio_eq_start")
            self._apply_eq(norm_tmp, final_tmp)

            if not os.path.exists(final_tmp) or os.path.getsize(final_tmp) <= 0:
                raise RuntimeError("Final audio was not generated correctly")
            os.replace(raw_tmp, raw_path)
            os.replace(norm_tmp, norm_path)
            os.replace(final_tmp, final_path)
        except Exception:
            # Best-effort temp cleanup to avoid leaving partial artifacts around.
            for tmp_path in (raw_tmp, norm_tmp, final_tmp):
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
            raise

        if not os.path.exists(final_path) or os.path.getsize(final_path) <= 0:
            raise RuntimeError("Final audio was not generated correctly")
        self.logger.info("audio_mix_done", final_path=final_path, bytes=os.path.getsize(final_path))
        return AudioMixResult(raw_path=raw_path, norm_path=norm_path, final_path=final_path)

