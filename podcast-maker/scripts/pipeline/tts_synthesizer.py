#!/usr/bin/env python3
from __future__ import annotations

"""Text-to-speech synthesis orchestration with checkpoint resume support.

This module turns validated script lines into segment MP3 files, tracks
manifest state for resumability, and exports structured summaries for the
audio stage.
"""

import json
import os
import re
import hashlib
import math
import shutil
import subprocess
import threading
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .audio_checkpoint import AudioCheckpointStore
from .config import AudioConfig, ReliabilityConfig, config_fingerprint
from .errors import (
    ERROR_KIND_RESUME_BLOCKED,
    ERROR_KIND_STUCK,
    ERROR_KIND_TIMEOUT,
    ERROR_KIND_UNKNOWN,
    TTSBatchError,
    TTSOperationError,
    classify_tts_exception,
    is_stuck_error_kind,
    summarize_failure_kinds,
)
from .logging_utils import Logger
from .openai_client import OpenAIClient
from .schema import content_hash


DEFAULT_HOST1_INSTR = (
    "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | "
    "Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief"
)
DEFAULT_HOST2_INSTR = (
    "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | "
    "Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief"
)

PHASE_INTRO = "intro"
PHASE_BODY = "body"
PHASE_CLOSING = "closing"
VALID_SPEECH_PHASES = {PHASE_INTRO, PHASE_BODY, PHASE_CLOSING}
INSTRUCTION_FIELDS_ORDER = (
    "Voice Affect",
    "Tone",
    "Pacing",
    "Emotion",
    "Pronunciation",
    "Pauses",
)
INSTRUCTION_FIELD_KEY_MAP = {re.sub(r"[^a-z]", "", field.lower()): field for field in INSTRUCTION_FIELDS_ORDER}
PHASE_STYLE_OVERRIDES: Dict[str, Dict[str, str]] = {
    PHASE_INTRO: {
        "Tone": "Conversational and inviting",
        "Pacing": "Brisk but clear",
        "Emotion": "Enthusiasm",
        "Pauses": "Brief",
    },
    PHASE_BODY: {
        "Tone": "Conversational and analytical",
        "Pacing": "Measured",
        "Emotion": "Focus",
        "Pauses": "Balanced",
    },
    PHASE_CLOSING: {
        "Tone": "Warm and appreciative",
        "Pacing": "Calm",
        "Emotion": "Gratitude",
        "Pauses": "Slightly longer",
    },
}


def _clamp_tts_speed(value: Any, *, fallback: float = 1.0) -> float:
    """Clamp speed values to API-safe range."""
    try:
        parsed = float(value)
        if not math.isfinite(parsed):
            parsed = float(fallback)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if not math.isfinite(parsed):
        parsed = 1.0
    return round(max(0.25, min(4.0, parsed)), 3)


def _normalize_phase(value: str) -> str:
    """Normalize speech phase labels to known values."""
    phase = str(value or "").strip().lower()
    if phase in VALID_SPEECH_PHASES:
        return phase
    return PHASE_BODY


def _normalize_instruction_key(value: str) -> str:
    """Normalize instruction keys to canonical field names."""
    token = re.sub(r"[^a-z]", "", str(value or "").strip().lower())
    return INSTRUCTION_FIELD_KEY_MAP.get(token, "")


def _parse_instruction_fields(instructions: str) -> Tuple[Dict[str, str], List[str]]:
    """Parse instruction text into canonical fields plus extras."""
    fields: Dict[str, str] = {}
    extras: List[str] = []
    for raw_part in str(instructions or "").split("|"):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            extras.append(part)
            continue
        key_raw, value_raw = part.split(":", 1)
        canonical_key = _normalize_instruction_key(key_raw)
        value = str(value_raw or "").strip()
        if canonical_key and value:
            fields[canonical_key] = value
        else:
            extras.append(part)
    return fields, extras


def _render_instruction_fields(fields: Dict[str, str], extras: Optional[List[str]] = None) -> str:
    """Render ordered instruction fields back into single-line format."""
    parts: List[str] = []
    for key in INSTRUCTION_FIELDS_ORDER:
        value = str(fields.get(key, "")).strip()
        if value:
            parts.append(f"{key}: {value}")
    if extras:
        parts.extend(part for part in extras if str(part).strip())
    return " | ".join(parts).strip()


def _default_instruction_fields(role: str) -> Dict[str, str]:
    """Return baseline instruction fields for Host1/Host2."""
    base = DEFAULT_HOST2_INSTR if str(role or "").strip() == "Host2" else DEFAULT_HOST1_INSTR
    fields, _extras = _parse_instruction_fields(base)
    return dict(fields)


def _split_long_tts_sentence(sentence: str, max_chars: int) -> List[str]:
    """Split long sentences at natural punctuation before hard cuts."""
    out: List[str] = []
    remaining = str(sentence or "").strip()
    if not remaining:
        return out
    safe_max = max(1, int(max_chars))
    while len(remaining) > safe_max:
        window = remaining[: safe_max + 1]
        min_strong_idx = int(safe_max * 0.55)
        min_secondary_idx = int(safe_max * 0.45)
        cut_idx = -1
        for match in re.finditer(r"[.!?](?=\s|$)", window):
            if match.end() >= min_strong_idx:
                cut_idx = match.end()
        if cut_idx < 0:
            for match in re.finditer(r"[,;:](?=\s|$)", window):
                if match.end() >= min_secondary_idx:
                    cut_idx = match.end()
        if cut_idx < 0:
            fallback_space = window.rfind(" ")
            if fallback_space >= int(safe_max * 0.60):
                cut_idx = fallback_space
        if cut_idx < 0:
            cut_idx = safe_max
        if cut_idx <= 0:
            cut_idx = safe_max
        chunk = remaining[:cut_idx].strip()
        if chunk:
            out.append(chunk)
        remaining = remaining[cut_idx:].strip()
    if remaining:
        out.append(remaining)
    return out


def _env_int(name: str, default: int) -> int:
    """Read integer env var with fallback."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    """Read string env var with trim + fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default


FIRST_NAME_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
FEMALE_NAME_HINTS = {
    "adriana",
    "ana",
    "carla",
    "carmen",
    "camila",
    "diana",
    "elena",
    "gabriela",
    "isabel",
    "julia",
    "laura",
    "lucia",
    "maria",
    "marta",
    "natalia",
    "paula",
    "patricia",
    "sofia",
    "valeria",
}
MALE_NAME_HINTS = {
    "adrian",
    "alejandro",
    "antonio",
    "carlos",
    "david",
    "diego",
    "fernando",
    "francisco",
    "javier",
    "jorge",
    "juan",
    "luis",
    "manuel",
    "marcos",
    "miguel",
    "pablo",
    "pedro",
    "rafael",
    "ricardo",
    "sergio",
}


def _normalize_name_token(value: str) -> str:
    """Normalize names to ascii lowercase tokens for matching."""
    token = str(value or "").strip()
    if not token:
        return ""
    ascii_token = (
        unicodedata.normalize("NFKD", token)
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .lower()
    )
    return ascii_token


def _first_name_token(name: str) -> str:
    """Extract first name token from speaker string."""
    match = FIRST_NAME_TOKEN_RE.search(str(name or ""))
    if match is None:
        return ""
    return _normalize_name_token(match.group(0))


def _infer_name_gender(name: str) -> str:
    """Infer coarse gender hint from known first-name sets."""
    token = _first_name_token(name)
    if not token:
        return ""
    if token in FEMALE_NAME_HINTS:
        return "female"
    if token in MALE_NAME_HINTS:
        return "male"
    return ""


def _role_voice_defaults() -> Dict[str, str]:
    """Resolve default voices per host role from environment."""
    host1_default = _env_str("TTS_HOST1_VOICE", "cedar")
    host2_default = _env_str("TTS_HOST2_VOICE", "marin")
    return {"Host1": host1_default, "Host2": host2_default}


def voice_for(
    role_or_speaker: str,
    *,
    speaker_name: str = "",
    role_speakers: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve voice using role defaults and optional speaker-gender hints."""
    role_or_name = str(role_or_speaker or "").strip()
    role_defaults = _role_voice_defaults()
    fallback_voice = role_defaults.get(
        role_or_name,
        _env_str("TTS_DEFAULT_VOICE", role_defaults.get("Host1", "cedar")),
    )
    mode = _env_str("TTS_VOICE_ASSIGNMENT_MODE", "auto").lower()
    if mode not in {"auto", "role", "speaker_gender"}:
        mode = "auto"
    if mode == "role":
        return fallback_voice

    candidate_name = str(speaker_name or "").strip()
    if not candidate_name and role_or_name in {"Host1", "Host2"} and role_speakers:
        candidate_name = str(role_speakers.get(role_or_name, "")).strip()
    if not candidate_name and role_or_name not in {"Host1", "Host2"}:
        candidate_name = role_or_name

    inferred_gender = _infer_name_gender(candidate_name)
    if inferred_gender == "female":
        return _env_str("TTS_FEMALE_VOICE", "marin")
    if inferred_gender == "male":
        return _env_str("TTS_MALE_VOICE", "cedar")
    return fallback_voice


def split_text_for_tts(text: str, max_chars: int) -> List[str]:
    """Split text into sentence-aware chunks bounded by character budget."""
    safe_max_chars = max(1, int(max_chars))
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    if len(compact) <= safe_max_chars:
        return [compact]

    sentences = re.split(r"(?<=[.!?])\s+", compact)
    out: List[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > safe_max_chars:
            long_parts = _split_long_tts_sentence(sentence, safe_max_chars)
            for chunk in long_parts:
                if current:
                    out.append(current)
                    current = ""
                out.append(chunk)
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) > safe_max_chars:
            if current:
                out.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        out.append(current)
    return out if out else [compact]


@dataclass
class TTSSynthesisResult:
    """Artifacts produced by TTS synthesis stage."""

    segment_files: List[str]
    manifest_path: str
    summary_path: str
    checkpoint_dir: str


@dataclass
class TTSSynthesizer:
    """Checkpoint-aware TTS orchestration across segment groups."""

    config: AudioConfig
    reliability: ReliabilityConfig
    logger: Logger
    client: OpenAIClient

    def _resolve_phase_for_line(self, *, line_index: int, total_lines: int) -> str:
        """Map line position to intro/body/closing speech phase."""
        safe_total = max(1, int(total_lines))
        safe_index = max(1, min(safe_total, int(line_index)))
        intro_ratio = max(0.0, min(0.45, float(self.config.tts_phase_intro_ratio)))
        closing_ratio = max(0.0, min(0.45, float(self.config.tts_phase_closing_ratio)))
        intro_count = int(math.floor(float(safe_total) * intro_ratio))
        closing_count = int(math.floor(float(safe_total) * closing_ratio))
        if safe_total >= 2:
            # With at least two lines, guarantee at least one intro and one closing.
            intro_count = max(1, intro_count)
            closing_count = max(1, closing_count)
        if intro_count + closing_count > safe_total:
            overflow = (intro_count + closing_count) - safe_total
            while overflow > 0:
                if intro_count >= closing_count and intro_count > 1:
                    intro_count -= 1
                elif closing_count > 1:
                    closing_count -= 1
                else:
                    break
                overflow -= 1
        if safe_index <= intro_count:
            return PHASE_INTRO
        if safe_index > (safe_total - closing_count):
            return PHASE_CLOSING
        return PHASE_BODY

    def _speed_for_phase(self, phase: str) -> float:
        """Resolve effective TTS speed for a phase."""
        normalized = _normalize_phase(phase)
        phase_speeds = {
            PHASE_INTRO: self.config.tts_speed_intro,
            PHASE_BODY: self.config.tts_speed_body,
            PHASE_CLOSING: self.config.tts_speed_closing,
        }
        return _clamp_tts_speed(
            phase_speeds.get(normalized, self.config.tts_speed_default),
            fallback=self.config.tts_speed_default,
        )

    def _refine_instructions_for_phase(self, *, instructions: str, role: str, phase: str) -> str:
        """Merge phase style defaults with line-level instruction overrides."""
        normalized_phase = _normalize_phase(phase)
        parsed_fields, extras = _parse_instruction_fields(instructions)
        if not parsed_fields:
            extras = []
        merged = _default_instruction_fields(role)
        merged.update(PHASE_STYLE_OVERRIDES.get(normalized_phase, PHASE_STYLE_OVERRIDES[PHASE_BODY]))
        # Preserve explicit per-line guidance when it is already structured.
        merged.update(parsed_fields)
        rendered = _render_instruction_fields(merged, extras)
        if rendered:
            return rendered
        return DEFAULT_HOST2_INSTR if str(role or "").strip() == "Host2" else DEFAULT_HOST1_INSTR

    def _phase_speed_metrics(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute phase and speed statistics from manifest segments."""
        phase_counts: Dict[str, int] = {PHASE_INTRO: 0, PHASE_BODY: 0, PHASE_CLOSING: 0}
        phase_speeds: Dict[str, List[float]] = {PHASE_INTRO: [], PHASE_BODY: [], PHASE_CLOSING: []}
        all_speeds: List[float] = []
        for seg in segments:
            phase = _normalize_phase(str(seg.get("phase", "")))
            fallback_speed = self._speed_for_phase(phase)
            speed = _clamp_tts_speed(seg.get("speed"), fallback=fallback_speed)
            phase_counts[phase] = int(phase_counts.get(phase, 0)) + 1
            phase_speeds.setdefault(phase, []).append(speed)
            all_speeds.append(speed)

        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            return {
                "min": round(min(values), 3),
                "avg": round(sum(values) / max(1, len(values)), 3),
                "max": round(max(values), 3),
            }

        return {
            "phase_counts": phase_counts,
            "speed_stats": _stats(all_speeds),
            "phase_speed_stats": {
                phase: _stats(values)
                for phase, values in phase_speeds.items()
                if values
            },
        }

    def _apply_phase_metrics_to_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Attach phase/speed metrics into manifest and return them."""
        metrics = self._phase_speed_metrics(list(manifest.get("segments", [])))
        manifest["tts_phase_counts"] = metrics["phase_counts"]
        manifest["tts_speed_stats"] = metrics["speed_stats"]
        manifest["tts_phase_speed_stats"] = metrics["phase_speed_stats"]
        return metrics

    def _build_segments(self, lines: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert script lines into segment-level synthesis units."""
        segments: List[Dict[str, Any]] = []
        role_speakers: Dict[str, str] = {}
        total_valid_lines = 0
        for line in lines:
            role_hint = str(line.get("role", "")).strip()
            speaker_hint = str(line.get("speaker", "")).strip()
            if role_hint in {"Host1", "Host2"} and speaker_hint and role_hint not in role_speakers:
                role_speakers[role_hint] = speaker_hint
            if speaker_hint and str(line.get("text", "")).strip():
                total_valid_lines += 1
        idx = 0
        valid_line_position = 0
        for line_idx, line in enumerate(lines, start=1):
            speaker = str(line.get("speaker", "")).strip()
            role = str(line.get("role", "")).strip()
            text = str(line.get("text", "")).strip()
            instructions = str(line.get("instructions", "")).strip()
            if not speaker or not text:
                continue
            valid_line_position += 1
            phase = self._resolve_phase_for_line(
                line_index=valid_line_position,
                total_lines=total_valid_lines,
            )
            instructions = self._refine_instructions_for_phase(
                instructions=instructions,
                role=role,
                phase=phase,
            )
            voice = voice_for(role or speaker, speaker_name=speaker, role_speakers=role_speakers)
            speed = self._speed_for_phase(phase)
            parts = split_text_for_tts(text, self.config.tts_max_chars_per_segment)
            for part in parts:
                idx += 1
                segment_id = f"{idx:04d}"
                chunk_id = (
                    ((line_idx - 1) // self.config.chunk_lines) + 1
                    if self.config.chunk_lines > 0
                    else 1
                )
                segments.append(
                    {
                        "segment_id": segment_id,
                        "index": idx,
                        "line_index": line_idx,
                        "chunk_id": chunk_id,
                        "speaker": speaker,
                        "role": role,
                        "voice": voice,
                        "instructions": instructions,
                        "phase": phase,
                        "speed": speed,
                        "text": part,
                        "text_len": len(part),
                        "status": "pending",
                        "attempts": 0,
                        "error": "",
                        "error_kind": "",
                        "file_name": f"seg_{segment_id}.mp3",
                    }
                )
        if not segments:
            raise RuntimeError("No valid lines found for TTS synthesis")
        return segments

    def _write_audio_atomic(self, path: str, content: bytes) -> None:
        """Write synthesized audio bytes atomically."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(content)
        os.replace(tmp, path)

    def _file_sha256(self, path: str) -> str:
        """Compute SHA-256 checksum for an audio artifact."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _manifest_checksum(self, manifest: Dict[str, Any]) -> str:
        """Build a deterministic checksum for manifest state tracking."""
        segments_view = []
        for seg in manifest.get("segments", []):
            segments_view.append(
                {
                    "segment_id": seg.get("segment_id"),
                    "index": seg.get("index"),
                    "file_name": seg.get("file_name", ""),
                    "status": seg.get("status"),
                    "attempts": seg.get("attempts"),
                    "error": seg.get("error", ""),
                    "error_kind": seg.get("error_kind", ""),
                    "phase": _normalize_phase(str(seg.get("phase", ""))),
                    "speed": _clamp_tts_speed(seg.get("speed"), fallback=self.config.tts_speed_default),
                    "checksum_sha256": seg.get("checksum_sha256", ""),
                }
            )
        payload = {
            "checkpoint_version": manifest.get("checkpoint_version"),
            "episode_id": manifest.get("episode_id"),
            "status": manifest.get("status", "running"),
            "segments": segments_view,
        }
        return content_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _segment_counts(self, manifest: Dict[str, Any]) -> Dict[str, int]:
        """Count manifest segments by status."""
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        for seg in manifest.get("segments", []):
            st = seg.get("status", "pending")
            if st not in counts:
                counts[st] = 0
            counts[st] += 1
        return counts

    def _ensure_chunk_metadata(self, manifest: Dict[str, Any]) -> bool:
        """Backfill missing segment metadata in legacy manifests."""
        changed = False
        segments = [seg for seg in manifest.get("segments", []) if isinstance(seg, dict)]
        total_segments = max(1, len(segments))
        for ordinal, seg in enumerate(segments, start=1):
            if "line_index" not in seg:
                seg["line_index"] = int(seg.get("index", 0) or 0)
                changed = True
            if "chunk_id" not in seg:
                if self.config.chunk_lines > 0:
                    seg["chunk_id"] = ((int(seg["line_index"]) - 1) // self.config.chunk_lines) + 1
                else:
                    seg["chunk_id"] = 1
                changed = True
            if "error_kind" not in seg:
                seg_error = str(seg.get("error", "")).strip()
                seg["error_kind"] = (
                    classify_tts_exception(RuntimeError(seg_error)) if seg_error else ""
                )
                changed = True
            raw_phase = str(seg.get("phase", "")).strip()
            normalized_phase = _normalize_phase(raw_phase)
            if not raw_phase:
                seg["phase"] = self._resolve_phase_for_line(
                    line_index=ordinal,
                    total_lines=total_segments,
                )
                changed = True
            elif raw_phase != normalized_phase:
                seg["phase"] = normalized_phase
                changed = True
            current_phase = _normalize_phase(str(seg.get("phase", PHASE_BODY)))
            expected_speed = self._speed_for_phase(current_phase)
            current_speed = _clamp_tts_speed(seg.get("speed"), fallback=expected_speed)
            if not isinstance(seg.get("speed"), (int, float)) or abs(float(current_speed) - float(seg.get("speed", 0.0))) > 1e-9:
                seg["speed"] = current_speed
                changed = True
        return changed

    def _pending_groups(self, manifest: Dict[str, Any]) -> List[Tuple[int, List[int]]]:
        """Group pending segments by chunk id for execution strategy."""
        pending: List[Tuple[int, int]] = []
        for idx, seg in enumerate(manifest.get("segments", [])):
            if seg.get("status") == "done":
                continue
            pending.append((int(seg.get("chunk_id", 1)), idx))
        if not pending:
            return []
        if self.config.chunk_lines <= 0:
            return [(1, [idx for _, idx in pending])]

        grouped: Dict[int, List[int]] = {}
        for chunk_id, idx in pending:
            grouped.setdefault(chunk_id, []).append(idx)
        return [(cid, grouped[cid]) for cid in sorted(grouped.keys())]

    def _create_pause_file(self, out_path: str, duration_ms: int) -> bool:
        """Generate optional silent pause MP3 between spoken segments."""
        if duration_ms <= 0:
            return False
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
        if shutil.which("ffmpeg") is None:
            self.logger.warn("pause_generation_skipped_no_ffmpeg")
            return False
        duration_s = max(0.05, duration_ms / 1000.0)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            f"{duration_s:.3f}",
            "-q:a",
            "9",
            "-acodec",
            "libmp3lame",
            out_path,
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            self.logger.warn(
                "pause_generation_failed",
                returncode=proc.returncode,
                stderr=(proc.stderr or "")[-400:],
            )
            return False
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0

    def _build_output_segment_files(
        self,
        manifest: Dict[str, Any],
        store: AudioCheckpointStore,
    ) -> List[str]:
        """Build final ordered list of output segment files (with pauses)."""
        done_segments = [seg for seg in manifest["segments"] if seg.get("status") == "done"]
        sorted_done = sorted(done_segments, key=lambda x: int(x["index"]))
        files: List[str] = []
        total = len(sorted_done)
        for i, seg in enumerate(sorted_done):
            files.append(os.path.join(store.segments_dir, seg["file_name"]))
            if self.config.pause_between_segments_ms > 0 and i < (total - 1):
                pause_name = f"pause_{seg['segment_id']}.mp3"
                pause_path = os.path.join(store.segments_dir, pause_name)
                if self._create_pause_file(pause_path, self.config.pause_between_segments_ms):
                    files.append(pause_path)
        return files

    def synthesize(
        self,
        *,
        lines: List[Dict[str, str]],
        episode_id: str,
        resume: bool = False,
        resume_force: bool = False,
        force_unlock: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> TTSSynthesisResult:
        """Synthesize all pending segments and persist manifest/summary state."""
        store = AudioCheckpointStore(
            base_dir=self.config.checkpoint_dir,
            episode_id=episode_id,
            reliability=self.reliability,
        )
        store.acquire_lock(force_unlock=force_unlock)
        started = time.time()
        lock = threading.Lock()

        script_hash = content_hash(json.dumps(lines, ensure_ascii=False, sort_keys=True))
        cfg_fp = config_fingerprint(
            audio_cfg=self.config,
            reliability_cfg=self.reliability,
            extra={"component": "tts_synthesizer"},
        )

        try:
            def persist_manifest() -> None:
                self._apply_phase_metrics_to_manifest(manifest)
                manifest["manifest_checksum_sha256"] = self._manifest_checksum(manifest)
                store.save(manifest)

            manifest = store.load()
            if resume:
                # Resume path validates manifest ownership/fingerprint before any
                # new synthesis attempt can proceed.
                if manifest is None:
                    if store.last_corrupt_backup_path:
                        if resume_force:
                            self.logger.warn(
                                "resume_force_after_corrupt_manifest",
                                backup_path=store.last_corrupt_backup_path,
                                error=store.last_corrupt_error,
                            )
                        else:
                            raise TTSOperationError(
                                "Resume requested but audio manifest was corrupt and moved to "
                                f"{store.last_corrupt_backup_path}. "
                                "Fix or discard that backup, then rerun with --resume-force "
                                "or without --resume.",
                                error_kind=ERROR_KIND_RESUME_BLOCKED,
                            )
                    else:
                        raise TTSOperationError(
                            "Resume requested but no audio manifest exists",
                            error_kind=ERROR_KIND_RESUME_BLOCKED,
                        )
                if manifest is not None:
                    try:
                        migrated = store.validate_resume(
                            manifest,
                            config_fingerprint=cfg_fp,
                            script_hash=script_hash,
                            resume_force=resume_force,
                        )
                    except RuntimeError as exc:
                        raise TTSOperationError(
                            str(exc),
                            error_kind=ERROR_KIND_RESUME_BLOCKED,
                        ) from exc
                    if migrated:
                        manifest["updated_at"] = int(time.time())
                        persist_manifest()
                        self.logger.info("audio_manifest_version_migrated", manifest=store.manifest_path)
            if manifest is None or not resume:
                if manifest is None and store.last_corrupt_backup_path:
                    self.logger.warn(
                        "audio_manifest_corrupt_quarantined",
                        backup_path=store.last_corrupt_backup_path,
                        error=store.last_corrupt_error,
                    )
                segments = self._build_segments(lines)
                manifest = store.init_manifest(
                    config_fingerprint=cfg_fp,
                    script_hash=script_hash,
                    segments=segments,
                )
                persist_manifest()
            else:
                # Mark missing files as pending so resume can heal partial states.
                for seg in manifest.get("segments", []):
                    file_name = str(seg.get("file_name", "")).strip()
                    if not file_name:
                        seg_id = str(seg.get("segment_id", "")).strip()
                        file_name = f"seg_{seg_id}.mp3" if seg_id else f"seg_{int(seg.get('index', 0) or 0):04d}.mp3"
                        seg["file_name"] = file_name
                    seg_path = os.path.join(store.segments_dir, file_name)
                    status = str(seg.get("status", "")).strip().lower()
                    has_audio_file = bool(os.path.exists(seg_path) and os.path.getsize(seg_path) > 0)
                    expected_checksum = str(seg.get("checksum_sha256", "")).strip().lower()
                    actual_checksum = ""
                    checksum_mismatch = False
                    checksum_read_error = ""
                    if has_audio_file:
                        try:
                            actual_checksum = self._file_sha256(seg_path)
                        except Exception as exc:  # noqa: BLE001
                            checksum_read_error = str(exc)
                            has_audio_file = False
                        if has_audio_file and expected_checksum and expected_checksum != actual_checksum:
                            checksum_mismatch = True
                    if status == "done":
                        # "done" must still be validated against actual files to
                        # avoid trusting stale manifest flags after crashes.
                        if checksum_read_error:
                            seg["status"] = "pending"
                            seg["error"] = "output_unreadable_on_resume"
                            seg["error_kind"] = ERROR_KIND_RESUME_BLOCKED
                            seg["updated_at"] = int(time.time())
                            self.logger.warn(
                                "tts_resume_segment_output_unreadable",
                                segment_id=seg.get("segment_id", ""),
                                path=seg_path,
                                error=checksum_read_error,
                            )
                        elif not has_audio_file:
                            seg["status"] = "pending"
                            seg["error"] = "output_missing_on_resume"
                            seg["error_kind"] = ERROR_KIND_RESUME_BLOCKED
                            seg["updated_at"] = int(time.time())
                        elif checksum_mismatch:
                            seg["status"] = "pending"
                            seg["error"] = "checksum_mismatch_on_resume"
                            seg["error_kind"] = ERROR_KIND_RESUME_BLOCKED
                            seg["updated_at"] = int(time.time())
                            self.logger.warn(
                                "tts_resume_checksum_mismatch",
                                segment_id=seg.get("segment_id", ""),
                                expected_checksum=expected_checksum,
                                actual_checksum=actual_checksum,
                            )
                        else:
                            seg["checksum_sha256"] = actual_checksum
                            seg["output_path"] = seg_path
                    elif status == "running":
                        # Do not trust "running" state blindly: only promote to done when
                        # checksum exists and matches (prevents accepting truncated artifacts).
                        if checksum_read_error:
                            seg["status"] = "pending"
                            seg["error"] = "output_unreadable_on_resume"
                            seg["error_kind"] = ERROR_KIND_RESUME_BLOCKED
                            seg["updated_at"] = int(time.time())
                            self.logger.warn(
                                "tts_resume_running_segment_output_unreadable",
                                segment_id=seg.get("segment_id", ""),
                                path=seg_path,
                                error=checksum_read_error,
                            )
                        elif has_audio_file and expected_checksum and not checksum_mismatch:
                            seg["status"] = "done"
                            seg["error"] = ""
                            seg["error_kind"] = ""
                            seg["output_path"] = seg_path
                            seg["checksum_sha256"] = actual_checksum
                            seg["updated_at"] = int(time.time())
                        else:
                            seg["status"] = "pending"
                            if checksum_mismatch:
                                seg["error"] = "checksum_mismatch_on_resume"
                                self.logger.warn(
                                    "tts_resume_running_segment_checksum_mismatch",
                                    segment_id=seg.get("segment_id", ""),
                                    expected_checksum=expected_checksum,
                                    actual_checksum=actual_checksum,
                                )
                            elif has_audio_file and not expected_checksum:
                                seg["error"] = "running_segment_requeued_on_resume"
                                self.logger.warn(
                                    "tts_resume_running_segment_requeued_no_checksum",
                                    segment_id=seg.get("segment_id", ""),
                                )
                            elif not str(seg.get("error", "")).strip():
                                seg["error"] = "output_missing_on_resume"
                            seg["error_kind"] = ERROR_KIND_RESUME_BLOCKED
                            seg["updated_at"] = int(time.time())
                if self._ensure_chunk_metadata(manifest):
                    self.logger.info("audio_manifest_chunk_metadata_repaired")
                manifest["updated_at"] = int(time.time())
                persist_manifest()
            counts = self._segment_counts(manifest)
            self.logger.info("tts_start", segments_total=len(manifest["segments"]), **counts)
            pending_groups = self._pending_groups(manifest)
            if not pending_groups:
                self.logger.info("tts_resume_no_pending_segments")
            if self.config.cross_chunk_parallel and pending_groups:
                # Optional mode: treat all pending segments as a single parallel
                # pool instead of chunk-by-chunk execution.
                merged_indexes: List[int] = []
                for _, group_indexes in pending_groups:
                    merged_indexes.extend(group_indexes)
                pending_groups = [(0, merged_indexes)]
                self.logger.info(
                    "tts_cross_chunk_parallel_enabled",
                    pending_segments=len(merged_indexes),
                    workers=max(1, self.config.max_concurrent),
                )

            def heartbeat_status() -> Dict[str, object]:
                with lock:
                    c = self._segment_counts(manifest)
                return {
                    "pending": c.get("pending", 0),
                    "running": c.get("running", 0),
                    "done": c.get("done", 0),
                    "failed": c.get("failed", 0),
                }

            def worker(seg_index: int) -> Tuple[int, str, int]:
                with lock:
                    seg = manifest["segments"][seg_index]
                    seg["status"] = "running"
                    seg["attempts"] = int(seg.get("attempts", 0)) + 1
                    seg["updated_at"] = int(time.time())
                    persist_manifest()
                    seg_id = seg["segment_id"]
                    text = seg["text"]
                    voice = seg["voice"]
                    phase = _normalize_phase(str(seg.get("phase", PHASE_BODY)))
                    speed = _clamp_tts_speed(seg.get("speed"), fallback=self._speed_for_phase(phase))
                    seg["phase"] = phase
                    seg["speed"] = speed
                    instructions = seg["instructions"]
                    file_name = str(seg.get("file_name", "")).strip()
                    if not file_name:
                        file_name = f"seg_{str(seg_id).strip() or int(seg.get('index', 0) or 0):0>4}.mp3"
                        seg["file_name"] = file_name
                    out_file = os.path.join(store.segments_dir, file_name)
                stage = f"tts_segment_{seg_id}"
                timeout_override = None
                if self.config.global_timeout_seconds > 0:
                    remaining = int(self.config.global_timeout_seconds - (time.time() - started))
                    if remaining <= 0:
                        raise TTSOperationError(
                            "TTS global timeout reached",
                            error_kind=ERROR_KIND_TIMEOUT,
                        )
                    timeout_override = min(self.config.timeout_seconds, max(1, remaining))
                if cancel_check is not None:
                    # Keep each individual network attempt bounded so cancellation
                    # does not wait an unbounded request timeout per worker.
                    interruptible_timeout = max(
                        1,
                        _env_int("TTS_INTERRUPTIBLE_REQUEST_TIMEOUT_SECONDS", 15),
                    )
                    base_timeout = self.config.timeout_seconds if timeout_override is None else int(timeout_override)
                    timeout_override = max(1, min(base_timeout, interruptible_timeout))
                synth_kwargs: Dict[str, Any] = {
                    "text": text,
                    "instructions": instructions,
                    "voice": voice,
                    "speed": speed,
                    "stage": stage,
                    "timeout_seconds_override": timeout_override,
                }
                if cancel_check is not None:
                    synth_kwargs["cancel_check"] = cancel_check
                while True:
                    try:
                        audio = self.client.synthesize_speech(**synth_kwargs)
                        break
                    except TypeError as exc:
                        # Compatibility path for mocked/custom clients that do not
                        # support newer keyword arguments.
                        msg = str(exc)
                        removed = False
                        for compat_key in ("cancel_check", "speed"):
                            if compat_key in msg and compat_key in synth_kwargs:
                                synth_kwargs.pop(compat_key, None)
                                removed = True
                        if not removed:
                            raise
                self._write_audio_atomic(out_file, audio)
                return seg_index, out_file, len(audio)

            with self.logger.heartbeat("tts_synthesis", status_fn=heartbeat_status):
                for chunk_id, group_indexes in pending_groups:
                    if cancel_check and cancel_check():
                        with lock:
                            manifest["status"] = "interrupted"
                            manifest["updated_at"] = int(time.time())
                            persist_manifest()
                        raise InterruptedError("Interrupted by signal before chunk processing")
                    self.logger.info(
                        "tts_chunk_start",
                        chunk_id=chunk_id if chunk_id > 0 else "parallel_all",
                        pending_segments=len(group_indexes),
                    )
                    group_start = time.time()
                    last_progress_at = group_start

                    executor = ThreadPoolExecutor(max_workers=max(1, self.config.max_concurrent))
                    abort_chunk = False
                    try:
                        future_map = {executor.submit(worker, idx): idx for idx in group_indexes}
                        pending_futures = set(future_map.keys())

                        while pending_futures:
                            if cancel_check and cancel_check():
                                abort_chunk = True
                                for fut in pending_futures:
                                    fut.cancel()
                                with lock:
                                    manifest["status"] = "interrupted"
                                    manifest["updated_at"] = int(time.time())
                                    persist_manifest()
                                raise InterruptedError("Interrupted by signal during TTS chunk processing")
                            done, not_done = wait(
                                pending_futures,
                                timeout=1.0,
                                return_when=FIRST_COMPLETED,
                            )
                            if not done:
                                now = time.time()
                                if (
                                    self.config.global_timeout_seconds > 0
                                    and (now - started) > self.config.global_timeout_seconds
                                ):
                                    # Global timeout controls whole synth stage,
                                    # not just individual request attempts.
                                    abort_chunk = True
                                    for fut in not_done:
                                        fut.cancel()
                                    raise TTSOperationError(
                                        "TTS global timeout reached",
                                        error_kind=ERROR_KIND_TIMEOUT,
                                    )
                                # Stuck detector by no progress in this chunk.
                                inactivity_limit = max(
                                    15,
                                    self.config.timeout_seconds * max(2, self.config.retries),
                                )
                                if (now - last_progress_at) > inactivity_limit:
                                    # No completed futures for too long usually
                                    # indicates external API/network deadlock.
                                    abort_chunk = True
                                    for fut in not_done:
                                        fut.cancel()
                                    raise TTSOperationError(
                                        f"TTS stuck detected in chunk {chunk_id} "
                                        f"(no progress for {int(now - last_progress_at)}s)",
                                        error_kind=ERROR_KIND_STUCK,
                                    )
                                continue

                            for fut in done:
                                pending_futures.remove(fut)
                                seg_idx = future_map[fut]
                                with lock:
                                    seg = manifest["segments"][seg_idx]
                                try:
                                    finished_idx, out_path, byte_count = fut.result()
                                    with lock:
                                        seg = manifest["segments"][finished_idx]
                                        seg["status"] = "done"
                                        seg["error"] = ""
                                        seg["error_kind"] = ""
                                        seg["bytes"] = byte_count
                                        seg["output_path"] = out_path
                                        seg["checksum_sha256"] = self._file_sha256(out_path)
                                        seg["updated_at"] = int(time.time())
                                        manifest["updated_at"] = int(time.time())
                                        # Persist after each segment completion so
                                        # resume can continue from exact progress.
                                        persist_manifest()
                                    last_progress_at = time.time()
                                    self.logger.info(
                                        "tts_segment_done",
                                        chunk_id=(
                                            chunk_id
                                            if chunk_id > 0
                                            else int(seg.get("chunk_id", 1) or 1)
                                        ),
                                        segment_id=seg["segment_id"],
                                        phase=_normalize_phase(str(seg.get("phase", ""))),
                                        speed=_clamp_tts_speed(
                                            seg.get("speed"),
                                            fallback=self.config.tts_speed_default,
                                        ),
                                        bytes=byte_count,
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    if isinstance(exc, InterruptedError):
                                        abort_chunk = True
                                        for pending in pending_futures:
                                            pending.cancel()
                                        with lock:
                                            manifest["status"] = "interrupted"
                                            manifest["updated_at"] = int(time.time())
                                            persist_manifest()
                                        raise
                                    error_kind = classify_tts_exception(exc)
                                    with lock:
                                        seg = manifest["segments"][seg_idx]
                                        seg["status"] = "failed"
                                        seg["error"] = str(exc)
                                        seg["error_kind"] = error_kind
                                        seg["updated_at"] = int(time.time())
                                        manifest["updated_at"] = int(time.time())
                                        persist_manifest()
                                    self.logger.error(
                                        "tts_segment_failed",
                                        chunk_id=(
                                            chunk_id
                                            if chunk_id > 0
                                            else int(seg.get("chunk_id", 1) or 1)
                                        ),
                                        segment_id=seg.get("segment_id", "unknown"),
                                        error_kind=error_kind,
                                        error=str(exc),
                                    )
                    finally:
                        if abort_chunk:
                            try:
                                # Do not wait on aborted workers: cancellation is
                                # best-effort and we want fast interruption paths.
                                executor.shutdown(wait=False, cancel_futures=True)
                            except TypeError:
                                executor.shutdown(wait=False)
                            self.logger.warn(
                                "tts_chunk_abort_without_wait",
                                chunk_id=(chunk_id if chunk_id > 0 else "parallel_all"),
                            )
                        else:
                            executor.shutdown(wait=True)
                    self.logger.info(
                        "tts_chunk_done",
                        chunk_id=chunk_id if chunk_id > 0 else "parallel_all",
                        elapsed_s=round(time.time() - group_start, 2),
                    )

            done_segments = [
                seg for seg in manifest["segments"] if seg.get("status") == "done"
            ]
            failed_segments = [
                seg for seg in manifest["segments"] if seg.get("status") == "failed"
            ]
            manifest["status"] = "failed" if failed_segments else "completed"
            manifest["updated_at"] = int(time.time())
            persist_manifest()
            phase_metrics = self._phase_speed_metrics(manifest["segments"])

            summary = {
                "component": "tts_synthesizer",
                "episode_id": episode_id,
                "status": manifest["status"],
                "segments_total": len(manifest["segments"]),
                "segments_done": len(done_segments),
                "segments_failed": len(failed_segments),
                "requests_made": self.client.requests_made,
                "estimated_cost_usd": round(self.client.estimated_cost_usd, 4),
                "elapsed_seconds": round(time.time() - started, 2),
                "tts_phase_counts": phase_metrics["phase_counts"],
                "tts_speed_stats": phase_metrics["speed_stats"],
                "tts_phase_speed_stats": phase_metrics["phase_speed_stats"],
            }
            if failed_segments:
                # Aggregate failure kinds keeps orchestrator policy simple and
                # still preserves per-segment detail inside manifest.
                failed_kinds = summarize_failure_kinds(
                    seg.get("error_kind", ERROR_KIND_UNKNOWN) for seg in failed_segments
                )
                summary["failure_kinds"] = failed_kinds
                summary["stuck_abort"] = any(is_stuck_error_kind(kind) for kind in failed_kinds)
            with open(store.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")

            if failed_segments:
                failed_kinds = summarize_failure_kinds(
                    seg.get("error_kind", ERROR_KIND_UNKNOWN) for seg in failed_segments
                )
                # Batch error carries per-segment context for orchestrator retry
                # policy and incident diagnosis.
                raise TTSBatchError(
                    manifest_path=store.manifest_path,
                    failed_segments=failed_segments,
                    failed_kinds=failed_kinds,
                )

            segment_files = self._build_output_segment_files(manifest, store)
            return TTSSynthesisResult(
                segment_files=segment_files,
                manifest_path=store.manifest_path,
                summary_path=store.summary_path,
                checkpoint_dir=store.run_dir,
            )
        except TTSOperationError as exc:
            try:
                segments_total = 0
                segments_done = 0
                segments_failed = 0
                phase_metrics: Dict[str, Any] = {
                    "phase_counts": {},
                    "speed_stats": {},
                    "phase_speed_stats": {},
                }
                current_failure_kind = str(exc.error_kind or ERROR_KIND_UNKNOWN).strip().lower()
                if not current_failure_kind:
                    current_failure_kind = ERROR_KIND_UNKNOWN
                failed_kinds = summarize_failure_kinds([current_failure_kind])
                # Resume-blocked failures preserve existing manifest content so
                # operators can inspect and decide whether to force-resume.
                preserve_existing_manifest = exc.error_kind == ERROR_KIND_RESUME_BLOCKED
                if "manifest" in locals() and isinstance(manifest, dict):
                    if not preserve_existing_manifest:
                        # For operational failures, mark unfinished segments as
                        # failed to leave explicit state for retries.
                        with lock:
                            segments = manifest.get("segments", [])
                            if isinstance(segments, list):
                                for seg in segments:
                                    if not isinstance(seg, dict):
                                        continue
                                    if seg.get("status") == "done":
                                        continue
                                    seg["status"] = "failed"
                                    if not str(seg.get("error", "")).strip():
                                        seg["error"] = str(exc)
                                    kind = str(seg.get("error_kind", "")).strip().lower()
                                    seg["error_kind"] = kind if kind else (exc.error_kind or ERROR_KIND_UNKNOWN)
                                    seg["updated_at"] = int(time.time())
                            manifest["status"] = "failed"
                            manifest["updated_at"] = int(time.time())
                            persist_manifest()
                    segments_total = len(manifest.get("segments", []))
                    segments_done = sum(
                        1 for seg in manifest.get("segments", []) if isinstance(seg, dict) and seg.get("status") == "done"
                    )
                    segments_failed = sum(
                        1 for seg in manifest.get("segments", []) if isinstance(seg, dict) and seg.get("status") == "failed"
                    )
                    manifest_failed_kinds = summarize_failure_kinds(
                        seg.get("error_kind", ERROR_KIND_UNKNOWN)
                        for seg in manifest.get("segments", [])
                        if isinstance(seg, dict) and seg.get("status") == "failed"
                    )
                    if manifest_failed_kinds:
                        failed_kinds = summarize_failure_kinds(
                            [exc.error_kind or ERROR_KIND_UNKNOWN, *manifest_failed_kinds]
                        )
                    phase_metrics = self._phase_speed_metrics(
                        [seg for seg in manifest.get("segments", []) if isinstance(seg, dict)]
                    )
                summary = {
                    "component": "tts_synthesizer",
                    "episode_id": episode_id,
                    "status": "failed",
                    "segments_total": segments_total,
                    "segments_done": segments_done,
                    "segments_failed": segments_failed,
                    "requests_made": self.client.requests_made,
                    "estimated_cost_usd": round(self.client.estimated_cost_usd, 4),
                    "elapsed_seconds": round(time.time() - started, 2),
                    "failure_kind": current_failure_kind,
                    "failure_kinds": failed_kinds,
                    "tts_phase_counts": phase_metrics["phase_counts"],
                    "tts_speed_stats": phase_metrics["speed_stats"],
                    "tts_phase_speed_stats": phase_metrics["phase_speed_stats"],
                    # Keep stuck signal anchored to the current operation error, not historical kinds.
                    "stuck_abort": is_stuck_error_kind(current_failure_kind),
                }
                with open(store.summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
                    f.write("\n")
            except Exception:
                pass
            raise
        except InterruptedError:
            try:
                if "manifest" in locals() and isinstance(manifest, dict):
                    with lock:
                        # Interruption status is persisted explicitly so upper
                        # layers can map this to exit code 130 semantics.
                        manifest["status"] = "interrupted"
                        manifest["updated_at"] = int(time.time())
                        persist_manifest()
                summary = {
                    "component": "tts_synthesizer",
                    "episode_id": episode_id,
                    "status": "interrupted",
                    "requests_made": self.client.requests_made,
                    "estimated_cost_usd": round(self.client.estimated_cost_usd, 4),
                    "elapsed_seconds": round(time.time() - started, 2),
                }
                with open(store.summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
                    f.write("\n")
            except Exception:
                pass
            raise
        finally:
            store.release_lock()

