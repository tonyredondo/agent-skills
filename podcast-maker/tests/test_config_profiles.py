import os
import sys
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import (  # noqa: E402
    AudioConfig,
    ReliabilityConfig,
    ScriptConfig,
    config_fingerprint,
    resolve_profile,
)


class ConfigProfilesTests(unittest.TestCase):
    def test_resolve_profile_unknown_defaults_standard(self) -> None:
        profile = resolve_profile("does-not-exist")
        self.assertEqual(profile.name, "standard")

    def test_script_config_uses_env_profile_when_not_explicit(self) -> None:
        with mock.patch.dict(os.environ, {"PODCAST_DURATION_PROFILE": "long"}, clear=False):
            cfg = ScriptConfig.from_env()
        self.assertEqual(cfg.profile_name, "long")
        self.assertGreaterEqual(cfg.target_minutes, 1.0)

    def test_script_config_clamps_max_words_to_min(self) -> None:
        cfg = ScriptConfig.from_env(
            profile_name="short",
            target_minutes=5,
            words_per_min=120,
            min_words=300,
            max_words=200,
        )
        self.assertEqual(cfg.min_words, 300)
        self.assertEqual(cfg.max_words, 300)

    def test_script_config_explicit_args_override_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"TARGET_MINUTES": "30", "WORDS_PER_MIN": "200", "PODCAST_DURATION_PROFILE": "long"},
            clear=False,
        ):
            cfg = ScriptConfig.from_env(profile_name="short", target_minutes=6, words_per_min=110)
        self.assertEqual(cfg.profile_name, "short")
        self.assertEqual(cfg.target_minutes, 6.0)
        self.assertEqual(cfg.words_per_min, 110.0)

    def test_audio_config_profile_drives_default_concurrency(self) -> None:
        short_cfg = AudioConfig.from_env(profile_name="short")
        long_cfg = AudioConfig.from_env(profile_name="long")
        self.assertGreaterEqual(long_cfg.max_concurrent, short_cfg.max_concurrent)

    def test_audio_config_clamps_negative_chunk_lines(self) -> None:
        with mock.patch.dict(os.environ, {"CHUNK_LINES": "-50"}, clear=False):
            cfg = AudioConfig.from_env(profile_name="short")
        self.assertEqual(cfg.chunk_lines, 0)

    def test_audio_config_speed_and_phase_defaults(self) -> None:
        cfg = AudioConfig.from_env(profile_name="short")
        self.assertEqual(cfg.tts_speed_default, 1.0)
        self.assertEqual(cfg.tts_speed_intro, 1.0)
        self.assertEqual(cfg.tts_speed_body, 1.0)
        self.assertEqual(cfg.tts_speed_closing, 1.0)
        self.assertEqual(cfg.tts_phase_intro_ratio, 0.15)
        self.assertEqual(cfg.tts_phase_closing_ratio, 0.15)
        self.assertTrue(cfg.tts_speed_hints_enabled)
        self.assertEqual(cfg.tts_speed_hints_max_delta, 0.08)

    def test_audio_config_speed_invalid_values_fallback_to_default_and_clamp(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TTS_SPEED_DEFAULT": "1.11",
                "TTS_SPEED_INTRO": "invalid",
                "TTS_SPEED_BODY": "nan",
                "TTS_SPEED_CLOSING": "9.7",
                "TTS_PHASE_INTRO_RATIO": "-2.0",
                "TTS_PHASE_CLOSING_RATIO": "1.2",
                "TTS_SPEED_HINTS_ENABLED": "yes",
                "TTS_SPEED_HINTS_MAX_DELTA": "2.4",
            },
            clear=False,
        ):
            cfg = AudioConfig.from_env(profile_name="short")
        self.assertEqual(cfg.tts_speed_default, 1.11)
        self.assertEqual(cfg.tts_speed_intro, 1.11)
        self.assertEqual(cfg.tts_speed_body, 1.11)
        self.assertEqual(cfg.tts_speed_closing, 4.0)
        self.assertEqual(cfg.tts_phase_intro_ratio, 0.0)
        self.assertEqual(cfg.tts_phase_closing_ratio, 0.45)
        self.assertTrue(cfg.tts_speed_hints_enabled)
        self.assertEqual(cfg.tts_speed_hints_max_delta, 1.0)

    def test_reliability_config_clamps_minimum_values(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "MIN_FREE_DISK_MB": "1",
                "MAX_CHECKPOINT_STORAGE_MB": "10",
                "MAX_LOG_STORAGE_MB": "10",
                "RETENTION_CHECKPOINT_DAYS": "0",
                "RETENTION_LOG_DAYS": "0",
                "RETENTION_INTERMEDIATE_AUDIO_DAYS": "0",
            },
            clear=False,
        ):
            cfg = ReliabilityConfig.from_env()
        self.assertEqual(cfg.min_free_disk_mb, 32)
        self.assertEqual(cfg.max_checkpoint_storage_mb, 256)
        self.assertEqual(cfg.max_log_storage_mb, 128)
        self.assertEqual(cfg.retention_checkpoint_days, 1)
        self.assertEqual(cfg.retention_log_days, 1)
        self.assertEqual(cfg.retention_intermediate_audio_days, 1)

    def test_config_fingerprint_is_deterministic(self) -> None:
        script_cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=120)
        audio_cfg = AudioConfig.from_env(profile_name="short")
        fp1 = config_fingerprint(script_cfg=script_cfg, audio_cfg=audio_cfg, extra={"component": "x"})
        fp2 = config_fingerprint(script_cfg=script_cfg, audio_cfg=audio_cfg, extra={"component": "x"})
        self.assertEqual(fp1, fp2)

    def test_config_fingerprint_changes_with_extra(self) -> None:
        script_cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=120)
        fp1 = config_fingerprint(script_cfg=script_cfg, extra={"v": 1})
        fp2 = config_fingerprint(script_cfg=script_cfg, extra={"v": 2})
        self.assertNotEqual(fp1, fp2)

    def test_config_fingerprint_ignores_speed_hint_runtime_toggles(self) -> None:
        base = AudioConfig.from_env(profile_name="short")
        variant = AudioConfig.from_env(profile_name="short")
        with mock.patch.dict(
            os.environ,
            {"TTS_SPEED_HINTS_ENABLED": "0", "TTS_SPEED_HINTS_MAX_DELTA": "0.33"},
            clear=False,
        ):
            variant = AudioConfig.from_env(profile_name="short")
        self.assertNotEqual(base.tts_speed_hints_enabled, variant.tts_speed_hints_enabled)
        fp1 = config_fingerprint(audio_cfg=base, extra={"component": "tts_synthesizer"})
        fp2 = config_fingerprint(audio_cfg=variant, extra={"component": "tts_synthesizer"})
        self.assertEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()

