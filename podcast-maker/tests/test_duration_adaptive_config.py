import os
import sys
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.config import ScriptConfig  # noqa: E402


class DurationAdaptiveConfigTests(unittest.TestCase):
    def test_adaptive_policy_matrix_2_to_60_minutes(self) -> None:
        durations = [2, 5, 15, 30, 45, 60]
        cfgs = []
        with mock.patch.dict(os.environ, {}, clear=True):
            for minutes in durations:
                cfgs.append(
                    ScriptConfig.from_env(
                        profile_name="standard",
                        target_minutes=minutes,
                        words_per_min=130,
                        min_words=max(60, int(minutes * 120)),
                        max_words=max(80, int(minutes * 140)),
                    )
                )
        for cfg in cfgs:
            self.assertGreaterEqual(cfg.chunk_target_minutes, 1.4)
            self.assertLessEqual(cfg.chunk_target_minutes, 3.8)
            self.assertGreaterEqual(cfg.max_context_lines, 10)
            self.assertLessEqual(cfg.max_context_lines, 44)
            self.assertGreaterEqual(cfg.max_continuations_per_chunk, 2)
            self.assertLessEqual(cfg.max_continuations_per_chunk, 7)
            self.assertGreaterEqual(cfg.no_progress_rounds, 2)
            self.assertLessEqual(cfg.no_progress_rounds, 6)
            self.assertGreaterEqual(cfg.min_word_delta, 20)
            self.assertLessEqual(cfg.min_word_delta, 90)
            self.assertGreaterEqual(cfg.timeout_seconds, 90)
            self.assertLessEqual(cfg.timeout_seconds, 240)
            self.assertGreaterEqual(cfg.max_output_tokens_initial, cfg.max_output_tokens_chunk)
            self.assertGreaterEqual(cfg.max_output_tokens_chunk, cfg.max_output_tokens_continuation)

        # Monotonic trend across duration bands.
        for idx in range(1, len(cfgs)):
            self.assertGreaterEqual(cfgs[idx].chunk_target_minutes, cfgs[idx - 1].chunk_target_minutes)
            self.assertGreaterEqual(cfgs[idx].max_context_lines, cfgs[idx - 1].max_context_lines)

    def test_adaptive_formula_values_for_standard_15m(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = ScriptConfig.from_env(
                profile_name="standard",
                target_minutes=15,
                words_per_min=130,
                min_words=1800,
                max_words=2100,
            )
        self.assertAlmostEqual(cfg.chunk_target_minutes, 2.15, places=2)
        self.assertEqual(cfg.max_context_lines, 20)
        self.assertEqual(cfg.max_continuations_per_chunk, 3)
        self.assertEqual(cfg.no_progress_rounds, 3)
        self.assertGreaterEqual(cfg.min_word_delta, 20)
        self.assertLessEqual(cfg.min_word_delta, 90)

    def test_expected_token_budget_and_timeout_formula(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=5,
                words_per_min=130,
                min_words=650,
                max_words=780,
            )
        self.assertEqual(cfg.expected_tokens_per_chunk, max(900, round(cfg.expected_words_per_chunk * 2.0)))
        self.assertEqual(cfg.max_output_tokens_chunk, round(cfg.expected_tokens_per_chunk * 2.0))
        self.assertEqual(cfg.max_output_tokens_initial, min(16000, round(cfg.expected_tokens_per_chunk * 2.5)))
        self.assertEqual(
            cfg.max_output_tokens_continuation,
            max(1400, min(10000, round(cfg.expected_tokens_per_chunk * 1.6))),
        )
        self.assertEqual(cfg.timeout_seconds, max(90, min(240, round(60 + (cfg.expected_tokens_per_chunk / 50.0)))))

    def test_expected_tokens_clamp_to_7000_and_timeout_scales(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = ScriptConfig.from_env(
                profile_name="short",
                target_minutes=2,
                words_per_min=130,
                min_words=9000,
                max_words=9000,
            )
        self.assertEqual(cfg.expected_tokens_per_chunk, 7000)
        self.assertEqual(cfg.timeout_seconds, 200)

    def test_presummary_parallel_workers_are_clamped(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SCRIPT_PRESUMMARY_PARALLEL": "1",
                "SCRIPT_PRESUMMARY_PARALLEL_WORKERS": "99",
            },
            clear=True,
        ):
            high_cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=120)
        self.assertTrue(high_cfg.pre_summary_parallel)
        self.assertEqual(high_cfg.pre_summary_parallel_workers, 4)

        with mock.patch.dict(
            os.environ,
            {
                "SCRIPT_PRESUMMARY_PARALLEL": "1",
                "SCRIPT_PRESUMMARY_PARALLEL_WORKERS": "0",
            },
            clear=True,
        ):
            low_cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=120)
        self.assertEqual(low_cfg.pre_summary_parallel_workers, 1)

    def test_adaptive_defaults_scale_with_target_minutes(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            short_cfg = ScriptConfig.from_env(
                profile_name="standard",
                target_minutes=2,
                words_per_min=130,
                min_words=220,
                max_words=280,
            )
            long_cfg = ScriptConfig.from_env(
                profile_name="standard",
                target_minutes=60,
                words_per_min=130,
                min_words=7000,
                max_words=8200,
            )

        self.assertTrue(short_cfg.adaptive_defaults_enabled)
        self.assertTrue(long_cfg.adaptive_defaults_enabled)
        self.assertGreater(long_cfg.chunk_target_minutes, short_cfg.chunk_target_minutes)
        self.assertGreater(long_cfg.max_context_lines, short_cfg.max_context_lines)
        self.assertGreaterEqual(long_cfg.max_continuations_per_chunk, short_cfg.max_continuations_per_chunk)
        self.assertGreaterEqual(long_cfg.expected_words_per_chunk, short_cfg.expected_words_per_chunk)
        self.assertGreaterEqual(long_cfg.max_output_tokens_chunk, short_cfg.max_output_tokens_chunk)
        self.assertGreaterEqual(long_cfg.timeout_seconds, short_cfg.timeout_seconds)
        self.assertGreaterEqual(short_cfg.timeout_seconds, 90)
        self.assertLessEqual(long_cfg.timeout_seconds, 240)

    def test_adaptive_defaults_can_be_disabled(self) -> None:
        with mock.patch.dict(os.environ, {"SCRIPT_ADAPTIVE_DEFAULTS": "0"}, clear=True):
            cfg = ScriptConfig.from_env(profile_name="long", target_minutes=30, words_per_min=130)
        self.assertFalse(cfg.adaptive_defaults_enabled)
        self.assertAlmostEqual(cfg.chunk_target_minutes, 3.0)
        self.assertEqual(cfg.max_output_tokens_initial, 14000)
        self.assertEqual(cfg.max_output_tokens_chunk, 8000)
        self.assertEqual(cfg.max_output_tokens_continuation, 6000)
        self.assertEqual(cfg.timeout_seconds, 120)

    def test_source_validation_env_is_parsed_and_clamped(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SCRIPT_SOURCE_VALIDATION_MODE": "bad",
                "SCRIPT_SOURCE_VALIDATION_WARN_RATIO": "2.0",
                "SCRIPT_SOURCE_VALIDATION_ENFORCE_RATIO": "-1.0",
            },
            clear=True,
        ):
            cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=130)
        self.assertEqual(cfg.source_validation_mode, "warn")
        self.assertGreaterEqual(cfg.source_validation_warn_ratio, 0.0)
        self.assertLessEqual(cfg.source_validation_warn_ratio, 1.0)
        self.assertGreaterEqual(cfg.source_validation_enforce_ratio, 0.0)
        self.assertLessEqual(cfg.source_validation_enforce_ratio, cfg.source_validation_warn_ratio)

    def test_source_validation_defaults_are_profile_and_duration_aware(self) -> None:
        cases = [
            ("short", 5, "warn", 0.35, 0.22),
            ("standard", 15, "enforce", 0.50, 0.35),
            ("long", 30, "enforce", 0.60, 0.45),
        ]
        with mock.patch.dict(os.environ, {}, clear=True):
            for profile, minutes, mode, warn_ratio, enforce_ratio in cases:
                with self.subTest(profile=profile, minutes=minutes):
                    cfg = ScriptConfig.from_env(profile_name=profile, target_minutes=minutes, words_per_min=130)
                    self.assertEqual(cfg.source_validation_mode, mode)
                    self.assertAlmostEqual(cfg.source_validation_warn_ratio, warn_ratio, places=3)
                    self.assertAlmostEqual(cfg.source_validation_enforce_ratio, enforce_ratio, places=3)

    def test_source_validation_duration_boundary_promotes_to_enforce_at_10_minutes(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            below = ScriptConfig.from_env(profile_name="short", target_minutes=9.9, words_per_min=130)
            at = ScriptConfig.from_env(profile_name="short", target_minutes=10.0, words_per_min=130)
        self.assertEqual(below.source_validation_mode, "warn")
        self.assertAlmostEqual(below.source_validation_warn_ratio, 0.35, places=3)
        self.assertAlmostEqual(below.source_validation_enforce_ratio, 0.22, places=3)
        self.assertEqual(at.source_validation_mode, "enforce")
        self.assertAlmostEqual(at.source_validation_warn_ratio, 0.50, places=3)
        self.assertAlmostEqual(at.source_validation_enforce_ratio, 0.35, places=3)

    def test_source_validation_invalid_mode_falls_back_to_dynamic_default(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SCRIPT_SOURCE_VALIDATION_MODE": "bad",
            },
            clear=True,
        ):
            standard_cfg = ScriptConfig.from_env(profile_name="standard", target_minutes=15, words_per_min=130)
            short_cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=130)
        self.assertEqual(standard_cfg.source_validation_mode, "enforce")
        self.assertEqual(short_cfg.source_validation_mode, "warn")

    def test_source_validation_enforce_ratio_is_clamped_to_warn_ratio(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SCRIPT_SOURCE_VALIDATION_WARN_RATIO": "0.2",
                "SCRIPT_SOURCE_VALIDATION_ENFORCE_RATIO": "0.5",
            },
            clear=True,
        ):
            cfg = ScriptConfig.from_env(profile_name="short", target_minutes=5, words_per_min=130)
        self.assertAlmostEqual(cfg.source_validation_warn_ratio, 0.2, places=3)
        self.assertAlmostEqual(cfg.source_validation_enforce_ratio, 0.2, places=3)


if __name__ == "__main__":
    unittest.main()

