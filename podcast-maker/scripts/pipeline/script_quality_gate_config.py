from __future__ import annotations

"""Configuration loader for script quality gate behavior.

This module centralizes environment parsing and default profiles so gate logic
can remain focused on evaluation instead of configuration plumbing.
"""

import math
import os
from dataclasses import dataclass

SUPPORTED_ACTIONS = {"off", "warn", "enforce"}
SUPPORTED_EVALUATORS = {"rules", "llm", "hybrid"}
SUPPORTED_GATE_PROFILES = {"default", "production_strict"}


def _env_str(name: str, default: str) -> str:
    """Read string env var with trim + fallback."""
    raw = os.environ.get(name)
    return default if raw is None else str(raw).strip()


def _env_bool(name: str, default: bool) -> bool:
    """Read bool env var from common truthy string values."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized == "":
        return default
    return normalized in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read integer env var with safe fallback."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read finite float env var with safe fallback."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
        if not math.isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp numeric value to an inclusive range."""
    return max(low, min(high, value))


def _profile_default_sample_rate(profile_name: str) -> float:
    """Return default LLM sampling ratio for profile."""
    normalized = str(profile_name or "standard").strip().lower()
    if normalized == "short":
        return 0.5
    return 1.0


def hard_fail_structural_only_enabled() -> bool:
    """Feature flag for structural-only hard-fail rollout."""
    return _env_bool("SCRIPT_QUALITY_GATE_HARD_FAIL_STRUCTURAL_ONLY", True)


def strict_score_blocking_enabled() -> bool:
    """Feature flag for critical-score-based blocking."""
    return _env_bool("SCRIPT_QUALITY_GATE_STRICT_SCORE_BLOCKING", False)


def critical_score_threshold() -> float:
    """Threshold used when strict score blocking is enabled."""
    return _clamp(_env_float("SCRIPT_QUALITY_GATE_CRITICAL_SCORE_THRESHOLD", 2.5), 0.0, 5.0)


@dataclass(frozen=True)
class ScriptQualityGateConfig:
    """Immutable runtime config for script quality evaluation/repair."""

    action: str
    evaluator: str
    llm_sample_rate: float
    min_words_ratio: float
    max_words_ratio: float
    max_consecutive_same_speaker: int
    max_repeat_line_ratio: float
    require_summary: bool
    require_closing: bool
    min_overall_score: float
    min_cadence_score: float
    min_logic_score: float
    min_clarity_score: float
    llm_max_output_tokens: int
    llm_max_prompt_chars: int
    auto_repair: bool
    repair_attempts: int
    repair_max_output_tokens: int
    repair_max_input_chars: int
    semantic_rule_fallback: bool = True
    semantic_min_confidence: float = 0.55
    semantic_tail_lines: int = 10
    semantic_max_output_tokens: int = 220
    repair_revert_on_fail: bool = True
    repair_min_word_ratio: float = 0.85
    max_turn_words: int = 58
    max_long_turn_count: int = 3
    max_question_ratio: float = 0.45
    max_question_streak: int = 2
    max_abrupt_transition_count: int = 2
    source_balance_enabled: bool = True
    source_balance_min_category_coverage: float = 0.6
    source_balance_max_topic_share: float = 0.65
    source_balance_min_lexical_hits: int = 4
    llm_rule_judgments_enabled: bool = True
    llm_rule_judgments_on_fail: bool = True
    llm_rule_judgments_min_confidence: float = 0.55

    @staticmethod
    def from_env(*, profile_name: str) -> "ScriptQualityGateConfig":
        """Build gate config from environment and profile defaults."""
        gate_profile = _env_str("SCRIPT_QUALITY_GATE_PROFILE", "default").lower()
        if gate_profile not in SUPPORTED_GATE_PROFILES:
            gate_profile = "default"
        is_production_strict = gate_profile == "production_strict"
        strict_alternation = _env_bool("SCRIPT_STRICT_HOST_ALTERNATION", True)

        default_action = "enforce" if is_production_strict else "warn"
        action = _env_str("SCRIPT_QUALITY_GATE_ACTION", default_action).lower()
        if action not in SUPPORTED_ACTIONS:
            action = default_action

        evaluator = _env_str("SCRIPT_QUALITY_GATE_EVALUATOR", "hybrid").lower()
        if evaluator not in SUPPORTED_EVALUATORS:
            evaluator = "hybrid"

        sample_default = 1.0 if is_production_strict else _profile_default_sample_rate(profile_name)
        sample_rate = _clamp(_env_float("SCRIPT_QUALITY_GATE_LLM_SAMPLE", sample_default), 0.0, 1.0)
        min_words_ratio = _clamp(_env_float("SCRIPT_QUALITY_MIN_WORDS_RATIO", 0.7), 0.0, 2.0)
        max_words_ratio = _clamp(_env_float("SCRIPT_QUALITY_MAX_WORDS_RATIO", 1.6), 0.1, 3.0)
        if max_words_ratio < min_words_ratio:
            max_words_ratio = min_words_ratio
        default_max_consecutive_same_speaker = 1 if strict_alternation else (2 if is_production_strict else 3)
        default_max_repeat_line_ratio = 0.12 if is_production_strict else 0.18
        default_min_overall = 4.0 if is_production_strict else 3.8
        default_min_cadence = 3.9 if is_production_strict else 3.7
        default_min_logic = 4.0 if is_production_strict else 3.8
        default_min_clarity = 4.0 if is_production_strict else 3.8
        default_repair_attempts = 2 if is_production_strict else 2
        return ScriptQualityGateConfig(
            action=action,
            evaluator=evaluator,
            llm_sample_rate=sample_rate,
            min_words_ratio=min_words_ratio,
            max_words_ratio=max_words_ratio,
            max_consecutive_same_speaker=max(
                1,
                _env_int(
                    "SCRIPT_QUALITY_MAX_CONSECUTIVE_SAME_SPEAKER",
                    default_max_consecutive_same_speaker,
                ),
            ),
            max_repeat_line_ratio=_clamp(
                _env_float("SCRIPT_QUALITY_MAX_REPEAT_LINE_RATIO", default_max_repeat_line_ratio),
                0.0,
                1.0,
            ),
            require_summary=_env_bool("SCRIPT_QUALITY_REQUIRE_SUMMARY", True),
            require_closing=_env_bool("SCRIPT_QUALITY_REQUIRE_CLOSING", True),
            min_overall_score=_clamp(_env_float("SCRIPT_QUALITY_MIN_OVERALL_SCORE", default_min_overall), 0.0, 5.0),
            min_cadence_score=_clamp(_env_float("SCRIPT_QUALITY_MIN_CADENCE_SCORE", default_min_cadence), 0.0, 5.0),
            min_logic_score=_clamp(_env_float("SCRIPT_QUALITY_MIN_LOGIC_SCORE", default_min_logic), 0.0, 5.0),
            min_clarity_score=_clamp(_env_float("SCRIPT_QUALITY_MIN_CLARITY_SCORE", default_min_clarity), 0.0, 5.0),
            llm_max_output_tokens=max(128, _env_int("SCRIPT_QUALITY_LLM_MAX_OUTPUT_TOKENS", 1400)),
            llm_max_prompt_chars=max(2000, _env_int("SCRIPT_QUALITY_LLM_MAX_PROMPT_CHARS", 12000)),
            auto_repair=_env_bool("SCRIPT_QUALITY_GATE_AUTO_REPAIR", True),
            repair_attempts=max(0, _env_int("SCRIPT_QUALITY_GATE_REPAIR_ATTEMPTS", default_repair_attempts)),
            repair_max_output_tokens=max(
                256,
                _env_int("SCRIPT_QUALITY_GATE_REPAIR_MAX_OUTPUT_TOKENS", 5200),
            ),
            repair_max_input_chars=max(
                4000,
                _env_int("SCRIPT_QUALITY_GATE_REPAIR_MAX_INPUT_CHARS", 30000),
            ),
            semantic_rule_fallback=_env_bool("SCRIPT_QUALITY_GATE_SEMANTIC_FALLBACK", True),
            semantic_min_confidence=_clamp(
                _env_float("SCRIPT_QUALITY_GATE_SEMANTIC_MIN_CONFIDENCE", 0.55),
                0.0,
                1.0,
            ),
            semantic_tail_lines=max(
                3,
                min(24, _env_int("SCRIPT_QUALITY_GATE_SEMANTIC_TAIL_LINES", 10)),
            ),
            semantic_max_output_tokens=max(
                96,
                _env_int("SCRIPT_QUALITY_GATE_SEMANTIC_MAX_OUTPUT_TOKENS", 440),
            ),
            repair_revert_on_fail=_env_bool("SCRIPT_QUALITY_GATE_REPAIR_REVERT_ON_FAIL", True),
            repair_min_word_ratio=_clamp(
                _env_float("SCRIPT_QUALITY_GATE_REPAIR_MIN_WORD_RATIO", 0.85),
                0.0,
                2.0,
            ),
            max_turn_words=max(
                8,
                _env_int("SCRIPT_QUALITY_MAX_TURN_WORDS", 58),
            ),
            max_long_turn_count=max(
                0,
                _env_int("SCRIPT_QUALITY_MAX_LONG_TURN_COUNT", 3),
            ),
            max_question_ratio=_clamp(
                _env_float("SCRIPT_QUALITY_MAX_QUESTION_RATIO", 0.45),
                0.0,
                1.0,
            ),
            max_question_streak=max(
                1,
                _env_int("SCRIPT_QUALITY_MAX_QUESTION_STREAK", 2),
            ),
            max_abrupt_transition_count=max(
                0,
                _env_int("SCRIPT_QUALITY_MAX_ABRUPT_TRANSITIONS", 2),
            ),
            source_balance_enabled=_env_bool("SCRIPT_QUALITY_SOURCE_BALANCE_ENABLED", True),
            source_balance_min_category_coverage=_clamp(
                _env_float("SCRIPT_QUALITY_SOURCE_MIN_CATEGORY_COVERAGE", 0.6),
                0.0,
                1.0,
            ),
            source_balance_max_topic_share=_clamp(
                _env_float("SCRIPT_QUALITY_SOURCE_MAX_TOPIC_SHARE", 0.65),
                0.0,
                1.0,
            ),
            source_balance_min_lexical_hits=max(
                1,
                _env_int("SCRIPT_QUALITY_SOURCE_MIN_LEXICAL_HITS", 4),
            ),
            llm_rule_judgments_enabled=_env_bool(
                "SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS",
                True,
            ),
            llm_rule_judgments_on_fail=_env_bool(
                "SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS_ON_FAIL",
                True,
            ),
            llm_rule_judgments_min_confidence=_clamp(
                _env_float("SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS_MIN_CONFIDENCE", 0.55),
                0.0,
                1.0,
            ),
        )

    @property
    def enabled(self) -> bool:
        """Whether quality gate should execute for this stage."""
        return self.action in {"warn", "enforce"}
