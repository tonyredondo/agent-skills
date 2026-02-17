from __future__ import annotations

"""Resolve effective script quality gate actions across env layers."""

import os

SUPPORTED_GATE_ACTIONS = {"off", "warn", "enforce"}


def default_script_gate_action(*, script_profile_name: str) -> str:
    """Return default stage action based on profile and gate profile."""
    gate_profile = str(os.environ.get("SCRIPT_QUALITY_GATE_PROFILE", "") or "").strip().lower()
    if gate_profile == "production_strict":
        return "enforce"
    return "warn"


def resolve_script_gate_action(*, script_profile_name: str, fallback_action: str) -> str:
    """Resolve effective action with explicit env overrides first.

    Precedence:
    1) `SCRIPT_QUALITY_GATE_SCRIPT_ACTION`
    2) `SCRIPT_QUALITY_GATE_ACTION`
    3) caller fallback (except enforcing by accident)
    4) computed default
    """
    default_action = default_script_gate_action(script_profile_name=script_profile_name)
    explicit_script_action = str(os.environ.get("SCRIPT_QUALITY_GATE_SCRIPT_ACTION", "") or "").strip().lower()
    if explicit_script_action in SUPPORTED_GATE_ACTIONS:
        return explicit_script_action
    raw_global_action = os.environ.get("SCRIPT_QUALITY_GATE_ACTION")
    if raw_global_action is not None:
        global_action = str(raw_global_action or "").strip().lower()
        if global_action in SUPPORTED_GATE_ACTIONS:
            return global_action
        return default_action
    fallback = str(fallback_action or "").strip().lower()
    if fallback in SUPPORTED_GATE_ACTIONS and fallback != "enforce":
        return fallback
    return default_action
