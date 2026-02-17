from __future__ import annotations

import os

SUPPORTED_GATE_ACTIONS = {"off", "warn", "enforce"}


def default_script_gate_action(*, script_profile_name: str) -> str:
    gate_profile = str(os.environ.get("SCRIPT_QUALITY_GATE_PROFILE", "") or "").strip().lower()
    if gate_profile == "production_strict":
        return "enforce"
    profile = str(script_profile_name or "standard").strip().lower()
    if profile in {"standard", "long"}:
        return "enforce"
    return "warn"


def resolve_script_gate_action(*, script_profile_name: str, fallback_action: str) -> str:
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
