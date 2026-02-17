import os
import sys
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.gate_action import default_script_gate_action, resolve_script_gate_action  # noqa: E402


class GateActionResolutionTests(unittest.TestCase):
    def test_default_action_uses_profile(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_script_gate_action(script_profile_name="short"), "warn")
            self.assertEqual(default_script_gate_action(script_profile_name="standard"), "enforce")
            self.assertEqual(default_script_gate_action(script_profile_name="long"), "enforce")

    def test_default_action_production_strict_forces_enforce(self) -> None:
        with mock.patch.dict(os.environ, {"SCRIPT_QUALITY_GATE_PROFILE": "production_strict"}, clear=True):
            self.assertEqual(default_script_gate_action(script_profile_name="short"), "enforce")

    def test_resolve_prefers_script_specific_action(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"SCRIPT_QUALITY_GATE_SCRIPT_ACTION": "off", "SCRIPT_QUALITY_GATE_ACTION": "enforce"},
            clear=True,
        ):
            action = resolve_script_gate_action(script_profile_name="standard", fallback_action="enforce")
        self.assertEqual(action, "off")

    def test_resolve_uses_global_action_when_script_override_missing(self) -> None:
        with mock.patch.dict(os.environ, {"SCRIPT_QUALITY_GATE_ACTION": "warn"}, clear=True):
            action = resolve_script_gate_action(script_profile_name="standard", fallback_action="enforce")
        self.assertEqual(action, "warn")

    def test_resolve_honors_non_enforce_fallback_when_env_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            action = resolve_script_gate_action(script_profile_name="standard", fallback_action="warn")
        self.assertEqual(action, "warn")


if __name__ == "__main__":
    unittest.main()
