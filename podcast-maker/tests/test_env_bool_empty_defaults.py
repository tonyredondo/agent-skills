import os
import sys
import unittest
from unittest import mock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import make_podcast  # noqa: E402
import make_script  # noqa: E402
from pipeline.config import _env_bool as _config_env_bool  # noqa: E402
from pipeline.script_quality_gate_config import _env_bool as _gate_env_bool  # noqa: E402


class EnvBoolEmptyDefaultsTests(unittest.TestCase):
    def test_make_podcast_env_bool_empty_uses_default(self) -> None:
        with mock.patch.dict(os.environ, {"TEST_BOOL_EMPTY_MAKE_PODCAST": ""}, clear=False):
            self.assertTrue(make_podcast._env_bool("TEST_BOOL_EMPTY_MAKE_PODCAST", True))  # noqa: SLF001

    def test_make_script_env_bool_empty_uses_default(self) -> None:
        with mock.patch.dict(os.environ, {"TEST_BOOL_EMPTY_MAKE_SCRIPT": ""}, clear=False):
            self.assertTrue(make_script._env_bool("TEST_BOOL_EMPTY_MAKE_SCRIPT", True))  # noqa: SLF001

    def test_config_env_bool_empty_uses_default(self) -> None:
        with mock.patch.dict(os.environ, {"TEST_BOOL_EMPTY_CONFIG": ""}, clear=False):
            self.assertTrue(_config_env_bool("TEST_BOOL_EMPTY_CONFIG", True))

    def test_script_quality_gate_env_bool_empty_uses_default(self) -> None:
        with mock.patch.dict(os.environ, {"TEST_BOOL_EMPTY_GATE": ""}, clear=False):
            self.assertTrue(_gate_env_bool("TEST_BOOL_EMPTY_GATE", True))

    def test_empty_value_does_not_override_false_default(self) -> None:
        with mock.patch.dict(os.environ, {"TEST_BOOL_EMPTY_FALSE_DEFAULT": ""}, clear=False):
            self.assertFalse(_config_env_bool("TEST_BOOL_EMPTY_FALSE_DEFAULT", False))


if __name__ == "__main__":
    unittest.main()
