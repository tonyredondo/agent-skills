import argparse
import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import make_script  # noqa: E402
from pipeline.config import LoggingConfig  # noqa: E402
from pipeline.logging_utils import Logger  # noqa: E402


class MakeScriptSourceFallbackTests(unittest.TestCase):
    def test_autodowngrade_uses_short_profile_when_standard_source_would_block(self) -> None:
        args = argparse.Namespace(
            profile="standard",
            target_minutes=None,
            words_per_min=130,
            min_words=None,
            max_words=None,
        )
        logger = Logger.create(LoggingConfig.from_env())
        source_text = ("contenido breve " * 45).strip()
        script_cfg = make_script.ScriptConfig.from_env(profile_name="standard", target_minutes=15, words_per_min=130)

        downgraded = make_script._maybe_autodowngrade_source_profile(  # noqa: SLF001
            args=args,
            source_text=source_text,
            script_cfg=script_cfg,
            logger=logger,
        )

        self.assertEqual(downgraded.profile_name, "short")

    def test_autodowngrade_does_not_override_explicit_target_request(self) -> None:
        args = argparse.Namespace(
            profile="standard",
            target_minutes=12.0,
            words_per_min=130,
            min_words=None,
            max_words=None,
        )
        logger = Logger.create(LoggingConfig.from_env())
        source_text = ("contenido breve " * 45).strip()
        script_cfg = make_script.ScriptConfig.from_env(profile_name="standard", target_minutes=15, words_per_min=130)

        downgraded = make_script._maybe_autodowngrade_source_profile(  # noqa: SLF001
            args=args,
            source_text=source_text,
            script_cfg=script_cfg,
            logger=logger,
        )

        self.assertEqual(downgraded.profile_name, "standard")


if __name__ == "__main__":
    unittest.main()
