import contextlib
import io
import os
import sys
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import make_podcast  # noqa: E402
import make_script  # noqa: E402


class EntryPointArgsTests(unittest.TestCase):
    def _assert_parse_rejected(self, parser, argv: list[str]) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                parser(argv)
        self.assertEqual(ctx.exception.code, 2)

    def test_script_rejects_engine_flag(self) -> None:
        self._assert_parse_rejected(make_script.parse_args, ["--engine", "default", "in.txt", "out.json"])

    def test_script_rejects_unknown_flag(self) -> None:
        self._assert_parse_rejected(make_script.parse_args, ["--compat-mode", "in.txt", "out.json"])

    def test_audio_rejects_engine_flag(self) -> None:
        self._assert_parse_rejected(
            make_podcast.parse_args,
            ["--engine", "default", "script.json", "outdir", "ep"],
        )

    def test_audio_rejects_unknown_flag(self) -> None:
        self._assert_parse_rejected(
            make_podcast.parse_args,
            ["--compat-mode", "script.json", "outdir", "ep"],
        )

    def test_audio_rejects_path_like_basename(self) -> None:
        self._assert_parse_rejected(
            make_podcast.parse_args,
            ["script.json", "outdir", "../escape"],
        )

    def test_script_accepts_regular_args(self) -> None:
        args = make_script.parse_args(["--profile", "standard", "in.txt", "out.json"])
        self.assertEqual(args.profile, "standard")
        self.assertEqual(args.input_path, "in.txt")
        self.assertEqual(args.output_path, "out.json")

    def test_audio_accepts_regular_args(self) -> None:
        args = make_podcast.parse_args(["--profile", "standard", "script.json", "outdir", "ep"])
        self.assertEqual(args.profile, "standard")
        self.assertEqual(args.script_path, "script.json")
        self.assertEqual(args.outdir, "outdir")
        self.assertEqual(args.basename, "ep")


if __name__ == "__main__":
    unittest.main()

