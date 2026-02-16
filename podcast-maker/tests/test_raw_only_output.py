import os
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from make_podcast import _write_raw_only_mp3  # noqa: E402


class RawOnlyOutputTests(unittest.TestCase):
    def test_raw_only_concat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seg1 = os.path.join(tmp, "a.mp3")
            seg2 = os.path.join(tmp, "b.mp3")
            with open(seg1, "wb") as f:
                f.write(b"AAA")
            with open(seg2, "wb") as f:
                f.write(b"BBB")
            out = os.path.join(tmp, "out.mp3")
            final = _write_raw_only_mp3([seg1, seg2], out)
            self.assertEqual(final, out)
            with open(out, "rb") as f:
                content = f.read()
            self.assertEqual(content, b"AAABBB")

    def test_raw_only_concat_cleans_tmp_when_segment_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seg1 = os.path.join(tmp, "a.mp3")
            with open(seg1, "wb") as f:
                f.write(b"AAA")
            missing = os.path.join(tmp, "missing.mp3")
            out = os.path.join(tmp, "out.mp3")
            with self.assertRaises(FileNotFoundError):
                _write_raw_only_mp3([seg1, missing], out)
            self.assertFalse(os.path.exists(f"{out}.tmp"))


if __name__ == "__main__":
    unittest.main()

