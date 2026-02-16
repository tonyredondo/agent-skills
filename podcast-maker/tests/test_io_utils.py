import os
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pipeline.io_utils import read_text_file_with_fallback  # noqa: E402


class IOUtilsTests(unittest.TestCase):
    def test_encoding_fallback_cp1252(self) -> None:
        text = "Información útil para prueba de codificación."
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "input_cp1252.txt")
            with open(path, "wb") as f:
                f.write(text.encode("cp1252"))
            out, enc = read_text_file_with_fallback(path)
            self.assertEqual(out, text)
            self.assertIn(enc, {"cp1252", "latin-1", "utf-8-sig", "utf-8"})


if __name__ == "__main__":
    unittest.main()

