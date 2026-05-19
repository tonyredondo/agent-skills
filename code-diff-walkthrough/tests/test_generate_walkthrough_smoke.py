"""Smoke tests for the code-diff walkthrough generator."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
"""Absolute path to the checked-out skill under test."""

GENERATOR = SKILL_ROOT / "scripts" / "generate_walkthrough.py"
"""Generator entrypoint used by the smoke tests."""

APPROVED_CDN_PREFIX = "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/"
"""Only external asset prefix allowed in generated walkthrough HTML."""


class GenerateWalkthroughSmokeTests(unittest.TestCase):
    """Validate that the generator produces parseable review artifacts."""

    def test_generates_bilingual_artifact_from_git_diff(self) -> None:
        """Create a tiny git diff and verify the generated artifact contract."""
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as out_dir:
            repo = Path(repo_dir)
            out = Path(out_dir)
            self._run(["git", "init", "-q"], cwd=repo)
            self._run(["git", "config", "user.email", "codex@example.com"], cwd=repo)
            self._run(["git", "config", "user.name", "Codex"], cwd=repo)

            sample = repo / "sample.py"
            sample.write_text('def greet(name):\n    return f"hello {name}"\n', encoding="utf-8")
            self._run(["git", "add", "sample.py"], cwd=repo)
            self._run(["git", "commit", "-q", "-m", "initial sample"], cwd=repo)

            sample.write_text(
                'def greet(name):\n    normalized = name.strip()\n    return f"hello {normalized}"\n',
                encoding="utf-8",
            )
            self._run(["git", "add", "sample.py"], cwd=repo)
            self._run(["git", "commit", "-q", "-m", "normalize greeting input"], cwd=repo)

            self._run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--repo",
                    str(repo),
                    "--base",
                    "HEAD^",
                    "--head",
                    "HEAD",
                    "--out",
                    str(out),
                    "--title",
                    "Sample Walkthrough",
                ],
                cwd=SKILL_ROOT,
            )

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(1, manifest["file_count"])

            expected_files = {
                "index.html",
                "index.es.html",
                "pr.diff",
                "diff-stat.txt",
                "commits.txt",
                "manifest.json",
            }
            self.assertEqual(expected_files, {path.name for path in out.iterdir() if path.is_file()})

            for html_name in ("index.html", "index.es.html"):
                page = (out / html_name).read_text(encoding="utf-8")
                payload = self._extract_walkthrough_data(page)
                self.assertEqual(1, len(payload["files"]))
                self.assertEqual("sample.py", payload["files"][0]["new_path"])
                self.assertEqual([], self._unapproved_external_assets(page))

    def _extract_walkthrough_data(self, page: str) -> dict[str, object]:
        """Return the parsed embedded JSON payload from a generated HTML page."""
        match = re.search(r'<script id="walkthrough-data" type="application/json">(.*?)</script>', page, re.S)
        self.assertIsNotNone(match)
        assert match is not None
        return json.loads(match.group(1))

    def _unapproved_external_assets(self, page: str) -> list[str]:
        """Return generated external URLs that are not on the approved CDN."""
        urls = re.findall(r'https://[^"\']+', page)
        return [url for url in urls if not url.startswith(APPROVED_CDN_PREFIX)]

    def _run(self, args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        """Run a subprocess and fail the test with captured output on error."""
        return subprocess.run(
            args,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )


if __name__ == "__main__":
    unittest.main()
