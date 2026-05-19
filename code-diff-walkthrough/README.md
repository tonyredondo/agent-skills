# Code Diff Walkthrough

`code-diff-walkthrough` generates a self-contained review artifact for a pull request, branch comparison, or exact git diff range. It writes an English HTML walkthrough, a Spanish HTML walkthrough, the raw diff, diff stats, commit metadata, and a manifest.

The skill is designed for code review preparation: it keeps the real diff visible, annotates each hunk with proportional rationale, and lets reviewers track reviewed files and draft inline follow-up comments in the browser.

## Contents

- `SKILL.md`: Codex-facing workflow and quality rules.
- `agents/openai.yaml`: UI metadata for exposing the skill in Codex.
- `scripts/generate_walkthrough.py`: standalone generator with no build step.
- `tests/test_generate_walkthrough_smoke.py`: smoke coverage for generating and parsing the artifact.

## Requirements

- Python 3.10 or newer.
- Git.
- GitHub CLI (`gh`) only when using `--pr`.

No Node.js, bundler, or package installation is required. Generated HTML is plain HTML, CSS, and JavaScript, with pinned highlight.js CDN assets for syntax highlighting.

## Common Commands

Generate a walkthrough for a branch comparison:

```bash
python3 ./scripts/generate_walkthrough.py --repo /path/to/repo --base main --head HEAD
```

Generate a walkthrough for a GitHub pull request from inside the target checkout:

```bash
python3 ./scripts/generate_walkthrough.py --repo /path/to/repo --pr 1234
```

Generate a walkthrough with extra implementation context:

```bash
python3 ./scripts/generate_walkthrough.py \
  --repo /path/to/repo \
  --base main \
  --head HEAD \
  --context docs/implementation-plan.md
```

By default, output is written to a temporary directory outside the reviewed repository. Use `--out` only when a specific output location is needed.

## Validation

Run the local smoke tests from the skill directory:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

The smoke test creates a temporary git repository, generates both HTML files, parses the embedded JSON payloads, checks the manifest file count, and verifies that external assets are limited to the approved pinned highlight.js CDN URLs.

## Notes JSON

For non-trivial changes, prefer `--notes-json` instead of editing generated HTML. Notes JSON is the explanation layer for file-level rationale, hunk-specific patterns, architecture diagrams, and glossary entries.

Keep notes precise:

- file notes explain the role of the file,
- hunk notes explain only the visible changed block,
- imports, metadata, and mechanical wiring should not inherit broad runtime claims,
- architecture entries must be grounded in actual caller and callee code.
