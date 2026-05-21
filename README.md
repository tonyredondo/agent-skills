# agent-skills

Repository for centralizing reusable agent skills (Cursor/Codex) in one place.

## Purpose

This repo keeps each skill's source code, documentation, and tests together so it is easy to:

- version changes with clear history,
- maintain and improve skills over time,
- reuse skills across projects,
- preserve operational knowledge.

## Repository Structure

Each skill lives in its own top-level directory.

```text
.
├── README.md
├── .gitignore
├── code-diff-walkthrough/
    ├── SKILL.md
    ├── README.md
    ├── agents/
    ├── scripts/
    └── tests/
├── podcast-maker/
    ├── SKILL.md
    ├── README.md
    ├── scripts/
    └── tests/
└── setup-codex-prerequisites/
    ├── SKILL.md
    ├── README.md
    ├── agents/
    └── scripts/
```

## Available Skills

### `code-diff-walkthrough`

Skill for generating self-contained bilingual HTML walkthroughs for pull requests, branches, or local git diffs, including:

- English and Spanish review pages from the real git diff,
- file and hunk-level explanations,
- optional architecture and glossary diagrams from notes JSON,
- inline saved review comments and reviewed-file tracking in browser `localStorage`,
- syntax-highlighted diff rendering with pinned highlight.js CDN assets.

### `podcast-maker`

Skill for generating podcast scripts and audio from source text, including:

- script generation and post-processing pipeline,
- TTS synthesis with per-segment control,
- quality gates and validation checks,
- checkpoint/debug utilities and regression tests.

### `setup-codex-prerequisites`

Skill for preparing a Windows, macOS, or Debian/Ubuntu Linux workstation with baseline Codex tools, including:

- bootstrapping `winget`, Homebrew, `apt`, and Python when applicable,
- `uv`, `pipx`, `PyYAML`, and isolated Python CLIs,
- repository and environment tools such as `git`, `gh`, `rg`, `pwsh`, `node`, `npm`, and `pnpm`,
- inspection, build, and light-audit utilities such as `jq`, `yq`, `fd`, `fzf`, `bat`, `delta`, `7z`, `just`, `cmake`, `ninja`, `gitleaks`, `shellcheck`, `shfmt`, and `hadolint`.

## Standards For Each Skill

Every new skill must include, at minimum:

1. `SKILL.md` with the skill's usage interface.
2. `README.md` with technical context and local operation details.
3. A `scripts/` directory with entry points and pipeline code.
4. Tests in `tests/` when applicable.

## Repository Hygiene Rules

- Do not commit temporary artifacts: checkpoints, logs, bundles, caches, `.DS_Store`, etc.
- Keep the root `.gitignore` and each skill's `.gitignore` aligned with their real artifacts.
- Avoid local environment files (`.env`, credentials, private keys).

## How To Add A New Skill

1. Create a top-level directory with the skill name, for example `my-skill/`.
2. Add `SKILL.md`, `README.md`, `scripts/`, and optionally `tests/`.
3. Define or update `.gitignore` for the skill's own artifacts.
4. Add a section to this `README.md` under "Available Skills".
5. Validate that `git status` has no temporary files before committing.
