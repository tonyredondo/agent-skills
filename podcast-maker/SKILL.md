---
name: podcast-maker
description: Create a two-host podcast script and MP3 episode from source text with resumable script/audio generation, quality checks, and short/standard/long profiles.
---

# Podcast Maker

Use this skill when the user wants to turn source text into a two-host podcast script JSON and MP3 episode, especially when they need resumable generation, a script-plus-audio pipeline, or script-quality-gated TTS.

Run commands from the skill root (`.../podcast-maker`).

## Prerequisites

- Python `3.10+`
- `OPENAI_API_KEY` set directly or available via `~/.codex/auth.json`
- `ffmpeg` is optional for generation but required for full concat/loudnorm/EQ post-processing

## Preferred Workflow

For normal end-to-end runs, use the orchestrator so script and audio stages share the same run identity.

```bash
python3 ./scripts/run_podcast.py --profile standard /path/to/source.txt /output/dir episode_name
```

Expected outputs:

- `/output/dir/episode_name_norm_eq.mp3`
- `/output/dir/episode_name_raw_only.mp3` when `ffmpeg` is unavailable and `--allow-raw-only` or `ALLOW_RAW_ONLY=1` is set
- checkpoint and quality artifacts under `./.script_checkpoints` and `outdir/.audio_checkpoints`

## Split Stages

Use split commands when you need the script JSON first or want to regenerate audio from an existing script.

```bash
./scripts/make_script.py --profile standard /path/to/source.txt /path/to/script.json
./scripts/make_podcast.py --profile standard /path/to/script.json /output/dir episode_name
```

For shell portability, the wrapper also works:

```bash
./scripts/make_podcast.sh /path/to/script.json /output/dir episode_name --profile standard --resume
```

## Naming Rules

`episode_name`, `basename`, and explicit `--episode-id` values must be plain file-name tokens only.

- valid: `episode_001`
- invalid: `foo/bar`, `../episode`, `/tmp/name`

If you split script and audio generation across separate commands, pass the same explicit `--episode-id` to both stages so checkpoints, manifests, and reports stay grouped.

## When To Use Which Entrypoint

| Need | Command |
| --- | --- |
| Normal end-to-end generation with shared run identity | `python3 ./scripts/run_podcast.py` |
| Generate only the script JSON | `./scripts/make_script.py` |
| Regenerate audio from an existing script JSON | `./scripts/make_podcast.py` |

Use only flags documented in each script's `--help`.

## Core Runtime Behavior

- Profiles: `short` (~5 min), `standard` (~15 min, default), `long` (~30 min)
- Recommended source sizing is profile-aware:
  - `short`: prefer at least `max(120, target_words * 0.35)` words
  - `standard`: prefer at least `max(120, target_words * 0.50)` words
  - `long`: prefer at least `max(120, target_words * 0.60)` words
- Script generation defaults to `gpt-5.4` and still honors `SCRIPT_MODEL` / `MODEL` overrides
- `make_script.py` is expected to persist `quality_report.json`; if that artifact is missing, the script run fails instead of triggering a hidden fallback repair/eval path
- Audio generation performs a pre-TTS script quality check and can run in `off`, `warn`, or `enforce` mode
- Resume is supported for both script and audio stages

## Script Expectations

When generating or reviewing scripts, keep these constraints in force:

- spoken text should alternate `Host1` and `Host2`
- avoid explicit section labels like `Bloque 1`, `Section 2`, or `Part 3`
- avoid internal workflow disclosures, shell commands, or document-meta narration in spoken lines
- keep the ending complete and natural: no dangling connectors or clipped endings, and no forced recap/farewell formulas unless the user explicitly asks for them
- use natural phrasing in the target language and avoid repeated line openers
- make `Host2` genuinely interactive: ask for clarification, push on tradeoffs, or challenge vague claims instead of only agreeing
- if the source covers multiple topics, move into topic one with a natural spoken transition instead of a forced roadmap

Expected JSON shape:

```json
{
  "lines": [
    {
      "speaker": "Carlos",
      "role": "Host1",
      "instructions": "Speak in a warm, confident, conversational tone. Keep pacing measured and clear with brief pauses.",
      "pace_hint": "steady",
      "text": "Hola y bienvenidos..."
    }
  ]
}
```

## Common Commands

Resume a stopped run:

```bash
./scripts/make_script.py --resume input.txt script.json
./scripts/make_podcast.py --resume script.json outdir episode_name
```

Force resume when inputs changed intentionally:

```bash
./scripts/make_script.py --resume --resume-force input.txt script.json
```

Override duration directly:

```bash
TARGET_MINUTES=15 WORDS_PER_MIN=130 ./scripts/make_script.py input.txt script.json
```

## When To Read References

- Read [`references/env.md`](references/env.md) for canonical environment variables, defaults, and artifact paths
- Read [`references/operations.md`](references/operations.md) for incident handling, retention, SLOs, and rollout guidance
- Read [`references/quality-and-golden-tests.md`](references/quality-and-golden-tests.md) for quality gate behavior, quality-loop comparisons, and golden regression commands

## Notes

- `make_podcast.py` defaults the basename to `episode` if omitted
- `CHECKPOINT_VERSION=3` is the current checkpoint schema major
- input decoding falls back through `utf-8`, `utf-8-sig`, `cp1252`, and `latin-1`
- the pipeline writes detailed logs to `stderr` and prints the final output path to `stdout`
