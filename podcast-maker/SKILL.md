---
name: podcast-maker
description: Create podcast scripts and MP3 episodes with a resilient pipeline (adaptive chunking/tokens, checkpoints, resume, detailed logs). Supports short/standard/long profiles and recommended 2-60 minute range.
---

# Podcast Maker

Generate a podcast-style MP3 from a JSON script (`Host1`/`Host2`) with:
- chunked script generation
- checkpoint/resume for script + audio
- pre-audio script quality gate (`off|warn|enforce`)
- retries/backoff and detailed debug logs
- loudness normalization + bass EQ final pass

Run all commands from the skill root (`.../podcast-maker`).
Requires Python `3.10+`.

## Quick workflow

1) Preferred end-to-end command (shared run identity):
```
./scripts/run_podcast.py /path/to/source.txt /output/dir episode_name
```

2) Or split stages manually.

Generate conversational script JSON:
```
./scripts/make_script.py /path/to/source.txt /path/to/script.json
```

Generate podcast audio:
```
./scripts/make_podcast.py /path/to/script.json /output/dir episode_name
./scripts/make_podcast.sh /path/to/script.json /output/dir episode_name --profile standard --resume
```

`episode_name` (`basename`) must be a file name only (no path separators).
For split runs, pass the same `--episode-id` in both commands to avoid artifact collisions.
`make_podcast.sh` is the official wrapper and forwards extra flags to `make_podcast.py`.

Output:
- `/output/dir/episode_name_norm_eq.mp3` (default final)
- `/output/dir/episode_name_raw_only.mp3` (when ffmpeg is unavailable and `--allow-raw-only` or `ALLOW_RAW_ONLY=1` is set)
- `<script_checkpoint_dir>/<episode>/quality_report_initial.json` (script-stage gate, initial evaluation)
- `<script_checkpoint_dir>/<episode>/quality_report.json` (script-stage gate, final report)
- `<audio_checkpoint_dir>/<episode>/quality_report.json` (pre-audio gate report)
- `<audio_checkpoint_dir>/<episode>/normalized_script.json` (exact normalized script consumed by TTS)

## Entrypoints

- `make_script.py` and `make_podcast.py` are the production entrypoints.
- `run_podcast.py` orchestrates script + audio with a shared `episode_id`.
- Use only the flags documented in `--help`.

## Duration profiles

Use `--profile` or env `PODCAST_DURATION_PROFILE`:
- `short` -> tuned for ~5 minutes
- `standard` -> tuned for ~15 minutes (default, sweet spot)
- `long` -> tuned for ~30 minutes

## Source length requirement (important for LLM)

Before generating long episodes, ensure the input source is proportional to target length.
Operational guideline for the model:
- `short` (~5 min): prefer at least `max(120, target_words * 0.35)` words (`warn` mode by default; enforce threshold `0.22`)
- `standard` (~15 min): prefer at least `max(120, target_words * 0.50)` words (default mode `enforce`, hard block below `0.35`)
- `long` (~30 min): prefer at least `max(120, target_words * 0.60)` words (default mode `enforce`, hard block below `0.45`)
- if source is clearly too short for requested duration, expand source first (do not rely only on repair)
- if source metadata includes attribution, keep it explicit in the text (for example: `Autor: ...`, `Autores: ...`, `Author: ...`) so the podcast can reference the author(s) naturally

## Script structure requirements (important for LLM)

When generating script content, enforce these rules from the first draft:
- do not use explicit section labels in spoken text (for example `Bloque 1`, `Bloque 2`, `Section 3`, `Part 4`)
- alternate turns strictly between `Host1` and `Host2` (no consecutive turns by the same role)
- avoid internal workflow/tooling disclosures in spoken text (for example script paths, shell commands, `DailyRead` pipeline notes, `Tavily`, `Serper`)
- avoid repetitive line openers across consecutive turns (especially repeated `Y ...`)
- prefer natural Spanish technical phrasing and avoid unnecessary anglicisms (for example `donante adicional` vs `donor extra`)
- the final chunk must contain a coherent recap + farewell (do not leave closing to chance)
- avoid abrupt endings (`...`, dangling connectors such as trailing `y`, clipped phrases)
- every line must end as a complete sentence
- if source includes multiple topics or an explicit index/agenda, opening turns should include a brief spoken roadmap (for example `hoy hablaremos de...`) and then a smooth pivot to the first topic (`comenzamos con...`)

Examples:
```
./scripts/make_script.py --profile short input.txt short.json
./scripts/make_script.py --profile standard input.txt standard.json
./scripts/make_script.py --profile long input.txt long.json
```

You can still override words/minutes directly:
```
TARGET_MINUTES=15 WORDS_PER_MIN=130 \
  ./scripts/make_script.py input.txt script.json
```

## Resume and checkpoints

- Script resume:
```
./scripts/make_script.py --resume input.txt script.json
```

- Audio resume:
```
./scripts/make_podcast.py --resume script.json outdir episode
```

Checkpoint dirs (default):
- Script: `./.script_checkpoints`
- Audio: `./.audio_checkpoints` (or `outdir/.audio_checkpoints` if env not set)

Checkpoint format defaults to v3 (`CHECKPOINT_VERSION=3`) in this major release.
Older checkpoints from previous majors are not guaranteed to resume.

If inputs/config changed and you still want resume:
```
./scripts/make_script.py --resume --resume-force input.txt script.json
```

## Debugging and observability

Useful flags:
- `--verbose`
- `--debug`
- `--force-unlock` (when previous run crashed and left lock file)
- `--dry-run-cleanup`
- `--force-clean` (override cleanup protection for recent failed runs)
- `--allow-raw-only` (audio only, continue without ffmpeg post-processing)

Useful env vars:
- Canonical reference: `ENV_REFERENCE.md` (single source for defaults from `config.py` + entrypoints)
- `LOG_LEVEL=INFO|DEBUG`
- `LOG_HEARTBEAT_SECONDS=15`
- `LOG_DEBUG_EVENTS=1`
- `SCRIPT_MODEL` / `MODEL` (default script model: `gpt-5.2`)
- `SCRIPT_REASONING_EFFORT=low|medium|high` (default `low`)
- `SCRIPT_TONE_PROFILE=balanced|energetic|broadcast` (default `balanced`)
- `SCRIPT_TRANSITION_STYLE=subtle|explicit` (default `subtle`)
- `SCRIPT_PRECISION_PROFILE=strict|balanced` (default `strict`)
- `SCRIPT_CLOSING_STYLE=brief|warm` (default `brief`)
- `SCRIPT_RETRIES`, `SCRIPT_TIMEOUT_SECONDS`
- Script orchestrated retries (entrypoint-level):
  - `SCRIPT_ORCHESTRATED_RETRY_ENABLED=0|1` (default `1`)
  - `SCRIPT_ORCHESTRATED_MAX_ATTEMPTS` (default `2`)
  - `SCRIPT_ORCHESTRATED_RETRY_BACKOFF_MS` (default `400`)
  - `SCRIPT_ORCHESTRATED_RETRY_FAILURE_KINDS` (default `openai_empty_output,invalid_schema,script_quality_rejected,script_completeness_failed`)
- `SCRIPT_ADAPTIVE_DEFAULTS=0|1` (default `1`)
- `SCRIPT_PRE_SUMMARY_TRIGGER_WORDS`, `SCRIPT_PRE_SUMMARY_TARGET_WORDS`
- `SCRIPT_PRE_SUMMARY_MAX_ROUNDS`
- `SCRIPT_PRESUMMARY_PARALLEL=0|1`, `SCRIPT_PRESUMMARY_PARALLEL_WORKERS`
- `SCRIPT_MAX_CONTEXT_LINES`, `SCRIPT_NO_PROGRESS_ROUNDS`, `SCRIPT_MIN_WORD_DELTA`
- `SCRIPT_REPAIR_MAX_ATTEMPTS`
- `SCRIPT_PARSE_REPAIR_ATTEMPTS` (base parse-repair attempts, default `2`)
- `SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS` (extra attempts for likely truncation, default `2`)
- `SCRIPT_PARSE_REPAIR_OUTPUT_TOKENS_GROWTH` (default `1.35`)
- `SCRIPT_PARSE_REPAIR_MAX_OUTPUT_TOKENS` (default `10000`)
- `SCRIPT_PARSE_REPAIR_MAX_INPUT_CHARS` (default `120000`)
- `SCRIPT_MAX_OUTPUT_TOKENS_INITIAL`, `SCRIPT_MAX_OUTPUT_TOKENS_CHUNK`, `SCRIPT_MAX_OUTPUT_TOKENS_CONTINUATION`
- `SCRIPT_SOURCE_VALIDATION_MODE=off|warn|enforce`, `SCRIPT_SOURCE_VALIDATION_WARN_RATIO`, `SCRIPT_SOURCE_VALIDATION_ENFORCE_RATIO`
  - default policy (when unset): `short=warn(0.35/0.22)`, `standard=enforce(0.50/0.35)`, `long=enforce(0.60/0.45)`
- Reliability v3 feature flags (default `1`):
  - `RUN_MANIFEST_V2`
  - `SCRIPT_RECOVERY_LADDER_V2`
  - `SCRIPT_COMPLETENESS_CHECK_V2`
- `TTS_RETRIES`, `TTS_TIMEOUT_SECONDS`
- `TTS_MAX_CONCURRENT`
- `TTS_RETRY_BACKOFF_BASE_MS`, `TTS_RETRY_BACKOFF_MAX_MS`
- `TTS_GLOBAL_TIMEOUT_SECONDS`
- `TTS_CROSS_CHUNK_PARALLEL=0|1`
- `CHUNK_LINES`
- `PAUSE_BETWEEN_SEGMENTS_MS`
- TTS cadence controls (speed clamps to `0.25..4.0`):
  - `TTS_SPEED_DEFAULT` (default `1.0`)
  - `TTS_SPEED_INTRO` (default `1.0`, inherits neutral speed)
  - `TTS_SPEED_BODY` (default `1.0`, inherits neutral speed)
  - `TTS_SPEED_CLOSING` (default `1.0`, inherits neutral speed)
  - `TTS_PHASE_INTRO_RATIO` (default `0.15`)
  - `TTS_PHASE_CLOSING_RATIO` (default `0.15`)
  - invalid phase speed values fallback to `TTS_SPEED_DEFAULT`
- `OPENAI_CIRCUIT_BREAKER_FAILURES`
- `ESTIMATED_COST_PER_SCRIPT_REQUEST_USD`, `ESTIMATED_COST_PER_TTS_REQUEST_USD`
- `SCRIPT_QUALITY_GATE_PROFILE=default|production_strict` (default `default`)
- `SCRIPT_QUALITY_GATE_ACTION=off|warn|enforce` (default `enforce`)
- `SCRIPT_QUALITY_GATE_SCRIPT_ACTION=off|warn|enforce` (profile default: `short=warn`, `standard/long=enforce`)
- `SCRIPT_QUALITY_GATE_EVALUATOR=rules|hybrid|llm` (default `hybrid`)
- `SCRIPT_QUALITY_GATE_LLM_SAMPLE` (profile defaults: `0.5` short, `1.0` standard/long)
- `SCRIPT_QUALITY_GATE_SEMANTIC_FALLBACK=0|1` (default `1`; used in `rules`/`hybrid`/`llm`)
- `SCRIPT_QUALITY_GATE_SEMANTIC_MIN_CONFIDENCE` (default `0.55`)
- `SCRIPT_QUALITY_GATE_SEMANTIC_TAIL_LINES` (default `10`)
- `SCRIPT_QUALITY_GATE_SEMANTIC_MAX_OUTPUT_TOKENS` (default `440`)
- `SCRIPT_QUALITY_GATE_AUTO_REPAIR=0|1` (default `1`)
- `SCRIPT_QUALITY_GATE_REPAIR_ATTEMPTS` (default `2`)
- `SCRIPT_QUALITY_GATE_REPAIR_MAX_OUTPUT_TOKENS` (default `5200`), `SCRIPT_QUALITY_GATE_REPAIR_MAX_INPUT_CHARS`
- `SCRIPT_QUALITY_GATE_REPAIR_OUTPUT_TOKENS_HARD_CAP` (default `6400`)
- `SCRIPT_QUALITY_GATE_REPAIR_REVERT_ON_FAIL=0|1` (default `1`)
- `SCRIPT_QUALITY_GATE_REPAIR_MIN_WORD_RATIO` (default `0.85`)
- `SCRIPT_QUALITY_MIN_WORDS_RATIO`, `SCRIPT_QUALITY_MAX_WORDS_RATIO`
- `SCRIPT_QUALITY_MAX_CONSECUTIVE_SAME_SPEAKER`, `SCRIPT_QUALITY_MAX_REPEAT_LINE_RATIO`
- `SCRIPT_QUALITY_REQUIRE_SUMMARY`, `SCRIPT_QUALITY_REQUIRE_CLOSING`
- `SCRIPT_QUALITY_MIN_OVERALL_SCORE`, `SCRIPT_QUALITY_MIN_CADENCE_SCORE`, `SCRIPT_QUALITY_MIN_LOGIC_SCORE`, `SCRIPT_QUALITY_MIN_CLARITY_SCORE`
- `SCRIPT_QUALITY_LLM_MAX_OUTPUT_TOKENS` (default `1400`), `SCRIPT_QUALITY_LLM_MAX_PROMPT_CHARS`
- Audio orchestrated retries (entrypoint-level):
  - `AUDIO_ORCHESTRATED_RETRY_ENABLED=0|1` (default `1`)
  - `AUDIO_ORCHESTRATED_MAX_ATTEMPTS` (default `2`)
  - `AUDIO_ORCHESTRATED_RETRY_BACKOFF_MS` (default `1200`)
  - `AUDIO_ORCHESTRATED_RETRY_FAILURE_KINDS` (default `timeout,network,rate_limit`)
- Full orchestrator:
  - `RUN_PODCAST_SCRIPT_ATTEMPTS` (default `1`, retries script stage in `run_podcast.py`)
- `ALLOW_RAW_ONLY=1`
- `RETENTION_CHECKPOINT_DAYS`, `RETENTION_LOG_DAYS`, `RETENTION_INTERMEDIATE_AUDIO_DAYS`

Recommended production preset:
```
export SCRIPT_QUALITY_GATE_PROFILE=production_strict
```

SLO gate env (optional):
- `SLO_GATE_MODE=off|warn|enforce`
- `SLO_WINDOW_SIZE`
- `SLO_REQUIRED_FAILED_WINDOWS`
- `SLO_HISTORY_PATH`
- `ACTUAL_COST_USD` (optional, enables cost estimation error KPI)

Budget/guardrails (optional):
- `MAX_REQUESTS_PER_RUN`
- `MAX_ESTIMATED_COST_USD`
- `MIN_FREE_DISK_MB`

Run summaries are saved in checkpoint folders as JSON for post-mortem.
Run summaries include phase timings and source validation metrics.
Audio run summaries/manifests also include cadence metrics: `tts_phase_counts`, `tts_speed_stats`, `tts_phase_speed_stats`.
Script checkpoint folder also stores:
- `run_manifest.json` (stage states: script/audio/bundle)
- `pipeline_summary.json` (overall state with explicit `not_started`, `started`, `running`, `partial`, `interrupted`, `failed`, `completed`)
Audio failures are classified with structured `error_kind` values and propagated as `failure_kind` in SLO events.
Quality-gate enforcement failures are emitted as `failure_kind=script_quality_rejected`.
Script failures propagate structured run-summary signals (`stuck_abort`, `invalid_schema`, `failure_kind`) to SLO events.
Source-validation enforce blocks are emitted as `failure_kind=source_too_short`.
`stuck_abort` is computed from structured failure kinds/signals instead of log wording.

## JSON format

Expected `script.json`:
```json
{
  "lines": [
    {
      "speaker": "Carlos",
      "role": "Host1",
      "instructions": "Voice Affect: Warm and confident | Tone: Conversational | Pacing: Brisk | Emotion: Curiosity | Pronunciation: Clear | Pauses: Brief",
      "text": "Hola y bienvenidos..."
    },
    {
      "speaker": "Lucía",
      "role": "Host2",
      "instructions": "Voice Affect: Bright and friendly | Tone: Conversational | Pacing: Measured | Emotion: Enthusiasm | Pronunciation: Clear | Pauses: Brief",
      "text": "Gracias, hoy hablaremos de..."
    }
  ]
}
```

## Notes

- Requires `OPENAI_API_KEY` (or `~/.codex/auth.json` containing `OPENAI_API_KEY`).
- `ffmpeg` is required for full post-processing (concat/loudnorm/EQ); if unavailable, use `--allow-raw-only`.
- Script generation defaults to `gpt-5.2` and can be overridden with `SCRIPT_MODEL` / `MODEL`.
- Voice assignment defaults to `TTS_VOICE_ASSIGNMENT_MODE=auto`:
  - if speaker name hints are clear, use `TTS_FEMALE_VOICE` (default `marin`) / `TTS_MALE_VOICE` (default `cedar`)
  - otherwise fallback to role voices (`TTS_HOST1_VOICE` default `cedar`, `TTS_HOST2_VOICE` default `marin`)
  - force role-only behavior with `TTS_VOICE_ASSIGNMENT_MODE=role`
- Recommended cadence presets by profile:
  - `short`: start neutral (`TTS_SPEED_INTRO=1.0`, `TTS_SPEED_BODY=1.0`, `TTS_SPEED_CLOSING=1.0`)
  - `standard`: keep neutral by default (`1.0`, `1.0`, `1.0`)
  - `long`: only adjust in small steps (`0.98`-`1.02`) after listening checks
  - if voices sound robotic with per-phase speed, keep all phase speeds at `1.0`
- This flow does not expose SSML/pitch parameters directly; tune expressiveness with line `instructions` plus segment `speed`.
- Input file decoding supports fallback (`utf-8`, `utf-8-sig`, `cp1252`, `latin-1`).
- The pipeline keeps detailed logs in `stderr`, while `stdout` prints final output path.
- Final audio is normalized to around **-16 LUFS** and applies bass EQ (`+3 dB @ 100 Hz` by default).
- Golden regression flow:
  - `python3 ./scripts/run_golden_pipeline.py --candidate-dir ./.golden_candidates`
  - `python3 ./scripts/check_golden_suite.py --candidate-dir ./.golden_candidates`
  - `run_golden_pipeline.py` requires valid OpenAI credentials and sources sized for current source-validation policy; if local constraints block generation, use `check_golden_suite.py --allow-fixture-fallback` for deterministic structural checks
- Debug bundle export:
  - `python3 ./scripts/export_debug_bundle.py <episode_id> --script-checkpoint-dir ./.script_checkpoints --audio-checkpoint-dir ./out/.audio_checkpoints --script-path ./script.json --source-path ./source.txt --log-path ./podcast_run_logs.txt`
  - use `<episode_id>` equal to explicit `--episode-id` (or audio basename if no override was used)
  - if manifest-based paths resolve to a different run id, check `debug_bundle_metadata.json -> resolved_episode_id`
  - always pass `--script-path` so the exporter can also collect script checkpoints by script basename when it differs from `<episode_id>`
  - inspect `collection_report.json` (`found` / `missing` / `read_error` / `not_applicable`) before concluding artifacts are missing
