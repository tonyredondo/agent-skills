# Podcast Maker Environment Reference

This document is the single reference for runtime env vars and defaults used by:

- `scripts/pipeline/config.py` (`LoggingConfig`, `ReliabilityConfig`, `ScriptConfig`, `AudioConfig`)
- `scripts/pipeline/script_quality_gate_config.py` (`ScriptQualityGateConfig`)
- `scripts/make_script.py`, `scripts/make_podcast.py`, `scripts/run_podcast.py` (entrypoint-level controls)

If this file and code diverge, code is the source of truth and this file must be updated.

## Gate Action Resolution

`SCRIPT_QUALITY_GATE_SCRIPT_ACTION` / `SCRIPT_QUALITY_GATE_ACTION` are resolved by `scripts/pipeline/gate_action.py`:

1. `SCRIPT_QUALITY_GATE_SCRIPT_ACTION` if valid (`off|warn|enforce`)
2. else `SCRIPT_QUALITY_GATE_ACTION` if set and valid
3. else gate-profile default (`default=warn`, `production_strict=enforce`)

## Core Runtime (config.py)

| Variable | Default | Notes |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | Logger level |
| `LOG_HEARTBEAT_SECONDS` | `15` | Heartbeat cadence |
| `LOG_DEBUG_EVENTS` | `0` | Enables debug event logs |
| `LOG_INCLUDE_EVENT_IDS` | `1` | Adds event IDs to logs |
| `CHECKPOINT_VERSION` | `3` | Checkpoint schema major |
| `LOCK_TTL_SECONDS` | `1800` | Lock expiration |
| `MAX_REQUESTS_PER_RUN` | `0` | `0` means unlimited |
| `MAX_ESTIMATED_COST_USD` | `0.0` | `0.0` means unlimited |
| `MIN_FREE_DISK_MB` | `512` | Minimum free disk |
| `RESUME_REQUIRE_MATCHING_FINGERPRINT` | `1` | Resume safety gate |
| `MAX_CHECKPOINT_STORAGE_MB` | `4096` | Checkpoint cleanup cap |
| `MAX_LOG_STORAGE_MB` | `1024` | Log cleanup cap |
| `RETENTION_CHECKPOINT_DAYS` | `14` | Cleanup policy |
| `RETENTION_LOG_DAYS` | `7` | Cleanup policy |
| `RETENTION_INTERMEDIATE_AUDIO_DAYS` | `3` | Cleanup policy |

## Script Generation (config.py)

| Variable | Default | Notes |
| --- | --- | --- |
| `SCRIPT_MODEL` / `MODEL` | `gpt-5.2` | Script model |
| `PODCAST_DURATION_PROFILE` | `standard` | `short|standard|long` |
| `TARGET_MINUTES` | profile default | `short=5`, `standard=15`, `long=30` |
| `WORDS_PER_MIN` | `130.0` | Target WPM |
| `MIN_WORDS` / `MAX_WORDS` | derived | Derived from target/WPM if unset |
| `SCRIPT_ADAPTIVE_DEFAULTS` | `1` | Adaptive script defaults |
| `SCRIPT_CHUNK_TARGET_MINUTES` | adaptive/profile | Chunk sizing |
| `SCRIPT_MAX_CONTEXT_LINES` | adaptive/profile | Context window |
| `SCRIPT_MAX_CONTINUATIONS_PER_CHUNK` | adaptive/profile | Continuation cap |
| `SCRIPT_NO_PROGRESS_ROUNDS` | adaptive/profile | Stuck detector rounds |
| `SCRIPT_MIN_WORD_DELTA` | adaptive/profile | Minimum growth |
| `SCRIPT_TIMEOUT_SECONDS` / `OPENAI_TIMEOUT` | adaptive or `120` | Request timeout |
| `SCRIPT_RETRIES` / `OPENAI_RETRIES` | `3` | Client retries |
| `SCRIPT_CHECKPOINT_DIR` | `./.script_checkpoints` | Script checkpoints |
| `SCRIPT_PRE_SUMMARY_TRIGGER_WORDS` | `6000` | Pre-summary activation |
| `SCRIPT_PRE_SUMMARY_CHUNK_TARGET_MINUTES` | `6.0` | Pre-summary chunk size |
| `SCRIPT_PRE_SUMMARY_TARGET_WORDS` | `1800` | Pre-summary target |
| `SCRIPT_PRE_SUMMARY_MAX_ROUNDS` | `2` | Pre-summary loops |
| `SCRIPT_REPAIR_MAX_ATTEMPTS` | `1` | Generator schema repair |
| `SCRIPT_MAX_OUTPUT_TOKENS_INITIAL` | adaptive or `14000` | Initial token budget |
| `SCRIPT_MAX_OUTPUT_TOKENS_CHUNK` | adaptive or `8000` | Chunk token budget |
| `SCRIPT_MAX_OUTPUT_TOKENS_CONTINUATION` | adaptive or `6000` | Continuation budget |
| `SCRIPT_SOURCE_VALIDATION_MODE` | profile policy | `off|warn|enforce` |
| `SCRIPT_SOURCE_VALIDATION_WARN_RATIO` | profile policy | Default warn threshold |
| `SCRIPT_SOURCE_VALIDATION_ENFORCE_RATIO` | profile policy | Default enforce threshold |
| `SCRIPT_PRESUMMARY_PARALLEL` | `0` | Parallel pre-summary |
| `SCRIPT_PRESUMMARY_PARALLEL_WORKERS` | `2` | Clamped `1..4` |
| `SCRIPT_TOPIC_COVERAGE_MIN_RATIO` | `0.75` | Prevents max-word early-stop when multi-topic coverage is still narrow |

Profile source-validation defaults:

- `short`: `warn`, warn ratio `0.35`, enforce ratio `0.22`
- `standard`: `enforce`, warn ratio `0.50`, enforce ratio `0.35`
- `long`: `enforce`, warn ratio `0.60`, enforce ratio `0.45`

## OpenAI Client Runtime (openai_client.py)

| Variable | Default | Notes |
| --- | --- | --- |
| `SCRIPT_REASONING_EFFORT` | `low` | Base reasoning effort for script requests |
| `SCRIPT_QUALITY_EVAL_REASONING_EFFORT` | `high` | Stage override for `script_quality_eval` freeform call |

## Audio Synthesis/Mix (config.py)

| Variable | Default | Notes |
| --- | --- | --- |
| `TTS_MODEL` | `gpt-4o-mini-tts` | Speech model |
| `TTS_TIMEOUT_SECONDS` / `TTS_TIMEOUT` | `60` | TTS timeout |
| `TTS_RETRIES` | `3` | TTS retries |
| `TTS_MAX_CONCURRENT` | profile default | Short/standard=1, long=2 |
| `AUDIO_CHECKPOINT_DIR` | `./.audio_checkpoints` | Audio checkpoints |
| `TTS_MAX_CHARS_PER_SEGMENT` | `450` | Segment split size |
| `TTS_RETRY_BACKOFF_BASE_MS` | `800` | Retry backoff base |
| `TTS_RETRY_BACKOFF_MAX_MS` | `8000` | Retry backoff cap |
| `TTS_GLOBAL_TIMEOUT_SECONDS` | `0` | `0` disables global timeout |
| `PAUSE_BETWEEN_SEGMENTS_MS` | `0` | Inter-segment pause |
| `LOUDNORM_LUFS` | `-16.0` | Mix loudness target |
| `LOUDNORM_TP` | `-1.5` | True peak |
| `LOUDNORM_LRA` | `11.0` | Loudness range |
| `BASS_EQ_FREQ` | `100` | Bass EQ frequency |
| `BASS_EQ_GAIN` | `3.0` | Bass EQ gain dB |
| `FFMPEG_LOGLEVEL` | `warning` | ffmpeg verbosity |
| `CHUNK_LINES` | `0` | `0` means auto |
| `TTS_CROSS_CHUNK_PARALLEL` | `0` | Cross-chunk parallel mode |
| `TTS_SPEED_DEFAULT` | `1.0` | Baseline speed |
| `TTS_SPEED_INTRO` | `1.0` | Intro speed |
| `TTS_SPEED_BODY` | `1.0` | Body speed |
| `TTS_SPEED_CLOSING` | `1.0` | Closing speed |
| `TTS_PHASE_INTRO_RATIO` | `0.15` | Intro segment ratio |
| `TTS_PHASE_CLOSING_RATIO` | `0.15` | Closing segment ratio |

## Script Quality Gate (script_quality_gate_config.py)

| Variable | Default | Notes |
| --- | --- | --- |
| `SCRIPT_QUALITY_GATE_PROFILE` | `default` | `default|production_strict` |
| `SCRIPT_QUALITY_GATE_ACTION` | `warn` | Global gate action (`production_strict` defaults to `enforce`) |
| `SCRIPT_QUALITY_GATE_SCRIPT_ACTION` | resolved | Stage override (`script`/`audio`) |
| `SCRIPT_QUALITY_GATE_EVALUATOR` | `hybrid` | `rules|hybrid|llm` |
| `SCRIPT_QUALITY_GATE_LLM_SAMPLE` | profile default | `short=0.5`, others `1.0` |
| `SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS` | `1` | Enables structured `rule_judgments` application |
| `SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS_ON_FAIL` | `1` | Allows hybrid evaluator on eligible structural failures |
| `SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS_MIN_CONFIDENCE` | `0.55` | Minimum confidence to apply LLM rule judgments |
| `SCRIPT_STRICT_HOST_ALTERNATION` | `1` | Influences speaker-run defaults |
| `SCRIPT_QUALITY_MIN_WORDS_RATIO` | `0.7` | Min words ratio |
| `SCRIPT_QUALITY_MAX_WORDS_RATIO` | `1.6` | Max words ratio |
| `SCRIPT_QUALITY_MAX_CONSECUTIVE_SAME_SPEAKER` | profile default | Max same-speaker run |
| `SCRIPT_QUALITY_MAX_REPEAT_LINE_RATIO` | profile default | Repetition threshold |
| `SCRIPT_QUALITY_MAX_TURN_WORDS` | `58` | Line-length threshold per spoken turn |
| `SCRIPT_QUALITY_MAX_LONG_TURN_COUNT` | `3` | Allowed turns above line-length threshold |
| `SCRIPT_QUALITY_MAX_QUESTION_RATIO` | `0.45` | Maximum question-line ratio |
| `SCRIPT_QUALITY_MAX_QUESTION_STREAK` | `2` | Max consecutive question turns |
| `SCRIPT_QUALITY_MAX_ABRUPT_TRANSITIONS` | `2` | Allowed abrupt transition count (for longer scripts) |
| `SCRIPT_QUALITY_SOURCE_BALANCE_ENABLED` | `1` | Enables source-aware topic balance check (script-stage) |
| `SCRIPT_QUALITY_SOURCE_MIN_CATEGORY_COVERAGE` | `0.6` | Minimum covered source-category ratio |
| `SCRIPT_QUALITY_SOURCE_MAX_TOPIC_SHARE` | `0.65` | Maximum concentration allowed in one source category |
| `SCRIPT_QUALITY_SOURCE_MIN_LEXICAL_HITS` | `4` | Minimum lexical overlap hits to mark source-balance as applicable |
| `SCRIPT_QUALITY_REQUIRE_SUMMARY` | `1` | Require summary signal |
| `SCRIPT_QUALITY_REQUIRE_CLOSING` | `1` | Require closing signal |
| `SCRIPT_QUALITY_MIN_OVERALL_SCORE` | profile default | LLM score guard |
| `SCRIPT_QUALITY_MIN_CADENCE_SCORE` | profile default | LLM score guard |
| `SCRIPT_QUALITY_MIN_LOGIC_SCORE` | profile default | LLM score guard |
| `SCRIPT_QUALITY_MIN_CLARITY_SCORE` | profile default | LLM score guard |
| `SCRIPT_QUALITY_LLM_MAX_OUTPUT_TOKENS` | `1400` | LLM eval output cap |
| `SCRIPT_QUALITY_LLM_MAX_PROMPT_CHARS` | `12000` | LLM eval prompt cap |
| `SCRIPT_QUALITY_GATE_AUTO_REPAIR` | `1` | Enables repair attempts |
| `SCRIPT_QUALITY_GATE_REPAIR_ATTEMPTS` | `2` | Repair attempts |
| `SCRIPT_QUALITY_GATE_REPAIR_MAX_OUTPUT_TOKENS` | `5200` | Repair output cap |
| `SCRIPT_QUALITY_GATE_REPAIR_MAX_INPUT_CHARS` | `30000` | Repair input cap |
| `SCRIPT_QUALITY_GATE_REPAIR_REVERT_ON_FAIL` | `1` | Keep original on failed repair |
| `SCRIPT_QUALITY_GATE_REPAIR_MIN_WORD_RATIO` | `0.85` | Shrink guardrail |
| `SCRIPT_QUALITY_GATE_SEMANTIC_FALLBACK` | `1` | Semantic summary/closing fallback |
| `SCRIPT_QUALITY_GATE_SEMANTIC_MIN_CONFIDENCE` | `0.55` | Semantic confidence threshold |
| `SCRIPT_QUALITY_GATE_SEMANTIC_TAIL_LINES` | `10` | Tail lines analyzed semantically |
| `SCRIPT_QUALITY_GATE_SEMANTIC_MAX_OUTPUT_TOKENS` | `440` | Semantic evaluator cap |
| `SCRIPT_QUALITY_GATE_HARD_FAIL_STRUCTURAL_ONLY` | `1` | Hard-fail rollout behavior |
| `SCRIPT_QUALITY_GATE_STRICT_SCORE_BLOCKING` | `0` | Optional strict score blocking |
| `SCRIPT_QUALITY_GATE_CRITICAL_SCORE_THRESHOLD` | `2.5` | Strict score threshold |
| `SCRIPT_QUALITY_GATE_REPAIR_OUTPUT_TOKENS_HARD_CAP` | `6400` | Dynamic repair hard cap |

## Entrypoint Runtime Controls

| Variable | Default | Entrypoint | Notes |
| --- | --- | --- | --- |
| `RUN_MANIFEST_V2` | `1` | `make_script.py`, `make_podcast.py` | Manifest/pipeline summary flow |
| `SCRIPT_COMPLETENESS_CHECK_V2` | `1` | `make_script.py`, `make_podcast.py` | Pre-gate completeness check |
| `SCRIPT_ORCHESTRATED_RETRY_ENABLED` | `1` | `make_script.py` | Enables orchestrated retries |
| `SCRIPT_ORCHESTRATED_MAX_ATTEMPTS` | `2` | `make_script.py` | Max orchestrated attempts |
| `SCRIPT_ORCHESTRATED_RETRY_BACKOFF_MS` | `400` | `make_script.py` | Retry backoff |
| `SCRIPT_ORCHESTRATED_RETRY_FAILURE_KINDS` | `openai_empty_output,invalid_schema,script_quality_rejected,script_completeness_failed` | `make_script.py` | Recoverable failure kinds |
| `SCRIPT_QUALITY_GATE_REPAIR_TOTAL_TIMEOUT_SECONDS` | `300` | `make_script.py` | Total repair timeout |
| `SCRIPT_QUALITY_GATE_REPAIR_ATTEMPT_TIMEOUT_SECONDS` | `90` | `make_script.py` | Per-attempt repair timeout |
| `AUDIO_ORCHESTRATED_RETRY_ENABLED` | `1` | `make_podcast.py` | Enables orchestrated retries |
| `AUDIO_ORCHESTRATED_MAX_ATTEMPTS` | `2` | `make_podcast.py` | Max orchestrated attempts |
| `AUDIO_ORCHESTRATED_RETRY_BACKOFF_MS` | `1200` | `make_podcast.py` | Retry backoff |
| `AUDIO_ORCHESTRATED_RETRY_FAILURE_KINDS` | `timeout,network,rate_limit` | `make_podcast.py` | Recoverable failure kinds |
| `ALLOW_RAW_ONLY` | `0` | `make_podcast.py` | Raw-only fallback when ffmpeg missing |
| `RUN_PODCAST_SCRIPT_ATTEMPTS` | `1` | `run_podcast.py` | Script-stage attempts in orchestrator |

## Artifact Paths (Quality Reports)

- Script stage (`make_script.py`):
  - `<script_checkpoint_dir>/<episode>/quality_report_initial.json`
  - `<script_checkpoint_dir>/<episode>/quality_report.json`
- Audio stage (`make_podcast.py`):
  - `<audio_checkpoint_dir>/<episode>/quality_report.json`
