# Podcast Maker Operations

Use this reference for incident response, resumability issues, rollout decisions, retention, and SLO interpretation.

## Contents

- Operational defaults
- SLO targets
- Rollout stages
- Common failure modes
- Error kind map
- Debug bundle export

## Operational Defaults

- Checkpoint format default: `CHECKPOINT_VERSION=3`
- Script checkpoints: `./.script_checkpoints`
- Audio checkpoints: `./.audio_checkpoints` or `outdir/.audio_checkpoints`
- SLO history path default: `./.podcast_slo_history.jsonl`
- Pre-audio quality gate default: `SCRIPT_QUALITY_GATE_ACTION=warn`
- Strict production preset: `SCRIPT_QUALITY_GATE_PROFILE=production_strict`

## SLO Targets

### Profile targets

- `short` (5 min): success rate `>= 98%`, p95 total runtime `<= 8 min`, p95 resume runtime `<= 4 min`
- `standard` (15 min): success rate `>= 97%`, p95 total runtime `<= 20 min`, p95 resume runtime `<= 10 min`
- `long` (30 min): success rate `>= 95%`, p95 total runtime `<= 40 min`, p95 resume runtime `<= 18 min`

### Technical quality SLOs

- `retry_rate` per stage `< 15%`
- `stuck_abort_rate < 2%`
- `invalid_schema_rate < 3%`
- `script_quality_rejected_rate < 5%` for the audio component
- `p90 cost_estimation_error <= 25%` when `ACTUAL_COST_USD` is present

Rollback trigger: if SLOs are missed for two consecutive windows, roll back to the last known-good release until fixed.

## Rollout Stages

1. Canary
   - run golden pipeline plus gate before production changes
   - monitor SLO windows in warn mode
2. Enforced canary
   - set `SLO_GATE_MODE=enforce`
   - promote only when windows stay healthy
3. Stable
   - keep current defaults
   - keep script quality gate enabled
   - run golden checks on each release candidate
   - keep SLO monitoring active

Promotion requirements:

- no open critical incidents
- SLOs healthy for two windows
- golden regression suite passes

Recommended canary gate setup:

```bash
export SLO_GATE_MODE=enforce
export SLO_WINDOW_SIZE=20
export SLO_REQUIRED_FAILED_WINDOWS=2
python3 ./scripts/run_golden_pipeline.py --candidate-dir ./.golden_candidates
python3 ./scripts/check_golden_suite.py --candidate-dir ./.golden_candidates
```

## Common Failure Modes

### Script without progress

Symptoms:

- repeated continuation or recovery attempts without meaningful script growth
- `word_delta` below threshold
- run ends with "No progress while expanding script"

Actions:

1. Re-run with `--debug` and inspect script `run_summary.json`
2. Increase `SCRIPT_MAX_CONTEXT_LINES` or lower `SCRIPT_MIN_WORD_DELTA`
3. Increase `SCRIPT_NO_PROGRESS_ROUNDS` if content grows slowly but still moves
4. If the source is too short, enrich input and use `--resume-force`
5. If failures are transient or quality-related, tune `SCRIPT_ORCHESTRATED_*`

### TTS stuck or very slow

Actions:

1. Check network and API status
2. Reduce `TTS_MAX_CONCURRENT` to `1`
3. Increase `TTS_TIMEOUT_SECONDS`
4. Inspect `audio_manifest.json` and confirm `error_kind`
5. Resume with `./scripts/make_podcast.py --resume script.json outdir episode`

### Checkpoint corrupted or lock orphaned

Actions:

1. Use `--force-unlock`
2. Inspect quarantined `*.corrupt.<timestamp>.json` files
3. Retry with `--resume-force` only if the state mismatch is intentional

### ffmpeg failure

Actions:

1. Verify `ffmpeg` is installed and executable
2. Re-run with `--debug` and inspect stderr command logs
3. If `ffmpeg` is unavailable, use `--allow-raw-only`

### Script quality gate blocks audio

Actions:

1. Inspect script-stage or audio-stage `quality_report.json`
2. Review failed checks under `rules` and `scores`
3. Confirm source sizing in script `run_summary.json`
4. For exploratory runs only, temporarily set `SCRIPT_QUALITY_GATE_ACTION=warn`
5. For stricter rollouts, use `SCRIPT_QUALITY_GATE_PROFILE=production_strict`

### Invalid basename or episode id

Actions:

1. Use a plain filename token such as `episode_001`
2. Keep path routing in `outdir`
3. Do not pass `foo/bar`, `../episode`, or absolute paths as basename values

## Error Kind Map

Use structured `failure_kind` / `error_kind` values for triage:

- `timeout`: global or per-segment timeout
- `stuck`: no progress in TTS or script expansion
- `source_too_short`: source validation enforce block
- `invalid_schema`: schema or JSON validation failure
- `script_quality_rejected`: script quality gate reject before TTS
- `run_mismatch`: manifest or handoff mismatch between script and audio
- `rate_limit`: API throttling
- `network`: connection failure
- `resume_blocked`: fingerprint or resume mismatch
- `interrupted`: controlled operator interruption
- `unknown`: uncategorized failure

Use these artifacts during triage:

- script signals: `<script_checkpoint_dir>/<episode>/run_summary.json`
- pipeline state: `<script_checkpoint_dir>/<episode>/run_manifest.json`, `pipeline_summary.json`
- audio signals: `<audio_checkpoint_dir>/<episode>/audio_manifest.json`
- quality report: `<audio_checkpoint_dir>/<episode>/quality_report.json`
- event history: `SLO_HISTORY_PATH`

## Debug Bundle Export

Use this when support or on-call needs a single bundle with diagnostics:

```bash
python3 ./scripts/export_debug_bundle.py <episode_id> \
  --script-checkpoint-dir ./.script_checkpoints \
  --audio-checkpoint-dir ./out/.audio_checkpoints \
  --script-path ./script.json \
  --source-path ./source.txt \
  --log-path ./podcast_run_logs.txt
```

Before sharing the bundle:

1. Prefer a stable explicit `--episode-id` across script and audio stages
2. Always pass `--script-path`
3. Inspect `collection_report.json`
4. Confirm `debug_bundle_tree.txt` contains the expected manifests and summaries
