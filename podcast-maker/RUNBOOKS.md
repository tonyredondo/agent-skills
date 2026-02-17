# Podcast Maker Runbooks

## 1) Script without progress

Symptoms:
- repeated continuation rounds
- `word_delta` below threshold
- run ends with "No progress while expanding script"

Actions:
1. Re-run with `--debug` and inspect checkpoint `run_summary.json`.
2. Increase `SCRIPT_MAX_CONTEXT_LINES` and/or lower `SCRIPT_MIN_WORD_DELTA`.
3. Increase `SCRIPT_NO_PROGRESS_ROUNDS` when content grows slowly but still makes progress.
4. If source is too short, enrich input and use `--resume-force` (guide by default profile: `short=0.35`, `standard=0.50`, `long=0.60` target-word ratio).
5. If failures are transient/quality-related, tune script orchestrated retry controls:
   - `SCRIPT_ORCHESTRATED_RETRY_ENABLED=0|1` (default `1`)
   - `SCRIPT_ORCHESTRATED_MAX_ATTEMPTS` (default `2`)
   - `SCRIPT_ORCHESTRATED_RETRY_BACKOFF_MS` (default `400`)
   - `SCRIPT_ORCHESTRATED_RETRY_FAILURE_KINDS`

## 2) TTS stuck or very slow

Symptoms:
- heartbeat shows no segment progress
- repeated retries on same segment

Actions:
1. Check network/API status.
2. Reduce `TTS_MAX_CONCURRENT` to `1`.
3. Increase `TTS_TIMEOUT_SECONDS`.
4. Inspect `audio_manifest.json` failed segments and confirm `error_kind` (`timeout`/`stuck`/`rate_limit`/`network`/`resume_blocked`).
5. Resume with:
   - `./scripts/make_podcast.py --resume script.json outdir episode`
   - if script/audio are split across commands, use the same `--episode-id` in both commands
6. Tune audio orchestrated retry controls:
   - `AUDIO_ORCHESTRATED_RETRY_ENABLED=0|1` (default `1`)
   - `AUDIO_ORCHESTRATED_MAX_ATTEMPTS` (default `2`)
   - `AUDIO_ORCHESTRATED_RETRY_BACKOFF_MS` (default `1200`)
   - `AUDIO_ORCHESTRATED_RETRY_FAILURE_KINDS`

## 3) Checkpoint corrupted or lock orphaned

Symptoms:
- lock errors even when no process is running
- malformed checkpoint JSON

Actions:
1. Use `--force-unlock`.
2. Corrupt files are auto-quarantined as `*.corrupt.<timestamp>.json`.
3. If you requested `--resume` and quarantine happened, inspect/fix backup then retry with `--resume-force` or without `--resume`.

## 4) Persistent rate-limit

Symptoms:
- repeated 429 errors

Actions:
1. Lower concurrency and retry later.
2. Increase backoff:
   - `TTS_RETRY_BACKOFF_BASE_MS`
   - `TTS_RETRY_BACKOFF_MAX_MS`
3. If request budget is hit, raise `MAX_REQUESTS_PER_RUN`.

## 5) ffmpeg failure

Symptoms:
- concat/loudnorm/eq command fails

Actions:
1. Verify `ffmpeg` is installed and executable.
2. Re-run with `--debug` and review stderr command logs.
3. If concat copy fails due to codec mismatch, the pipeline auto-fallback re-encodes before concat.
4. If `ffmpeg` is unavailable, run with `--allow-raw-only` (or `ALLOW_RAW_ONLY=1`) to produce `_raw_only.mp3`.

## 6) Canary SLO gate blocks promotion

Symptoms:
- run exits with SLO gate warning/error
- `slo_gate_triggered` appears in logs

Actions:
1. Inspect `SLO_HISTORY_PATH` JSONL history.
2. For canary enforce mode, export before running the pipeline:
   - `SLO_GATE_MODE=enforce`
   - `SLO_WINDOW_SIZE=20`
   - `SLO_REQUIRED_FAILED_WINDOWS=2`
3. Rebuild golden candidates and rerun gate:
   - `python3 ./scripts/run_golden_pipeline.py --candidate-dir ./.golden_candidates`
   - `python3 ./scripts/check_golden_suite.py --candidate-dir ./.golden_candidates`
4. If candidate generation fails due auth/network or source-sizing constraints, run deterministic fallback gate:
   - `python3 ./scripts/check_golden_suite.py --candidate-dir ./.golden_candidates --allow-fixture-fallback`

## 7) Script quality gate blocks audio start

Symptoms:
- run exits with code `4`
- log event `podcast_failed_script_quality_gate`
- no TTS segment request was sent
- optional script-side enforce mode exits `make_script.py` with `failure_kind=script_quality_rejected`

Actions:
1. Inspect quality report:
   - script-stage reports:
     - `<script_checkpoint_dir>/<episode>/quality_report_initial.json`
     - `<script_checkpoint_dir>/<episode>/quality_report.json`
   - audio-stage report:
     - `<outdir>/.audio_checkpoints/<episode>/quality_report.json`
2. Review failed checks under `rules` and score thresholds under `scores`.
3. Check repair metadata in the report:
   - `initial_pass`, `repair_attempted`, `repair_succeeded`, `repair_changed_script`, `repair_attempts_used`, `repair_history`
4. Confirm source sizing in script `run_summary.json` (`source_to_target_ratio`, `source_recommended_min_words`, `source_required_min_words`).
   - default policy: `short=warn(0.35/0.22)`, `standard=enforce(0.50/0.35)`, `long=enforce(0.60/0.45)`.
   - for 15-min standard runs, treat sources under ~35% of target as hard-stop and under ~50% as low-confidence.
5. If this is an exploratory run, use temporary override:
   - `SCRIPT_QUALITY_GATE_ACTION=warn`
6. If LLM evaluator is too strict/costly in current rollout, adjust:
   - `SCRIPT_QUALITY_GATE_EVALUATOR=rules|hybrid|llm`
   - `SCRIPT_QUALITY_GATE_LLM_SAMPLE` (hybrid only)
   - `SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS=0|1`
   - `SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS_ON_FAIL=0|1`
   - `SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS_MIN_CONFIDENCE`
   - `SCRIPT_QUALITY_EVAL_REASONING_EFFORT=low|medium|high` (quality-eval stage only; keep global `SCRIPT_REASONING_EFFORT` at default `low` unless needed)
  - semantic fallback (for summary/closing misses, independent from exact regex wording):
     - `SCRIPT_QUALITY_GATE_SEMANTIC_FALLBACK=1`
     - `SCRIPT_QUALITY_GATE_SEMANTIC_MIN_CONFIDENCE=0.55`
     - `SCRIPT_QUALITY_GATE_SEMANTIC_TAIL_LINES=10`
7. For script-stage quality validation (optional):
   - `SCRIPT_QUALITY_GATE_SCRIPT_ACTION=warn|enforce`
   - `SCRIPT_QUALITY_GATE_AUTO_REPAIR=0|1`
   - `SCRIPT_QUALITY_GATE_REPAIR_ATTEMPTS`
   - default repair attempts: `2` (increase for harder long-context runs)
   - repair time budgets (entrypoint-level defaults):
     - `SCRIPT_QUALITY_GATE_REPAIR_TOTAL_TIMEOUT_SECONDS=300`
     - `SCRIPT_QUALITY_GATE_REPAIR_ATTEMPT_TIMEOUT_SECONDS=90`
   - defaults by gate profile (when unset): `default=warn`, `production_strict=enforce`
   - monotonic repair defaults:
     - `SCRIPT_QUALITY_GATE_REPAIR_REVERT_ON_FAIL=1`
     - `SCRIPT_QUALITY_GATE_REPAIR_MIN_WORD_RATIO=0.85`
     - `SCRIPT_QUALITY_GATE_REPAIR_OUTPUT_TOKENS_HARD_CAP=6400`
   - monitor prompt pressure in logs (`script_quality_repair_prompt_stats`, `script_quality_tail_repair_prompt_stats`) when timeout repeats
8. For strict production rollout defaults in one switch:
   - `SCRIPT_QUALITY_GATE_PROFILE=production_strict`
9. Reliability v3 flags (rollback controls):
   - `SCRIPT_RECOVERY_LADDER_V2=0` (disable chunk subsplit fallback)
   - `SCRIPT_COMPLETENESS_CHECK_V2=0` (disable internal truncation check/repair)
   - `RUN_MANIFEST_V2=0` (disable run manifest writes)

## 8) Run orchestrator retries (`run_podcast.py`)

Symptoms:
- script stage fails intermittently in end-to-end runs
- rerun succeeds when invoking `make_script.py` manually

Actions:
1. Increase orchestrator script attempts:
   - `RUN_PODCAST_SCRIPT_ATTEMPTS` (default `1`)
2. Keep a stable `--episode-id` (or consistent basename) so retries reuse the same checkpoint lineage.
3. For shell portability, use:
   - `./scripts/make_podcast.sh script.json outdir episode_name --profile standard --resume`

## 9) Invalid basename argument

Symptoms:
- `make_podcast.py` fails immediately with argument parsing error
- message indicates invalid `basename`/path separators

Actions:
1. Use a plain filename for `basename` (for example: `episode_001`).
2. Do not pass path-like values such as `../episode` or `foo/bar`.
3. Keep directory routing in `outdir`; keep `basename` as name only.

## 10) Error kind map (triage rapido)

Use this table to classify failures consistently in incidents and rollbacks.
`stuck_abort` is derived from structured failure signals, not from log wording.

| Symptom / source error (example) | `error_kind` | `stuck_abort` | Operational action |
| --- | --- | --- | --- |
| `TTS global timeout reached` | `timeout` | `true` | Reduce concurrency, increase `TTS_TIMEOUT_SECONDS`, resume run |
| `TTS stuck detected in chunk ...` | `stuck` | `true` | Inspect chunk/segment pressure, tune timeout/concurrency, resume run |
| Script abort with `No progress while expanding script` | `stuck` | `true` | Increase source detail/tuning, then resume or rerun |
| Source validation enforce blocks run (`Provide at least ~... words`) | `source_too_short` | `false` | Expand source content or lower target duration before rerun |
| Script schema/JSON validation failures in run summary | `invalid_schema` | `false` | Review schema repairs; tune `SCRIPT_PARSE_REPAIR_ATTEMPTS`, `SCRIPT_PARSE_REPAIR_TRUNCATION_BONUS_ATTEMPTS`, and parse-repair output cap/growth; rerun |
| Script quality gate rejects script before TTS | `script_quality_rejected` | `false` | Inspect `quality_report.json`, fix script quality or run in `warn` for controlled tests |
| Run manifest/script handoff mismatch (`script_output_path` or `episode_id` mismatch) | `run_mismatch` | `false` | Re-run script/audio with the same explicit `--episode-id` and matching `script_path`; avoid mixing stale checkpoints |
| `HTTP 429`, `rate limit` | `rate_limit` | `false` | Increase backoff, lower throughput, retry later |
| `urlopen error`, DNS/connection failures | `network` | `false` | Validate network/API reachability and rerun |
| `Resume blocked ...` (fingerprint/script mismatch, corrupt resume path) | `resume_blocked` | `false` | Use `--resume-force` only when intentional, or restart without resume |
| `KeyboardInterrupt` / `InterruptedError` | `interrupted` | `false` | Expected controlled stop; rerun with `--resume` |
| Any unmapped failure | `unknown` | `false` | Escalate with manifest + summaries; add classifier if recurring |

Quick inspection:

- Segment-level kinds: `<outdir>/.audio_checkpoints/<episode>/audio_manifest.json`
- Quality gate report: `<outdir>/.audio_checkpoints/<episode>/quality_report.json`
- Script-level signals: `<script_checkpoint_dir>/<episode>/run_summary.json`
- Pipeline-level status: `<script_checkpoint_dir>/<episode>/run_manifest.json` and `pipeline_summary.json`
- Run-level event kind: `SLO_HISTORY_PATH` JSONL field `failure_kind`

## 11) Quick jq summaries

Use these commands to spot trends quickly from the latest events.

```bash
HIST="${SLO_HISTORY_PATH:-./.podcast_slo_history.jsonl}"
tail -n 200 "$HIST" | jq -r 'select(.component=="audio") | (.failure_kind // "none")' | sort | uniq -c | sort -nr
```

```bash
HIST="${SLO_HISTORY_PATH:-./.podcast_slo_history.jsonl}"
tail -n 200 "$HIST" | jq -s '
{
  total_events: length,
  stuck_abort_events: (map(select(.stuck_abort == true)) | length),
  failure_kinds: (
    map(.failure_kind // "none")
    | group_by(.)
    | map({kind: .[0], count: length})
    | sort_by(.count)
    | reverse
  )
}'
```

## 12) Export debug bundle for on-call

Use this when support/on-call asks for a single bundle with all diagnostics.

```bash
python3 ./scripts/export_debug_bundle.py <episode_id> \
  --script-checkpoint-dir ./.script_checkpoints \
  --audio-checkpoint-dir ./out/.audio_checkpoints \
  --script-path ./script.json \
  --source-path ./source.txt \
  --log-path ./podcast_run_logs.txt
```

Completeness checklist (before sharing bundle):
1. Prefer a stable explicit `--episode-id` across script/audio runs; pass that as `<episode_id>`.
2. Always pass `--script-path`; this lets exporter resolve script checkpoint alias from script basename when it differs from `<episode_id>`.
3. Open `debug_bundle_tree.txt` and verify these files are present:
   - script `run_summary.json`, `quality_report.json`, `script_checkpoint.json`, `run_manifest.json`, `pipeline_summary.json`
   - audio `podcast_run_summary.json`, `audio_manifest.json`, `quality_report.json`, `normalized_script.json`
4. Inspect `collection_report.json` for per-candidate status (`found`, `missing`, `read_error`, `not_applicable`) and `debug_bundle_metadata.json -> collection_complete` before assuming data loss.
5. If some files are still missing, inspect `debug_bundle_metadata.json -> missing_candidates` and `resolved_episode_id`, then rerun export with corrected paths/checkpoint dirs.

Bundle includes:

- script/audio checkpoints and summaries (when present)
- quality reports
- optional source/script/log files
- `collection_report.json` with explicit collection semantics (`not_applicable` distinguishes not executed from missing)
- `debug_bundle_metadata.json` with sanitized env snapshot, missing files list, effective params, and git commit/skill version (if available)

