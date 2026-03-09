# Podcast Maker Quality And Golden Tests

Use this reference for script-quality behavior, repair tuning, and release-gate regression checks.

## Pre-Audio Script Quality Gate

`make_podcast.py` validates the script before TTS:

- default action: `SCRIPT_QUALITY_GATE_ACTION=warn`
- evaluator default: `SCRIPT_QUALITY_GATE_EVALUATOR=hybrid`
- `short` profile default LLM sample: `0.5`
- `standard` and `long` default LLM sample: `1.0`

When evaluation fails:

- `enforce`: exit code `4`, no TTS call
- `warn`: log warning and continue
- `off`: skip the gate

Key defaults:

- `SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS=1`
- `SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS_ON_FAIL=1`
- `SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS_MIN_CONFIDENCE=0.55`
- `SCRIPT_QUALITY_GATE_SEMANTIC_FALLBACK=1`
- `SCRIPT_QUALITY_GATE_SEMANTIC_MIN_CONFIDENCE=0.55`
- `SCRIPT_QUALITY_GATE_SEMANTIC_TAIL_LINES=10`

Optional script-stage validation after `make_script.py` output:

- `SCRIPT_QUALITY_GATE_SCRIPT_ACTION=off|warn|enforce`
- `SCRIPT_QUALITY_GATE_AUTO_REPAIR=0|1`
- `SCRIPT_QUALITY_GATE_REPAIR_ATTEMPTS`
- `SCRIPT_QUALITY_GATE_REPAIR_TOTAL_TIMEOUT_SECONDS`
- `SCRIPT_QUALITY_GATE_REPAIR_ATTEMPT_TIMEOUT_SECONDS`

Recommended strict preset:

```bash
export SCRIPT_QUALITY_GATE_PROFILE=production_strict
```

## Quality Artifacts

- script-stage:
  - `<script_checkpoint_dir>/<episode>/quality_report_initial.json`
  - `<script_checkpoint_dir>/<episode>/quality_report.json`
- audio-stage:
  - `<audio_checkpoint_dir>/<episode>/quality_report.json`
- normalized script consumed by TTS:
  - `<audio_checkpoint_dir>/<episode>/normalized_script.json`

## Quality-Loop Comparison

Use this helper to compare baseline versus candidate script quality:

```bash
python3 ./scripts/evaluate_script_quality_loop.py \
  --source /path/to/source.txt \
  --baseline-dir /tmp/podcast_eval/baseline \
  --candidate-dir /tmp/podcast_eval/candidate \
  --out /tmp/podcast_eval/before_vs_after.json
```

Use this before changing prompts, repairs, or thresholds.

## Golden Regression Gate

Run both commands before promoting release candidates:

```bash
python3 ./scripts/run_golden_pipeline.py --candidate-dir ./.golden_candidates
python3 ./scripts/check_golden_suite.py --candidate-dir ./.golden_candidates
```

Notes:

- `run_golden_pipeline.py` requires valid OpenAI credentials
- candidate sources must satisfy the current source-validation policy
- if auth, network, or source-sizing constraints block candidate generation, use fixture fallback for structural validation

Fixture fallback:

```bash
python3 ./scripts/check_golden_suite.py --candidate-dir ./.golden_candidates --allow-fixture-fallback
```

Useful options:

- `run_golden_pipeline.py`: `--cases-path`, `--stop-on-error`, `--report-json`, `--debug`
- `check_golden_suite.py`: `--fixtures-dir`, `--baseline-path`, `--json-out`, `--allow-fixture-fallback`

The suite validates structural quality against:

- `tests/fixtures/golden/*.json`
- `tests/fixtures/golden/baseline_metrics.json`
- `tests/fixtures/golden/sources/*.txt`
- `tests/fixtures/golden/cases.json`
