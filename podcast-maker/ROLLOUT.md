# Podcast Maker Rollout Policy

## Stages

1. Canary:
   - run golden pipeline + gate before production changes
   - monitor SLO windows in warn mode

2. Enforced canary:
   - `SLO_GATE_MODE=enforce`
   - promote only when windows stay healthy

3. Stable:
   - keep current defaults
   - keep script quality gate enabled (`SCRIPT_QUALITY_GATE_ACTION=warn` by default; use `SCRIPT_QUALITY_GATE_PROFILE=production_strict` for blocking mode)
   - keep hybrid evaluator with structured LLM rule judgments enabled (`SCRIPT_QUALITY_GATE_EVALUATOR=hybrid`, `SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS=1`)
   - run `run_golden_pipeline.py` + `check_golden_suite.py` on each release candidate
   - keep SLO monitoring active for each release

## Promotion requirements

- No open critical incidents
- SLOs healthy for two windows (automatically evaluated from `SLO_HISTORY_PATH`)
- Golden regression suite passes:
  - `python3 ./scripts/run_golden_pipeline.py --candidate-dir ./.golden_candidates`
  - `python3 ./scripts/check_golden_suite.py --candidate-dir ./.golden_candidates`

Recommended canary gate setup:

```bash
export SLO_GATE_MODE=enforce
export SLO_WINDOW_SIZE=20
export SLO_REQUIRED_FAILED_WINDOWS=2
python3 ./scripts/run_golden_pipeline.py --candidate-dir ./.golden_candidates
python3 ./scripts/check_golden_suite.py --candidate-dir ./.golden_candidates
```

Quality-gate rollout notes:

- If hybrid LLM judgments over-correct in canary, tune confidence first (`SCRIPT_QUALITY_GATE_LLM_RULE_JUDGMENTS_MIN_CONFIDENCE`) before disabling judgments.
- If quality repair times out on long prompts, raise only entrypoint repair budgets first (`SCRIPT_QUALITY_GATE_REPAIR_ATTEMPT_TIMEOUT_SECONDS`, `SCRIPT_QUALITY_GATE_REPAIR_TOTAL_TIMEOUT_SECONDS`) and keep attempt count stable.

## Rollback

Immediate rollback to the last known-good release if:
- critical reliability regression appears
- SLO misses exceed threshold
- repeated checkpoint corruption or resume failures

