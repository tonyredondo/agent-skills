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
   - keep script quality gate enabled (`SCRIPT_QUALITY_GATE_ACTION=enforce`)
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

## Rollback

Immediate rollback to the last known-good release if:
- critical reliability regression appears
- SLO misses exceed threshold
- repeated checkpoint corruption or resume failures

