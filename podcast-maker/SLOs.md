# Podcast Maker SLOs

## Profile targets

- short (5 min):
  - success rate >= 98%
  - p95 total runtime <= 8 min
  - p95 resume runtime <= 4 min

- standard (15 min):
  - success rate >= 97%
  - p95 total runtime <= 20 min
  - p95 resume runtime <= 10 min

- long (30 min):
  - success rate >= 95%
  - p95 total runtime <= 40 min
  - p95 resume runtime <= 18 min

## Technical quality SLOs

- retry_rate per stage < 15%
- stuck_abort_rate < 2%
- invalid_schema_rate < 3%
- script_quality_rejected_rate < 5% (audio component)
- p90 cost estimation error <= 25%

## Rollback trigger

If SLOs are missed for two consecutive windows, rollback to the last known-good release until fixed.

Automation:

- Each script/audio run appends an event to `SLO_HISTORY_PATH` (default `./.podcast_slo_history.jsonl`).
- Failure events include `failure_kind` (for example: `timeout`, `stuck`, `rate_limit`, `resume_blocked`, `script_quality_rejected`).
- Gate mode is controlled with `SLO_GATE_MODE=off|warn|enforce`.
- Window controls:
  - `SLO_WINDOW_SIZE` (default `20`)
  - `SLO_REQUIRED_FAILED_WINDOWS` (default `2`)
- Automatic technical KPI checks per window:
  - `retry_rate < 15%`
  - `stuck_abort_rate < 2%`
  - `invalid_schema_rate < 3%`
  - `script_quality_rejected_rate < 5%`
  - `p90 cost_estimation_error_pct <= 25%` (evaluated only when `ACTUAL_COST_USD` is set)
- Resume runtime SLO is evaluated from events tagged with `is_resume=true`.

