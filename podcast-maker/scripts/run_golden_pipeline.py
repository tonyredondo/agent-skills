#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.abspath(os.path.join(scripts_dir, ".."))
    parser = argparse.ArgumentParser(
        description="Generate golden candidate scripts using the production pipeline"
    )
    parser.add_argument(
        "--cases-path",
        default=os.path.join(repo_dir, "tests", "fixtures", "golden", "cases.json"),
    )
    parser.add_argument(
        "--candidate-dir",
        default=os.environ.get("GOLDEN_CANDIDATE_DIR", os.path.join(repo_dir, ".golden_candidates")),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--report-json", default="")
    return parser.parse_args()


def _load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise RuntimeError("Cases file must be a JSON array")
    cases: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            cases.append(item)
    if not cases:
        raise RuntimeError("No valid cases found in cases file")
    return cases


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    cases_path = os.path.abspath(args.cases_path)
    candidate_dir = os.path.abspath(args.candidate_dir)
    os.makedirs(candidate_dir, exist_ok=True)
    script_entry = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_script.py")
    cases = _load_cases(cases_path)

    run_report: Dict[str, Any] = {
        "cases_path": cases_path,
        "candidate_dir": candidate_dir,
        "started_at": int(time.time()),
        "results": [],
    }

    failures = 0
    for idx, case in enumerate(cases, start=1):
        name = str(case.get("case_name", f"case_{idx}")).strip() or f"case_{idx}"
        source_path = str(case.get("source_path", "")).strip()
        if not source_path:
            failures += 1
            run_report["results"].append(
                {
                    "case_name": name,
                    "status": "failed",
                    "error": "missing source_path",
                }
            )
            if args.stop_on_error:
                break
            continue
        source_abs = source_path if os.path.isabs(source_path) else os.path.join(repo_root, source_path)
        output_path = os.path.join(candidate_dir, f"{name}.json")
        profile = str(case.get("profile", "standard"))
        cmd = [sys.executable, script_entry]
        if args.debug:
            cmd.append("--debug")
        cmd.extend(["--profile", profile])
        if case.get("target_minutes") is not None:
            cmd.extend(["--target-minutes", str(case["target_minutes"])])
        if case.get("words_per_min") is not None:
            cmd.extend(["--words-per-min", str(case["words_per_min"])])
        if case.get("min_words") is not None:
            cmd.extend(["--min-words", str(case["min_words"])])
        if case.get("max_words") is not None:
            cmd.extend(["--max-words", str(case["max_words"])])
        cmd.extend([source_abs, output_path])

        started = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed = round(time.time() - started, 2)
        ok = proc.returncode == 0 and os.path.exists(output_path)
        result = {
            "case_name": name,
            "profile": profile,
            "source_path": source_abs,
            "output_path": output_path,
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "status": "completed" if ok else "failed",
            "stdout_tail": (proc.stdout or "")[-500:],
            "stderr_tail": (proc.stderr or "")[-1000:],
        }
        run_report["results"].append(result)
        if not ok:
            failures += 1
            if args.stop_on_error:
                break

    run_report["completed_at"] = int(time.time())
    run_report["failures"] = failures
    run_report["overall_pass"] = failures == 0

    report_path = (
        os.path.abspath(args.report_json)
        if args.report_json
        else os.path.join(candidate_dir, "golden_pipeline_run_report.json")
    )
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"overall_pass": run_report["overall_pass"], "failures": failures, "report": report_path}))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

