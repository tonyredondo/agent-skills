#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

from pipeline.golden_gate import evaluate_golden_suite


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _print_summary(*, report: dict, candidate_dir: str) -> None:
    print(
        json.dumps(
            {
                "overall_pass": bool(report.get("overall_pass", False)),
                "cases": len(report.get("cases", [])),
                "candidate_dir": candidate_dir,
                **(
                    {"error": str(report.get("error", ""))}
                    if str(report.get("error", "")).strip()
                    else {}
                ),
            }
        )
    )


def parse_args() -> argparse.Namespace:
    repo_dir = _repo_root()
    parser = argparse.ArgumentParser(description="Run golden regression gate for podcast maker")
    parser.add_argument(
        "--candidate-dir",
        default=os.environ.get("GOLDEN_CANDIDATE_DIR", os.path.join(repo_dir, ".golden_candidates")),
        help="Directory with generated candidate script JSON files",
    )
    parser.add_argument(
        "--fixtures-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "tests",
            "fixtures",
            "golden",
        ),
    )
    parser.add_argument(
        "--baseline-path",
        default=None,
    )
    parser.add_argument(
        "--allow-fixture-fallback",
        action="store_true",
        help="Allow fallback to fixtures dir when candidate file is missing",
    )
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_dir = os.path.abspath(args.candidate_dir)
    fixtures_dir = os.path.abspath(args.fixtures_dir)
    baseline_path = (
        os.path.abspath(args.baseline_path)
        if args.baseline_path
        else os.path.join(fixtures_dir, "baseline_metrics.json")
    )
    fallback_fixtures = fixtures_dir if args.allow_fixture_fallback else None
    try:
        report = evaluate_golden_suite(
            baseline_path=baseline_path,
            candidate_dir=candidate_dir,
            fixtures_dir=fallback_fixtures,
        )
    except Exception as exc:  # noqa: BLE001
        report = {
            "overall_pass": False,
            "cases": [],
            "candidate_dir": candidate_dir,
            "baseline_path": baseline_path,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    _print_summary(report=report, candidate_dir=candidate_dir)
    return 0 if bool(report.get("overall_pass", False)) else 1


if __name__ == "__main__":
    sys.exit(main())

