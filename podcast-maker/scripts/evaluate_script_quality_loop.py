#!/usr/bin/env python3
from __future__ import annotations

"""Compare baseline vs candidate script quality metrics across runs."""

import argparse
import json
import math
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


META_LANGUAGE_RE = re.compile(
    r"(?:\bseg[uú]n\s+el\s+[ií]ndice\b|\ben\s+este\s+resumen\b|\ben\s+el\s+siguiente\s+tramo\b|\bruta\s+del\s+episodio\b|\btabla\s+de\s+contenidos?\b)",
    re.IGNORECASE,
)
FORCED_TEASE_RE = re.compile(
    r"(?:\bte\s+voy\s+a\s+(?:chinchar|pinchar|provocar|picar)\b|\bvoy\s+a\s+(?:chincharte|pincharte|provocarte|picarte)\b|\bte\s+pincho\s+un\s+poco\b)",
    re.IGNORECASE,
)
RECAP_TOKENS = (
    "en resumen",
    "en sintesis",
    "en conclusion",
    "nos quedamos con",
    "en pocas palabras",
    "in summary",
    "to sum up",
    "em resumo",
    "en bref",
)
FAREWELL_TOKENS = (
    "gracias por escuch",
    "hasta la proxima",
    "nos vemos",
    "nos escuchamos",
    "adios",
    "thank you",
    "thanks for listening",
    "see you",
    "goodbye",
    "next episode",
    "until next",
    "merci",
    "au revoir",
    "obrigad",
)
SOURCE_INDEX_ITEM_RE = re.compile(
    r"^\s*-\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+·\s+([^·]+)\s+·\s+(.+?)\s*(?:\([^)]*\))?\s*$"
)
TRANSITION_CONNECTOR_RE = re.compile(
    r"^(?:y\s+de\s+hecho|por\s+otro\s+lado|ahora\s+bien|dicho\s+esto|a\s+partir\s+de\s+ahi|pasando\s+a|en\s+paralelo|si\s+lo\s+conectamos|en\s+ese\s+sentido|por\s+cierto)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[^\W_]{3,}", re.UNICODE)
SOURCE_BALANCE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "about",
    "over",
    "under",
    "como",
    "para",
    "desde",
    "sobre",
    "entre",
    "hacia",
}


def _normalized_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    deaccented = "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )
    return re.sub(r"\s+", " ", deaccented)


def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall(_normalized_text(text))


def _question_cadence(lines: List[Dict[str, Any]]) -> Dict[str, float]:
    if not lines:
        return {"question_ratio": 0.0, "max_question_streak": 0.0}
    question_lines = 0
    max_streak = 0
    streak = 0
    for line in lines:
        is_question = "?" in str(line.get("text", ""))
        if is_question:
            question_lines += 1
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        "question_ratio": float(question_lines) / float(max(1, len(lines))),
        "max_question_streak": float(max_streak),
    }


def _long_turn_count(lines: List[Dict[str, Any]], *, max_turn_words: int) -> int:
    count = 0
    for line in lines:
        if len(re.findall(r"\b\w+\b", str(line.get("text", "")), re.UNICODE)) > max(8, int(max_turn_words)):
            count += 1
    return count


def _abrupt_transition_count(lines: List[Dict[str, Any]]) -> int:
    abrupt = 0
    for idx in range(1, len(lines)):
        prev_text = str(lines[idx - 1].get("text", "")).strip()
        curr_text = str(lines[idx].get("text", "")).strip()
        if not prev_text or not curr_text:
            continue
        prev_tokens = set(_tokenize(prev_text))
        curr_tokens = set(_tokenize(curr_text))
        if len(prev_tokens) < 4 or len(curr_tokens) < 4:
            continue
        overlap = prev_tokens.intersection(curr_tokens)
        overlap_ratio = float(len(overlap)) / float(max(1, len(curr_tokens)))
        has_connector = TRANSITION_CONNECTOR_RE.search(curr_text.lower()) is not None
        if overlap_ratio < 0.08 and not has_connector:
            abrupt += 1
    return abrupt


def _source_category_profiles(source_text: str) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for raw_line in str(source_text or "").splitlines()[:420]:
        line = str(raw_line or "").strip()
        if not line:
            continue
        match = SOURCE_INDEX_ITEM_RE.match(line)
        if match is None:
            continue
        category = str(match.group(1) or "").strip()
        title = str(match.group(2) or "").strip()
        if not category or not title:
            continue
        key = _normalized_text(category)
        profile = profiles.setdefault(
            key,
            {
                "category": category,
                "source_count": 0,
                "keywords": set(),
            },
        )
        profile["source_count"] = int(profile["source_count"]) + 1
        for token in _tokenize(title):
            if len(token) < 4 or token in SOURCE_BALANCE_STOPWORDS:
                continue
            profile["keywords"].add(token)
        for alias in (
            key,
            key.replace("science", "ciencia"),
            key.replace("technology", "tecnologia"),
            key.replace("health", "salud"),
            key.replace("business", "negocio"),
            key.replace("world", "mundo"),
            key.replace("culture", "cultura"),
        ):
            for token in _tokenize(alias):
                if len(token) >= 4:
                    profile["keywords"].add(token)
    return profiles


def _topic_share(lines: List[Dict[str, Any]], *, source_text: str) -> Tuple[Dict[str, float], float | None]:
    profiles = _source_category_profiles(source_text)
    if len(profiles) < 2:
        return {}, None
    scores: Dict[str, int] = {key: 0 for key in profiles}
    for line in lines:
        line_tokens = set(_tokenize(str(line.get("text", ""))))
        if len(line_tokens) < 3:
            continue
        for key, profile in profiles.items():
            keywords = set(profile.get("keywords", set()))
            if line_tokens.intersection(keywords):
                scores[key] = int(scores[key]) + 1
    total_hits = int(sum(scores.values()))
    if total_hits <= 0:
        return {}, None
    shares: Dict[str, float] = {}
    for key, score in scores.items():
        category_name = str(profiles[key].get("category", key))
        shares[category_name] = float(score) / float(total_hits)
    return shares, (max(shares.values()) if shares else None)


def _load_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    lines = payload.get("lines")
    if not isinstance(lines, list):
        raise ValueError(f"Expected key 'lines' array at {path}")
    return payload


def _compute_run_metrics(path: Path, *, source_text: str, max_turn_words: int) -> Dict[str, Any]:
    payload = _load_payload(path)
    lines = list(payload.get("lines", []))
    joined_text = " ".join(str(line.get("text", "")) for line in lines)
    normalized_joined = _normalized_text(joined_text)
    meta_language_hits = sum(
        1 for line in lines if META_LANGUAGE_RE.search(str(line.get("text", "")))
    )
    forced_tease_hits = sum(
        1 for line in lines if FORCED_TEASE_RE.search(str(line.get("text", "")))
    )
    question = _question_cadence(lines)
    long_turn_count = _long_turn_count(lines, max_turn_words=max_turn_words)
    abrupt_transition_count = _abrupt_transition_count(lines)
    topic_share, max_topic_share = _topic_share(lines, source_text=source_text)
    summary_signal_present = any(token in normalized_joined for token in RECAP_TOKENS)
    farewell_signal_present = any(token in normalized_joined for token in FAREWELL_TOKENS)
    return {
        "file": str(path),
        "line_count": len(lines),
        "meta_language_hits": meta_language_hits,
        "forced_tease_hits": forced_tease_hits,
        "topic_share": topic_share,
        "max_topic_share": max_topic_share,
        "question_cadence": {
            "question_ratio": round(float(question["question_ratio"]), 4),
            "max_question_streak": int(question["max_question_streak"]),
        },
        "long_turn_count": long_turn_count,
        "abrupt_transition_count": abrupt_transition_count,
        "summary_signal_present": bool(summary_signal_present),
        "farewell_signal_present": bool(farewell_signal_present),
    }


def _median(values: Iterable[float]) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return None
    return float(statistics.median(numeric))


def _aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_topic_values = [run.get("max_topic_share") for run in runs if run.get("max_topic_share") is not None]
    return {
        "run_count": len(runs),
        "median_meta_language_hits": _median(run.get("meta_language_hits") for run in runs),
        "median_forced_tease_hits": _median(run.get("forced_tease_hits") for run in runs),
        "median_max_topic_share": _median(max_topic_values),
        "median_abrupt_transition_count": _median(run.get("abrupt_transition_count") for run in runs),
        "median_long_turn_count": _median(run.get("long_turn_count") for run in runs),
        "median_question_ratio": _median(
            run.get("question_cadence", {}).get("question_ratio") for run in runs
        ),
        "summary_signal_rate": (
            float(sum(1 for run in runs if bool(run.get("summary_signal_present")))) / float(max(1, len(runs)))
        ),
        "farewell_signal_rate": (
            float(sum(1 for run in runs if bool(run.get("farewell_signal_present")))) / float(max(1, len(runs)))
        ),
    }


def _delta(candidate: float | None, baseline: float | None) -> Dict[str, float | None]:
    if candidate is None or baseline is None:
        return {"absolute": None, "relative": None}
    absolute = float(candidate) - float(baseline)
    relative = None if baseline == 0 else float(candidate) / float(baseline)
    return {"absolute": absolute, "relative": relative}


def _collect_runs(directory: Path, *, source_text: str, max_turn_words: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    files = sorted(path for path in directory.glob("*.json") if path.is_file())
    runs: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for path in files:
        try:
            runs.append(_compute_run_metrics(path, source_text=source_text, max_turn_words=max_turn_words))
        except Exception:
            skipped.append(str(path))
            continue
    return runs, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs candidate script quality loops.")
    parser.add_argument("--source", required=True, help="Source text path used for generation.")
    parser.add_argument("--baseline-dir", required=True, help="Directory with baseline script JSON runs.")
    parser.add_argument("--candidate-dir", required=True, help="Directory with candidate script JSON runs.")
    parser.add_argument("--out", required=True, help="Output JSON report path.")
    parser.add_argument("--max-turn-words", type=int, default=58, help="Long-turn threshold.")
    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    source_text = source_path.read_text(encoding="utf-8")
    baseline_runs, baseline_skipped = _collect_runs(
        baseline_dir,
        source_text=source_text,
        max_turn_words=args.max_turn_words,
    )
    candidate_runs, candidate_skipped = _collect_runs(
        candidate_dir,
        source_text=source_text,
        max_turn_words=args.max_turn_words,
    )
    if not baseline_runs:
        raise RuntimeError(f"No baseline .json files found in {baseline_dir}")
    if not candidate_runs:
        raise RuntimeError(f"No candidate .json files found in {candidate_dir}")

    baseline_summary = _aggregate(baseline_runs)
    candidate_summary = _aggregate(candidate_runs)

    checks = {
        "meta_language_reduced": bool(
            float(candidate_summary["median_meta_language_hits"] or 0.0)
            <= float(baseline_summary["median_meta_language_hits"] or 0.0) * 0.70
        ),
        "forced_tease_zero": bool(float(candidate_summary["median_forced_tease_hits"] or 0.0) == 0.0),
        "max_topic_share_reduced": (
            True
            if baseline_summary["median_max_topic_share"] is None or candidate_summary["median_max_topic_share"] is None
            else float(candidate_summary["median_max_topic_share"])
            <= float(baseline_summary["median_max_topic_share"]) * 0.90
        ),
        "abrupt_transitions_reduced": bool(
            float(candidate_summary["median_abrupt_transition_count"] or 0.0)
            <= float(baseline_summary["median_abrupt_transition_count"] or 0.0) * 0.80
        ),
        "summary_signal_100": bool(float(candidate_summary["summary_signal_rate"] or 0.0) >= 1.0),
        "farewell_signal_100": bool(float(candidate_summary["farewell_signal_rate"] or 0.0) >= 1.0),
    }

    report = {
        "source": str(source_path),
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "baseline_runs": baseline_runs,
        "candidate_runs": candidate_runs,
        "baseline_skipped_files": baseline_skipped,
        "candidate_skipped_files": candidate_skipped,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "deltas": {
            "meta_language_hits": _delta(
                candidate_summary["median_meta_language_hits"],
                baseline_summary["median_meta_language_hits"],
            ),
            "forced_tease_hits": _delta(
                candidate_summary["median_forced_tease_hits"],
                baseline_summary["median_forced_tease_hits"],
            ),
            "max_topic_share": _delta(
                candidate_summary["median_max_topic_share"],
                baseline_summary["median_max_topic_share"],
            ),
            "abrupt_transition_count": _delta(
                candidate_summary["median_abrupt_transition_count"],
                baseline_summary["median_abrupt_transition_count"],
            ),
        },
        "checks": checks,
        "improved": bool(all(checks.values())),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
