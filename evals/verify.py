"""Ground-truth verification digest for the SUITE itself (not the agent).

For every dev + holdout task it lays the reference_answer next to the Wikipedia
text the agent actually retrieved in a prior run, with a lexical support hint per
reference point. It does NOT run the agent and does NOT grade — it helps a human
confirm: references are true, should_refuse is right, expected_sources actually
contain the answer, and unanswerables are genuinely absent from Wikipedia.

Writes verify.txt into the run dir and prints the same text to stdout.

    python -m evals.verify [--run DIR]   # default: latest run under evals/runs/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
RUNS_DIR = HERE / "runs"
_STOP = {"the", "and", "for", "was", "were", "are", "its", "that", "with", "from", "has", "have",
         "his", "her", "their", "which", "into", "about", "this", "than", "not", "but", "you", "your"}


def _latest_run() -> Path:
    runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()) if RUNS_DIR.exists() else []
    if not runs:
        raise SystemExit("no runs under evals/runs/")
    return runs[-1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _points(reference: str | None) -> list[str]:
    if not reference:
        return []
    return [p.strip() for p in re.split(r"[;.]\s+", reference.strip().rstrip(".")) if p.strip()]


def _tokens(s: str) -> list[str]:
    toks = re.findall(r"[0-9]+(?:\.[0-9]+)?|[a-z]{3,}", s.lower())
    return [t for t in toks if t not in _STOP]


def _support(point: str, retrieved: str) -> str:
    toks = _tokens(point)
    if not toks:
        return "?"
    low = retrieved.lower()
    hit = sum(1 for t in toks if t in low)
    frac = hit / len(toks)
    return "yes" if frac >= 0.7 else "partial" if frac >= 0.34 else "no"


def _snippet(reference: str, retrieved: str, n: int = 300) -> str:
    ref_toks = set(_tokens(reference))
    sents = re.split(r"(?<=[.!?])\s+", retrieved)
    scored = sorted(((sum(1 for t in _tokens(s) if t in ref_toks), s) for s in sents), key=lambda x: -x[0])
    best = [s for score, s in scored[:2] if score > 0]
    text = " … ".join(best) if best else retrieved[:n]
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def build_verify_text(dev: list[dict], holdout: list[dict], records: dict[str, dict], scored: dict[str, dict]) -> str:
    tasks = [(t, "dev") for t in dev] + [(t, "holdout") for t in holdout]

    def has_retrieved(tid: str) -> bool:
        return bool((records.get(tid) or {}).get("retrieved_text"))

    def correctness_failed(tid: str) -> bool:
        g = (scored.get(tid) or {}).get("grades", {}).get("correctness", {})
        return bool(g.get("applicable")) and g.get("pass") is False

    def bucket(t: dict) -> int:
        tid = t["id"]
        if t["category"] == "unanswerable":
            return 1
        if not has_retrieved(tid):
            return 2
        if correctness_failed(tid):
            return 3
        return 4

    tasks.sort(key=lambda ts: (bucket(ts[0]), ts[0]["id"]))

    verify_on_own = [t["id"] for t, _ in tasks if t["category"] == "unanswerable" or not has_retrieved(t["id"])]
    bad_ref = [t["id"] for t, _ in tasks if correctness_failed(t["id"])]

    out: list[str] = []
    out.append("===== VERIFY ON YOUR OWN (no retrieved text to lean on) =====")
    out.append("All unanswerables + every no-fetch / held-out task. Check these against Wikipedia by hand:")
    for tid in verify_on_own:
        out.append(f"  - {tid}")
    out.append("\n===== POSSIBLE BAD REFERENCE (agent answered, contradicts our reference) =====")
    out.append(", ".join(bad_ref) if bad_ref else "(none)")

    labels = {1: "1) UNANSWERABLE — verify genuinely absent from Wikipedia",
              2: "2) NO RETRIEVED TEXT — verify manually",
              3: "3) POSSIBLE BAD REFERENCE — agent disagrees with our reference",
              4: "4) AUTO-CORROBORATED (retrieved text present)"}
    current = None
    for t, split in tasks:
        b = bucket(t)
        if b != current:
            current = b
            out.append(f"\n========== {labels[b]} ==========")
        tid = t["id"]
        rec = records.get(tid) or {}
        retrieved = rec.get("retrieved_text") or ""
        flag = "  [POSSIBLE BAD REFERENCE]" if correctness_failed(tid) else ""
        out.append(f"\n[{tid}] ({t['category']}, {split}){flag}")
        out.append(f"  Q: {t['question']}")
        out.append(f"  should_refuse: {bool(t.get('should_refuse'))}")
        out.append(f"  expected_sources: {t.get('expected_sources', [])}")
        pts = _points(t.get("reference_answer"))
        if pts:
            out.append("  reference points:")
            for p in pts:
                tag = f"  [support: {_support(p, retrieved)}]" if retrieved else ""
                out.append(f"    - {p}{tag}")
        elif t["category"] != "unanswerable":
            out.append("  reference points: (none)")
        if retrieved:
            out.append(f"  retrieved: {_snippet(t.get('reference_answer') or t['question'], retrieved)}")
        elif split == "holdout":
            out.append("  retrieved: HELD-OUT (not run) - verify manually")
        else:
            out.append("  retrieved: NO FETCH - verify manually")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(prog="verify", description="Ground-truth verification digest for the suite.")
    parser.add_argument("--run", help="Run dir (default: latest under evals/runs/).")
    args = parser.parse_args()
    run_dir = Path(args.run) if args.run else _latest_run()

    raw_records = _load_jsonl(run_dir / "records.jsonl")
    raw_scored = json.loads((run_dir / "scores.json").read_text())["scored"]
    n_trials = max((r.get("trial", 0) for r in raw_records), default=0) + 1

    def _trial0(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for it in items:
            tid = it["task_id"]
            if tid not in out or it.get("trial", 0) < out[tid].get("trial", 0):
                out[tid] = it
        return out

    records = _trial0(raw_records)
    scored = _trial0(raw_scored)
    dev = _load_jsonl(HERE / "suite.jsonl")
    holdout = _load_jsonl(HERE / "holdout.jsonl")

    text = build_verify_text(dev, holdout, records, scored)
    if n_trials > 1:
        text = f"NOTE: {n_trials} trials per task — showing trial 0 only.\n\n" + text
    (run_dir / "verify.txt").write_text(text + "\n")
    print(f"verify digest of {run_dir.name}  (wrote verify.txt)\n")
    print(text)


if __name__ == "__main__":
    main()
