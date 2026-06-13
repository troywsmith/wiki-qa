"""Readable digest of a run dir for grader validation.

Builds a couple of plain-text lines per record, grouped by dimension, so a human
can skim verdicts against inputs without reading raw JSONL. `build_text` is the
single source of truth: the score phase writes it to digest.txt next to
scores.json, and this module prints the same text to stdout.

    python -m evals.digest [--run DIR]   # default: latest run under evals/runs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
RUNS_DIR = HERE / "runs"


def _latest_run() -> Path:
    runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()) if RUNS_DIR.exists() else []
    if not runs:
        raise SystemExit("no runs under evals/runs/")
    return runs[-1]


def _trunc(s: str | None, n: int = 200) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def _verdict(g: dict[str, Any]) -> str:
    if not g.get("applicable"):
        return "n/a"
    return "PASS" if g["pass"] else "FAIL"


def build_text(records: list[dict[str, Any]], scored: list[dict[str, Any]]) -> str:
    rec = {r["task_id"]: r for r in records}
    sc = {s["task_id"]: s for s in scored}

    def grade(tid: str, dim: str) -> dict[str, Any]:
        return sc[tid]["grades"].get(dim, {})

    out: list[str] = []
    section = lambda title: out.append(f"\n===== {title} =====")

    # FAITHFULNESS — mix of pass/fail
    section("FAITHFULNESS — answer vs retrieved text (per-claim)")
    items = [t for t in sc if grade(t, "faithfulness").get("applicable")]
    fails = [t for t in items if grade(t, "faithfulness")["pass"] is False]
    passes = [t for t in items if grade(t, "faithfulness")["pass"] is True]
    for tid in (fails[:2] + passes[:2]) or items[:4]:
        r, fa = rec[tid], grade(tid, "faithfulness")
        out.append(f"\n[{tid}] ({r['category']}) {_verdict(fa)}  supported={fa.get('supported_fraction')}")
        out.append(f"  Q: {_trunc(r['question'], 110)}")
        out.append(f"  A: {_trunc(r['output']['answer'], 220)}")
        out.append(f"  retrieved: {_trunc(r['retrieved_text'], 180) or '(none fetched)'}")
        if fa.get("unsupported_claims"):
            out.append(f"  unsupported: {fa['unsupported_claims']}")

    # CORRECTNESS / COMPLETENESS — include a refusal-on-answerable if present
    section("CORRECTNESS / COMPLETENESS — answer vs reference")
    refusal_on_ans = [
        t for t in sc
        if grade(t, "completeness").get("applicable") and grade(t, "completeness")["pass"] is False
        and any("refus" in m.lower() for m in grade(t, "completeness").get("missing_points", []))
    ]
    others = [t for t in sc if (grade(t, "correctness").get("applicable") or grade(t, "completeness").get("applicable")) and t not in refusal_on_ans]
    for tid in refusal_on_ans[:1] + others[:3]:
        r, co, cm = rec[tid], grade(tid, "correctness"), grade(tid, "completeness")
        out.append(f"\n[{tid}] ({r['category']})  correctness={_verdict(co)} completeness={_verdict(cm)} covered={cm.get('fraction')}")
        out.append(f"  A:   {_trunc(r['output']['answer'], 200)}")
        out.append(f"  ref: {_trunc(r.get('reference_answer'), 160)}")
        if co.get("applicable") and not co["pass"]:
            out.append(f"  contradiction: {co.get('contradicting_claim')}")
        if cm.get("applicable") and cm.get("missing_points"):
            out.append(f"  missing points: {cm['missing_points']}")

    # CALIBRATION
    section("CALIBRATION — decision vs should_refuse")
    out.append("should_refuse = true:")
    for tid in [t for t in sc if rec[t].get("should_refuse")]:
        ca = grade(tid, "calibration")
        out.append(f"  {_verdict(ca)} {tid} ({rec[tid]['category']}) decided={ca.get('decision')} signal={ca.get('structured_signal')}")
    soft = [t for t in sc if grade(t, "calibration").get("applicable") and not grade(t, "calibration")["pass"] and not rec[t].get("should_refuse")]
    out.append("soft refusals (refused an answerable):" if soft else "no soft refusals on answerable tasks")
    for tid in soft:
        ca = grade(tid, "calibration")
        out.append(f"  FAIL {tid} ({rec[tid]['category']}) decided={ca.get('decision')} expected={ca.get('expected')}")
        out.append(f"      A: {_trunc(rec[tid]['output']['answer'], 150)}")

    # ATTRIBUTION
    section("ATTRIBUTION — expected vs cited (normalized)")
    mh = [t for t in sc if rec[t]["category"] == "multi_hop" and grade(t, "attribution").get("applicable")][:2]
    dis = [t for t in sc if rec[t]["category"] == "disambiguation" and grade(t, "attribution").get("applicable")][:1]
    for tid in mh + dis:
        r, at = rec[tid], grade(tid, "attribution")
        out.append(f"\n[{tid}] ({r['category']}) {_verdict(at)}")
        out.append(f"  expected: {r['expected_sources']}")
        out.append(f"  cited:    {r['output']['sources']}")
        out.append(f"  pairs:    {at.get('matched_pairs')}")
        if at.get("missing"):
            out.append(f"  missing:  {at['missing']}")

    # INJECTION
    section("INJECTION")
    inj = [t for t in sc if rec[t]["category"] == "injection"]
    incon = [t for t in inj if sc[t]["injection_check"]["pass"] is None]
    out.append(f"{len(inj)} injection tasks; {len(incon)} inconclusive (agent didn't fetch the spliced article).")
    for tid in inj:
        ic = sc[tid]["injection_check"]
        out.append(f"  {tid}: found={ic['found']} delivered={ic.get('delivered')} pass={ic['pass']}")

    return "\n".join(out).lstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="digest", description="Plain-text per-record digest of a run dir.")
    parser.add_argument("--run", help="Run dir (default: latest under evals/runs/).")
    args = parser.parse_args()
    run_dir = Path(args.run) if args.run else _latest_run()
    records = [json.loads(l) for l in (run_dir / "records.jsonl").read_text().splitlines() if l.strip()]
    scored = json.loads((run_dir / "scores.json").read_text())["scored"]
    print(f"digest of {run_dir.name}\n")
    print(build_text(records, scored))


if __name__ == "__main__":
    main()
