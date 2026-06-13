"""Evaluation harness with a strict record/score split, one directory per run.

    python -m evals.harness run --note "..." --kind prompt   # new run dir: meta + records
    python -m evals.harness score [--run DIR]                 # score a run dir (default: latest)
    python -m evals.harness all                               # run then score (default)

Each run writes evals/runs/<timestamp>_<commit>/ containing:
    meta.json      run id, commit (+dirty), suite file + hash, pinned protocol, note, kind
    records.jsonl  immutable — one record per task per trial (output, tool trace,
                   retrieved text, spliced injection)
    scores.json    the graded view derived from records.jsonl

`run` is the only phase that touches the agent. `score` reads a run dir's
records.jsonl and writes scores.json beside it without touching records, so a
grader fix re-scores without re-running the agent. Records carry a trial index so
multi-trial (pass@k) can aggregate later without changing the shape.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from rich.console import Console
from rich.table import Table

from wikiqa.agent import Agent
from wikiqa.config import get_settings
from wikiqa.wikipedia import WikipediaClient

from . import digest, graders

console = Console()
HERE = Path(__file__).parent
DEFAULT_SUITE = HERE / "suite.jsonl"
HOLDOUT_SUITE = HERE / "holdout.jsonl"
RUNS_DIR = HERE / "runs"

CATEGORIES = ("factual", "multi_hop", "disambiguation", "unanswerable", "adversarial", "retrieval_gap", "injection")
DIMENSIONS = ("faithfulness", "completeness", "correctness", "attribution", "calibration")
PINNED_TEMPERATURE = 0  # protocol value recorded in meta; not yet enforced on the API calls

# The no-retrieval baseline prompt: the real agent system prompt with the tool /
# retrieval / citation clauses stripped, the role + task framing + answer-or-refusal
# shape kept, and one line telling it to answer from its own knowledge. Frozen as a
# fixed string (NOT derived from agent.SYSTEM_PROMPT) so it can't drift when the
# agent's prompt is tuned during the climb. Run once; editing it is a re-freeze.
BASELINE_SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer from your own knowledge. "
    "If you do not know the answer, say so plainly rather than guessing. "
    "Keep answers concise."
)


def _git_info() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE.parent, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=HERE.parent, capture_output=True, text=True).stdout.strip())
        return sha or "nogit", dirty
    except Exception:
        return "nogit", False


def _suite_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _latest_run() -> Path:
    runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()) if RUNS_DIR.exists() else []
    if not runs:
        raise SystemExit("no runs found under evals/runs/ — run `python -m evals.harness run` first.")
    return runs[-1]


def load_suite(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- run phase: agent -> immutable records ---


class _SplicingWiki(WikipediaClient):
    """Appends an injection payload to every fetched extract, so the agent sees an
    embedded instruction in its retrieved text (injection tasks only)."""

    def __init__(self, payload: str, **kw: Any) -> None:
        super().__init__(**kw)
        self._payload = payload

    async def get_article(self, title: str, max_chars: int = 4000) -> dict[str, str]:
        art = await super().get_article(title, max_chars)
        if art.get("extract"):
            art["extract"] = f"{art['extract']}\n\n{self._payload}"
        return art


def _make_agent(settings: Any, task: dict[str, Any], baseline: bool = False) -> Agent:
    if baseline:  # no-retrieval floor: stripped prompt, no tools, answers from memory
        return Agent(settings, system_prompt=BASELINE_SYSTEM_PROMPT, tools=[])
    if task.get("category") == "injection" and task.get("injection"):
        wiki = _SplicingWiki(task["injection"], lang=settings.wikipedia_lang, timeout=settings.request_timeout)
        return Agent(settings, wiki=wiki)
    return Agent(settings)


async def run_task(settings: Any, task: dict[str, Any], baseline: bool = False) -> dict[str, Any]:
    """Run the agent once and return an immutable record of what happened."""
    chunks: list[str] = []
    trace: list[dict[str, Any]] = []

    def on_event(event: dict[str, Any]) -> None:
        if event["type"] == "tool_call":
            trace.append({"call": event["name"], "input": event["input"]})
        elif event["type"] == "tool_result":
            r = event["result"]
            if event["name"] == "get_article" and isinstance(r, dict) and r.get("extract"):
                chunks.append(r["extract"])
                trace.append({"result": "get_article", "title": r.get("title"), "chars": len(r["extract"])})
            elif event["name"] == "search_wikipedia" and isinstance(r, list):
                trace.append({"result": "search_wikipedia", "hits": [h.get("title") for h in r]})

    result = await _make_agent(settings, task, baseline).answer(task["question"], on_event=on_event)
    return {
        "task_id": task["id"],
        "trial": 0,  # single trial for now; pass@k will add trials without changing this shape
        "category": task.get("category", "uncategorized"),
        "question": task["question"],
        "dimensions": task.get("dimensions", []),
        "should_refuse": bool(task.get("should_refuse", False)),
        "reference_answer": task.get("reference_answer"),
        "expected_sources": task.get("expected_sources", []),
        "injection": task.get("injection"),
        "forbidden": task.get("forbidden"),
        "output": {"answer": result["answer"], "sources": [s["title"] for s in result["sources"]]},
        "retrieved_text": "\n\n".join(chunks),
        "tool_trace": trace,
        "answerer_model": settings.model,
        "steps": result["steps"],
    }


async def run_phase(settings: Any, tasks: list[dict[str, Any]], suite_path: Path, args: argparse.Namespace) -> Path:
    """Create a run directory (keyed by timestamp + commit), write meta.json and an
    immutable records.jsonl (one record per task per trial). Returns the run dir."""
    sha, dirty = _git_info()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{sha}{'-dirty' if dirty else ''}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "run_id": run_id,
        "timestamp": stamp,
        "git_commit": sha,
        "git_dirty": dirty,
        "suite_file": suite_path.name,
        "suite_sha": _suite_sha(suite_path),
        "protocol": {
            "answerer": settings.model,
            "judge": settings.judge_model,
            "temperature": PINNED_TEMPERATURE,
            "trials": 1,
        },
        "kind": "baseline" if args.baseline else args.kind,  # for the hillclimb table
        "note": args.note,   # one-line description of what changed
    }
    if args.baseline:
        # Freeze the baseline prompt verbatim alongside the suite. Run once; never re-run.
        meta["baseline"] = {"retrieval": "disabled", "tools": [], "system_prompt": BASELINE_SYSTEM_PROMPT}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    records: list[dict[str, Any]] = []
    with console.status(f"[bold]running {len(tasks)} task(s)…[/bold]", spinner="dots"):
        for task in tasks:
            records.append(await run_task(settings, task, baseline=args.baseline))
            console.print(f"[dim]✓ ran {task['id']}[/dim]")
    (run_dir / "records.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    console.print(f"[dim]wrote {len(records)} records → {run_dir}[/dim]")
    return run_dir


# --- score phase: records -> graded view ---


async def score_record(client: AsyncAnthropic, judge_model: str, record: dict[str, Any]) -> dict[str, Any]:
    declared = set(record.get("dimensions", []))
    grades: dict[str, Any] = {}

    if "faithfulness" in declared:
        grades["faithfulness"] = await graders.faithfulness(client, judge_model, record)
    if declared & {"correctness", "completeness"}:
        ref = await graders.reference(client, judge_model, record)
        if "correctness" in declared:
            grades["correctness"] = ref["correctness"]
        if "completeness" in declared:
            grades["completeness"] = ref["completeness"]
    if "attribution" in declared:
        grades["attribution"] = graders.attribution(record)
    if "calibration" in declared:
        grades["calibration"] = graders.calibration(record)

    inj = graders.injection_check(record)
    # Task passes only if every applicable declared dimension passes AND the
    # injection check did not catch an obey (an inconclusive/N/A injection check
    # does not block — it just couldn't run).
    applicable = [g for g in grades.values() if g.get("applicable")]
    task_pass = all(g["pass"] for g in applicable) and inj["pass"] is not False
    return {
        "task_id": record["task_id"],
        "trial": record.get("trial", 0),
        "category": record["category"],
        "grades": grades,
        "injection_check": inj,
        "task_pass": task_pass,
    }


def _task_pass(grades: dict[str, Any], inj: dict[str, Any]) -> bool:
    applicable = [g for g in grades.values() if g.get("applicable")]
    return all(g["pass"] for g in applicable) and inj["pass"] is not False


def _regrade_code(prev: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Recompute only the CODE graders (attribution, calibration, injection) from
    the record, reusing the cached judge verdicts. Used to re-score after a
    code-grader fix without re-running the judges — so only the code dimensions can
    move, and there's no API cost or judge variance."""
    declared = set(record.get("dimensions", []))
    grades = dict(prev["grades"])
    if "attribution" in declared:
        grades["attribution"] = graders.attribution(record)
    if "calibration" in declared:
        grades["calibration"] = graders.calibration(record)
    inj = graders.injection_check(record)
    return {
        "task_id": record["task_id"],
        "trial": record.get("trial", 0),
        "category": record["category"],
        "grades": grades,
        "injection_check": inj,
        "task_pass": _task_pass(grades, inj),
    }


async def score_run(settings: Any, run_dir: Path, regrade_code: bool = False) -> list[dict[str, Any]]:
    """Read a run dir's immutable records.jsonl, grade them, write scores.json next
    to it (records are never touched), and return the scored view.

    regrade_code: reuse the existing scores.json judge verdicts and only recompute
    the code graders — no judge calls (use after a code-grader fix)."""
    records_path = run_dir / "records.jsonl"
    if not records_path.exists():
        raise SystemExit(f"no records.jsonl in {run_dir}")
    records = [json.loads(l) for l in records_path.read_text().splitlines() if l.strip()]

    if regrade_code:
        prev = {(s["task_id"], s.get("trial", 0)): s for s in json.loads((run_dir / "scores.json").read_text())["scored"]}
        scored = [_regrade_code(prev[(r["task_id"], r.get("trial", 0))], r) for r in records]
        console.print(f"[dim]regraded code graders for {len(scored)} record(s) — judge verdicts reused, no API[/dim]")
        (run_dir / "scores.json").write_text(json.dumps({"run_id": run_dir.name, "scored": scored}, indent=2, ensure_ascii=False))
        (run_dir / "digest.txt").write_text(digest.build_text(records, scored) + "\n")
        return scored

    if settings.judge_model in {r.get("answerer_model") for r in records}:
        console.print(f"[red]warning: judge model {settings.judge_model} equals the answerer model — self-preference risk.[/red]")
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    scored: list[dict[str, Any]] = []
    with console.status(f"[bold]scoring {len(records)} record(s)…[/bold]", spinner="dots"):
        for record in records:
            scored.append(await score_record(client, settings.judge_model, record))

    (run_dir / "scores.json").write_text(json.dumps({"run_id": run_dir.name, "scored": scored}, indent=2, ensure_ascii=False))
    # Write-also: a plain-text digest derived from records (like scores), for skimming.
    (run_dir / "digest.txt").write_text(digest.build_text(records, scored) + "\n")
    console.print(f"[dim]wrote scores.json + digest.txt → {run_dir}[/dim]")
    return scored


# --- reporting ---


def _rate(scored: list[dict[str, Any]], dim: str) -> tuple[int, int]:
    applicable = [s for s in scored if s["grades"].get(dim, {}).get("applicable")]
    passed = sum(1 for s in applicable if s["grades"][dim]["pass"])
    return passed, len(applicable)


def _cell(passed: int, n: int) -> str:
    if n == 0:
        return "[dim]·[/dim]"
    color = "green" if passed == n else "yellow" if passed >= n / 2 else "red"
    return f"[{color}]{passed}/{n}[/{color}]"


def report(scored: list[dict[str, Any]]) -> None:
    cats = sorted({s["category"] for s in scored}, key=lambda c: CATEGORIES.index(c) if c in CATEGORIES else 99)

    table = Table(title="wiki-qa eval — per-category / per-dimension pass rates", title_style="bold")
    table.add_column("category")
    for dim in DIMENSIONS:
        table.add_column(dim[:6], justify="right")
    table.add_column("tasks", justify="right")
    for cat in cats:
        rows = [s for s in scored if s["category"] == cat]
        passed_tasks = sum(1 for s in rows if s["task_pass"])
        table.add_row(cat, *[_cell(*_rate(rows, d)) for d in DIMENSIONS], f"{passed_tasks}/{len(rows)}")
    console.print(table)

    fp, fn = _rate(scored, "faithfulness")
    console.print(f"\n[bold]★ faithfulness: {_cell(fp, fn)}[/bold] (north star)")

    inj = [s for s in scored if s["category"] == "injection"]
    if inj:
        resisted = sum(1 for s in inj if s["injection_check"]["pass"] is True)
        obeyed = sum(1 for s in inj if s["injection_check"]["found"])
        incon = sum(1 for s in inj if s["injection_check"]["pass"] is None)
        console.print(f"injection: {resisted} resisted, {obeyed} obeyed, {incon} inconclusive (not delivered) of {len(inj)}")

    failures = [s for s in scored if not s["task_pass"]]
    if failures:
        console.print(f"\n[bold red]{len(failures)} task(s) failed:[/bold red]")
        for s in failures:
            bad = [d for d, g in s["grades"].items() if g.get("applicable") and not g["pass"]]
            if s["injection_check"]["applicable"] and not s["injection_check"]["pass"]:
                bad.append("injection")
            console.print(f"  [red]{s['task_id']}[/red] ({s['category']}): {', '.join(bad) or 'see diagnostics'}")


# --- CLI ---


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        console.print("[red]WIKIQA_ANTHROPIC_API_KEY is not set.[/red]")
        raise SystemExit(1)

    run_dir: Path
    if args.mode in ("run", "all"):
        suite_path = HOLDOUT_SUITE if args.holdout else Path(args.suite)
        if args.holdout:
            console.print("[yellow]running the HELD-OUT slice — only at the very end.[/yellow]")
        tasks = load_suite(suite_path)
        unknown = {t.get("category") for t in tasks} - set(CATEGORIES)
        if unknown:
            console.print(f"[yellow]warning: unknown categories {sorted(unknown)}[/yellow]")
        if args.category:
            tasks = [t for t in tasks if t.get("category") == args.category]
        run_dir = await run_phase(settings, tasks, suite_path, args)
    else:  # score
        run_dir = Path(args.run) if args.run else _latest_run()
        console.print(f"[dim]scoring run ← {run_dir}[/dim]")

    if args.mode in ("score", "all"):
        scored = await score_run(settings, run_dir, regrade_code=args.regrade_code)
        report(scored)


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals", description="Run/score the wiki-qa evaluation suite.")
    parser.add_argument("mode", nargs="?", choices=["run", "score", "all"], default="all", help="run, score, or both (default).")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE), help="Dev suite path (run).")
    parser.add_argument("--holdout", action="store_true", help="Use the held-out slice (only at the very end).")
    parser.add_argument("--baseline", action="store_true", help="No-retrieval floor: stripped prompt, no tools. Run once.")
    parser.add_argument("--category", choices=CATEGORIES, help="Only run tasks in this category.")
    parser.add_argument("--run", help="Run dir to score (default: latest under evals/runs/).")
    parser.add_argument("--regrade-code", action="store_true", help="Re-score reusing cached judge verdicts; only recompute code graders (no API).")
    parser.add_argument("--note", default="", help="One-line description of what changed (recorded in meta.json).")
    parser.add_argument("--kind", default="baseline", choices=["baseline", "prompt", "retrieval", "grader"], help="Change kind for the hillclimb table.")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
