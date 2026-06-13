"""Evaluation harness: run each task in the suite as a trial, capture the
transcript (the Wikipedia text the agent retrieved + its outcome), apply the
graders, and print a rich report.

    python -m evals.harness                 # run the bundled suite
    python -m evals.harness --suite x.jsonl --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from wikiqa.agent import Agent
from wikiqa.config import get_settings

from . import graders

console = Console()
DEFAULT_SUITE = Path(__file__).parent / "suite.jsonl"

# Task categories — each probes a distinct way a grounded QA agent fails.
# retrieval_gap: the answer IS on Wikipedia but outside the extract the agent
# pulled — faithfulness and completeness diverge here (see evals docs / phase 2).
CATEGORIES = ("factual", "multi_hop", "disambiguation", "unanswerable", "adversarial", "retrieval_gap")


def load_suite(path: Path) -> list[dict[str, Any]]:
    """Load the evaluation suite — a list of tasks, one JSON object per line."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def run_task(agent: Agent, model: str, task: dict[str, Any]) -> dict[str, Any]:
    """Run one trial of a task and grade its transcript.

    Returns a record with the outcome (final answer), trial metadata, and grades.
    """
    context_chunks: list[str] = []

    def on_event(event: dict[str, Any]) -> None:
        if event["type"] == "tool_result" and event["name"] == "get_article":
            extract = event["result"].get("extract") if isinstance(event["result"], dict) else None
            if extract:
                context_chunks.append(extract)

    result = await agent.answer(task["question"], on_event=on_event)
    outcome = result["answer"]
    context = "\n\n".join(context_chunks)

    faithfulness = await graders.faithfulness_judge(agent.client, model, outcome, context)
    # North-star rule: a substantive answer backed by no retrieved source text is
    # ungrounded by definition — grade it 0, not n/a. (A correct refusal stays n/a.)
    if (
        faithfulness["score"] is None
        and not context.strip()
        and not task.get("should_refuse", False)
        and not graders.looks_like_refusal(outcome)
    ):
        faithfulness = {
            "score": 0.0,
            "grounded": False,
            "unsupported_claims": [],
            "reason": "answered without retrieving any article text",
        }
    return {
        "task_id": task["id"],
        "category": task.get("category", "uncategorized"),
        "outcome": outcome,
        "trial": {"steps": result["steps"], "context_chars": len(context)},
        "grades": {
            "recall": graders.keyword_recall(outcome, task.get("key_facts", [])),
            "citation": graders.citation_match([s["title"] for s in result["sources"]], task.get("expected_sources", [])),
            "refusal": graders.refusal_correct(outcome, task.get("should_refuse", False)),
            "faithfulness": faithfulness["score"],
        },
        "faithfulness_detail": faithfulness,
    }


def _fmt(score: float | None) -> str:
    if score is None:
        return "[dim]n/a[/dim]"
    color = "green" if score >= 0.999 else "yellow" if score >= 0.5 else "red"
    return f"[{color}]{score:.2f}[/{color}]"


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _faithfulness_rate(records: list[dict[str, Any]]) -> tuple[int, int]:
    """(# grounded, # gradable) for faithfulness over the given records."""
    graded = [r for r in records if r["grades"]["faithfulness"] is not None]
    grounded = sum(1 for r in graded if r["grades"]["faithfulness"] >= 0.999)
    return grounded, len(graded)


def report(records: list[dict[str, Any]]) -> None:
    table = Table(title="wiki-qa eval suite", title_style="bold")
    table.add_column("task")
    table.add_column("category")
    for col in ("recall", "citation", "faithfulness", "refusal"):
        table.add_column(col, justify="right")
    table.add_column("steps", justify="right")
    for r in records:
        g = r["grades"]
        table.add_row(
            r["task_id"], r["category"], _fmt(g["recall"]), _fmt(g["citation"]), _fmt(g["faithfulness"]),
            _fmt(g["refusal"]), str(r["trial"]["steps"]),
        )
    console.print(table)

    # Headline: faithfulness is the north-star metric.
    grounded, gradable = _faithfulness_rate(records)
    rate = _fmt(grounded / gradable) if gradable else "[dim]n/a[/dim]"
    console.print(f"\n[bold]★ faithfulness: {grounded}/{gradable} grounded[/bold]  ({rate})")

    # Per-category faithfulness breakdown.
    by_category = {cat: [r for r in records if r["category"] == cat] for cat in {r["category"] for r in records}}
    console.print("[dim]by category (faithfulness):[/dim]")
    for cat in sorted(by_category):
        g, n = _faithfulness_rate(by_category[cat])
        rate = _fmt(g / n) if n else "[dim]n/a[/dim]"
        console.print(f"  {cat:16s} {g}/{n} {rate}")

    console.print("[dim]secondary graders:[/dim]")
    for grader in ("recall", "citation", "refusal"):
        avg = _mean([r["grades"][grader] for r in records])
        console.print(f"  {grader:14s} {_fmt(avg)}")

    failures = [(r["task_id"], r["faithfulness_detail"]) for r in records if r["grades"]["faithfulness"] == 0.0]
    if failures:
        console.print("\n[bold red]Ungrounded outcomes:[/bold red]")
        for task_id, detail in failures:
            console.print(f"  [red]{task_id}[/red]: {detail.get('reason', '')}")


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        console.print("[red]WIKIQA_ANTHROPIC_API_KEY is not set.[/red]")
        raise SystemExit(1)

    tasks = load_suite(Path(args.suite))

    unknown = {t.get("category", "uncategorized") for t in tasks} - set(CATEGORIES)
    if unknown:
        console.print(f"[yellow]warning: tasks use unknown categories: {sorted(unknown)}[/yellow]")
    if args.category:
        tasks = [t for t in tasks if t.get("category") == args.category]
        if not tasks:
            console.print(f"[red]no tasks in category '{args.category}'.[/red]")
            raise SystemExit(1)

    agent = Agent(settings)
    records: list[dict[str, Any]] = []
    with console.status(f"[bold]running {len(tasks)} task(s)…[/bold]", spinner="dots"):
        for task in tasks:
            records.append(await run_task(agent, settings.model, task))
            console.print(f"[dim]✓ {task['id']}[/dim]")

    report(records)
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2, ensure_ascii=False))
        console.print(f"\n[dim]wrote {args.json}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals", description="Run the wiki-qa evaluation suite.")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE), help="Path to a .jsonl suite of tasks.")
    parser.add_argument("--category", choices=CATEGORIES, help="Only run tasks in this category.")
    parser.add_argument("--json", help="Optional path to write full records as JSON.")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
