"""Run the eval suite: drive the agent over each case, capture the Wikipedia
text it retrieved, score the result, and print a rich report.

    python -m evals.runner                # run the bundled dataset
    python -m evals.runner --dataset x.jsonl --json out.json
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

from . import scorers

console = Console()
DEFAULT_DATASET = Path(__file__).parent / "dataset.jsonl"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def run_case(agent: Agent, model: str, case: dict[str, Any]) -> dict[str, Any]:
    """Run one case and return the agent output, retrieved context, and scores."""
    context_chunks: list[str] = []

    def on_event(event: dict[str, Any]) -> None:
        if event["type"] == "tool_result" and event["name"] == "get_article":
            extract = event["result"].get("extract") if isinstance(event["result"], dict) else None
            if extract:
                context_chunks.append(extract)

    result = await agent.answer(case["question"], on_event=on_event)
    answer = result["answer"]
    context = "\n\n".join(context_chunks)

    faithfulness = await scorers.faithfulness_judge(agent.client, model, answer, context)
    # North-star rule: a substantive answer backed by no retrieved source text is
    # ungrounded by definition — score it 0, not n/a. (A correct refusal stays n/a.)
    if (
        faithfulness["score"] is None
        and not context.strip()
        and not case.get("should_refuse", False)
        and not scorers.looks_like_refusal(answer)
    ):
        faithfulness = {
            "score": 0.0,
            "grounded": False,
            "unsupported_claims": [],
            "reason": "answered without retrieving any article text",
        }
    return {
        "id": case["id"],
        "answer": answer,
        "steps": result["steps"],
        "scores": {
            "recall": scorers.keyword_recall(answer, case.get("key_facts", [])),
            "citation": scorers.citation_match([s["title"] for s in result["sources"]], case.get("expected_sources", [])),
            "refusal": scorers.refusal_correct(answer, case.get("should_refuse", False)),
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


def report(results: list[dict[str, Any]]) -> None:
    table = Table(title="wiki-qa eval", title_style="bold")
    table.add_column("case")
    for col in ("recall", "citation", "faithfulness", "refusal"):
        table.add_column(col, justify="right")
    table.add_column("steps", justify="right")
    for r in results:
        s = r["scores"]
        table.add_row(
            r["id"], _fmt(s["recall"]), _fmt(s["citation"]), _fmt(s["faithfulness"]), _fmt(s["refusal"]), str(r["steps"])
        )
    console.print(table)

    # Headline: faithfulness is the north-star metric.
    graded = [r for r in results if r["scores"]["faithfulness"] is not None]
    grounded = sum(1 for r in graded if r["scores"]["faithfulness"] >= 0.999)
    rate = _fmt(grounded / len(graded)) if graded else "[dim]n/a[/dim]"
    console.print(f"\n[bold]★ faithfulness: {grounded}/{len(graded)} grounded[/bold]  ({rate})")

    console.print("[dim]secondary metrics:[/dim]")
    for metric in ("recall", "citation", "refusal"):
        avg = _mean([r["scores"][metric] for r in results])
        console.print(f"  {metric:14s} {_fmt(avg)}")

    failures = [
        (r["id"], r["faithfulness_detail"])
        for r in results
        if r["scores"]["faithfulness"] == 0.0
    ]
    if failures:
        console.print("\n[bold red]Ungrounded answers:[/bold red]")
        for cid, detail in failures:
            console.print(f"  [red]{cid}[/red]: {detail.get('reason', '')}")


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        console.print("[red]WIKIQA_ANTHROPIC_API_KEY is not set.[/red]")
        raise SystemExit(1)

    cases = load_cases(Path(args.dataset))
    agent = Agent(settings)
    results: list[dict[str, Any]] = []
    with console.status(f"[bold]running {len(cases)} case(s)…[/bold]", spinner="dots"):
        for case in cases:
            results.append(await run_case(agent, settings.model, case))
            console.print(f"[dim]✓ {case['id']}[/dim]")

    report(results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        console.print(f"\n[dim]wrote {args.json}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals", description="Run the wiki-qa eval suite.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to a .jsonl dataset.")
    parser.add_argument("--json", help="Optional path to write full results as JSON.")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
