"""Terminal client for fast iteration: calls the agent in-process and streams
each Wikipedia tool call live, then renders the grounded answer with sources."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from .agent import Agent
from .config import get_settings

console = Console()


def _render_event(event: dict[str, Any]) -> None:
    if event["type"] == "tool_call":
        name, inp = event["name"], event["input"]
        if name == "search_wikipedia":
            console.print(f"[dim]→[/dim] [cyan]search[/cyan] {inp.get('query')!r}")
        elif name == "get_article":
            console.print(f"[dim]→[/dim] [cyan]read[/cyan] {inp.get('title')!r}")
        else:
            console.print(f"[dim]→[/dim] [cyan]{name}[/cyan] {inp}")
    elif event["type"] == "tool_result":
        result = event["result"]
        if event["name"] == "search_wikipedia" and isinstance(result, list):
            for hit in result:
                console.print(f"   [dim]·[/dim] {hit['title']}")
        elif event["name"] == "get_article" and isinstance(result, dict):
            chars = len(result.get("extract", ""))
            console.print(f"   [dim]· {chars} chars from[/dim] {result.get('title')}")


async def _ask(agent: Agent, question: str, verbose: bool) -> None:
    on_event = _render_event if verbose else None
    if verbose:
        console.print(Rule("[dim]agent trace[/dim]", style="dim"))
        result = await agent.answer(question, on_event=on_event)
        console.print(Rule(style="dim"))
    else:
        with console.status("[bold]thinking…[/bold]", spinner="dots"):
            result = await agent.answer(question)

    console.print(Panel(Markdown(result["answer"]), title="answer", border_style="green"))
    if result["sources"]:
        console.print("[bold]sources[/bold]")
        for s in result["sources"]:
            console.print(f"  [dim]·[/dim] {s['title']} — [blue underline]{s['url']}[/blue underline]")
    console.print(f"[dim]({result['steps']} step(s))[/dim]")


async def _repl(agent: Agent, verbose: bool) -> None:
    console.print("[bold]wiki-qa[/bold] — ask a question, or [dim]Ctrl-D / 'exit' to quit[/dim].")
    while True:
        try:
            question = console.input("[bold green]?[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        await _ask(agent, question, verbose)


def main() -> None:
    parser = argparse.ArgumentParser(prog="wiki-qa", description="Ask Wikipedia-grounded questions.")
    parser.add_argument("question", nargs="*", help="Question to ask. Omit for an interactive session.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Stream the agent's tool calls.")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.anthropic_api_key:
        console.print("[red]WIKIQA_ANTHROPIC_API_KEY is not set.[/red] Add it to .env or your environment.")
        raise SystemExit(1)

    agent = Agent(settings)
    if args.question:
        asyncio.run(_ask(agent, " ".join(args.question), args.verbose))
    else:
        asyncio.run(_repl(agent, args.verbose))


if __name__ == "__main__":
    main()
