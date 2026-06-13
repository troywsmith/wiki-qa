"""Tool-calling agent: Claude drives Wikipedia searches until it can answer, grounded."""

from __future__ import annotations

from typing import Any, Callable

from anthropic import AsyncAnthropic

from .config import Settings
from .wikipedia import TOOLS, WikipediaClient

SYSTEM_PROMPT = (
    "You are a question-answering assistant that answers strictly from Wikipedia. "
    "Workflow: use search_wikipedia to find relevant articles, then get_article to read them, "
    "then answer. Ground every claim in the fetched article text. If the sources do not contain "
    "the answer, say so plainly rather than guessing. Keep answers concise and end with a "
    "'Sources:' list of the article titles you relied on."
)


class Agent:
    def __init__(
        self,
        settings: Settings,
        wiki: WikipediaClient | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        # `wiki` is injectable so the eval harness can splice injection payloads
        # into retrieved text without changing agent behavior. `system_prompt` and
        # `tools` are overridable so the harness can run the no-retrieval baseline
        # (stripped prompt, no tools); both default to the real agent config.
        self.wiki = wiki or WikipediaClient(lang=settings.wikipedia_lang, timeout=settings.request_timeout)
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self.tools = tools if tools is not None else TOOLS

    async def _dispatch_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == "search_wikipedia":
            return await self.wiki.search(args["query"], min(int(args.get("limit", 5)), 10))
        if name == "get_article":
            return await self.wiki.get_article(args["title"])
        return {"error": f"unknown tool: {name}"}

    async def answer(
        self, question: str, on_event: Callable[[dict[str, Any]], None] | None = None
    ) -> dict[str, Any]:
        """Run the tool loop and return {answer, sources, steps}.

        on_event, if given, is called with {"type": "tool_call"|"tool_result", ...}
        as the agent works — used by the CLI to stream progress.
        """
        emit = on_event or (lambda _event: None)
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        sources: list[dict[str, str]] = []

        for step in range(self.settings.max_agent_steps):
            create_kwargs: dict[str, Any] = {
                "model": self.settings.model,
                "max_tokens": self.settings.max_tokens,
                "system": [{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
                "messages": messages,
            }
            if self.tools:  # omit entirely for the no-retrieval baseline
                create_kwargs["tools"] = self.tools
            resp = await self.client.messages.create(**create_kwargs)

            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                return {"answer": text.strip(), "sources": _dedupe(sources), "steps": step + 1}

            messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                emit({"type": "tool_call", "name": block.name, "input": block.input})
                result = await self._dispatch_tool(block.name, block.input)
                emit({"type": "tool_result", "name": block.name, "result": result})
                if block.name == "get_article" and isinstance(result, dict) and result.get("extract"):
                    sources.append({"title": result["title"], "url": result["url"]})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _stringify(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return {
            "answer": "I couldn't reach a grounded answer within the step budget.",
            "sources": _dedupe(sources),
            "steps": self.settings.max_agent_steps,
        }


def _stringify(result: Any) -> str:
    import json

    return json.dumps(result, ensure_ascii=False)


def _dedupe(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    seen, out = set(), []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            out.append(s)
    return out
