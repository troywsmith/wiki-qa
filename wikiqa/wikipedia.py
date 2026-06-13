"""Live Wikipedia retrieval client and the tool schemas exposed to the agent."""

from __future__ import annotations

from typing import Any

import httpx

USER_AGENT = "wiki-qa/0.1 (https://github.com/Snowbird-Labs/wiki-qa)"


def _api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def _page_url(lang: str, title: str) -> str:
    return f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"


class WikipediaClient:
    """Thin async wrapper over the MediaWiki action API."""

    def __init__(self, lang: str = "en", timeout: float = 15.0) -> None:
        self.lang = lang
        self.timeout = timeout

    async def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "format": "json"}
        async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": USER_AGENT}) as client:
            resp = await client.get(_api_url(self.lang), params=params)
            resp.raise_for_status()
            return resp.json()

    async def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        """Return a list of {title, snippet, url} matches for a query."""
        data = await self._get(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srprop": "snippet",
            }
        )
        results = []
        for hit in data.get("query", {}).get("search", []):
            title = hit["title"]
            snippet = _strip_html(hit.get("snippet", ""))
            results.append({"title": title, "snippet": snippet, "url": _page_url(self.lang, title)})
        return results

    async def get_article(self, title: str, max_chars: int = 12000) -> dict[str, str]:
        """Return {title, extract, url} for an article's plain-text body.

        Pulls the full plain-text extract (not just the lead) and truncates client
        side, so body-section facts are in what the agent can ground on. NOTE: the
        extracts API returns prose only — infobox tables are never included."""
        data = await self._get(
            {
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "explaintext": 1,
                "redirects": 1,
            }
        )
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                continue
            resolved = page.get("title", title)
            return {
                "title": resolved,
                "extract": page.get("extract", "").strip()[:max_chars],
                "url": _page_url(self.lang, resolved),
            }
        return {"title": title, "extract": "", "url": _page_url(self.lang, title)}


def _strip_html(text: str) -> str:
    out, depth = [], 0
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out).replace("&quot;", '"').replace("&amp;", "&").strip()


# Tool schemas advertised to Claude. Keep names in sync with agent dispatch.
TOOLS = [
    {
        "name": "search_wikipedia",
        "description": (
            "Search Wikipedia for articles matching a query. Returns candidate article "
            "titles with short snippets. Use this to discover which articles to read."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords."},
                "limit": {"type": "integer", "description": "Max results (1-10).", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_article",
        "description": (
            "Fetch the plain-text content of a Wikipedia article by its exact title "
            "(as returned by search_wikipedia). Use this to read source material before answering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Exact article title."},
            },
            "required": ["title"],
        },
    },
]
