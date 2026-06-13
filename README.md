# wiki-qa

Wikipedia-grounded question answering. A FastAPI service wraps a tool-calling
Claude agent that searches and reads **live Wikipedia** at request time, then
answers grounded in the fetched article text with a list of sources.

## How it works

1. `POST /api/ask` with a question.
2. The agent (Claude with tool use) decides when to call `search_wikipedia`
   and `get_article`, looping until it can answer.
3. The model answers strictly from retrieved text and lists the article titles
   it relied on. Articles actually fetched are returned as structured `sources`.

No database, no indexing — retrieval is always current via the MediaWiki API.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set WIKIQA_ANTHROPIC_API_KEY
```

## CLI (fastest way to iterate)

The `wiki-qa` command calls the agent in-process (no server) and, with `-v`,
streams each Wikipedia search/read as it happens — handy for tuning prompts and
tools.

```bash
wiki-qa "Who designed the first compiler?"   # one-shot
wiki-qa -v "..."                              # show the agent's tool trace
wiki-qa                                       # interactive session (Ctrl-D to quit)
```

## Run as a server

```bash
uvicorn wikiqa.main:app --reload
```

```bash
curl -s localhost:8000/api/ask \
  -H 'content-type: application/json' \
  -d '{"question": "Who designed the first compiler?"}' | jq
```

## Test

```bash
pytest            # all tests (HTTP is mocked; no network or API key needed)
pytest tests/test_wikipedia.py::test_search_parses_hits
```

## Evals

A standalone suite that runs the agent over a dataset and scores it. The
**north-star metric is faithfulness** — is every claim in the answer supported
by the Wikipedia text the agent actually retrieved? An answer given with no
retrieved source text scores 0 (ungrounded), not a pass. Recall, citation, and
refusal are secondary checks. Faithfulness is graded by an LLM judge; the rest
are deterministic. Runs live (real Wikipedia + Claude) and needs an API key.

```bash
python -m evals.runner                       # run evals/dataset.jsonl
python -m evals.runner --dataset my.jsonl --json results.json
```

Each dataset line: `{"id", "question", "key_facts": [...],
"expected_sources": [...], "should_refuse": bool}`.

## Deploy (Vercel)

Vercel detects the FastAPI app via the `[tool.vercel] entrypoint` in
`pyproject.toml` and routes all requests to it on Fluid Compute. Set the
`WIKIQA_ANTHROPIC_API_KEY` environment variable in the project settings
(or `vercel env add WIKIQA_ANTHROPIC_API_KEY`) before deploying.

## Configuration

All settings are environment variables prefixed `WIKIQA_` (see `.env.example`):
`ANTHROPIC_API_KEY`, `MODEL`, `MAX_AGENT_STEPS`, `MAX_TOKENS`,
`WIKIPEDIA_LANG`, `REQUEST_TIMEOUT`.
