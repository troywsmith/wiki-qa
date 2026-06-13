# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Wikipedia-grounded QA service: a FastAPI app fronting a tool-calling Claude
agent. The agent answers questions strictly from **live** Wikipedia content it
retrieves at request time — there is no vector store, index, or local corpus.

## Commands

All commands assume the venv is active (`source .venv/bin/activate`).

```bash
pip install -e ".[dev]"                              # install runtime + dev deps
uvicorn wikiqa.main:app --reload                     # run dev server (localhost:8000)
wiki-qa -v "a question"                              # in-process CLI, streams tool calls
pytest                                               # run all tests
pytest tests/test_wikipedia.py::test_search_parses_hits  # run a single test
python -m evals.harness                              # run the eval suite (live; needs API key)
```

Tests mock all HTTP with `respx` — they never hit the network or need an API key.
The eval harness, by contrast, runs live against real Wikipedia + Claude.

## Architecture

Request flow: `main.py` (HTTP) → `agent.py` (the tool loop) → `wikipedia.py`
(retrieval). `config.py` supplies settings to all of them.

- **`wikiqa/main.py`** — FastAPI app and the `POST /api/ask` + `GET /api/health`
  routes. Request/response Pydantic models live here. This is also the Vercel
  entrypoint (see Deployment).
- **`wikiqa/agent.py`** — the agent loop. Calls `client.messages.create` with the
  Wikipedia `TOOLS`, and while `stop_reason == "tool_use"` it dispatches each
  tool call back to the `WikipediaClient`, appends `tool_result` blocks, and
  re-calls the model. Loops until the model stops requesting tools or
  `max_agent_steps` is hit. The system prompt is sent with `cache_control`
  (prompt caching) since it's constant across the loop.
- **`wikiqa/wikipedia.py`** — `WikipediaClient` (async MediaWiki action API
  wrapper) plus the `TOOLS` schema list advertised to Claude. **Tool names in
  `TOOLS` must stay in sync with the `_dispatch_tool` dispatch in `agent.py`** —
  they're matched by string.
- **`wikiqa/config.py`** — `Settings` (pydantic-settings). All env vars are
  prefixed `WIKIQA_`. `get_settings()` is `lru_cache`d.
- **`wikiqa/cli.py`** — terminal client (`wiki-qa` entry point). Calls `Agent`
  in-process and uses the optional `on_event` callback on `Agent.answer` to
  stream tool calls. The agent stays decoupled — no `rich` import in `agent.py`.

### Evals

The `evals/` package follows Anthropic's eval nomenclature
(https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents):
- **`evals/suite.jsonl`** — the dev **suite**: one **task** (JSON object) per
  line, ~6 per category. Each task has a `category` (one of `CATEGORIES` in
  `harness.py`: factual, multi_hop, disambiguation, unanswerable, adversarial,
  retrieval_gap); the report breaks faithfulness down per category and
  `--category` filters to one. `retrieval_gap` (answer on Wikipedia but outside
  the agent's extract) is where faithfulness and completeness diverge.
- **`evals/holdout.jsonl`** — held-out slice (~2 per category). **Never loaded by
  default**; run only via `--holdout`, once, at the very end. Do not look at
  holdout results while iterating on the agent/prompt (see the eval methodology:
  build suite out → freeze → hillclimb one change at a time, logging each in a
  per-dimension README table; keep prompt changes separate from grader changes).
- **`evals/graders.py`** — **code-based graders** (citation→attribution,
  refusal→calibration) and **model-based graders** (faithfulness vs retrieved
  context; reference_judge vs `reference_answer` → correctness + completeness).
- **`evals/harness.py`** — runs each task as a **trial**, captures the
  **transcript**/**outcome** via the agent's `on_event` hook, grades, reports.

**Quality dimensions** (`DIMENSIONS` in `harness.py`): faithfulness (north star),
completeness, correctness, attribution, calibration. Each task declares which
apply via its `dimensions` field; only those are graded. correctness/completeness
are a precision/recall split vs the reference (`reference_answer` required for
them). Faithfulness is graded strictly against retrieved text — an outcome with
no retrieved source text grades 0. Keep the shared vocabulary
(task/trial/grader/transcript/outcome/suite/harness/dimension) when extending.

### Grounding contract

Two layers enforce "answer only from Wikipedia": the system prompt instructs the
model to cite and refuse when sources lack the answer, and `agent.py`
independently tracks which articles were actually fetched (non-empty extract) to
return as structured `sources` — so citations reflect real retrievals, not just
what the model claims.

## Conventions

- Everything on the request path is **async** (FastAPI, `AsyncAnthropic`,
  `httpx.AsyncClient`). Keep new I/O async.
- `requires-python` is `>=3.11` so the project installs on the local toolchain;
  Vercel runs it on its default 3.12+. Don't use 3.12-only syntax.
- When changing the model's capabilities, update the tool schema and its
  dispatcher together, and adjust the system prompt if the workflow changes.

## Deployment (Vercel)

Vercel detects the FastAPI app from `[tool.vercel] entrypoint = "wikiqa.main:app"`
in `pyproject.toml` and routes all traffic to it on Fluid Compute — there is no
`vercel.json` and no `api/` directory. Dependencies are installed from
`pyproject.toml`. The only required env var is `WIKIQA_ANTHROPIC_API_KEY`.
