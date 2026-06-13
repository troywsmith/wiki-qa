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

Terminology follows Anthropic's
[Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents):
the **harness** runs each **task** in the **suite** as a **trial**, captures the
**transcript** (the Wikipedia text the agent retrieved + its **outcome**), and
applies **graders**.

Each task is graded along the **quality dimensions** it declares. **Faithfulness
is the north star** — is every claim in the outcome supported by the text the
agent actually retrieved? An answer given with no retrieved source text grades 0
(ungrounded), not a pass. The five dimensions:

- **faithfulness** — claims supported by *retrieved* text (model-based, vs context)
- **correctness** — outcome asserts nothing contradicting the reference (precision)
- **completeness** — outcome covers the reference's key info (recall); a refusal
  scores 0 here, which is how `retrieval_gap` surfaces
- **attribution** — expected source articles were cited (code-based)
- **calibration** — refuses iff it should (code-based)

correctness + completeness come from one **reference-based** judge call against
the task's `reference_answer`. Runs live (real Wikipedia + Claude); needs an API
key.

```bash
python -m evals.harness                       # run evals/suite.jsonl
python -m evals.harness --category adversarial # run one category
python -m evals.harness --suite my.jsonl --json records.json
```

Each task line: `{"id", "category", "question", "dimensions": [...],
"reference_answer": "...", "expected_sources": [...], "should_refuse": bool}`.
A task is only graded on the dimensions it lists; `reference_answer` is required
for `correctness`/`completeness`.

Tasks are grouped into **categories**, each probing a distinct failure mode; the
report breaks faithfulness down per category:

- `factual` — single fact, one article (baseline competence)
- `multi_hop` — answer needs 2+ articles combined (retrieval + synthesis)
- `disambiguation` — ambiguous entity/sense (retrieval precision)
- `unanswerable` — not on Wikipedia / private / future (honesty, correct refusal)
- `adversarial` — false-premise or bait questions (hallucination resistance)
- `retrieval_gap` — answer is on Wikipedia but outside the extract the agent
  pulled (e.g. not in the lead). Distinct from `unanswerable`: the info is
  *obtainable*, so a faithful refusal here is really a retrieval miss. This is
  where faithfulness and completeness diverge — the yardstick for improving the
  agent's retrieval depth.

## Deploy (Vercel)

Vercel detects the FastAPI app via the `[tool.vercel] entrypoint` in
`pyproject.toml` and routes all requests to it on Fluid Compute. Set the
`WIKIQA_ANTHROPIC_API_KEY` environment variable in the project settings
(or `vercel env add WIKIQA_ANTHROPIC_API_KEY`) before deploying.

## Configuration

All settings are environment variables prefixed `WIKIQA_` (see `.env.example`):
`ANTHROPIC_API_KEY`, `MODEL`, `MAX_AGENT_STEPS`, `MAX_TOKENS`,
`WIKIPEDIA_LANG`, `REQUEST_TIMEOUT`.

## Next steps (parked)

Deliberately deferred until the eval suite is fully built out:

- **Multi-trial metrics (pass@k / pass^k).** Run each task as multiple trials and
  report pass@k / pass^k for stability against model variance. Parked until the
  suite has enough tasks to make repeated trials worthwhile.
- **Application / prompt grounding improvements.** The agent currently over-claims
  past its thin retrieved extracts and sometimes answers without reading an
  article. Candidate fixes: require reading before answering, and retrieve fuller
  article text. Parked until the eval suite is finalized so we can measure the
  impact rather than guess — the `retrieval_gap` category is the yardstick (those
  tasks should flip from refusal to correct answers once retrieval improves).
