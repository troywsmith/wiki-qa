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

## Scope

The contract for what wiki-qa does. **A task is only added to the eval suite if
it maps to an in-scope requirement and its expected behavior; otherwise flag it
rather than adding it.**

**In scope**
- Answer factual questions grounded in live Wikipedia text.
- Retrieve relevant article(s) via search, then fetch; combine across articles when needed.
- Resolve ambiguous entities to the intended sense, or surface the ambiguity.
- Abstain honestly when Wikipedia doesn't support an answer.
- Reject false premises instead of confabulating.
- Cite the articles actually used.
- Treat retrieved text as data, not instructions.

**Out of scope** (with reason)
- Non-Wikipedia / open-web knowledge — Wikipedia is the single source of truth; out-of-source facts are refused, not answered.
- Toxicity / bias / PII / jailbreak moderation — near-zero surface for public-encyclopedia QA; the only live safety axis is prompt injection via retrieved content, which is in scope.
- Opinion / advice / subjective questions.
- Math or inference beyond the source.
- Multi-turn dialogue — the unit is one question to one grounded answer.
- Recency reasoning beyond what the source says.

**Expected behavior per category**
- `factual` — retrieve the article, answer from it, cite it.
- `multi_hop` — use 2+ articles, synthesize only from retrieved text, cite all.
- `disambiguation` — answer the intended sense or surface the ambiguity; never answer the wrong sense.
- `unanswerable` — refuse honestly, no guessing.
- `adversarial` — reject the false premise; don't produce a fluent wrong answer.
- `retrieval_gap` — retrieve deeper or abstain; never answer from memory. A refusal here is a retrieval miss, not honesty — where faithfulness and completeness diverge.

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
python -m evals.harness                       # run the dev suite (evals/suite.jsonl)
python -m evals.harness --category adversarial # run one category
python -m evals.harness --json records.json    # write full records as JSON
python -m evals.harness --holdout              # run the held-out slice (ONLY at the very end)
```

The suite is the spec: the dev set (`evals/suite.jsonl`) has ~6 tasks per
category so every failure mode has real coverage. A held-out slice
(`evals/holdout.jsonl`, ~2 per category) is **never loaded by default** — it is
run once, at the end, via `--holdout`, to check we didn't overfit the dev set
while iterating.

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

## Design decisions

The choices that govern the agent and eval, each with its one-line tradeoff.
These are the locked decisions we build and hillclimb toward; where current code
diverges, closing the gap is hillclimbing work, not a doc change.

- **Always-retrieve on the factual path** (no model-decides-when-to-search) — guarantees a grounding attempt; pays for a search even when the model "knows" it.
- **Two-step search-then-fetch** — keeps retrieval legible and cheap; an extra round-trip vs one-shot.
- **Hard tool-call budget with graceful abstention on exhaustion** — bounds cost and latency; may abstain on genuinely hard multi-hop.
- **Citation-forced grounding** (verbatim contiguous span, fetched articles only) — makes faithfulness checkable; rejects valid paraphrase-only answers.
- **Intro-only extracts, escalation gated on eval evidence** — cheap and usually enough; misses buried facts (the `retrieval_gap` bet) until evidence justifies expanding.
- **Model choice** — a mid-tier answerer so grounding failures stay visible, and a stronger, different judge to decorrelate self-preference.

### How to read the output

- **Dimensions**: faithfulness (claims supported by retrieved text), correctness
  (nothing contradicting the reference), completeness (covers the reference),
  attribution (right sources cited), calibration (refuses iff it should).
- **Per subset, not blended**: scores are reported per category and per
  dimension, never as one number — a blended score hides which failure mode moved.
- **Good vs failure**: a good result is high faithfulness with correctness and
  completeness both high on answerable tasks. A failure is high correctness but
  low faithfulness (right answer, ungrounded), or a `retrieval_gap` task scoring
  completeness 0 (abstained on an obtainable answer).

## Caveats / limitations

What the numbers don't tell you:
- **Single-trial variance** — even at temperature 0, runs are not bit-for-bit
  deterministic; one-trial scores carry variance (multi-trial pass@k is parked).
- **Grader blind spots** — the code-based graders are approximate (e.g.
  calibration uses keyword refusal-detection; attribution is title substring
  matching) and can mis-grade edge cases.
- **Small held-out slice** — ~2 tasks per category; a coarse overfitting check,
  not a precise generalization estimate.
- **No change→result table yet** — the per-change iteration log arrives once we
  freeze the suite and start hillclimbing.

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
