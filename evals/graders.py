"""Graders for the evaluation suite.

Three code-based graders (recall, citation, refusal) plus one model-based grader
(faithfulness, LLM-as-judge against a rubric). Each returns a float in [0, 1], or
None when the grader does not apply to a given task.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import AsyncAnthropic

_REFUSAL_MARKERS = (
    "could not find",
    "couldn't find",
    "do not contain",
    "does not contain",
    "doesn't contain",
    "not contain the answer",
    "no information",
    "unable to",
    "couldn't reach",
    "i don't have",
    "cannot answer",
    "can't answer",
)


# --- Code-based graders (deterministic) ---


def keyword_recall(answer: str, key_facts: list[str]) -> float | None:
    """Fraction of required facts that appear (case-insensitive) in the answer."""
    if not key_facts:
        return None
    text = answer.lower()
    hits = sum(1 for fact in key_facts if fact.lower() in text)
    return hits / len(key_facts)


def citation_match(source_titles: list[str], expected: list[str]) -> float | None:
    """Fraction of expected source articles the agent actually cited."""
    if not expected:
        return None
    got = [t.lower() for t in source_titles]
    hits = sum(1 for e in expected if any(e.lower() in t or t in e.lower() for t in got))
    return hits / len(expected)


def looks_like_refusal(answer: str) -> bool:
    return any(marker in answer.lower() for marker in _REFUSAL_MARKERS)


def refusal_correct(answer: str, should_refuse: bool) -> float:
    """1.0 if the answer's refuse/answer behaviour matches expectation, else 0.0."""
    return 1.0 if looks_like_refusal(answer) == should_refuse else 0.0


# --- Model-based grader (LLM-as-judge) ---

# Rubric for the faithfulness grader.
_RUBRIC = """You grade whether an ANSWER is fully supported by SOURCE TEXT \
retrieved from Wikipedia. Do not use outside knowledge — judge only against the source.

SOURCE TEXT:
{context}

ANSWER:
{answer}

Is every factual claim in the ANSWER supported by the SOURCE TEXT? Reply with ONLY a \
JSON object: {{"grounded": true|false, "unsupported_claims": ["..."], "reason": "..."}}"""


async def faithfulness_judge(
    client: AsyncAnthropic, model: str, answer: str, context: str
) -> dict[str, Any]:
    """LLM-as-judge: is the answer grounded in the retrieved context? Returns
    {"score": 0|1, "grounded": bool, "unsupported_claims": [...], "reason": str}."""
    if not context.strip():
        # Nothing was retrieved — grounding is vacuous; treat as not-applicable.
        return {"score": None, "grounded": None, "unsupported_claims": [], "reason": "no context retrieved"}
    resp = await client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": _RUBRIC.format(context=context[:12000], answer=answer)}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    try:
        verdict = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return {"score": None, "grounded": None, "unsupported_claims": [], "reason": f"unparseable judge output: {raw[:120]}"}
    grounded = bool(verdict.get("grounded"))
    return {
        "score": 1.0 if grounded else 0.0,
        "grounded": grounded,
        "unsupported_claims": verdict.get("unsupported_claims", []),
        "reason": verdict.get("reason", ""),
    }
