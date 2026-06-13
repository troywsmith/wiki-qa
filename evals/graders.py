"""Graders for the evaluation suite, organized by quality dimension.

Dimensions and their graders:
  - faithfulness  → faithfulness_judge   (model-based, vs retrieved context)
  - correctness   → reference_judge      (model-based, vs reference_answer)
  - completeness  → reference_judge      (model-based, vs reference_answer)
  - attribution   → citation_match       (code-based)
  - calibration   → refusal_correct      (code-based)

correctness/completeness are a precision/recall split: one reference_judge call
returns both — correctness = the outcome asserts nothing contradicting the
reference; completeness = the outcome covers the reference's key information.

Each grader returns a float in [0, 1], or None when it does not apply to a task.
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


def citation_match(source_titles: list[str], expected: list[str]) -> float | None:
    """Attribution: fraction of expected source articles the agent actually cited."""
    if not expected:
        return None
    got = [t.lower() for t in source_titles]
    hits = sum(1 for e in expected if any(e.lower() in t or t in e.lower() for t in got))
    return hits / len(expected)


def looks_like_refusal(answer: str) -> bool:
    return any(marker in answer.lower() for marker in _REFUSAL_MARKERS)


def refusal_correct(answer: str, should_refuse: bool) -> float:
    """Calibration: 1.0 if the refuse/answer behaviour matches expectation, else 0.0."""
    return 1.0 if looks_like_refusal(answer) == should_refuse else 0.0


# --- Model-based graders (LLM-as-judge) ---

# Rubric for the faithfulness grader (graded against retrieved context).
_FAITHFULNESS_RUBRIC = """You grade whether an ANSWER is fully supported by SOURCE TEXT \
retrieved from Wikipedia. Do not use outside knowledge — judge only against the source.

SOURCE TEXT:
{context}

ANSWER:
{answer}

Is every factual claim in the ANSWER supported by the SOURCE TEXT? Reply with ONLY a \
JSON object: {{"grounded": true|false, "unsupported_claims": ["..."], "reason": "..."}}"""

# Rubric for the reference grader (graded against the gold reference_answer).
_REFERENCE_RUBRIC = """You compare an ANSWER to a gold REFERENCE answer along two axes:
- correctness (precision): does the ANSWER avoid asserting anything that contradicts \
the REFERENCE? An answer that states no relevant facts (e.g. a refusal) is vacuously \
correct (1.0) because it asserts nothing false.
- completeness (recall): does the ANSWER convey the key information in the REFERENCE? \
A refusal or non-answer covers nothing and scores 0.0.

REFERENCE:
{reference}

ANSWER:
{answer}

Reply with ONLY a JSON object: \
{{"correctness": 0.0-1.0, "completeness": 0.0-1.0, "reason": "..."}}"""


def _extract_json(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return None


async def faithfulness_judge(
    client: AsyncAnthropic, model: str, answer: str, context: str
) -> dict[str, Any]:
    """Faithfulness: is the answer grounded in the retrieved context? Returns
    {"score": 0|1|None, "grounded": bool|None, "unsupported_claims": [...], "reason": str}."""
    if not context.strip():
        return {"score": None, "grounded": None, "unsupported_claims": [], "reason": "no context retrieved"}
    resp = await client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": _FAITHFULNESS_RUBRIC.format(context=context[:12000], answer=answer)}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    verdict = _extract_json(raw)
    if verdict is None:
        return {"score": None, "grounded": None, "unsupported_claims": [], "reason": f"unparseable judge output: {raw[:120]}"}
    grounded = bool(verdict.get("grounded"))
    return {
        "score": 1.0 if grounded else 0.0,
        "grounded": grounded,
        "unsupported_claims": verdict.get("unsupported_claims", []),
        "reason": verdict.get("reason", ""),
    }


async def reference_judge(
    client: AsyncAnthropic, model: str, answer: str, reference: str
) -> dict[str, Any]:
    """Correctness + completeness vs the gold reference. Returns
    {"correctness": 0-1|None, "completeness": 0-1|None, "reason": str}."""
    resp = await client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": _REFERENCE_RUBRIC.format(reference=reference, answer=answer)}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    verdict = _extract_json(raw)
    if verdict is None:
        return {"correctness": None, "completeness": None, "reason": f"unparseable judge output: {raw[:120]}"}
    return {
        "correctness": _clamp(verdict.get("correctness")),
        "completeness": _clamp(verdict.get("completeness")),
        "reason": verdict.get("reason", ""),
    }


def _clamp(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
