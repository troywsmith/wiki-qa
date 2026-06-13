"""FastAPI app exposing the Wikipedia-grounded QA agent."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Agent
from .config import get_settings

app = FastAPI(title="wiki-qa", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    steps: int


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise HTTPException(status_code=503, detail="WIKIQA_ANTHROPIC_API_KEY is not configured.")
    result = await Agent(settings).answer(req.question)
    return AskResponse(**result)
