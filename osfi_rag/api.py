"""REST API for the OSFI Regulation Assistant.

Run locally:   uvicorn osfi_rag.api:app --port 8000
Interactive docs are served at /docs.
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          examples=["What is the minimum qualifying rate for uninsured mortgages?"])


class Source(BaseModel):
    n: int
    doc_id: str
    section: str
    url: str
    chunk_id: str
    relevance: float
    cited: bool


class AskResponse(BaseModel):
    question: str
    answer: str
    status: Literal["answered", "partial", "refused"]
    citations: list[int]
    sources: list[Source]
    warnings: list[str]
    seconds: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    k: int = Field(5, ge=1, le=20)
    method: Literal["keyword", "semantic", "hybrid", "hybrid_rerank"] = "hybrid_rerank"


class SearchHit(BaseModel):
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    section: str
    text: str
    url: str


def create_app(assistant=None) -> FastAPI:
    """Build the app. Pass an assistant to reuse one (or a test double); otherwise it loads on first use."""
    app = FastAPI(title="OSFI Regulation Assistant", version=__version__,
                  description="Answers questions about OSFI mortgage and model-risk guidelines, "
                              "using only the official documents, with citations.")
    state = {"assistant": assistant}

    def get_assistant():
        if state["assistant"] is None:
            from .service import load_assistant
            state["assistant"] = load_assistant()
        return state["assistant"]

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "loaded": state["assistant"] is not None}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        try:
            result = get_assistant().answer(req.question.strip())
        except Exception as exc:  # e.g. the LLM server is unreachable
            raise HTTPException(status_code=503, detail=f"Assistant unavailable: {exc}") from exc
        return AskResponse(**{k: v for k, v in result.to_dict().items()
                              if k in AskResponse.model_fields})

    @app.post("/search", response_model=list[SearchHit])
    def search(req: SearchRequest):
        hits = get_assistant().retriever.search(req.query.strip(), k=req.k, method=req.method)
        return [SearchHit(**h.to_dict()) for h in hits]

    return app


app = create_app()
