"""Settings, read from environment variables so the same code runs in Colab, Docker or locally."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _float_or_none(value: str | None) -> float | None:
    return None if value in (None, "", "none", "None") else float(value)


@dataclass(frozen=True)
class Settings:
    index_dir: str = os.getenv("OSFI_INDEX_DIR", "index")
    embed_model: str = os.getenv("OSFI_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = os.getenv("OSFI_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_model: str = os.getenv("OSFI_LLM_MODEL", "qwen2.5:7b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    top_k: int = int(os.getenv("OSFI_TOP_K", "5"))
    rerank_candidates: int = int(os.getenv("OSFI_RERANK_CANDIDATES", "50"))
    # Chosen in the Part 4 evaluation: refuses every unrelated question, never blocks an in-scope one
    min_relevance: float | None = _float_or_none(os.getenv("OSFI_MIN_RELEVANCE", "-1.99"))
