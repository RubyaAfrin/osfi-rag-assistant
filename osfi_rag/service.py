"""Load the models and index, and build the assistant. Kept separate so the API and
scripts share one setup path, and tests can skip it entirely."""
from __future__ import annotations

from .config import Settings
from .generator import OllamaClient, RAGAssistant
from .retriever import HybridRetriever


def device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_models(settings: Settings):
    from sentence_transformers import CrossEncoder, SentenceTransformer
    dev = device()
    return SentenceTransformer(settings.embed_model, device=dev), CrossEncoder(settings.rerank_model, device=dev)


def load_assistant(settings: Settings | None = None) -> RAGAssistant:
    settings = settings or Settings()
    embedder, reranker = load_models(settings)
    retriever = HybridRetriever.load(settings.index_dir, embedder, reranker, stem=True,
                                     rerank_candidates=settings.rerank_candidates)
    llm = OllamaClient(settings.llm_model, host=settings.ollama_host)
    return RAGAssistant(retriever, llm, k=settings.top_k, min_relevance=settings.min_relevance)
