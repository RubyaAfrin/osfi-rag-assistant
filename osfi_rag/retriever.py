"""Hybrid retrieval over OSFI guideline chunks (version 2).

Combines keyword search (BM25) with semantic search (embeddings + cosine
similarity), merges the two rankings with Reciprocal Rank Fusion, and can
re-rank the merged candidates with a cross-encoder.

Version 2 changes, from the Part 4 evaluation:
- Keyword search stems words, so "define", "defines" and "definition" match.
- The re-ranker sees 50 candidates instead of 20, since it can only promote
  passages that the first stage retrieved.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict

import faiss
import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

STOPWORDS = set("""a an and are as at be by can do does for from has have how i in is it its
of on or should that the their them these this to under was what when which who why will
with""".split())

# BGE embedding models work best when search queries carry this instruction
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


_stemmer = PorterStemmer()


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Lower-case words for BM25, keeping acronyms like GDS and numbers like 80.
    With stem=True, words are reduced to their root: define/defines/defining -> defin."""
    words = [w for w in re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower()) if w not in STOPWORDS]
    return [_stemmer.stem(w) for w in words] if stem else words


@dataclass
class SearchResult:
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    section: str
    text: str
    url: str

    def to_dict(self) -> dict:
        return asdict(self)


class HybridRetriever:
    """Keyword + semantic search over a list of chunk dictionaries."""

    def __init__(self, chunks: list[dict], embedder, reranker=None, rrf_k: int = 60,
                 stem: bool = True, rerank_candidates: int = 50):
        self.chunks = chunks
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.stem = stem
        self.rerank_candidates = rerank_candidates
        self.bm25 = BM25Okapi([tokenize(c["embed_text"], stem) for c in chunks])
        self.index = None

    # ---------- building and saving the vector index ----------
    def build_index(self, batch_size: int = 32) -> np.ndarray:
        vectors = self.embedder.encode([c["embed_text"] for c in self.chunks], batch_size=batch_size,
                                       normalize_embeddings=True, show_progress_bar=True)
        self._set_vectors(np.asarray(vectors, dtype="float32"))
        return self.vectors

    def _set_vectors(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
        # Inner product on unit-length vectors equals cosine similarity
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "embeddings.npy"), self.vectors)
        with open(os.path.join(folder, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, folder: str, embedder, reranker=None, **kwargs) -> "HybridRetriever":
        with open(os.path.join(folder, "chunks.jsonl"), encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        retriever = cls(chunks, embedder, reranker, **kwargs)
        retriever._set_vectors(np.load(os.path.join(folder, "embeddings.npy")))
        return retriever

    # ---------- individual search methods (return (chunk index, score) pairs) ----------
    def keyword_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query, self.stem))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def semantic_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        q = self.embedder.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def hybrid_search(self, query: str, k: int = 20, candidates: int | None = None) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: reward chunks that rank well in either list."""
        candidates = candidates or max(30, k)
        fused: dict[int, float] = {}
        for ranking in (self.keyword_search(query, candidates), self.semantic_search(query, candidates)):
            for rank, (i, _) in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

    def rerank(self, query: str, hits: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
        """A cross-encoder reads the question and each passage together, which is slower but more precise."""
        if self.reranker is None or not hits:
            return hits[:k]
        scores = self.reranker.predict([(query, self.chunks[i]["embed_text"]) for i, _ in hits])
        order = np.argsort(scores)[::-1][:k]
        return [(hits[j][0], float(scores[j])) for j in order]

    # ---------- the main entry point ----------
    def search(self, query: str, k: int = 5, method: str = "hybrid_rerank") -> list[SearchResult]:
        if method == "keyword":
            hits = self.keyword_search(query, k)
        elif method == "semantic":
            hits = self.semantic_search(query, k)
        elif method == "hybrid":
            hits = self.hybrid_search(query, k)
        elif method == "hybrid_rerank":
            hits = self.rerank(query, self.hybrid_search(query, self.rerank_candidates), k)
        else:
            raise ValueError(f"Unknown method: {method}")
        return [SearchResult(rank=r, score=round(s, 4), chunk_id=self.chunks[i]["chunk_id"],
                             doc_id=self.chunks[i]["doc_id"], section=self.chunks[i]["section"],
                             text=self.chunks[i]["text"], url=self.chunks[i]["url"])
                for r, (i, s) in enumerate(hits, start=1)]
