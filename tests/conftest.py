"""Shared test doubles: tiny deterministic stand-ins for the embedding model, the
re-ranker and the LLM, so tests run in seconds with no downloads or GPU."""
import hashlib
import re

import numpy as np
import pytest

from osfi_rag.retriever import HybridRetriever


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words get similar vectors."""
    def encode(self, texts, normalize_embeddings=False, **kwargs):
        vecs = np.zeros((len(texts), 256), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                vecs[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        if normalize_embeddings:
            vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        return vecs


class FakeReranker:
    """Scores a passage by how many question words it contains."""
    def predict(self, pairs):
        return np.array([len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs], float)


class FakeLLM:
    def __init__(self, reply="The limit is 65 percent [1]."):
        self.reply, self.calls = reply, 0

    def chat(self, messages, temperature=None):
        self.calls += 1
        return self.reply


def chunk(cid, doc, section, text):
    return {"chunk_id": cid, "doc_id": doc, "title": f"Guideline {doc}", "section": section,
            "url": f"https://example.org/{doc}", "text": text,
            "embed_text": f"Document: Guideline {doc}\nSection: {section}\n\n{text}"}


@pytest.fixture
def chunks():
    return [
        chunk("B-20-001", "B-20", "II. Principles › HELOCs",
              "OSFI expects lenders to limit the HELOC component to a loan-to-value ratio of 65 percent."),
        chunk("B-20-002", "B-20", "II. Principles › Debt service coverage",
              "Lenders should calculate GDS and TDS ratios including property taxes and heating costs."),
        chunk("E-23-001", "E-23", "A. Overview › A.4 Key Terms › Model",
              "An application of theoretical, empirical or statistical techniques which processes input data."),
        chunk("E-23-002", "E-23", "A. Overview › A.4 Key Terms › Model risk",
              "Model risk is the risk of model error in model design and model use of a model."),
    ]


@pytest.fixture
def retriever(chunks):
    r = HybridRetriever(chunks, FakeEmbedder(), FakeReranker())
    r.build_index()
    return r
