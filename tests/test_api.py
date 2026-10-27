from fastapi.testclient import TestClient

from osfi_rag.api import create_app
from osfi_rag.generator import RAGAssistant
from tests.conftest import FakeLLM


class BrokenLLM:
    def chat(self, messages, temperature=None):
        raise ConnectionError("LLM server unreachable")


def client(retriever, llm=None):
    return TestClient(create_app(RAGAssistant(retriever, llm or FakeLLM(), k=3)))


def test_health(retriever):
    r = client(retriever).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ask_returns_answer_with_sources(retriever):
    r = client(retriever).post("/ask", json={"question": "What is the HELOC LTV limit?"})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "answered" and body["citations"] == [1]
    assert body["sources"][0]["doc_id"] == "B-20"


def test_ask_rejects_empty_question(retriever):
    assert client(retriever).post("/ask", json={"question": ""}).status_code == 422


def test_ask_reports_llm_outage_as_503(retriever):
    r = client(retriever, BrokenLLM()).post("/ask", json={"question": "HELOC limit?"})
    assert r.status_code == 503


def test_search_endpoint(retriever):
    r = client(retriever).post("/search", json={"query": "GDS TDS", "k": 2, "method": "keyword"})
    assert r.status_code == 200 and r.json()[0]["chunk_id"] == "B-20-002"


def test_search_rejects_unknown_method(retriever):
    r = client(retriever).post("/search", json={"query": "GDS", "method": "magic"})
    assert r.status_code == 422
