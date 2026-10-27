"""Grounded answer generation: retrieve OSFI passages, then ask an LLM to answer
using only those passages, with numbered citations that are checked afterwards.

Version history (from the evaluation in notebooks/osfi_rag_part4_evaluation.ipynb):
- v2: partial answers get their own status, so a hedged answer is not counted as a refusal.
- v3: partial answers are detected with or without a colon, and the prompt discourages
  adding "not covered" notes by habit.
- v4: an answer that STARTS with the refusal sentence is a refusal, even if the model
  also attached citation numbers to it.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

import requests

NOT_FOUND = "I can't find this in the OSFI documents provided."
PARTIAL_MARKER = "The passages do not cover:"
_PARTIAL_PATTERN = re.compile(r"passages do not (cover|contain|include|address)", re.IGNORECASE)

SYSTEM_PROMPT = f"""You are an assistant that answers questions about Canadian financial regulation
using ONLY the numbered passages from OSFI guidelines provided by the user.

Rules:
1. Use only facts stated in the passages. Do not use outside knowledge.
2. After every sentence that uses a passage, cite it with its number in square brackets, like [1] or [2][3].
3. If NONE of the passages helps answer the question, reply with exactly this sentence and nothing else:
   {NOT_FOUND}
4. If the question asks for something specific that the passages do not contain, answer the rest with
   citations, then end with one sentence starting "{PARTIAL_MARKER}" naming that specific thing.
   Do NOT add this sentence just because more detail could exist: if the passages answer what was asked,
   simply answer. Never use the sentence from rule 3 in an answer that also contains information.
5. Each passage shows its document and section heading. A passage under a heading such as
   "Key Terms > Model" is the definition of that term, even if the term itself is not repeated in the text.
6. Be concise: at most 5 sentences, in plain language."""


class OllamaClient:
    """Minimal client for a local Ollama server's chat endpoint."""

    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.0,
                 num_ctx: int = 8192, timeout: int = 300):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.options = {"temperature": temperature, "num_ctx": num_ctx}

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        options = dict(self.options)
        if temperature is not None:
            options["temperature"] = temperature
        r = requests.post(f"{self.host}/api/chat", timeout=self.timeout,
                          json={"model": self.model, "messages": messages, "stream": False,
                                "options": options})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


@dataclass
class Answer:
    question: str
    answer: str
    status: str                      # "answered", "partial" or "refused"
    citations: list[int]
    invalid_citations: list[int]
    sources: list[dict]
    top_relevance: float | None
    seconds: float
    warnings: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.status == "refused"

    def to_dict(self) -> dict:
        return asdict(self)


def classify(text: str, valid_citations: list[int]) -> str:
    """Decide whether an LLM response is an answer, a partial answer, or a refusal."""
    lowered = text.strip().lower()
    refusal = NOT_FOUND.lower().rstrip(".")
    if lowered.startswith(refusal) or (refusal in lowered and not valid_citations):
        return "refused"
    if refusal in lowered or _PARTIAL_PATTERN.search(text):
        return "partial"
    return "answered"


class RAGAssistant:
    """Retrieve passages, build a grounded prompt, call the LLM, and validate the citations."""

    def __init__(self, retriever, llm, k: int = 5, min_relevance: float | None = None):
        self.retriever, self.llm, self.k = retriever, llm, k
        self.min_relevance = min_relevance   # guardrail on the re-ranker score

    @staticmethod
    def format_passages(results) -> str:
        return "\n\n".join(f"[{r.rank}] ({r.doc_id} > {r.section.replace(' › ', ' > ')})\n{r.text}"
                           for r in results)

    def answer(self, question: str) -> Answer:
        t0 = time.time()
        results = self.retriever.search(question, k=self.k, method="hybrid_rerank")
        top = results[0].score if results else None

        # Guardrail 1: if nothing relevant was retrieved, don't ask the LLM at all
        if not results or (self.min_relevance is not None and top < self.min_relevance):
            return Answer(question, NOT_FOUND, "refused", [], [], [], top, round(time.time() - t0, 2),
                          ["No passage passed the relevance threshold; the LLM was not called."])

        user_msg = f"Passages:\n\n{self.format_passages(results)}\n\nQuestion: {question}"
        text = self.llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user_msg}])

        # Guardrail 2: every citation must point to a passage that was actually provided
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
        valid = [n for n in cited if 1 <= n <= len(results)]
        invalid = [n for n in cited if n not in valid]
        status = classify(text, valid)
        if status == "refused":
            valid = []                   # citations attached to a refusal don't support anything

        warnings = []
        if invalid:
            warnings.append(f"Answer cites passages that were not provided: {invalid}")
        if status != "refused" and not valid:
            warnings.append("Answer has no citations, so it may not be grounded in the documents.")

        sources = [{"n": r.rank, "doc_id": r.doc_id, "section": r.section, "url": r.url,
                    "chunk_id": r.chunk_id, "relevance": r.score, "cited": r.rank in valid}
                   for r in results]
        return Answer(question, text, status, valid, invalid, sources, top,
                      round(time.time() - t0, 2), warnings)
