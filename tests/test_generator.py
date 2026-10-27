from osfi_rag.generator import NOT_FOUND, RAGAssistant, classify
from tests.conftest import FakeLLM


# Real response patterns observed during evaluation (notebooks, Part 3 and Part 4)
def test_plain_refusal():
    assert classify(NOT_FOUND, []) == "refused"


def test_refusal_with_attached_citations_is_still_a_refusal():
    # Part 4, round 2: near-topic questions were wrongly counted as "partial"
    text = f"{NOT_FOUND} [1][2][3] The passages do not cover: cyber security risk."
    assert classify(text, [1, 2, 3]) == "refused"


def test_answer_followed_by_refusal_sentence_is_partial():
    # Part 3: a good GDS answer was wrongly counted as a refusal
    text = f"Include property taxes and heating costs [1][2]. {NOT_FOUND}"
    assert classify(text, [1, 2]) == "partial"


def test_partial_marker_detected_with_or_without_colon():
    # Part 4, round 1: "The passages do not cover the specific formulae" had no colon
    assert classify("Calculate conservatively [1]. The passages do not cover the formulae.", [1]) == "partial"
    assert classify("Calculate conservatively [1]. The passages do not cover: formulae.", [1]) == "partial"


def test_cited_answer():
    assert classify("The limit is 65 percent [1].", [1]) == "answered"


def test_answer_with_valid_citations(retriever):
    a = RAGAssistant(retriever, FakeLLM("HELOCs are limited to 65 percent LTV [1]."), k=3).answer("HELOC limit?")
    assert a.status == "answered" and a.citations == [1] and not a.warnings
    assert a.sources[0]["cited"] is True


def test_invalid_citation_is_flagged(retriever):
    a = RAGAssistant(retriever, FakeLLM("The limit is 65 percent [1][9]."), k=3).answer("HELOC limit?")
    assert a.invalid_citations == [9]
    assert any("not provided" in w for w in a.warnings)


def test_uncited_answer_is_flagged(retriever):
    a = RAGAssistant(retriever, FakeLLM("The limit is 65 percent."), k=3).answer("HELOC limit?")
    assert any("no citations" in w for w in a.warnings)


def test_relevance_threshold_refuses_without_calling_the_llm(retriever):
    llm = FakeLLM()
    a = RAGAssistant(retriever, llm, k=3, min_relevance=1000).answer("banana bread recipe")
    assert a.status == "refused" and llm.calls == 0


def test_refusal_drops_citations(retriever):
    a = RAGAssistant(retriever, FakeLLM(f"{NOT_FOUND} [1][2]"), k=3).answer("cyber risk?")
    assert a.status == "refused" and a.citations == []
    assert not any(s["cited"] for s in a.sources)
