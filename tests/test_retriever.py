from osfi_rag.retriever import HybridRetriever, tokenize
from tests.conftest import FakeEmbedder, FakeReranker


def test_stemming_matches_word_variants():
    # Part 4 finding: without stemming, "define" and "defines" never matched
    assert tokenize("define")[0] == tokenize("defines")[0] == tokenize("defining")[0]


def test_tokenize_keeps_acronyms_and_drops_stopwords():
    assert tokenize("What is the GDS ratio?", stem=False) == ["gds", "ratio"]


def test_each_search_method_finds_the_heloc_passage(retriever):
    for method in ["keyword", "semantic", "hybrid", "hybrid_rerank"]:
        top = retriever.search("HELOC loan-to-value limit", k=2, method=method)
        assert top[0].chunk_id == "B-20-001", method


def test_results_are_ranked_and_carry_citation_metadata(retriever):
    hits = retriever.search("GDS TDS ratios", k=3)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert hits[0].doc_id == "B-20" and hits[0].section and hits[0].url


def test_unknown_method_is_rejected(retriever):
    import pytest
    with pytest.raises(ValueError):
        retriever.search("anything", method="magic")


def test_save_and_load_round_trip(retriever, tmp_path):
    retriever.save(tmp_path)
    loaded = HybridRetriever.load(tmp_path, FakeEmbedder(), FakeReranker())
    assert len(loaded.chunks) == len(retriever.chunks)
    assert loaded.search("HELOC", k=1)[0].chunk_id == "B-20-001"


def test_reranker_pool_size_is_configurable(chunks):
    small = HybridRetriever(chunks, FakeEmbedder(), FakeReranker(), rerank_candidates=1)
    small.build_index()
    # With a pool of 1, only the first-stage winner can reach the re-ranker
    assert len(small.search("model", k=4)) == 1
