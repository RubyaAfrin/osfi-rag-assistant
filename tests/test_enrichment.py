from osfi_rag.enrichment import defined_term, enrich_definitions


def test_defined_term_reads_the_last_heading_under_key_terms():
    assert defined_term("A. Overview › A.4 Key Terms › Model") == "Model"
    assert defined_term("Glossary › Loan-to-value ratio") == "Loan-to-value ratio"


def test_ordinary_sections_are_not_definitions():
    assert defined_term("II. Principles › Debt service coverage") is None
    assert defined_term("A.4 Key Terms") is None   # the heading itself, with no term under it


def test_enrichment_changes_search_text_only(chunks):
    enriched, n = enrich_definitions(chunks)
    assert n == 2
    model = next(c for c in enriched if c["chunk_id"] == "E-23-001")
    original = next(c for c in chunks if c["chunk_id"] == "E-23-001")
    assert 'Definition of the term "Model"' in model["embed_text"]
    assert model["text"] == original["text"]          # what the LLM and users see is unchanged
    assert "Definition" not in original["embed_text"]  # input chunks are not modified
