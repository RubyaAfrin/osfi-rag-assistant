"""Make definition passages findable.

Definitions rarely repeat the term they define ("Model: An application of theoretical,
empirical... techniques"). The term appears only in the heading, so search ranks the
definition below neighbouring passages that mention the word many times. This adds an
explicit label to the text used for search."""
from __future__ import annotations

DEFINITION_SECTIONS = ("key terms", "definitions", "glossary", "defined terms")


def defined_term(section: str) -> str | None:
    """Return the term if this section is a definition entry, e.g. 'A.4 Key Terms › Model' -> 'Model'."""
    parts = [p.strip() for p in section.split("›")]
    for i, part in enumerate(parts[:-1]):
        if any(key in part.lower() for key in DEFINITION_SECTIONS):
            return parts[-1]
    return None


def enrich_definitions(chunks: list[dict]) -> tuple[list[dict], int]:
    """Return copies of the chunks with definition passages labelled, plus how many were changed."""
    enriched, changed = [], 0
    for c in chunks:
        c = dict(c)
        term = defined_term(c["section"])
        if term:
            label = f'Definition of the term "{term}". This passage defines what "{term}" means.'
            c["embed_text"] = f"{label}\n{c['embed_text']}"
            changed += 1
        enriched.append(c)
    return enriched, changed
