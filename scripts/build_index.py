"""Build the final search index from the chunks produced in Parts 1-2.

Applies the definition labels (Part 4 fix), embeds every chunk, and saves
chunks.jsonl + embeddings.npy to the output folder that the API loads.

Usage:  python scripts/build_index.py --chunks path/to/chunks.jsonl --out index
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osfi_rag.config import Settings                  # noqa: E402
from osfi_rag.enrichment import enrich_definitions    # noqa: E402
from osfi_rag.retriever import HybridRetriever        # noqa: E402
from osfi_rag.service import load_models              # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", required=True, help="chunks.jsonl from the Part 2 index folder")
    parser.add_argument("--out", default="index", help="output folder for the final index")
    args = parser.parse_args()

    with open(args.chunks, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    chunks, labelled = enrich_definitions(chunks)

    settings = Settings()
    embedder, reranker = load_models(settings)
    retriever = HybridRetriever(chunks, embedder, reranker, stem=True,
                                rerank_candidates=settings.rerank_candidates)
    retriever.build_index()
    retriever.save(args.out)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"embed_model": settings.embed_model, "rerank_model": settings.rerank_model,
                   "chunks": len(chunks), "definition_passages_labelled": labelled,
                   "stemming": True, "rerank_candidates": settings.rerank_candidates}, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {args.out} ({labelled} definition passages labelled)")


if __name__ == "__main__":
    main()
