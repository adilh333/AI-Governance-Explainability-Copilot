"""
Knowledge base for the RAG pipeline.

Loads the markdown governance documents in rag/documents/, splits them into
section-level chunks, and builds a TF-IDF index for retrieval.

TF-IDF is used deliberately here rather than a neural embedding model: it
has zero external dependencies and no API cost, makes the retrieval step
fully transparent and debuggable, and is genuinely sufficient for a small,
curated document set like this one. The README explains how this component
could be swapped for sentence-transformer embeddings with a FAISS index as
a drop-in replacement, the rest of the pipeline does not need to change.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")


@dataclass
class Chunk:
    source: str
    heading: str
    text: str


def _load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(DOCS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split on markdown ## headings into sections.
        sections = re.split(r"\n## ", content)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            lines = section.split("\n", 1)
            heading = lines[0].lstrip("# ").strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            if body:
                chunks.append(Chunk(source=filename, heading=heading, text=body))
    return chunks


class GovernanceKnowledgeBase:
    def __init__(self):
        self.chunks = _load_chunks()
        texts = [f"{c.heading}. {c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]
