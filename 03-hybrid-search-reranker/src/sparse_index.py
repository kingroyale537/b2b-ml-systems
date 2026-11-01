"""Sparse BM25 Lexical Index for Exact Keyword & Acronym Retrieval."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .data import SearchDocument


class SparseBM25Index:
    """Okapi BM25 Index with domain-aware tokenization preserving acronyms."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.corpus_docs: List[SearchDocument] = []
        self.bm25: BM25Okapi = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize preserving technical hyphenated words and alphanumeric acronyms (e.g., SOC2, HNSW)."""
        clean = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", clean)
        return tokens

    def index(self, documents: List[SearchDocument]) -> None:
        """Build BM25 inverted index over document corpus."""
        self.corpus_docs = documents
        self.doc_ids = [doc.doc_id for doc in documents]
        tokenized_corpus = [self.tokenize(doc.to_index_text()) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Retrieve top_k documents with raw BM25 scores."""
        if self.bm25 is None or not self.doc_ids:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score > 0.0:  # Only retain lexical matches with non-zero TF-IDF weight
                results.append((self.doc_ids[idx], score))

        return results
