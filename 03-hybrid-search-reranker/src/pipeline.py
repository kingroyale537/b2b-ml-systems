"""Unified Two-Stage Hybrid Information Retrieval Pipeline.

Combines:
  1. Dense HNSW Vector Search (FAISS)
  2. Sparse Lexical Search (BM25)
  3. Reciprocal Rank Fusion (RRF)
  4. Cross-Encoder Full Cross-Attention Re-Ranking (<25ms)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .data import SearchDocument
from .dense_index import DenseHNSWIndex
from .reranker import CrossEncoderReranker
from .rrf import ReciprocalRankFusion
from .sparse_index import SparseBM25Index


class HybridSearchPipeline:
    """Production Two-Stage Hybrid IR Engine."""

    def __init__(
        self,
        dense_index: Optional[DenseHNSWIndex] = None,
        sparse_index: Optional[SparseBM25Index] = None,
        rrf: Optional[ReciprocalRankFusion] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        enable_reranker: bool = True,
    ) -> None:
        self.dense_index = dense_index if dense_index is not None else DenseHNSWIndex()
        self.sparse_index = sparse_index if sparse_index is not None else SparseBM25Index()
        self.rrf = rrf if rrf is not None else ReciprocalRankFusion(k=60)
        self.reranker = reranker if reranker is not None else CrossEncoderReranker()
        self.enable_reranker = enable_reranker
        self.doc_lookup: Dict[str, SearchDocument] = {}

    def index_corpus(self, documents: List[SearchDocument]) -> Dict[str, float]:
        """Index document corpus across dense HNSW and sparse BM25 indices."""
        self.doc_lookup = {d.doc_id: d for d in documents}
        t_dense = self.dense_index.index(documents)
        t0 = time.perf_counter()
        self.sparse_index.index(documents)
        t_sparse = time.perf_counter() - t0

        return {
            "dense_index_time_sec": t_dense,
            "sparse_index_time_sec": t_sparse,
            "total_docs_indexed": float(len(documents)),
        }

    def search(
        self,
        query: str,
        top_k_candidates: int = 50,
        top_k_final: int = 10,
        mode: str = "hybrid_rerank",  # 'bm25_only', 'dense_only', 'hybrid_rrf', 'hybrid_rerank'
    ) -> Dict[str, Any]:
        """Execute end-to-end multi-stage retrieval with exact latency instrumentation."""
        total_t0 = time.perf_counter()

        timings: Dict[str, float] = {}

        # 1. Sparse BM25 retrieval
        t0 = time.perf_counter()
        sparse_results = self.sparse_index.search(query, top_k=top_k_candidates)
        timings["sparse_latency_ms"] = (time.perf_counter() - t0) * 1000.0

        if mode == "bm25_only":
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            return {
                "results": sparse_results[:top_k_final],
                "timings": {**timings, "total_latency_ms": total_ms},
            }

        # 2. Dense HNSW retrieval
        t0 = time.perf_counter()
        dense_results = self.dense_index.search(query, top_k=top_k_candidates)
        timings["dense_latency_ms"] = (time.perf_counter() - t0) * 1000.0

        if mode == "dense_only":
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            return {
                "results": dense_results[:top_k_final],
                "timings": {**timings, "total_latency_ms": total_ms},
            }

        # 3. Reciprocal Rank Fusion
        t0 = time.perf_counter()
        fused_candidates = self.rrf.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k_candidates,
        )
        timings["rrf_fusion_latency_ms"] = (time.perf_counter() - t0) * 1000.0

        if mode == "hybrid_rrf" or not self.enable_reranker:
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            return {
                "results": fused_candidates[:top_k_final],
                "timings": {**timings, "total_latency_ms": total_ms},
            }

        # 4. Cross-Encoder Re-Ranking
        reranked_results, rerank_ms = self.reranker.rerank(
            query=query,
            candidates=fused_candidates,
            doc_lookup=self.doc_lookup,
            top_k=top_k_final,
        )
        timings["rerank_latency_ms"] = rerank_ms

        total_ms = (time.perf_counter() - total_t0) * 1000.0
        timings["total_latency_ms"] = total_ms

        return {
            "results": reranked_results,
            "timings": timings,
        }
