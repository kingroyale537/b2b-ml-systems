"""Reciprocal Rank Fusion (RRF) for Combining Sparse and Dense Retrievers.

Combines disjoint or overlapping rank positions mathematically:
  RRF_score(d) = sum_{m in {dense, sparse}} w_m / (k + rank_m(d))
"""

from __future__ import annotations

import collections
from typing import Dict, List, Tuple


class ReciprocalRankFusion:
    """Combines ranked candidate lists using Reciprocal Rank Fusion."""

    def __init__(
        self,
        k: int = 60,
        weight_dense: float = 1.0,
        weight_sparse: float = 1.0,
    ) -> None:
        self.k = k
        self.weight_dense = weight_dense
        self.weight_sparse = weight_sparse

    def fuse(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Compute RRF score across dense and sparse ranked candidates."""
        rrf_scores: Dict[str, float] = collections.defaultdict(float)

        # 1-based ranks for dense
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] += self.weight_dense / float(self.k + rank)

        # 1-based ranks for sparse
        for rank, (doc_id, _) in enumerate(sparse_results, start=1):
            rrf_scores[doc_id] += self.weight_sparse / float(self.k + rank)

        # Sort descending by fused score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_candidates[:top_k]
