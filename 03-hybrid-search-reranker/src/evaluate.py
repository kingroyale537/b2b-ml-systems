"""Evaluation Engine for Information Retrieval Pipelines.

Calculates:
  1. NDCG@K (Normalized Discounted Cumulative Gain)
  2. MRR@K (Mean Reciprocal Rank)
  3. Precision@K and Recall@K
  4. Latency benchmarks (p50, p95, p99)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .data import EvaluationQuery
from .pipeline import HybridSearchPipeline


class IREvaluator:
    """Evaluates IR system quality across standard trec_eval style benchmarks."""

    @staticmethod
    def compute_dcg(relevance_scores: List[int], k: int = 10) -> float:
        """Compute Discounted Cumulative Gain at K."""
        dcg = 0.0
        for i, rel in enumerate(relevance_scores[:k], start=1):
            if rel > 0:
                dcg += (2.0 ** rel - 1.0) / math.log2(i + 1)
        return dcg

    @classmethod
    def compute_ndcg(
        cls, retrieved_doc_ids: List[str], qrels: Dict[str, int], k: int = 10
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain at K."""
        actual_scores = [qrels.get(d_id, 0) for d_id in retrieved_doc_ids[:k]]
        dcg = cls.compute_dcg(actual_scores, k=k)

        # Ideal DCG
        ideal_scores = sorted(list(qrels.values()), reverse=True)
        idcg = cls.compute_dcg(ideal_scores, k=k)

        if idcg <= 0.0:
            return 1.0 if dcg <= 0.0 else 0.0
        return dcg / idcg

    @staticmethod
    def compute_mrr(
        retrieved_doc_ids: List[str], qrels: Dict[str, int], k: int = 10
    ) -> float:
        """Compute Mean Reciprocal Rank at K (rank of first document with rel > 0)."""
        for i, d_id in enumerate(retrieved_doc_ids[:k], start=1):
            if qrels.get(d_id, 0) > 0:
                return 1.0 / float(i)
        return 0.0

    @staticmethod
    def compute_precision_recall(
        retrieved_doc_ids: List[str], qrels: Dict[str, int], k: int = 10
    ) -> Tuple[float, float]:
        """Compute Precision@K and Recall@K."""
        retrieved_k = retrieved_doc_ids[:k]
        relevant_in_k = sum(1 for d_id in retrieved_k if qrels.get(d_id, 0) > 0)
        total_relevant = sum(1 for score in qrels.values() if score > 0)

        precision = relevant_in_k / float(max(len(retrieved_k), 1))
        recall = relevant_in_k / float(max(total_relevant, 1))
        return precision, recall

    @classmethod
    def evaluate_pipeline(
        cls,
        pipeline: HybridSearchPipeline,
        queries: List[EvaluationQuery],
        k: int = 10,
        mode: str = "hybrid_rerank",
    ) -> Dict[str, Any]:
        """Run batch evaluation across all queries and aggregate IR metrics."""
        ndcgs = []
        mrrs = []
        precisions = []
        recalls = []
        latencies = []

        for q in queries:
            search_out = pipeline.search(q.query_text, top_k_final=k, mode=mode)
            retrieved_ids = [doc_id for doc_id, _ in search_out["results"]]

            ndcg = cls.compute_ndcg(retrieved_ids, q.relevant_doc_scores, k=k)
            mrr = cls.compute_mrr(retrieved_ids, q.relevant_doc_scores, k=k)
            prec, rec = cls.compute_precision_recall(retrieved_ids, q.relevant_doc_scores, k=k)

            ndcgs.append(ndcg)
            mrrs.append(mrr)
            precisions.append(prec)
            recalls.append(rec)
            latencies.append(search_out["timings"]["total_latency_ms"])

        return {
            "mode": mode,
            "num_queries": len(queries),
            "ndcg_at_k": float(np.mean(ndcgs)),
            "mrr_at_k": float(np.mean(mrrs)),
            "precision_at_k": float(np.mean(precisions)),
            "recall_at_k": float(np.mean(recalls)),
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }
