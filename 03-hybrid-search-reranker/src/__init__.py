"""Hybrid Information Retrieval & Cross-Encoder Re-Ranking Pipeline."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .data import SearchDocument, EvaluationQuery, CORPUS_DOCUMENTS, EVALUATION_QUERIES
from .sparse_index import SparseBM25Index
from .dense_index import DenseHNSWIndex
from .rrf import ReciprocalRankFusion
from .reranker import CrossEncoderReranker
from .pipeline import HybridSearchPipeline
from .evaluate import IREvaluator

__all__ = [
    "SearchDocument",
    "EvaluationQuery",
    "CORPUS_DOCUMENTS",
    "EVALUATION_QUERIES",
    "SparseBM25Index",
    "DenseHNSWIndex",
    "ReciprocalRankFusion",
    "CrossEncoderReranker",
    "HybridSearchPipeline",
    "IREvaluator",
]
