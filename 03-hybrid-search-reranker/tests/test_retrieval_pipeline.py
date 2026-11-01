"""Unit Tests for Hybrid IR Pipeline & Cross-Encoder Reranker."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
from src.data import CORPUS_DOCUMENTS, SearchDocument
from src.dense_index import DenseHNSWIndex
from src.evaluate import IREvaluator
from src.pipeline import HybridSearchPipeline
from src.reranker import CrossEncoderReranker
from src.rrf import ReciprocalRankFusion
from src.sparse_index import SparseBM25Index


def test_sparse_bm25_index():
    index = SparseBM25Index()
    index.index(CORPUS_DOCUMENTS)

    # Search for exact acronym
    results = index.search("SOC2 compliance", top_k=5)
    assert len(results) > 0
    # doc_vanta_01 should be the top match
    assert results[0][0] == "doc_vanta_01"


def test_reciprocal_rank_fusion():
    rrf = ReciprocalRankFusion(k=60)
    dense_res = [("doc_A", 0.9), ("doc_B", 0.8), ("doc_C", 0.7)]
    sparse_res = [("doc_B", 12.0), ("doc_A", 10.0), ("doc_D", 8.0)]

    fused = rrf.fuse(dense_res, sparse_res, top_k=3)
    assert len(fused) == 3

    # doc_A and doc_B appear in both lists, so they should rank highest
    top_doc_ids = [doc_id for doc_id, _ in fused]
    assert "doc_A" in top_doc_ids[:2]
    assert "doc_B" in top_doc_ids[:2]


def test_dcg_and_ndcg_metrics():
    qrels = {"doc_1": 3, "doc_2": 2, "doc_3": 1}

    # Perfect ranking
    perfect = ["doc_1", "doc_2", "doc_3"]
    ndcg_perfect = IREvaluator.compute_ndcg(perfect, qrels, k=3)
    assert abs(ndcg_perfect - 1.0) < 1e-4

    # Sub-optimal ranking
    suboptimal = ["doc_3", "doc_2", "doc_1"]
    ndcg_sub = IREvaluator.compute_ndcg(suboptimal, qrels, k=3)
    assert ndcg_sub < 1.0
    assert ndcg_sub > 0.0


def test_mrr_metric():
    qrels = {"doc_relevant": 2}

    # Relevant document at rank 1 -> MRR = 1.0
    assert IREvaluator.compute_mrr(["doc_relevant", "doc_other"], qrels, k=2) == 1.0
    # Relevant document at rank 2 -> MRR = 0.5
    assert IREvaluator.compute_mrr(["doc_other", "doc_relevant"], qrels, k=2) == 0.5
    # Relevant document not retrieved -> MRR = 0.0
    assert IREvaluator.compute_mrr(["doc_other1", "doc_other2"], qrels, k=2) == 0.0


def test_end_to_end_hybrid_pipeline():
    pipeline = HybridSearchPipeline()
    pipeline.index_corpus(CORPUS_DOCUMENTS)

    out = pipeline.search("Rust vector database with HNSW indexing", top_k_final=3)
    assert "results" in out
    assert "timings" in out
    assert len(out["results"]) > 0

    top_doc = out["results"][0][0]
    # doc_qdrant_01 should be the top retrieved document
    assert top_doc == "doc_qdrant_01"
