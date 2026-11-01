"""Benchmark Runner for Two-Stage Hybrid IR Pipeline.

Compares:
  1. BM25 Sparse Lexical Only
  2. Dense HNSW Semantic Only
  3. Hybrid Retrieval (Reciprocal Rank Fusion)
  4. Hybrid Retrieval + Cross-Encoder Neural Re-Ranking
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table

from src.data import CORPUS_DOCUMENTS, EVALUATION_QUERIES
from src.dense_index import DenseHNSWIndex
from src.evaluate import IREvaluator
from src.pipeline import HybridSearchPipeline
from src.reranker import CrossEncoderReranker
from src.rrf import ReciprocalRankFusion
from src.sparse_index import SparseBM25Index


def run_benchmark():
    console = Console()
    console.print("\n[bold cyan]====================================================================[/bold cyan]")
    console.print("[bold cyan] Scalable Information Retrieval & Cross-Encoder Benchmark Suite     [/bold cyan]")
    console.print("[bold cyan]====================================================================[/bold cyan]\n")

    console.print(f"[*] Initializing Pipeline & Indexing {len(CORPUS_DOCUMENTS)} Enterprise B2B Documents...")
    pipeline = HybridSearchPipeline()
    index_stats = pipeline.index_corpus(CORPUS_DOCUMENTS)
    console.print(f"[green]✓ Documents indexed. Dense HNSW: {index_stats['dense_index_time_sec']:.2f}s, Sparse BM25: {index_stats['sparse_index_time_sec']*1000:.1f}ms.[/green]\n")

    modes = [
        ("BM25 Only (Lexical)", "bm25_only"),
        ("Dense HNSW Only (Semantic)", "dense_only"),
        ("Hybrid (RRF Fusion, k=60)", "hybrid_rrf"),
        ("Hybrid + Cross-Encoder Reranker", "hybrid_rerank"),
    ]

    results = []
    for label, mode in modes:
        console.print(f"[*] Evaluating Mode: [bold yellow]{label}[/bold yellow]...")
        metrics = IREvaluator.evaluate_pipeline(
            pipeline=pipeline, queries=EVALUATION_QUERIES, k=10, mode=mode
        )
        metrics["label"] = label
        results.append(metrics)

    # Print benchmark table
    table = Table(title="Information Retrieval Benchmark: Standard IR Evaluation Suite (k=10)")
    table.add_column("Retrieval Architecture", style="cyan", no_wrap=True)
    table.add_column("NDCG@10", style="magenta")
    table.add_column("MRR@10", style="bold green")
    table.add_column("Precision@10", style="white")
    table.add_column("Recall@10", style="white")
    table.add_column("Latency (p50 / p95)", style="yellow")

    for res in results:
        table.add_row(
            res["label"],
            f"{res['ndcg_at_k']:.4f}",
            f"{res['mrr_at_k']:.4f}",
            f"{res['precision_at_k']:.4f}",
            f"{res['recall_at_k']:.4f}",
            f"{res['p50_latency_ms']:.1f} ms / {res['p95_latency_ms']:.1f} ms",
        )

    console.print("\n")
    console.print(table)
    console.print("\n[bold green]✓ IR benchmark completed successfully.[/bold green]\n")


if __name__ == "__main__":
    run_benchmark()
