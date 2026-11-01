"""Benchmark Runner for B2B Entity Resolution Engine.

Evaluates:
  1. Two-Stage Blocking Layer efficiency and candidate recall
  2. Pairwise Linkage Precision, Recall, and F1
  3. BCubed Global Cluster Consistency
  4. End-to-end throughput and latency scaling
"""

import sys
import os
import time

# Ensure package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table
from src.data import B2BDataGenerator
from src.blocking import TwoStageBlocker
from src.pipeline import EntityResolutionPipeline
from src.evaluate import EntityResolutionEvaluator


def run_benchmark():
    console = Console()
    console.print("\n[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan] B2B Entity Resolution & Record Linkage Engine: Benchmark Suite [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]\n")

    # Generate test corpus
    num_canonical = 50
    duplicates_per_entity = 4
    console.print(f"[*] Generating synthetic corporate corpus ({num_canonical} canonical entities, ~{num_canonical * 5} records)...")
    gen = B2BDataGenerator(seed=1337)
    records, clusters = gen.generate_corpus(
        num_canonical=num_canonical, duplicates_per_entity=duplicates_per_entity
    )
    console.print(f"[green]✓ Generated {len(records)} corporate records across {len(clusters)} ground truth clusters.[/green]\n")

    # Step 1: Benchmark Two-Stage Blocking Layer
    console.print("[bold yellow][Stage 1] Benchmarking Two-Stage Blocking Layer (MinHash LSH + Sparse Tokens)...[/bold yellow]")
    blocker = TwoStageBlocker(num_perm=64, threshold=0.35, max_candidates_per_record=20)
    idx_time = blocker.fit_index(records)
    candidate_pairs, blocking_stats = blocker.get_candidate_pairs(records)
    blocking_eval = EntityResolutionEvaluator.evaluate_blocking(records, candidate_pairs)

    b_table = Table(title="Stage 1: Blocking Layer Performance Metrics")
    b_table.add_column("Metric", style="cyan", no_wrap=True)
    b_table.add_column("Value", style="magenta")
    b_table.add_column("Target Spec", style="green")

    b_table.add_row("Total Ingested Records", str(len(records)), "N")
    b_table.add_row("All Possible O(N^2) Pairs", f"{(len(records)*(len(records)-1))//2:,}", "-")
    b_table.add_row("Candidate Pairs Retained", f"{len(candidate_pairs):,}", "Top 20 per rec")
    b_table.add_row("Pairwise Reduction Ratio", f"{blocking_eval['reduction_ratio']*100:.3f}%", "> 99.0%")
    b_table.add_row("Candidate Recall (True Dupes)", f"{blocking_eval['candidate_recall']*100:.2f}%", "> 95.0%")
    b_table.add_row("Total Blocking Time", f"{blocking_stats['total_blocking_time_sec']*1000:.1f} ms", "-")
    b_table.add_row("Avg Latency per Record", f"{blocking_stats['latency_per_record_ms']:.2f} ms", "Sub-10 ms")

    console.print(b_table)
    console.print()

    # Step 2 & 3: End-to-End Pipeline
    console.print("[bold yellow][Stage 2 & 3] Benchmarking Transformer Bi-Encoder & Graph Clustering...[/bold yellow]")
    pipeline = EntityResolutionPipeline(blocking_threshold=0.35, match_threshold=0.75, batch_size=32)
    pipeline_res = pipeline.run(records)

    # Extract predicted pairs (scored above match threshold)
    predicted_pairs = {
        (id_a, id_b) if id_a < id_b else (id_b, id_a)
        for id_a, id_b, score in pipeline_res["scored_pairs"]
        if score >= 0.75
    }

    # Evaluate Metrics
    pairwise_eval = EntityResolutionEvaluator.evaluate_pairwise(records, predicted_pairs)
    bcubed_eval = EntityResolutionEvaluator.evaluate_bcubed(records, pipeline_res["record_to_cluster"])

    res_table = Table(title="End-to-End Resolution Benchmark: PyTorch Bi-Encoder + Graph Clustering")
    res_table.add_column("Metric", style="cyan", no_wrap=True)
    res_table.add_column("Score", style="bold green")
    res_table.add_column("Notes", style="white")

    res_table.add_row("Pairwise Precision", f"{pairwise_eval['precision']*100:.2f}%", "Ratio of true matches among predicted")
    res_table.add_row("Pairwise Recall", f"{pairwise_eval['recall']*100:.2f}%", "Ratio of true duplicates linked")
    res_table.add_row("Pairwise F1-Score", f"{pairwise_eval['f1_score']*100:.2f}%", "Harmonic mean of precision & recall")
    res_table.add_row("BCubed Cluster Precision", f"{bcubed_eval['bcubed_precision']*100:.2f}%", "Item-level cluster purity")
    res_table.add_row("BCubed Cluster Recall", f"{bcubed_eval['bcubed_recall']*100:.2f}%", "Item-level cluster completeness")
    res_table.add_row("BCubed Cluster F1", f"{bcubed_eval['bcubed_f1']*100:.2f}%", "Overall cluster quality metric")
    res_table.add_row("Ground Truth Entities", str(num_canonical), "Original distinct companies")
    res_table.add_row("Discovered Global Entity IDs", str(len(pipeline_res["canonical_profiles"])), "Graph-clustered entities")
    res_table.add_row("Throughput", f"{pipeline_res['metrics']['throughput_records_per_sec']:.1f} rec/s", "End-to-end processing speed")

    console.print(res_table)
    console.print("\n[bold green]✓ Benchmark completed successfully.[/bold green]\n")


if __name__ == "__main__":
    run_benchmark()
