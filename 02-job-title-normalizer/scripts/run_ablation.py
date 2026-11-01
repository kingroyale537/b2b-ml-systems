"""Ablation Study Runner for Job Title Normalization.

Compares:
  - Baseline Zero-Shot Multilingual Model
  - Fine-Tuned Model with MNRL + Hard Negative Mining + Hierarchical Dual-Head
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table

from src.data import TitleDataGenerator
from src.evaluate import TitleEvaluator
from src.inference import TitleNormalizerInference
from src.model import HierarchicalTitleNormalizer
from src.train import train_title_normalizer


def run_ablation():
    console = Console()
    console.print("\n[bold cyan]==================================================================[/bold cyan]")
    console.print("[bold cyan] Multilingual Job Title Normalizer: Ablation & Benchmark Suite   [/bold cyan]")
    console.print("[bold cyan]==================================================================[/bold cyan]\n")

    # 1. Generate test samples
    console.print("[*] Generating multilingual test dataset across EN, FR, DE, ES...")
    gen = TitleDataGenerator(seed=999)
    test_samples = gen.generate_corpus(samples_per_role=15)
    console.print(f"[green]✓ Generated {len(test_samples)} evaluation titles.[/green]\n")

    # 2. Evaluate Baseline Zero-Shot
    console.print("[bold yellow][1/2] Evaluating Baseline Zero-Shot Multilingual Model...[/bold yellow]")
    baseline_engine = TitleNormalizerInference()
    baseline_metrics = TitleEvaluator.evaluate(baseline_engine, test_samples)

    # 3. Train Fine-Tuned Model with MNRL + Hard Negatives + Hierarchical Head
    console.print("\n[bold yellow][2/2] Training & Evaluating Fine-Tuned Model (MNRL + Hard Negatives)...[/bold yellow]")
    train_title_normalizer(
        samples_per_role=25,
        epochs=3,
        batch_size=32,
        output_dir="checkpoints",
    )

    # Load fine-tuned weights
    fine_tuned_model = HierarchicalTitleNormalizer()
    ckpt_path = "checkpoints/best_title_model.pt"
    if os.path.exists(ckpt_path):
        import torch
        fine_tuned_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        console.print("[green]✓ Loaded best fine-tuned model checkpoint.[/green]")

    ft_engine = TitleNormalizerInference(model=fine_tuned_model)
    ft_metrics = TitleEvaluator.evaluate(ft_engine, test_samples)

    # 4. Print Comparative Ablation Table
    table = Table(title="Ablation Benchmark: Baseline vs Fine-Tuned Title Normalizer")
    table.add_column("Evaluation Metric", style="cyan", no_wrap=True)
    table.add_column("Baseline (Zero-Shot)", style="red")
    table.add_column("Fine-Tuned (MNRL + Hard Negs)", style="bold green")
    table.add_column("Absolute Gain", style="yellow")

    top1_gain = (ft_metrics["top1_accuracy"] - baseline_metrics["top1_accuracy"]) * 100
    top3_gain = (ft_metrics["top3_accuracy"] - baseline_metrics["top3_accuracy"]) * 100
    hard_gain = (ft_metrics["hard_negative_accuracy"] - baseline_metrics["hard_negative_accuracy"]) * 100
    dept_gain = (ft_metrics["department_micro_f1"] - baseline_metrics["department_micro_f1"]) * 100
    sen_gain = (ft_metrics["seniority_macro_f1"] - baseline_metrics["seniority_macro_f1"]) * 100

    table.add_row(
        "Top-1 Canonical Role Accuracy",
        f"{baseline_metrics['top1_accuracy']*100:.2f}%",
        f"{ft_metrics['top1_accuracy']*100:.2f}%",
        f"+{top1_gain:.2f}%",
    )
    table.add_row(
        "Top-3 Canonical Role Accuracy",
        f"{baseline_metrics['top3_accuracy']*100:.2f}%",
        f"{ft_metrics['top3_accuracy']*100:.2f}%",
        f"+{top3_gain:.2f}%",
    )
    table.add_row(
        "Hard Negative Disambiguation Acc",
        f"{baseline_metrics['hard_negative_accuracy']*100:.2f}%",
        f"{ft_metrics['hard_negative_accuracy']*100:.2f}%",
        f"+{hard_gain:.2f}%",
    )
    table.add_row(
        "Department Micro-F1 (12 Classes)",
        f"{baseline_metrics['department_micro_f1']*100:.2f}%",
        f"{ft_metrics['department_micro_f1']*100:.2f}%",
        f"+{dept_gain:.2f}%",
    )
    table.add_row(
        "Seniority Macro-F1 (7 Tiers)",
        f"{baseline_metrics['seniority_macro_f1']*100:.2f}%",
        f"{ft_metrics['seniority_macro_f1']*100:.2f}%",
        f"+{sen_gain:.2f}%",
    )

    console.print("\n")
    console.print(table)

    # Language breakdown table
    lang_table = Table(title="Multilingual Top-1 Accuracy Breakdown by Language")
    lang_table.add_column("Language", style="cyan")
    lang_table.add_column("Baseline (Zero-Shot)", style="red")
    lang_table.add_column("Fine-Tuned (MNRL)", style="bold green")

    for lang in ["en", "fr", "de", "es"]:
        b_acc = baseline_metrics["language_breakdown"].get(lang, {}).get("top1_acc", 0.0) * 100
        f_acc = ft_metrics["language_breakdown"].get(lang, {}).get("top1_acc", 0.0) * 100
        lang_table.add_row(lang.upper(), f"{b_acc:.2f}%", f"{f_acc:.2f}%")

    console.print("\n")
    console.print(lang_table)
    console.print("\n[bold green]✓ Ablation benchmark complete.[/bold green]\n")


if __name__ == "__main__":
    run_ablation()
