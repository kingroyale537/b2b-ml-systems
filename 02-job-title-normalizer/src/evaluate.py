"""Evaluation and Ablation Suite for Job Title Normalization.

Benchmarks:
  1. Top-1 and Top-3 Accuracy on Canonical Roles
  2. Language-Specific Breakdown (EN, FR, DE, ES)
  3. Hard Negative Disambiguation (e.g. Software Engineer vs Staff Engineer vs EM)
  4. Department Micro-F1 & Seniority Macro-F1.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score

from .data import CONFUSION_CLUSTERS, TitleSample
from .inference import TitleNormalizerInference


class TitleEvaluator:
    """Evaluates title normalization, ranking accuracy, and hierarchical classification."""

    @staticmethod
    def evaluate(
        inference_engine: TitleNormalizerInference, test_samples: List[TitleSample]
    ) -> Dict[str, Any]:
        """Compute evaluation metrics across test samples."""
        if not test_samples:
            return {}

        raw_titles = [s.raw_title for s in test_samples]
        predictions = inference_engine.normalize_batch(raw_titles, top_k=3)

        top1_correct = 0
        top3_correct = 0
        dept_correct = 0
        sen_correct = 0

        # Per-language stats
        lang_stats: Dict[str, Dict[str, int]] = collections.defaultdict(
            lambda: {"total": 0, "top1": 0, "top3": 0}
        )

        # Hard negative confusion stats
        hard_confusion_titles = {
            item[0] for cluster in CONFUSION_CLUSTERS for item in cluster
        }
        hard_total = 0
        hard_correct = 0

        true_depts = []
        pred_depts = []
        true_sens = []
        pred_sens = []

        for sample, pred in zip(test_samples, predictions):
            cand_titles = [c["canonical_title"] for c in pred["top_k_candidates"]]

            is_top1 = (pred["top_canonical_role"] == sample.canonical_title)
            is_top3 = (sample.canonical_title in cand_titles)

            if is_top1:
                top1_correct += 1
            if is_top3:
                top3_correct += 1

            if pred["predicted_department"] == sample.department:
                dept_correct += 1
            if pred["predicted_seniority"] == sample.seniority:
                sen_correct += 1

            # Language breakdown
            lang = sample.language
            lang_stats[lang]["total"] += 1
            if is_top1:
                lang_stats[lang]["top1"] += 1
            if is_top3:
                lang_stats[lang]["top3"] += 1

            # Hard confusion subset
            if sample.canonical_title in hard_confusion_titles:
                hard_total += 1
                if is_top1:
                    hard_correct += 1

            true_depts.append(sample.department)
            pred_depts.append(pred["predicted_department"])
            true_sens.append(sample.seniority)
            pred_sens.append(pred["predicted_seniority"])

        n = len(test_samples)
        top1_acc = top1_correct / float(n)
        top3_acc = top3_correct / float(n)
        dept_acc = dept_correct / float(n)
        sen_acc = sen_correct / float(n)

        dept_micro_f1 = float(f1_score(true_depts, pred_depts, average="micro"))
        sen_macro_f1 = float(f1_score(true_sens, pred_sens, average="macro"))

        lang_breakdown = {}
        for l, counts in lang_stats.items():
            tot = max(counts["total"], 1)
            lang_breakdown[l] = {
                "top1_acc": counts["top1"] / float(tot),
                "top3_acc": counts["top3"] / float(tot),
                "count": counts["total"],
            }

        hard_acc = hard_correct / max(hard_total, 1)

        return {
            "num_test_samples": n,
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
            "department_accuracy": dept_acc,
            "department_micro_f1": dept_micro_f1,
            "seniority_accuracy": sen_acc,
            "seniority_macro_f1": sen_macro_f1,
            "hard_negative_accuracy": hard_acc,
            "language_breakdown": lang_breakdown,
        }
