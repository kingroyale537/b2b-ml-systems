"""Comprehensive Evaluation Suite for Record Linkage & Entity Resolution.

Calculates:
  1. Pairwise Precision, Recall, and F1-score
  2. Blocking Layer Candidate Recall & Reduction Ratio
  3. BCubed Clustering Precision, Recall, and F1
  4. Latency breakdown and records/sec throughput
"""

from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from .data import CompanyRecord


class EntityResolutionEvaluator:
    """Evaluates entity resolution precision, recall, and cluster consistency."""

    @staticmethod
    def get_ground_truth_pairs(records: List[CompanyRecord]) -> Set[Tuple[str, str]]:
        """Extract all ground truth positive duplicate pairs (same canonical_id)."""
        cluster_map: Dict[str, List[str]] = collections.defaultdict(list)
        for r in records:
            cluster_map[r.canonical_id].append(r.id)

        true_pairs: Set[Tuple[str, str]] = set()
        for members in cluster_map.values():
            n = len(members)
            for i in range(n):
                for j in range(i + 1, n):
                    p = (members[i], members[j]) if members[i] < members[j] else (members[j], members[i])
                    true_pairs.add(p)
        return true_pairs

    @classmethod
    def evaluate_blocking(
        cls, records: List[CompanyRecord], candidate_pairs: Set[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Compute Candidate Recall and Reduction Ratio of the blocking layer."""
        true_pairs = cls.get_ground_truth_pairs(records)
        n = len(records)
        all_possible_pairs = (n * (n - 1)) / 2

        if not true_pairs:
            return {"candidate_recall": 1.0, "reduction_ratio": 1.0, "total_true_pairs": 0}

        true_positives_in_candidates = len(candidate_pairs.intersection(true_pairs))
        candidate_recall = true_positives_in_candidates / float(len(true_pairs))
        reduction_ratio = 1.0 - (len(candidate_pairs) / max(all_possible_pairs, 1.0))

        return {
            "candidate_recall": candidate_recall,
            "reduction_ratio": reduction_ratio,
            "num_candidates": len(candidate_pairs),
            "num_true_pairs": len(true_pairs),
            "captured_true_pairs": true_positives_in_candidates,
        }

    @classmethod
    def evaluate_pairwise(
        cls,
        records: List[CompanyRecord],
        predicted_pairs: Set[Tuple[str, str]],
    ) -> Dict[str, float]:
        """Calculate pairwise Precision, Recall, and F1-score."""
        true_pairs = cls.get_ground_truth_pairs(records)

        tp = len(predicted_pairs.intersection(true_pairs))
        fp = len(predicted_pairs - true_pairs)
        fn = len(true_pairs - predicted_pairs)

        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    @staticmethod
    def evaluate_bcubed(
        records: List[CompanyRecord], record_to_cluster: Dict[str, str]
    ) -> Dict[str, float]:
        """Calculate BCubed Precision, Recall, and F1 across cluster partitions.

        BCubed penalizes cluster fragmentation and impure mergers at the item level.
        """
        ground_truth: Dict[str, Set[str]] = collections.defaultdict(set)
        for r in records:
            ground_truth[r.canonical_id].add(r.id)

        predicted_clusters: Dict[str, Set[str]] = collections.defaultdict(set)
        for r in records:
            cid = record_to_cluster.get(r.id, f"SOLO_{r.id}")
            predicted_clusters[cid].add(r.id)

        precisions: List[float] = []
        recalls: List[float] = []

        rec_to_true = {r.id: ground_truth[r.canonical_id] for r in records}
        rec_to_pred = {r.id: predicted_clusters[record_to_cluster.get(r.id, f"SOLO_{r.id}")] for r in records}

        for r in records:
            c_true = rec_to_true[r.id]
            c_pred = rec_to_pred[r.id]
            common = len(c_true.intersection(c_pred))
            precisions.append(common / float(len(c_pred)))
            recalls.append(common / float(len(c_true)))

        bcubed_precision = float(np.mean(precisions))
        bcubed_recall = float(np.mean(recalls))
        bcubed_f1 = (
            2 * (bcubed_precision * bcubed_recall) / max(bcubed_precision + bcubed_recall, 1e-9)
        )

        return {
            "bcubed_precision": bcubed_precision,
            "bcubed_recall": bcubed_recall,
            "bcubed_f1": bcubed_f1,
        }
