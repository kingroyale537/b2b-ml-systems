"""End-to-End Enterprise Entity Resolution Pipeline.

Combines:
  1. Two-Stage Blocking Layer (Sub-10ms pruning)
  2. Fine-Tuned PyTorch Bi-Encoder Matcher (Dense semantic similarity)
  3. Graph Clustering (Global Entity ID assignment & canonical profile synthesis)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from .blocking import TwoStageBlocker
from .clustering import EntityGraphClustering
from .data import CompanyRecord
from .model import CompanyBiEncoder


class EntityResolutionPipeline:
    """Production Entity Resolution Engine for B2B deduplication at scale."""

    def __init__(
        self,
        model: Optional[CompanyBiEncoder] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        blocking_threshold: float = 0.35,
        match_threshold: float = 0.75,
        max_candidates_per_record: int = 20,
        batch_size: int = 64,
    ) -> None:
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        )
        self.model = (
            model
            if model is not None
            else CompanyBiEncoder("sentence-transformers/all-MiniLM-L6-v2")
        )
        self.model.to(self.device)
        self.model.eval()

        self.blocker = TwoStageBlocker(
            threshold=blocking_threshold,
            max_candidates_per_record=max_candidates_per_record,
        )
        self.clusterer = EntityGraphClustering(match_threshold=match_threshold)
        self.match_threshold = match_threshold
        self.batch_size = batch_size

    def score_candidate_pairs(
        self, records: List[CompanyRecord], candidate_pairs: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str, float]], float]:
        """Score candidate pairs using the Transformer Bi-Encoder in batches."""
        if not candidate_pairs:
            return [], 0.0

        record_map = {r.id: r for r in records}
        scored_pairs: List[Tuple[str, str, float]] = []

        t0 = time.perf_counter()

        with torch.no_grad():
            for i in range(0, len(candidate_pairs), self.batch_size):
                batch = candidate_pairs[i : i + self.batch_size]
                texts_a = [record_map[id_a].to_representation() for id_a, _ in batch]
                texts_b = [record_map[id_b].to_representation() for _, id_b in batch]

                inputs_a = self.tokenizer(
                    texts_a, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)
                inputs_b = self.tokenizer(
                    texts_b, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)

                emb_a = self.model.encode(inputs_a["input_ids"], inputs_a["attention_mask"])
                emb_b = self.model.encode(inputs_b["input_ids"], inputs_b["attention_mask"])

                # Cosine similarity between normalized vectors
                sims = torch.sum(emb_a * emb_b, dim=1).cpu().numpy()

                for (id_a, id_b), sim in zip(batch, sims):
                    scored_pairs.append((id_a, id_b, float(sim)))

        scoring_time = time.perf_counter() - t0
        return scored_pairs, scoring_time

    def run(
        self, records: List[CompanyRecord]
    ) -> Dict[str, Any]:
        """Execute complete deduplication workflow on incoming batch of records."""
        total_t0 = time.perf_counter()

        # Step 1: Blocking
        self.blocker.fit_index(records)
        candidate_pairs_set, blocking_stats = self.blocker.get_candidate_pairs(records)
        candidate_pairs = list(candidate_pairs_set)

        # Step 2: Bi-Encoder Scoring
        scored_pairs, scoring_time = self.score_candidate_pairs(records, candidate_pairs)

        # Step 3: Graph Clustering
        t_cluster_start = time.perf_counter()
        record_to_cluster = self.clusterer.cluster(records, scored_pairs)
        canonical_profiles = self.clusterer.synthesize_canonical_profiles(
            records, record_to_cluster
        )
        clustering_time = time.perf_counter() - t_cluster_start

        total_time = time.perf_counter() - total_t0

        return {
            "record_to_cluster": record_to_cluster,
            "canonical_profiles": canonical_profiles,
            "scored_pairs": scored_pairs,
            "metrics": {
                "total_records": len(records),
                "candidate_pairs_evaluated": len(candidate_pairs),
                "resolved_canonical_entities": len(canonical_profiles),
                "blocking_stats": blocking_stats,
                "scoring_time_sec": scoring_time,
                "clustering_time_sec": clustering_time,
                "total_pipeline_time_sec": total_time,
                "throughput_records_per_sec": len(records) / max(total_time, 1e-6),
            },
        }
