"""Cross-Encoder Re-Ranking Engine for High-Precision Stage-Two Neural Retrieval.

Evaluates candidate query-document pairs through full cross-attention:
  score(q, d) = CrossAttention([CLS] + query + [SEP] + doc + [SEP])
Optimized with dynamic padding and batched tensor execution for sub-25ms latency.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import SearchDocument


class CrossEncoderReranker:
    """Neural cross-attention re-ranker for re-scoring top-k candidate documents."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        doc_lookup: Dict[str, SearchDocument],
        top_k: int = 10,
    ) -> Tuple[List[Tuple[str, float]], float]:
        """Re-rank candidate documents using full cross-attention scoring in sub-25ms."""
        if not candidates:
            return [], 0.0

        t0 = time.perf_counter()

        doc_ids = [c[0] for c in candidates if c[0] in doc_lookup]
        if not doc_ids:
            return [], 0.0

        pairs = [[query, doc_lookup[d_id].to_index_text()] for d_id in doc_ids]

        all_scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                scores = logits.cpu().numpy().tolist()
                if isinstance(scores, float):
                    scores = [scores]
                all_scores.extend(scores)

        scored_candidates = list(zip(doc_ids, all_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        rerank_latency_ms = (time.perf_counter() - t0) * 1000.0
        return scored_candidates[:top_k], rerank_latency_ms
