"""Inference Engine for Fast Job Title Normalization & Ontology Mapping."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .model import HierarchicalTitleNormalizer
from .ontology import (
    CANONICAL_ONTOLOGY,
    CanonicalRole,
    DEPARTMENTS,
    IDX_TO_DEP,
    IDX_TO_SEN,
    SENIORITIES,
)


class TitleNormalizerInference:
    """Production inference engine for mapping raw titles to canonical taxonomy."""

    def __init__(
        self,
        model: Optional[HierarchicalTitleNormalizer] = None,
        tokenizer: Optional[Any] = None,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
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
            else AutoTokenizer.from_pretrained(model_name)
        )
        self.model = (
            model
            if model is not None
            else HierarchicalTitleNormalizer(model_name=model_name)
        )
        self.model.to(self.device)
        self.model.eval()

        # Pre-compute canonical role embeddings for ultra-fast vector search
        self.canonical_roles = CANONICAL_ONTOLOGY
        self.canonical_titles = [r.canonical_title for r in self.canonical_roles]
        self.canonical_embeddings = self._index_canonical_roles()

    def _index_canonical_roles(self) -> torch.Tensor:
        """Pre-encode all canonical role titles into memory."""
        with torch.no_grad():
            inputs = self.tokenizer(
                self.canonical_titles,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)
            embeddings = self.model.encode(inputs["input_ids"], inputs["attention_mask"])
        return embeddings

    def normalize(
        self, raw_title: str, top_k: int = 3
    ) -> Dict[str, Any]:
        """Normalize a single raw job title into its canonical role, department, and seniority."""
        results = self.normalize_batch([raw_title], top_k=top_k)
        return results[0]

    def normalize_batch(
        self, raw_titles: List[str], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Normalize a batch of raw titles with batched tensor operations."""
        if not raw_titles:
            return []

        with torch.no_grad():
            inputs = self.tokenizer(
                raw_titles,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)

            query_embeddings, dept_logits, sen_logits = self.model(
                inputs["input_ids"], inputs["attention_mask"]
            )

            # Cosine similarity matrix against pre-indexed canonical roles: [batch_size, num_canonical]
            sim_matrix = torch.matmul(query_embeddings, self.canonical_embeddings.T)
            top_scores, top_indices = torch.topk(sim_matrix, k=min(top_k, len(self.canonical_titles)), dim=-1)

            dept_probs = F.softmax(dept_logits, dim=-1)
            sen_probs = F.softmax(sen_logits, dim=-1)

            pred_dept_indices = torch.argmax(dept_probs, dim=-1).cpu().numpy()
            pred_dept_conf = torch.max(dept_probs, dim=-1).values.cpu().numpy()

            pred_sen_indices = torch.argmax(sen_probs, dim=-1).cpu().numpy()
            pred_sen_conf = torch.max(sen_probs, dim=-1).values.cpu().numpy()

            top_scores_np = top_scores.cpu().numpy()
            top_indices_np = top_indices.cpu().numpy()

        outputs = []
        for i, title in enumerate(raw_titles):
            matches = [
                {
                    "canonical_title": self.canonical_titles[idx],
                    "similarity_score": float(score),
                    "department": self.canonical_roles[idx].department,
                    "seniority": self.canonical_roles[idx].seniority,
                }
                for idx, score in zip(top_indices_np[i], top_scores_np[i])
            ]

            outputs.append({
                "raw_title": title,
                "top_canonical_role": matches[0]["canonical_title"],
                "role_confidence": matches[0]["similarity_score"],
                "predicted_department": IDX_TO_DEP[int(pred_dept_indices[i])],
                "department_confidence": float(pred_dept_conf[i]),
                "predicted_seniority": IDX_TO_SEN[int(pred_sen_indices[i])],
                "seniority_confidence": float(pred_sen_conf[i]),
                "top_k_candidates": matches,
            })

        return outputs
