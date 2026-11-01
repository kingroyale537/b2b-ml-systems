"""Hierarchical Multilingual Job Title Normalizer Model.

Combines a Multilingual Transformer encoder backbone with dual hierarchical
classification heads:
  1. Department Head: P(Department | Title)
  2. Seniority Head: P(Seniority | Title, Department)
  3. Metric Projection: L2-normalized embedding for similarity search against canonical roles.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .ontology import DEPARTMENTS, SENIORITIES


class HierarchicalTitleNormalizer(nn.Module):
    """Multilingual Transformer with Hierarchical Multi-Label Classification Heads."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        num_departments: int = len(DEPARTMENTS),
        num_seniorities: int = len(SENIORITIES),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Department classification head
        self.dept_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_departments),
        )

        # Seniority classification head conditioned on token embedding + department soft distribution
        self.sen_head = nn.Sequential(
            nn.Linear(hidden_dim + num_departments, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_seniorities),
        )

    def mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode raw title into L2-normalized dense metric embedding."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs, attention_mask)
        return F.normalize(pooled, p=2, dim=1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing normalized embedding, department logits, and seniority logits."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs, attention_mask)
        normalized_emb = F.normalize(pooled, p=2, dim=1)

        dropped = self.dropout(pooled)
        dept_logits = self.dept_head(dropped)

        # Condition seniority head on department probability distribution
        dept_probs = F.softmax(dept_logits, dim=-1)
        sen_input = torch.cat([dropped, dept_probs], dim=-1)
        sen_logits = self.sen_head(sen_input)

        return normalized_emb, dept_logits, sen_logits
