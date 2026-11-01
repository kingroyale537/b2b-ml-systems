"""Loss Functions for Job Title Representation & Hierarchical Classification.

Implements:
  1. MultipleNegativesRankingLoss (MNRL) with explicit hard negatives
  2. Hierarchical Multi-Task Cross-Entropy Loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    """Multiple Negatives Ranking Loss (MNRL) with in-batch and explicit hard negatives.

    For batch of (query, positive, [hard_negative]):
      Similarity matrix S_ij = (u_i @ v_j) / tau
      Loss = CrossEntropy(S, labels = [0, 1, 2, ..., B-1])
    """

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """Args:

        query_emb: [batch_size, dim] normalized query embeddings pos_emb:
        [batch_size, dim] normalized positive embeddings neg_emb: [batch_size,
        dim] optional normalized hard negative embeddings
        """
        # In-batch dot product: [batch_size, batch_size]
        scores = torch.matmul(query_emb, pos_emb.T) / self.temperature

        if neg_emb is not None:
            # Diagonal dot product for hard negatives: [batch_size, 1]
            neg_scores = torch.sum(query_emb * neg_emb, dim=1, keepdim=True) / self.temperature
            # Concatenate hard negatives as additional column: [batch_size, batch_size + 1]
            scores = torch.cat([scores, neg_scores], dim=1)

        labels = torch.arange(len(query_emb), device=query_emb.device, dtype=torch.long)
        return self.cross_entropy(scores, labels)


class HierarchicalMultiTaskLoss(nn.Module):
    """Joint multi-task loss combining metric embedding alignment with hierarchical classification."""

    def __init__(
        self,
        temperature: float = 0.05,
        weight_mnrl: float = 1.0,
        weight_dep: float = 0.5,
        weight_sen: float = 0.5,
    ) -> None:
        super().__init__()
        self.mnrl = MultipleNegativesRankingLoss(temperature=temperature)
        self.ce_dep = nn.CrossEntropyLoss()
        self.ce_sen = nn.CrossEntropyLoss()
        self.w_mnrl = weight_mnrl
        self.w_dep = weight_dep
        self.w_sen = weight_sen

    def forward(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        dep_logits: torch.Tensor,
        sen_logits: torch.Tensor,
        target_dep: torch.Tensor,
        target_sen: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_mnrl = self.mnrl(query_emb, pos_emb, neg_emb)
        loss_dep = self.ce_dep(dep_logits, target_dep)
        loss_sen = self.ce_sen(sen_logits, target_sen)

        total_loss = (
            self.w_mnrl * loss_mnrl
            + self.w_dep * loss_dep
            + self.w_sen * loss_sen
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_mnrl": loss_mnrl.item(),
            "loss_dep": loss_dep.item(),
            "loss_sen": loss_sen.item(),
        }
        return total_loss, metrics
