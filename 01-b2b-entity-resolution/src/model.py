"""PyTorch Bi-Encoder Matcher for Record Linkage.

Encodes structured company representations into a dense d-dimensional metric space
optimized with Triplet Loss and Contrastive Margins:
  u = Normalize(MeanPool(Transformer(record_A)))
  v = Normalize(MeanPool(Transformer(record_B)))
  Similarity = u @ v.T
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class CompanyBiEncoder(nn.Module):
    """Transformer Bi-Encoder for learning dense company entity representations."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        freeze_backbone_layers: int = 0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_backbone_layers > 0 and hasattr(self.encoder, "encoder"):
            for layer in self.encoder.encoder.layer[:freeze_backbone_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Calibrated match probability projection parameters: P(match) = sigmoid(scale * cos_sim + bias)
        self.scale = nn.Parameter(torch.tensor([10.0]))
        self.bias = nn.Parameter(torch.tensor([-5.0]))

    def mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply attention-masked mean pooling over token embeddings."""
        token_embeddings = model_output[0]  # First element of model_output contains hidden state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through encoder returning L2-normalized unit embeddings."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs, attention_mask)
        # Normalize to unit hypersphere S^(d-1) for fast inner product search
        normalized = F.normalize(pooled, p=2, dim=1)
        return normalized

    def forward(
        self,
        anchor_ids: torch.Tensor,
        anchor_mask: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute embeddings and predicted match probability for a batch of pairs."""
        u = self.encode(anchor_ids, anchor_mask)
        v = self.encode(candidate_ids, candidate_mask)
        cosine_sim = torch.sum(u * v, dim=1)  # Dot product of normalized vectors
        logits = self.scale * cosine_sim + self.bias
        match_prob = torch.sigmoid(logits)
        return cosine_sim, match_prob


class TripletMarginLossWithHardNegatives(nn.Module):
    """Triplet Margin Loss over normalized embeddings with Euclidean or Cosine distance.

    L(a, p, n) = max(0, D(a, p) - D(a, n) + margin)
    """

    def __init__(self, margin: float = 0.35, distance_metric: str = "cosine") -> None:
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.distance_metric == "cosine":
            # Cosine distance = 1 - cos_sim
            dist_pos = 1.0 - torch.sum(anchor_emb * positive_emb, dim=1)
            dist_neg = 1.0 - torch.sum(anchor_emb * negative_emb, dim=1)
        else:
            # Squared Euclidean distance
            dist_pos = torch.sum((anchor_emb - positive_emb) ** 2, dim=1)
            dist_neg = torch.sum((anchor_emb - negative_emb) ** 2, dim=1)

        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return torch.mean(loss)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for pair verification:

    L = y * D^2 + (1 - y) * max(0, margin - D)^2
    """

    def __init__(self, margin: float = 0.50) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self, emb1: torch.Tensor, emb2: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        euclidean_dist = F.pairwise_distance(emb1, emb2)
        loss = labels * torch.pow(euclidean_dist, 2) + (1.0 - labels) * torch.pow(
            torch.clamp(self.margin - euclidean_dist, min=0.0), 2
        )
        return torch.mean(loss)
