"""Unit Tests for Multilingual Job Title Normalizer & Hierarchy Classifier."""

import pytest
import torch
from src.ontology import (
    CANONICAL_ONTOLOGY,
    DEPARTMENTS,
    SENIORITIES,
    DEP_TO_IDX,
    SEN_TO_IDX,
)
from src.data import TitleDataGenerator, TitleSample
from src.loss import MultipleNegativesRankingLoss, HierarchicalMultiTaskLoss
from src.model import HierarchicalTitleNormalizer
from src.inference import TitleNormalizerInference


def test_ontology_integrity():
    assert len(DEPARTMENTS) == 12
    assert len(SENIORITIES) == 7
    assert len(CANONICAL_ONTOLOGY) > 0

    for role in CANONICAL_ONTOLOGY:
        assert role.department in DEP_TO_IDX
        assert role.seniority in SEN_TO_IDX
        assert len(role.aliases_en) > 0
        assert len(role.aliases_fr) > 0
        assert len(role.aliases_de) > 0
        assert len(role.aliases_es) > 0


def test_data_generation_and_hard_negatives():
    gen = TitleDataGenerator(seed=42)
    corpus = gen.generate_corpus(samples_per_role=5)

    assert len(corpus) > 0
    sample = corpus[0]
    assert isinstance(sample, TitleSample)
    assert sample.language in ["en", "fr", "de", "es"]

    hard_negs = gen.mine_hard_negatives(sample)
    assert isinstance(hard_negs, list)
    assert len(hard_negs) > 0
    # Hard negative should not be identical to sample's canonical title
    assert sample.canonical_title not in hard_negs


def test_mnrl_loss_computation():
    loss_fn = MultipleNegativesRankingLoss(temperature=0.05)

    # Batch of 4 samples, embedding dim 8
    q = torch.randn(4, 8)
    q = torch.nn.functional.normalize(q, p=2, dim=1)

    # Identical positives -> diagonal should have maximum similarity
    p = q.clone()

    loss_no_neg = loss_fn(q, p)
    assert loss_no_neg.item() < 0.1  # Very low loss for perfect alignment

    # Add hard negative orthogonal or opposite
    neg = -q
    loss_with_neg = loss_fn(q, p, neg)
    assert loss_with_neg.item() < 0.1


def test_model_forward_pass():
    model = HierarchicalTitleNormalizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
    model.eval()

    input_ids = torch.randint(0, 1000, (4, 16))
    attention_mask = torch.ones((4, 16), dtype=torch.long)

    with torch.no_grad():
        emb, dept_logits, sen_logits = model(input_ids, attention_mask)

    assert emb.shape == (4, 384)
    assert dept_logits.shape == (4, len(DEPARTMENTS))
    assert sen_logits.shape == (4, len(SENIORITIES))

    # Check normalization
    norms = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_inference_normalization():
    model = HierarchicalTitleNormalizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
    inference = TitleNormalizerInference(
        model=model, model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    result = inference.normalize("Responsable des opérations de revenus", top_k=3)

    assert "top_canonical_role" in result
    assert "predicted_department" in result
    assert "predicted_seniority" in result
    assert len(result["top_k_candidates"]) == 3
