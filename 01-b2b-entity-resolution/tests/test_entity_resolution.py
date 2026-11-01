"""Unit Tests for B2B Entity Resolution & Record Linkage Engine."""

import pytest
import torch
from src.data import B2BDataGenerator, CompanyRecord
from src.blocking import MinHashLSHBlocker, SparseTokenBlocker, TwoStageBlocker
from src.model import CompanyBiEncoder, TripletMarginLossWithHardNegatives
from src.clustering import EntityGraphClustering
from src.evaluate import EntityResolutionEvaluator


@pytest.fixture
def sample_records():
    gen = B2BDataGenerator(seed=42)
    records, clusters = gen.generate_corpus(num_canonical=10, duplicates_per_entity=3)
    return records, clusters


def test_minhash_lsh_blocking(sample_records):
    records, clusters = sample_records
    lsh = MinHashLSHBlocker(num_perm=32, threshold=0.35)
    lsh.index(records)

    # Test candidate query for first record
    query_rec = records[0]
    candidates = lsh.query_candidates(query_rec, max_candidates=10)
    assert isinstance(candidates, list)
    assert len(candidates) <= 10
    # Cand IDs should not contain the query record itself
    assert query_rec.id not in [c[0] for c in candidates]


def test_two_stage_blocking_efficiency(sample_records):
    records, _ = sample_records
    blocker = TwoStageBlocker(num_perm=32, threshold=0.30, max_candidates_per_record=15)
    blocker.fit_index(records)
    candidate_pairs, stats = blocker.get_candidate_pairs(records)

    assert len(candidate_pairs) > 0
    assert stats["reduction_ratio"] > 0.85
    assert stats["latency_per_record_ms"] < 20.0  # Sub-20ms per record in python uncompiled


def test_biencoder_embedding_normalization():
    model = CompanyBiEncoder("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()

    # Dummy inputs
    input_ids = torch.randint(0, 1000, (4, 16))
    attention_mask = torch.ones((4, 16), dtype=torch.long)

    with torch.no_grad():
        emb = model.encode(input_ids, attention_mask)

    # Check shape [batch_size, embedding_dim]
    assert emb.shape == (4, 384)
    # Check L2 unit normalization: ||emb||_2 == 1.0
    norms = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_triplet_margin_loss():
    loss_fn = TripletMarginLossWithHardNegatives(margin=0.3, distance_metric="cosine")

    # Anchor and Positive identical, Negative orthogonal
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    n = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    loss = loss_fn(a, p, n)
    # D(a, p) = 0, D(a, n) = 1.0 -> 0 - 1.0 + 0.3 = -0.7 -> clamped to 0.0
    assert loss.item() == 0.0


def test_graph_clustering():
    records = [
        CompanyRecord("rec_1", "CAN_1", "Stripe Inc", "stripe.com", "SF", "Fintech", ""),
        CompanyRecord("rec_2", "CAN_1", "Stripe LLC", "stripe.com", "San Francisco", "Fintech", ""),
        CompanyRecord("rec_3", "CAN_2", "Datadog HQ", "datadog.com", "NYC", "Monitoring", ""),
    ]

    scored_pairs = [
        ("rec_1", "rec_2", 0.95),  # Strong match
        ("rec_1", "rec_3", 0.12),  # Distant non-match
    ]

    clusterer = EntityGraphClustering(match_threshold=0.70)
    mapping = clusterer.cluster(records, scored_pairs)

    assert mapping["rec_1"] == mapping["rec_2"]
    assert mapping["rec_1"] != mapping["rec_3"]

    profiles = clusterer.synthesize_canonical_profiles(records, mapping)
    assert len(profiles) == 2
