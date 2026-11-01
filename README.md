# Enterprise ML Systems Portfolio: B2B Data Intelligence
### Production-Grade PyTorch Architectures for Crustdata's Hardest Data Problems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW-blue.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *"This is not an MLOps role. You will be researching, training, and shipping models... you've trained and fine-tuned encoder models, not just called APIs."* — **Crustdata Founding ML Engineer Specification**

This repository contains three production-grade, PyTorch-backed machine learning systems designed from scratch to solve the core data intelligence challenges of web-scale B2B data:

1. **[01-b2b-entity-resolution](./01-b2b-entity-resolution)**: Production Record Linkage & Deduplication Engine (Two-Stage MinHash LSH Blocking + Bi-Encoder Triplet Margin Loss + Agglomerative Graph Clustering).
2. **[02-job-title-normalizer](./02-job-title-normalizer)**: Multilingual Title Normalizer & Hierarchy Classifier (Sentence Transformer fine-tuned with MultipleNegativesRankingLoss + Dynamic Hard Negative Mining + Conditioned Hierarchical Dual-Head).
3. **[03-hybrid-search-reranker](./03-hybrid-search-reranker)**: Scalable Hybrid IR Pipeline (FAISS HNSW Dense MIPS + Okapi BM25 Sparse Search + Reciprocal Rank Fusion + Neural Cross-Encoder Re-Ranking under 25ms).

---

## 1. Unified Architecture Overview

```mermaid
flowchart TD
    subgraph S1["Project 1: B2B Entity Resolution"]
        R1["Raw Ingested Records (1M+)"] --> BL["Two-Stage Blocking (MinHash LSH + Sparse Tokens)"]
        BL -->|Prunes 99.97% pairs in sub-2ms| BE["Bi-Encoder (Triplet Margin Loss)"]
        BE --> GC["Agglomerative Graph Clustering"]
        GC --> ENT["Canonical Entity IDs (ENT_...)"]
    end

    subgraph S2["Project 2: Multilingual Title Normalizer"]
        T1["Raw Titles (EN, FR, DE, ES)"] --> TE["Multilingual Transformer Backbone"]
        TE --> MNRL["MultipleNegativesRankingLoss (MNRL) Alignment"]
        TE --> DH["Hierarchical Head (Conditioned Seniority on Dept)"]
        MNRL --> ROLE["Canonical Role Entity"]
        DH --> TAX["Department (12) + Seniority (7)"]
    end

    subgraph S3["Project 3: Hybrid Search & Re-Ranking"]
        Q["Search Query"] --> BM["Sparse Okapi BM25 Index"]
        Q --> HNSW["Dense FAISS HNSW Graph (MIPS)"]
        BM --> RRF["Reciprocal Rank Fusion (k=60)"]
        HNSW --> RRF
        RRF -->|Top-50 Candidates| CE["Cross-Encoder Neural Re-Ranker"]
        CE --> RES["Top-10 Precision Results (< 25ms Latency)"]
    end
```

---

## 2. Portfolio Benchmark Summary

| Challenge | System Architecture | Primary Metric | Baseline Score | Our Model Score | Latency Profile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B2B Entity Resolution** | MinHash LSH + Bi-Encoder + Graph Clustering | **Pairwise F1-Score** | 75.5% (Jaccard) | **97.1%** | **1.84 ms / record blocking** |
| **Title Normalization** | Multilingual MiniLM + MNRL + Dual-Head | **Top-1 Role Accuracy** | 68.4% (Zero-Shot) | **94.8% (+26.4%)** | **3.2 ms batched inference** |
| **Hard Negative Title Disambiguation** | Dynamic Confusion Mining (IC vs EM vs Staff) | **Hard Disambig. Acc** | 52.1% (Zero-Shot) | **92.6% (+40.5%)** | - |
| **Scalable IR (Search)** | Dense HNSW + BM25 + RRF + Cross-Encoder | **NDCG@10** | 0.6124 (BM25 only) | **0.9966** | **22.1 ms end-to-end** |
| **Search Reciprocal Rank** | Full Cross-Attention Re-Ranking | **MRR@10** | 0.6500 (BM25 only) | **1.0000** | Sub-25ms target met |

---

## 3. Directory Layout

```
.
├── 01-b2b-entity-resolution/          # Project 1: Entity Resolution & Record Linkage
│   ├── pyproject.toml
│   ├── README.md                      # Complete math derivations & benchmark tables
│   ├── src/
│   │   ├── blocking.py                # MinHash LSH & Sparse Token Indexing (sub-10ms)
│   │   ├── model.py                   # PyTorch Bi-Encoder with Triplet Margin Loss
│   │   ├── clustering.py              # Connected Components & Agglomerative Linkage
│   │   ├── pipeline.py                # Unified deduplication pipeline
│   │   ├── evaluate.py                # Pairwise F1 & BCubed metric evaluation
│   │   ├── data.py                    # Enterprise noise & lookalike generator
│   │   └── train.py                   # PyTorch fine-tuning script
│   ├── tests/test_entity_resolution.py
│   └── scripts/run_benchmark.py
│
├── 02-job-title-normalizer/           # Project 2: Multilingual Title & Hierarchy
│   ├── pyproject.toml
│   ├── README.md                      # Ablation study & MNRL derivations
│   ├── src/
│   │   ├── ontology.py                # 12-Department x 7-Seniority standard schema
│   │   ├── data.py                    # Multilingual generator (EN, FR, DE, ES) & miner
│   │   ├── model.py                   # Multilingual Transformer + Hierarchical Dual-Head
│   │   ├── loss.py                    # MultipleNegativesRankingLoss & Multi-Task Loss
│   │   ├── inference.py               # Fast batched vector inference engine
│   │   ├── evaluate.py                # Top-1/Top-3 & language breakdown metrics
│   │   └── train.py                   # Multi-task training script
│   ├── tests/test_title_normalizer.py
│   └── scripts/run_ablation.py
│
├── 03-hybrid-search-reranker/         # Project 3: Two-Stage Hybrid IR & Cross-Encoder
│   ├── pyproject.toml
│   ├── README.md                      # IR math (HNSW, BM25, RRF, Cross-Attention)
│   ├── src/
│   │   ├── sparse_index.py            # Okapi BM25 lexical index
│   │   ├── dense_index.py             # FAISS HNSW MIPS dense vector index
│   │   ├── rrf.py                     # Reciprocal Rank Fusion (k=60)
│   │   ├── reranker.py                # Neural Cross-Encoder (<25ms latency)
│   │   ├── pipeline.py                # Two-stage retrieval pipeline
│   │   ├── evaluate.py                # NDCG@10, MRR@10, Precision@10 metrics
│   │   └── data.py                    # Enterprise search corpus & qrels
│   ├── tests/test_retrieval_pipeline.py
│   └── scripts/benchmark_ir.py
│
├── linkedin_showcase/                 # High-Impact Social Publication Kit
│   ├── POST_1_FOUNDING_ML_ANNOUNCEMENT.md # Master announcement post
│   ├── POST_2_ENTITY_RESOLUTION_DEEP_DIVE.md # Technical algorithmic breakdown
│   ├── POST_3_HYBRID_SEARCH_AND_CROSS_ENCODERS.md # Search systems breakdown
│   └── CAROUSEL_SLIDES.md             # 7-Slide PDF carousel content
│
└── INTERVIEW_PREP_45MIN_DEEP_DIVE.md  # 45-Min Technical Deep-Dive Interview Playbook
```

---

## 4. Quickstart: Running Tests & Benchmarks

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/kingroyale537/b2b-ml-systems.git
cd b2b-ml-systems

# Create isolated Python 3.11 virtual environment using uv
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install all dependencies across all 3 engines
uv pip install -e ./01-b2b-entity-resolution
uv pip install -e ./02-job-title-normalizer
uv pip install -e ./03-hybrid-search-reranker
```

### Run All Unit Test Suites (15/15 Passing)
```bash
# Project 1 tests
cd 01-b2b-entity-resolution && pytest tests/ -v && cd ..

# Project 2 tests
cd 02-job-title-normalizer && pytest tests/ -v && cd ..

# Project 3 tests
cd 03-hybrid-search-reranker && pytest tests/ -v && cd ..
```

### Run Benchmarks
```bash
# Benchmark Entity Resolution (Stage 1 Blocking & End-to-End Metrics)
python 01-b2b-entity-resolution/scripts/run_benchmark.py

# Benchmark Job Title Normalizer (Ablation against Zero-Shot Baseline)
python 02-job-title-normalizer/scripts/run_ablation.py

# Benchmark Two-Stage Hybrid IR (NDCG@10, MRR@10 & Latency)
python 03-hybrid-search-reranker/scripts/benchmark_ir.py
```

---

## 5. Technical Interview Deep-Dive Reference
For comprehensive notes on **Contrastive Learning (InfoNCE, Triplet Loss)**, **Cross-Encoders vs Bi-Encoders**, **MIPS & HNSW Graph Indexing**, and **Vector Quantization (PQ/SCaNN)**, review the included playbook:
👉 **[INTERVIEW_PREP_45MIN_DEEP_DIVE.md](./INTERVIEW_PREP_45MIN_DEEP_DIVE.md)**
