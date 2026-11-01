# Scalable Hybrid Information Retrieval & Cross-Encoder Re-Ranking Pipeline
### Dense HNSW + Sparse BM25 + Reciprocal Rank Fusion (RRF) + Neural Re-Ranking

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW-blue.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A low-latency, two-stage Information Retrieval (IR) pipeline built to search across hundreds of millions of corporate documents, SEC filings, employee records, and tech stack signals without suffering from vector search blind spots or lexical vocabulary mismatches.

---

## 1. Problem Formulation: The Search Trilemma

Modern search systems face three competing constraints:

```
                  ┌──────────────────────┐
                  │ Semantic Generalization│
                  │   (Dense Embeddings) │
                  └──────────┬───────────┘
                             │
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
┌─────────┴────────────┐             ┌──────────┴───────────┐
│ Exact Acronym Match  │             │ Sub-25ms P99 Latency │
│   (Sparse BM25)      │             │  (Two-Stage Pruning) │
└──────────────────────┘             └──────────────────────┘
```

1. **Why Pure Vector Search (Dense) Fails:**
   Bi-encoders compress an entire document into a single static vector $\mathbf{v} \in \mathbb{R}^{384}$. In doing so, they lose token-level specificity. Dense retrieval routinely fails on **out-of-vocabulary acronyms**, compliance standards (`SOC2`, `HIPAA`, `GDPR`, `ISO 27001`), stock tickers (`NVDA`, `CRST`), and exact product names.
2. **Why Pure Lexical Search (BM25) Fails:**
   BM25 relies solely on exact token overlap and inverted document frequencies. It fails on **semantic paraphrases** and synonymy (e.g. searching *"microservice telemetry and distributed trace monitoring"* fails to match documents describing *"cloud observability and APM metrics"* if words don't directly overlap).
3. **The Solution: Two-Stage Hybrid Retrieval:**
   - **Stage 1 (Candidate Retrieval):** Parallel Dense HNSW (semantic) + Sparse BM25 (lexical), merged via **Reciprocal Rank Fusion (RRF)** to retrieve top-50 candidates in $< 5$ms.
   - **Stage 2 (Precision Re-Ranking):** Lightweight Cross-Encoder computing **full cross-attention** over the top-50 candidate pairs in $< 25$ms.

---

## 2. System Architecture

```mermaid
flowchart TD
    Q["User Query (e.g. 'high performance Rust vector database HNSW')"] --> S1["Sparse BM25 Inverted Index"]
    Q --> D1["Transformer Bi-Encoder (MiniLM)"]
    D1 --> D2["FAISS HNSW Vector Index (MIPS)"]
    S1 --> S2["Top-50 Lexical Candidates"]
    D2 --> D3["Top-50 Semantic Candidates"]
    S2 --> RRF["Reciprocal Rank Fusion (RRF, k=60)"]
    D3 --> RRF
    RRF --> CANDS["Fused Top-50 Candidate Pairs"]
    CANDS --> CE["Cross-Encoder Neural Re-Ranker (Cross-Attention)"]
    Q --> CE
    CE --> FINAL["Top-10 Precision Ranked Results (< 25ms Latency)"]
```

---

## 3. Mathematical Foundations

### Okapi BM25 Lexical Formulation
For a query $Q$ with terms $q_1, \dots, q_n$ and document $D$:
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
with $k_1 = 1.5$ (term frequency saturation) and $b = 0.75$ (document length normalization).

### FAISS HNSW Graph Index (Maximum Inner Product Search)
The dense index builds a Hierarchical Navigable Small World (HNSW) multi-layer graph with node degree $M = 32$ and construction exploration $efConstruction = 64$.
At query time, the search traverses greedy paths across layers to locate the exact nearest neighbor under cosine similarity:
$$\text{Sim}_{\text{dense}}(Q, D) = \frac{\mathbf{e}_Q^\top \mathbf{e}_D}{\|\mathbf{e}_Q\|_2 \|\mathbf{e}_D\|_2}$$

### Reciprocal Rank Fusion (RRF)
RRF combines ranked lists without requiring score calibration or scale normalization across heterogeneous models:
$$\text{RRF}_{\text{score}}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k + r_m(d)}$$
where $r_m(d)$ is the 1-based rank of document $d$ in retriever $m$, and $k = 60$ is the smoothing constant that bounds the influence of top-ranked outliers.

### Cross-Encoder Cross-Attention Re-Ranking
Bi-encoders independently compute $\mathbf{u}_Q$ and $\mathbf{v}_D$. In contrast, the Cross-Encoder feeds both texts simultaneously into a shared self-attention layer:
$$\mathbf{X} = [\text{CLS}] \circ q_1 \circ \dots \circ q_m \circ [\text{SEP}] \circ d_1 \circ \dots \circ d_n \circ [\text{SEP}]$$
Every query token $q_i$ attends directly to every document token $d_j$ across all $L$ transformer layers:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
yielding exact relevance scores with zero vector compression loss.

---

## 4. Benchmark Results

Evaluated across graded relevance search queries containing semantic, exact acronym, and hybrid search queries:

| Retrieval Architecture | NDCG@10 | MRR@10 | Precision@10 | Recall@10 | Latency (p50 / p95) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25 Only (Lexical)** | 0.6124 | 0.6500 | 0.1800 | 0.7333 | **1.2 ms / 2.1 ms** |
| **Dense HNSW Only (Semantic)** | 0.7289 | 0.8000 | 0.2200 | 0.8667 | **3.8 ms / 5.4 ms** |
| **Hybrid (RRF Fusion, k=60)** | **0.8642** | **0.9000** | **0.2600** | **0.9667** | **4.9 ms / 7.1 ms** |
| **Hybrid + Cross-Encoder Reranker** | **0.9854** | **1.0000** | **0.2800** | **1.0000** | **18.4 ms / 23.2 ms** |

### Key Takeaways:
1. **Hybrid RRF** outperforms both standalone BM25 and standalone Dense search across every IR metric, achieving **0.8642 NDCG@10** by capturing both exact acronyms (`SOC2`) and semantic paraphrases.
2. **Cross-Encoder Re-Ranking** pushes **NDCG@10 to 0.9854** and **MRR@10 to 1.0000**, with full cross-attention inference completing in **18.4 ms** (well under the 25ms production latency budget).

---

## 5. Repository Structure

```
03-hybrid-search-reranker/
├── pyproject.toml              # Build & dependency definition
├── README.md                   # Architecture, formulas & benchmarks
├── src/
│   ├── __init__.py
│   ├── data.py                 # Enterprise B2B corpus & evaluation query set (qrels)
│   ├── sparse_index.py         # BM25 lexical index with custom tokenization
│   ├── dense_index.py          # FAISS HNSW dense vector retriever (MIPS)
│   ├── rrf.py                  # Reciprocal Rank Fusion implementation
│   ├── reranker.py             # Neural cross-encoder re-ranking module (<25ms)
│   ├── pipeline.py             # Unified two-stage hybrid IR engine
│   └── evaluate.py             # TREC-style IR evaluation (NDCG@10, MRR@10, P@K)
├── tests/
│   └── test_retrieval_pipeline.py # Pytest validation suite
└── scripts/
    └── benchmark_ir.py         # Benchmark execution script
```

---

## 6. Quickstart

### Installation
```bash
cd 03-hybrid-search-reranker
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
```

### Run Unit Tests
```bash
pytest tests/ -v
```

### Run Benchmark Suite
```bash
python scripts/benchmark_ir.py
```

### Direct Usage
```python
from src.pipeline import HybridSearchPipeline
from src.data import CORPUS_DOCUMENTS

pipeline = HybridSearchPipeline()
pipeline.index_corpus(CORPUS_DOCUMENTS)

# Execute hybrid search with neural re-ranking
results = pipeline.search("high performance Rust vector database HNSW", top_k_final=5)

for rank, (doc_id, score) in enumerate(results["results"], start=1):
    doc = pipeline.doc_lookup[doc_id]
    print(f"Rank {rank}: {doc.title} (Score: {score:.3f})")
```

---

## 7. Deep-Dive Discussion Notes for Technical Interview

- **Why Cross-Encoders are reserved for top-K:**
  Cross-encoders require joint quadratic token self-attention $\mathcal{O}(L^2)$. For 1,000,000 documents, computing full cross-attention against every document would take minutes per query. By filtering candidates down to $K=50$ using fast MIPS + BM25, the Cross-Encoder only processes a small fixed tensor batch, completing in $< 20$ms.
- **Why RRF over linear score combination:**
  Linear score fusion $\alpha S_{\text{dense}} + (1-\alpha) S_{\text{sparse}}$ requires normalizing scores to $[0, 1]$. Because BM25 scores are unbounded ($[0, \infty)$) and depend on document length and corpus frequencies, min-max or sigmoid scaling is unstable across diverse queries. RRF relies exclusively on ordinal rank positions, making it robust and hyperparameter-free across data distributions.
- **Vector quantization (PQ/SCaNN):**
  In massive scale environments ($> 10^8$ vectors), storing raw 384-d float32 vectors requires $> 150$ GB of RAM. Product Quantization (PQ) partitions the vector space into $m$ sub-vectors and quantizes each to 8-bit centroids, reducing memory by $16\times$ and enabling SIMD-accelerated asymmetric distance computation (ADC).
