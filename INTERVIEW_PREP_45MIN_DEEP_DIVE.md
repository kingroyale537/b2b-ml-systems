# Crustdata Founding ML Engineer: 45-Min Technical Deep-Dive Master Guide
### ML Systems Design, Encoder Architectures & Mathematical Foundations

This document is your preparation playbook for Crustdata's **45-minute technical deep-dive (ML systems design + paper discussion)** with the Founders / Tech Leads. It covers every concept, trade-off, mathematical formulation, and architecture pattern required to excel in the interview.

---

## 1. Contrastive Learning & Metric Learning Mechanics

### Mathematical Formulations

#### 1. Triplet Margin Loss
Given an anchor $\mathbf{x}_a$, a positive $\mathbf{x}_p$, and a negative $\mathbf{x}_n$:
$$\mathcal{L}_{\text{triplet}}(a, p, n) = \max\left(0, D(\mathbf{x}_a, \mathbf{x}_p) - D(\mathbf{x}_a, \mathbf{x}_n) + \alpha\right)$$
where $D(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ (cosine distance) and $\alpha > 0$ is the minimum enforcement margin.

- **Easy Triplets:** $D(a, n) > D(a, p) + \alpha \implies \mathcal{L} = 0$. (Provide no gradient; waste compute).
- **Hard Triplets:** $D(a, n) < D(a, p) \implies$ Negative is closer than the positive. (High gradient, can destabilize early training).
- **Semi-Hard Triplets:** $D(a, p) < D(a, n) < D(a, p) + \alpha \implies$ Negative is farther than positive, but within the safety margin $\alpha$. (Most stable gradient signal for fine-tuning).

#### 2. InfoNCE / MultipleNegativesRankingLoss (MNRL)
For a batch of $B$ pairs $\{(\mathbf{q}_i, \mathbf{p}_i)\}_{i=1}^B$:
$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp\left(\frac{\mathbf{q}_i^\top \mathbf{p}_i}{\tau}\right)}{\sum_{j=1}^B \exp\left(\frac{\mathbf{q}_i^\top \mathbf{p}_j}{\tau}\right) + \sum_{k \in \mathcal{N}_i} \exp\left(\frac{\mathbf{q}_i^\top \mathbf{n}_{ik}}{\tau}\right)}$$
- **Temperature parameter $\tau$ (e.g. 0.05):** Acts as a hardness amplifier. Small $\tau$ exponentiates differences between top candidates, penalizing confusing negatives much more sharply.
- **Why MNRL dominates Triplet Loss in practice:** In a batch of size $B$, Triplet Loss provides $B$ comparisons. MNRL reuses all in-batch items as negatives, providing $B \times (B-1)$ negative signals per forward pass with zero additional forward passes through the encoder.

---

## 2. Modern Encoders: Bi-Encoder vs. Cross-Encoder

| Architecture Dimension | Bi-Encoder (Dual Encoder) | Cross-Encoder |
| :--- | :--- | :--- |
| **Input Structure** | $\text{Enc}(Q)$ and $\text{Enc}(D)$ independently | $\text{Enc}([CLS] \circ Q \circ [SEP] \circ D \circ [SEP])$ |
| **Cross-Attention** | None. Vectors interact only via dot product $\mathbf{u}^\top \mathbf{v}$ | Full token-level cross-attention across all layers |
| **Computational Cost** | $O(N \cdot L_D^2 + |Q| \cdot L_Q^2)$ indexing; $O(|Q| \cdot d)$ retrieval | $O(|Q| \cdot |D| \cdot (L_Q + L_D)^2)$ per query |
| **Indexability** | Offline $O(1)$ pre-computed vector index (FAISS, HNSW) | Unindexable offline (must evaluate pair at query time) |
| **Semantic Accuracy** | Moderate/High (compressed into fixed $\mathbb{R}^d$) | State-of-the-Art (captures nuanced term interactions) |
| **Production Role** | **Stage 1 Retrieval:** Prune $10^8 \to 50$ candidates | **Stage 2 Re-Ranking:** Re-score top 50 in $< 25$ms |

### Late Interaction: ColBERT
ColBERT bridges the gap between Bi-Encoders and Cross-Encoders using token-level embeddings and a **MaxSim** operator:
$$\text{Score}(Q, D) = \sum_{i \in Q} \max_{j \in D} \left(\mathbf{E}_{Q, i} \cdot \mathbf{E}_{D, j}^\top\right)$$
- Retains token-level representations without paying the full $O((L_Q + L_D)^2)$ cross-attention cost.
- Indexable via residual quantization (PLAID engine).

---

## 3. Dense Retrieval Bottlenecks & Approximate Nearest Neighbor (ANN)

### Maximum Inner Product Search (MIPS)
- For normalized vectors ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$), Cosine Similarity and Squared Euclidean Distance are monotonically related:
$$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2 \mathbf{u}^\top \mathbf{v} = 2 - 2 \langle \mathbf{u}, \mathbf{v} \rangle$$
Maximizing inner product is identical to minimizing Euclidean distance.

### Hierarchical Navigable Small World (HNSW)
HNSW builds a multi-layer geometric graph inspired by skip-lists:
1. **Top Layers:** Long-range sparse links for fast spatial navigation across vector space clusters.
2. **Bottom Layer ($L_0$):** Dense local graph containing all data points for precise local convergence.
3. **Key Hyperparameters:**
   - $M$ (Number of bi-directional links per node, typically 16–64): Higher $M$ increases recall and graph construction time.
   - $efConstruction$ (Size of priority queue during index build): Controls build quality.
   - $efSearch$ (Size of priority queue during query search): Directly trades off query latency vs recall.

### Vector Quantization (PQ vs. SCaNN)
- **Product Quantization (PQ):**
  Splits $d = 384$ vector into $m = 48$ sub-vectors of 8 dimensions each. Learns a codebook of 256 centroids per sub-space (8 bits).
  - Storage: $384 \times 4 \text{ bytes} = 1536 \text{ bytes} \implies 48 \text{ bytes}$ ($32\times$ memory compression).
  - Asymmetric Distance Computation (ADC): Pre-computes distance table between query sub-vectors and codebook centroids; searches in CPU cache with SIMD instructions.
- **SCaNN (Score-aware Anisotropic Vector Quantization):**
  Standard PQ minimizes reconstruction loss $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$. SCaNN penalizes parallel reconstruction error more heavily than orthogonal error, ensuring inner product ranking is preserved even at aggressive compression.

---

## 4. Why Pure Vector Search Fails in B2B Data

| Failure Mode | Concrete Real-World Example | Why Vectors Fail | Why Hybrid Solves It |
| :--- | :--- | :--- | :--- |
| **Exact Acronyms & Standards** | `"SOC2 Type II compliance"` | Acronyms mapped to generic security cluster | BM25 gives high IDF weight to exact token `SOC2` |
| **Financial / Metric Qualifiers** | `"Series B startup ARR > $10M"` | Numbers and financial terms lose magnitude in embedding space | Lexical/Metadata filtering handles exact filters |
| **Company Legal Entities** | `"Stripe Inc"` vs `"Stripey Ltd"` | Lexical and semantic overlap confuse Bi-Encoder | MinHash LSH blocking + Triplet hard-negative mining |
| **Multilingual Job Titles** | `"Directeur des opérations de revenus"` | Monolingual embeddings lack cross-lingual alignment | Multilingual backbone (`bge-m3` / `MiniLM-L12`) + MNRL |

---

## 5. Reciprocal Rank Fusion (RRF) Deep-Dive

### Mathematical Formulation
$$RRF(d) = \sum_{m \in \mathcal{M}} \frac{w_m}{k + r_m(d)}$$
where $\mathcal{M} = \{\text{dense}, \text{sparse}\}$, $r_m(d)$ is the 1-based rank position, and $k = 60$.

### Why RRF Over Convex Linear Combination:
1. **Unbounded Scale Mismatch:** BM25 scores range in $[0, \infty)$ and vary drastically based on document length and corpus statistics. Dense cosine similarities range in $[-1, 1]$.
2. **Min-Max Normalization Instability:** If query $A$ has BM25 scores $[2.1, 2.3]$ and query $B$ has scores $[18.5, 45.0]$, standard min-max scaling severely distorts relative importance across diverse queries.
3. **Hyperparameter Invariance:** RRF operates purely on ordinal ranks, making it completely invariant to the score distributions of underlying engines.

---

## 6. How to Frame Your Answers in the 45-Min Technical Interview

1. **When asked about Entity Resolution at scale:**
   - *"We never do $O(N^2)$ pairwise comparisons. We use a three-stage cascade: First, a two-stage heuristic blocking layer (MinHash LSH with $b=16, r=4$ and sparse domain/name token inverted indexes) to prune $10^6$ records down to $\sim 20$ candidate pairs per record in sub-10ms. Second, a fine-tuned Transformer Bi-Encoder trained with Triplet Margin Loss and hard lookalike negative mining. Third, an Agglomerative Graph Clustering step to resolve transitivity without suffering from false-positive chain linking."*

2. **When asked about Job Title Normalization:**
   - *"Keyword matching fails because of linguistic variations ('Responsable des opérations' vs 'Head of RevOps') and seniority nuance ('Software Engineer' vs 'Staff Software Engineer'). We fine-tune a multilingual sentence transformer using Multiple Negatives Ranking Loss (MNRL) with temperature $\tau=0.05$ and explicit hard negative mining. Additionally, we use a hierarchical multi-task head where the Seniority classification is conditioned directly on the predicted Department soft distribution."*

3. **When asked about Scalable Information Retrieval:**
   - *"Dense search alone has well-known blind spots around exact acronyms (SOC2, HIPAA) and part numbers, while sparse search fails on semantic paraphrasing. We deploy a hybrid two-stage retrieval pipeline: Stage 1 executes parallel FAISS HNSW dense search and BM25 sparse search, fused via Reciprocal Rank Fusion (RRF, $k=60$) to obtain the top 50 candidates in $< 5$ms. Stage 2 executes a Cross-Encoder re-ranker that evaluates full cross-attention over query-document pairs in $< 20$ms, achieving $>0.98$ NDCG@10."*
