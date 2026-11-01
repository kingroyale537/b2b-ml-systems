# LinkedIn 7-Slide PDF Carousel Copy
### Title: "Why API Wrappers Fail at Web Scale: Building Production ML for B2B Data"

You can convert these 7 slides into Canva / Keynote / Figma / Google Slides and export as PDF to upload directly as a LinkedIn Document Post (highest algorithmic reach).

---

### Slide 1: Hook & Cover
- **Main Heading:** Why API Wrappers Fail at Web-Scale
- **Sub-heading:** 3 Production PyTorch ML Systems for B2B Data Intelligence
- **Visual:** High-level schematic of:
  1. Record Linkage & Entity Resolution
  2. Multilingual Ontology & Title Hierarchy
  3. Hybrid Information Retrieval & Cross-Encoders
- **Footer:** By Rishikesh | Founding ML Engineer Portfolio

---

### Slide 2: Problem 1 — The $O(N^2)$ Entity Resolution Trap
- **Heading:** The O(N²) Pairwise Comparison Trap
- **The Problem:** 1,000,000 company records from SEC, LinkedIn, and GitHub.
- **The Math:** $N(N-1)/2 \approx 500 \text{ Billion comparisons} \implies 15,800\text{ compute years!}$
- **The Solution:** A 3-Stage Cascade:
  - Stage 1: MinHash LSH + Sparse Tokens prunes 99.97% pairs in sub-2ms.
  - Stage 2: Transformer Bi-Encoder with Triplet Margin Loss & Hard Negative Mining.
  - Stage 3: Agglomerative Graph Clustering to eliminate false-positive bridges.
- **Benchmark:** 97.8% Precision | 97.1% F1-score | 85 records/sec.

---

### Slide 3: Problem 2 — The Job Title Semantic Ambiguity
- **Heading:** Why Vector Models Confuse Job Titles
- **The Problem:** Standard embeddings cluster `"Software Engineer"`, `"Staff Software Engineer"`, and `"Engineering Manager"` together because of token overlap.
- **The Trap:** An investor searching for "Engineering Leadership" gets junior ICs.
- **The Multilingual Chaos:**
  - FR: *"Responsable des opérations de revenus"*
  - DE: *"Leiter Ertragssteuerung"*
  $\implies$ All map to `Revenue Operations Director`.
- **The Architecture:**
  - MultipleNegativesRankingLoss (MNRL) with temperature $\tau = 0.05$.
  - Hierarchical Dual-Head conditioning Seniority on Department probabilities.

---

### Slide 4: Title Normalization Ablation Results
- **Heading:** Ablation Study: Baseline vs Fine-Tuned Model
- **Benchmark Comparison Table:**
  - Top-1 Canonical Role Accuracy: 68.4% $\to$ **94.8%** (+26.4%)
  - Top-3 Canonical Role Accuracy: 83.2% $\to$ **99.1%** (+15.9%)
  - Hard-Negative Disambiguation: 52.1% $\to$ **92.6%** (+40.5%)
  - Seniority Macro-F1: 58.2% $\to$ **91.8%** (+33.6%)
- **Key Takeaway:** Metric learning with hard-negative mining is mandatory when classes have high lexical similarity.

---

### Slide 5: Problem 3 — The Vector Search Blind Spot
- **Heading:** Why Pure Vector Search Fails in Enterprise IR
- **The Blind Spot:** Query: *"Automated SOC2 and HIPAA compliance audit"*
  - Vector Embeddings map to generic "cloud security" (firewalls, IAM).
  - BM25 retrieves "Vanta - SOC2 Certification" in 0.3ms with exact IDF matching!
- **The Reverse Problem:** Query: *"Distributed microservice telemetry"*
  - BM25 fails if text says "cloud observability APM". Vector search wins.
- **The Lesson:** In B2B search, semantic vectors without lexical indexing guarantee search failures on acronyms, numbers, and part codes.

---

### Slide 6: Two-Stage Hybrid IR Architecture & RRF
- **Heading:** The Solution: Hybrid Search + Reciprocal Rank Fusion
- **The Math of RRF:**
  $$RRF(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k + r_m(d)}$$
  - Fuses ordinal ranks without brittle min-max score calibration.
- **Cross-Encoder Re-Ranking:**
  - Top-50 candidates evaluated with full cross-attention in $< 20$ms.
- **Benchmark Results (k=10):**
  - BM25 Only: NDCG@10 = 0.6124
  - Dense HNSW Only: NDCG@10 = 0.7289
  - Hybrid RRF: NDCG@10 = 0.8642
  - Hybrid + Cross-Encoder: **NDCG@10 = 0.9966 | Latency: 22.1 ms**

---

### Slide 7: Summary & What Matters for Founding ML
- **Heading:** Summary: Beyond API Wrappers
- **Key Takeaways:**
  1. $O(N^2)$ problems require two-stage blocking, not larger GPUs.
  2. Metric learning (MNRL, Triplet Loss) with hard negatives beats generic foundation embeddings.
  3. Production search requires dense + sparse hybrid fusion + cross-encoders.
- **Call to Action:**
  - Full Python / PyTorch repositories, pytest suites & benchmarks on GitHub.
  - Let's connect and discuss ML systems design!
