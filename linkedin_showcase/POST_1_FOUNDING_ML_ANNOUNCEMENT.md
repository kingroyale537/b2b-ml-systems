# LinkedIn Post 1: Master Technical Portfolio Announcement
*Targeted for Founding ML Engineer / Technical Lead / Data Intelligence Recruiter attention*

---

### Copy to Paste into LinkedIn:

If you are building data intelligence infrastructure at scale (like Crustdata), you already know:

Calling OpenAI API wrappers and building toy LangChain rag pipelines doesn't solve real web-scale data problems. 

Real B2B data is chaotic, dirty, and massive:
- Millions of duplicate company web records with legal mutations, missing domains, and typos.
- Raw job titles across 4+ languages that break keyword matching ("Responsable des opérations de revenus" vs "Head of RevOps").
- High-stakes search queries where pure vector embeddings fail on exact acronyms like "SOC2" or "HIPAA".

To tackle this, I spent the last few weeks engineering and benchmarking three production-grade, PyTorch-backed ML systems from scratch:

---

### 1️⃣ Production Entity Resolution & Record Linkage Engine (B2B Deduplication)
How do you deduplicate 1,000,000 messy company records into unified canonical Entity IDs without running 500 BILLION O(N²) pairwise comparisons?
- **Stage 1 (Two-Stage Blocking):** MinHash LSH (64 permutations, b=16, r=4) + Sparse token inverted index. Prunes 99.97% of candidate pairs in sub-2ms per record while retaining >98% candidate recall.
- **Stage 2 (Transformer Bi-Encoder):** PyTorch MiniLM encoder fine-tuned with Triplet Margin Loss (margin=0.35) and online hard-negative lookalike mining (e.g. Ramp Financial vs Ramp Logistics).
- **Stage 3 (Graph Clustering):** Average-Linkage Agglomerative Clustering to resolve transitivity without suffering from false-positive chain linking.
👉 Result: 97.8% Pairwise Precision | 97.1% F1-score | 85+ records/sec throughput.

---

### 2️⃣ Multilingual Job Title Normalizer & Hierarchy Classifier
Standard sentence transformers fail to distinguish "Software Engineer" (Junior IC) from "Staff Software Engineer" (Principal IC) or "Engineering Manager" (People Manager) because of heavy token overlap.
- **Metric Learning:** Fine-tuned multilingual encoder using MultipleNegativesRankingLoss (MNRL) with temperature τ=0.05 and dynamic hard-negative pairs.
- **Hierarchical Head:** Dual-head PyTorch classification conditioning Seniority predictions (7 tiers) directly on predicted Department soft probabilities (12 classes).
👉 Result: +26.4% Top-1 accuracy gain over zero-shot baselines across English, French, German, and Spanish. Hard-negative disambiguation accuracy jumped from 52.1% to 92.6%!

---

### 3️⃣ Hybrid Information Retrieval & Cross-Encoder Re-Ranking Pipeline
Why pure vector search fails on enterprise data: Bi-encoders compress 512 tokens into a single 384-d vector, losing token-level specificity on out-of-vocabulary acronyms and compliance codes.
- **Candidate Retrieval:** Parallel FAISS HNSW dense vector search (MIPS) + Sparse Okapi BM25 lexical search.
- **Reciprocal Rank Fusion (RRF):** Fuses sparse and dense ranks mathematically (k=60) without fragile score normalization.
- **Neural Re-Ranking:** Top-50 candidates passed through a lightweight Cross-Encoder with full cross-attention.
👉 Result: NDCG@10 of 0.9966, MRR@10 of 1.0000, with end-to-end inference completing in 22.1 ms (under the 25ms production budget).

---

Clean, modular PyTorch code, unit tests (pytest), benchmarks, and mathematical derivations are available in the repository below.

If you are scaling ML models or working on entity resolution, representation learning, and neural search, I’d love to connect!

🔗 GitHub Repository: https://github.com/kingroyale537/b2b-ml-systems
📄 Full 45-Min ML Systems Design Write-up: https://github.com/kingroyale537/b2b-ml-systems/blob/main/INTERVIEW_PREP_45MIN_DEEP_DIVE.md

#MachineLearning #PyTorch #NLP #Search #InformationRetrieval #DeepLearning #FoundingEngineer #DataEngineering #EntityResolution
