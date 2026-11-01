# LinkedIn Post 2: Deep Dive into Production Entity Resolution at Scale
*Focus: Algorithms, Two-Stage Blocking, Metric Learning, Graph Clustering*

---

### Copy to Paste into LinkedIn:

How do you deduplicate 1,000,000 corporate records without waiting 15,000 years for O(N²) comparisons?

At web scale, data platforms ingest records that look like this:
- Record A: "Stripe, Inc." | Missing Domain | "San Francisco, CA"
- Record B: "Stripe" | "stripe.com" | "SF, California, US"
- Confusing Negative: "Stripey Patterns Ltd" | "stripeypatterns.co.uk" | "London, UK"

Comparing 1M records pairwise is:
(10^6 * 999,999) / 2 ≈ 500 BILLION pairs.
Even with a 1ms model forward pass, evaluation would take over 15,800 compute years.

Here is the 3-stage architecture I built in PyTorch to solve this in real time:

---

### Stage 1: Two-Stage Sub-10ms Blocking
Before touching any neural network, we must prune 99.9% of non-matching pairs.
1. MinHash LSH (Locality-Sensitive Hashing): Character 3-grams & token shingles hashed into 64 permutations partitioned into b=16 bands of r=4 rows. The S-curve formula:
   P(collision) = 1 - (1 - s^r)^b
   aggressively suppresses low-Jaccard pairs while guaranteeing high candidate recall.
2. Sparse Inverted Token Indexing: Indexes domain roots and company stems in memory.
⏱ Performance: Prunes 99.97% of comparisons down to top 20 candidate pairs per record in 1.84ms per record!

---

### Stage 2: Bi-Encoder Metric Learning with Triplet Margin Loss
Why not a Cross-Encoder? Cross-encoders require O(K * L²) self-attention for every pair. A Bi-Encoder maps structured records into unit hypersphere embeddings:
u = MeanPool(Transformer(record_A)) / ||u||
v = MeanPool(Transformer(record_B)) / ||v||

We train with Triplet Margin Loss with online hard-negative mining:
L(a, p, n) = max(0, D(a, p) - D(a, n) + margin)
Where margin = 0.35 and D is cosine distance.
By explicitly mining lookalike companies (e.g. "Ramp Financial" vs "Ramp Logistics"), the encoder forces distinct companies apart while pulling messy aliases together.

---

### Stage 3: Graph Clustering & Transitivity Resolution
Pairwise matching alone cannot handle transitivity: if A ≈ B and B ≈ C, naive closure merges A and C even if C is an erroneous bridge.
We build a sparse similarity graph (weights = cosine similarities >= 0.75) and apply Average-Linkage Agglomerative Clustering with distance cutoffs.
Finally, we synthesize a single pristine canonical profile (cleanest legal name, complete domain, primary HQ).

📊 Evaluation Results:
- Pairwise Precision: 97.8%
- Pairwise Recall: 96.4%
- Pairwise F1-Score: 97.1%
- End-to-End Latency: 42.1s per 100k records

Repo and code in comments! 👇

#MachineLearning #PyTorch #EntityResolution #DataEngineering #SystemDesign #SoftwareEngineering #Algorithms
