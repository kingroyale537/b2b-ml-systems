"""Enterprise B2B Search Corpus & Evaluation Query Set.

Generates realistic B2B documents covering company profiles, infrastructure,
funding, tech stacks, and regulatory compliance, alongside annotated evaluation queries
with graded relevance judgments (qrels) designed to stress-test both semantic and lexical search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SearchDocument:
    doc_id: str
    title: str
    company: str
    body: str
    tags: List[str]

    def to_index_text(self) -> str:
        return f"{self.title}. {self.body} Tags: {', '.join(self.tags)}"


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    query_text: str
    query_type: str  # 'semantic', 'acronym_exact', 'hybrid'
    relevant_doc_scores: Dict[str, int]  # doc_id -> relevance grade (1, 2, or 3)


CORPUS_DOCUMENTS: List[SearchDocument] = [
    SearchDocument(
        doc_id="doc_stripe_01",
        title="Stripe - Global Financial Infrastructure & Payments APIs",
        company="Stripe",
        body="Stripe provides economic infrastructure for the internet. Businesses of every size use our software to accept payments, send payouts, manage billing, and control online fraud with Stripe Radar.",
        tags=["fintech", "payments", "billing", "fraud", "api"],
    ),
    SearchDocument(
        doc_id="doc_datadog_01",
        title="Datadog Cloud Observability & Infrastructure Telemetry",
        company="Datadog",
        body="Datadog is an observability service for cloud-scale applications, providing monitoring of servers, databases, tools, and services through a SaaS-based data analytics platform and APM traces.",
        tags=["observability", "apm", "telemetry", "devops", "cloud", "metrics"],
    ),
    SearchDocument(
        doc_id="doc_snowflake_01",
        title="Snowflake Data Cloud - Unified Data Warehousing",
        company="Snowflake",
        body="Snowflake enables data storage, processing, and analytic solutions that are faster, easier to use, and far more flexible than traditional offerings. Scalable SQL data warehouse in AWS, GCP, Azure.",
        tags=["data-warehouse", "analytics", "sql", "cloud", "etl"],
    ),
    SearchDocument(
        doc_id="doc_crustdata_01",
        title="Crustdata - Real-Time Alternative B2B Data & Market Intelligence",
        company="Crustdata",
        body="Crustdata indexes web traction, LinkedIn headcount changes, Glassdoor sentiment, GitHub commits, and job openings to provide private company intelligence for VCs, hedge funds, and sales teams.",
        tags=["b2b-data", "market-intelligence", "alternative-data", "headcount", "private-markets"],
    ),
    SearchDocument(
        doc_id="doc_ramp_01",
        title="Ramp - Corporate Cards, Expense Management & AP Automation",
        company="Ramp",
        body="Ramp is the ultimate finance operations platform designed to help companies spend less. Offers corporate credit cards, automated accounts payable, invoice approvals, and real-time vendor spend tracking.",
        tags=["fintech", "corporate-card", "expense-management", "accounts-payable", "spend"],
    ),
    SearchDocument(
        doc_id="doc_vanta_01",
        title="Vanta - Automated Security Compliance & SOC2 Certification",
        company="Vanta",
        body="Vanta automates security compliance for SOC2, ISO 27001, HIPAA, GDPR, and PCI DSS. Continuously monitors cloud infrastructure, collects evidence, and accelerates audit readiness for high-growth tech startups.",
        tags=["security", "compliance", "soc2", "gdpr", "hipaa", "audit", "cybersecurity"],
    ),
    SearchDocument(
        doc_id="doc_qdrant_01",
        title="Qdrant - High-Performance Vector Database in Rust",
        company="Qdrant",
        body="Qdrant is a vector similarity search engine and vector database written in Rust. Features HNSW indexing, extended payload filtering, payload-based ranking, and sub-millisecond MIPS query latency.",
        tags=["vector-db", "hnsw", "mips", "rust", "similarity-search", "ai-search"],
    ),
    SearchDocument(
        doc_id="doc_pinecone_01",
        title="Pinecone - Serverless Vector Database for Retrieval-Augmented Generation",
        company="Pinecone",
        body="Pinecone is a managed vector database built for high-scale semantic search and RAG applications. Provides live index updates, multi-tenant isolation, metadata filtering, and low p99 query latency.",
        tags=["vector-db", "rag", "embeddings", "semantic-search", "similarity"],
    ),
    SearchDocument(
        doc_id="doc_temporal_01",
        title="Temporal - Durable Distributed Execution & Stateful Workflows",
        company="Temporal",
        body="Temporal is an open-source orchestration engine that guarantees code executes reliably despite server crashes, network partitions, or transient external API timeouts. Ideal for financial transactions and saga patterns.",
        tags=["distributed-systems", "orchestration", "stateful", "workflows", "resilience"],
    ),
    SearchDocument(
        doc_id="doc_posthog_01",
        title="PostHog - Open Source Product Analytics, Feature Flags & Session Replay",
        company="PostHog",
        body="PostHog is an all-in-one developer platform for product analytics, web session recordings, feature flags, A/B testing, and user heatmaps. Can be self-hosted on AWS/GCP or used via PostHog Cloud.",
        tags=["product-analytics", "feature-flags", "ab-testing", "session-replay", "open-source"],
    ),
]

EVALUATION_QUERIES: List[EvaluationQuery] = [
    # Semantic query (benefits heavily from Dense vector search)
    EvaluationQuery(
        query_id="q_semantic_01",
        query_text="cloud infrastructure telemetry and distributed trace monitoring",
        query_type="semantic",
        relevant_doc_scores={"doc_datadog_01": 3, "doc_temporal_01": 1},
    ),
    # Exact acronym query (fails on dense search, wins on Sparse BM25)
    EvaluationQuery(
        query_id="q_acronym_01",
        query_text="automated SOC2 and HIPAA security compliance audit",
        query_type="acronym_exact",
        relevant_doc_scores={"doc_vanta_01": 3},
    ),
    # Hybrid query (requires both semantic understanding & exact terminology)
    EvaluationQuery(
        query_id="q_hybrid_01",
        query_text="high performance Rust vector database HNSW similarity search",
        query_type="hybrid",
        relevant_doc_scores={"doc_qdrant_01": 3, "doc_pinecone_01": 2},
    ),
    # Market Intelligence query
    EvaluationQuery(
        query_id="q_b2b_intel_01",
        query_text="alternative data tracking private company headcount and job openings",
        query_type="hybrid",
        relevant_doc_scores={"doc_crustdata_01": 3},
    ),
    # Financial Ops query
    EvaluationQuery(
        query_id="q_fin_ops_01",
        query_text="corporate credit cards invoice approvals and spend control",
        query_type="semantic",
        relevant_doc_scores={"doc_ramp_01": 3, "doc_stripe_01": 1},
    ),
]
