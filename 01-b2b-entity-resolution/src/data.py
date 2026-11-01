"""Realistic B2B Company Record and Triplet Dataset Generator.

Generates synthetic yet highly realistic corporate records mirroring raw web scrapes
(Crunchbase, LinkedIn, SEC filings, GitHub, AngelList) with realistic noise:
- Legal suffix mutations (Inc, LLC, GmbH, Corp, Pte Ltd, S.A.)
- Subdomain, protocol, and TLD variants
- Typographical noise and keyboard adjacent flips
- Geo / location formatting inconsistencies
- Hard negative company pairs (e.g. Ramp Financial vs Ramp Logistics)
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import uuid


@dataclass(frozen=True)
class CompanyRecord:
    id: str
    canonical_id: str
    name: str
    domain: Optional[str]
    location: str
    industry: str
    description: str

    def to_representation(self) -> str:
        """Format record into structured input string for the Bi-Encoder matcher."""
        dom = self.domain if self.domain else "UNKNOWN"
        return (
            f"[COMPANY] {self.name} "
            f"[DOMAIN] {dom} "
            f"[LOCATION] {self.location} "
            f"[INDUSTRY] {self.industry}"
        )


CANONICAL_COMPANIES = [
    {
        "name": "Stripe",
        "domain": "stripe.com",
        "location": "San Francisco, CA, USA",
        "industry": "Financial Technology & Payments Infrastructure",
        "description": "Financial infrastructure platform for businesses of all sizes to accept payments and manage revenue.",
    },
    {
        "name": "Datadog",
        "domain": "datadoghq.com",
        "location": "New York, NY, USA",
        "industry": "Cloud Observability & Security Monitoring",
        "description": "Monitoring and security platform for cloud applications and infrastructure metrics.",
    },
    {
        "name": "Snowflake",
        "domain": "snowflake.com",
        "location": "Bozeman, MT, USA",
        "industry": "Data Cloud & Enterprise Warehousing",
        "description": "Cloud-based data warehouse and analytics platform enabling unified data architecture.",
    },
    {
        "name": "Crustdata",
        "domain": "crustdata.com",
        "location": "San Francisco, CA, USA",
        "industry": "B2B Data Intelligence & Market Graph",
        "description": "Real-time alternative data intelligence on private companies, headcount, and web traction.",
    },
    {
        "name": "Ramp",
        "domain": "ramp.com",
        "location": "New York, NY, USA",
        "industry": "Corporate Finance & Spend Management",
        "description": "Corporate card and financial operations platform designed to control enterprise spend.",
    },
    {
        "name": "Brex",
        "domain": "brex.com",
        "location": "Salt Lake City, UT, USA",
        "industry": "Corporate Banking & Expense Automation",
        "description": "Financial services platform offering corporate credit cards and automated treasury management.",
    },
    {
        "name": "Figma",
        "domain": "figma.com",
        "location": "San Francisco, CA, USA",
        "industry": "Collaborative Design & Prototyping",
        "description": "Browser-based collaborative interface design and prototyping vector tool.",
    },
    {
        "name": "Notion",
        "domain": "notion.so",
        "location": "San Francisco, CA, USA",
        "industry": "Connected Workspace & Productivity Software",
        "description": "Unified workspace for wiki, docs, notes, and agile project management.",
    },
    {
        "name": "Vercel",
        "domain": "vercel.com",
        "location": "San Francisco, CA, USA",
        "industry": "Frontend Cloud & Serverless Infrastructure",
        "description": "Cloud platform for static sites and serverless functions powered by Next.js.",
    },
    {
        "name": "Supabase",
        "domain": "supabase.com",
        "location": "Singapore",
        "industry": "Open Source Backend & PostgreSQL Cloud",
        "description": "Open source Firebase alternative providing Postgres databases, auth, and real-time APIs.",
    },
    {
        "name": "Linear",
        "domain": "linear.app",
        "location": "San Francisco, CA, USA",
        "industry": "Project Management & Issue Tracking",
        "description": "Purpose-built project management tool for modern software teams with speed and focus.",
    },
    {
        "name": "Retool",
        "domain": "retool.com",
        "location": "San Francisco, CA, USA",
        "industry": "Low-Code Internal Tools Development",
        "description": "Development platform for building custom internal applications, dashboards, and workflows.",
    },
    {
        "name": "Pinecone",
        "domain": "pinecone.io",
        "location": "New York, NY, USA",
        "industry": "Managed Vector Database & AI Infrastructure",
        "description": "High-performance vector database optimized for similarity search and LLM retrieval.",
    },
    {
        "name": "Qdrant",
        "domain": "qdrant.tech",
        "location": "Berlin, Germany",
        "industry": "Vector Search Engine & Neural Databases",
        "description": "Open source vector search engine with extended filtering capabilities written in Rust.",
    },
    {
        "name": "Temporal",
        "domain": "temporal.io",
        "location": "Seattle, WA, USA",
        "industry": "Durable Workflow Orchestration",
        "description": "Open source execution platform for reliable, long-running stateful distributed applications.",
    },
    {
        "name": "PostHog",
        "domain": "posthog.com",
        "location": "San Francisco, CA, USA",
        "industry": "Product Analytics & Feature Flags Suite",
        "description": "All-in-one product analytics, session recording, feature flags, and A/B testing suite.",
    },
]

# Confusing lookalike companies for hard-negative generation
LOOKALIKE_PAIRS = [
    ("Ramp", {
        "name": "Ramp Logistics Inc",
        "domain": "ramplogistics.com",
        "location": "Chicago, IL, USA",
        "industry": "Third-Party Logistics & Freight Brokerage",
        "description": "Supply chain logistics and temperature-controlled freight forwarding services."
    }),
    ("Stripe", {
        "name": "Stripey Patterns Ltd",
        "domain": "stripeypatterns.co.uk",
        "location": "London, UK",
        "industry": "Textile Manufacturing & Interior Design",
        "description": "Luxury bespoke striped textiles and woven fabrics for hospitality interiors."
    }),
    ("Snowflake", {
        "name": "Snowflake Mountain Resort",
        "domain": "snowflakeresort.com",
        "location": "Aspen, CO, USA",
        "industry": "Hospitality, Ski Lodging & Tourism",
        "description": "Luxury ski-in ski-out vacation rentals and mountain lodging in Colorado."
    }),
    ("Pinecone", {
        "name": "Pinecone Forestry Consulting",
        "domain": "pineconeforestry.com",
        "location": "Portland, OR, USA",
        "industry": "Timberland Management & Ecological Conservation",
        "description": "Forestry planning, wildlife habitat surveys, and sustainable timber appraisals."
    }),
    ("Temporal", {
        "name": "Temporal Sound & Acoustics",
        "domain": "temporalsound.de",
        "location": "Munich, Germany",
        "industry": "Audio Engineering & Architectural Acoustics",
        "description": "Acoustic consulting, noise control, and studio soundproofing installations."
    }),
    ("Linear", {
        "name": "Linear Motion Systems Corp",
        "domain": "linearmotionsys.com",
        "location": "Detroit, MI, USA",
        "industry": "Industrial Automation & Mechanical Actuators",
        "description": "Precision linear guides, ball screws, and industrial automation robotics."
    }),
    ("Datadog", {
        "name": "DataDog Training Academy",
        "domain": "datadogtraining.org",
        "location": "Austin, TX, USA",
        "industry": "K9 Obedience & Service Dog Certification",
        "description": "Professional canine behavior modification and certified service dog instruction."
    }),
    ("Crustdata", {
        "name": "Crust Bakery Data Systems",
        "domain": "crustbakerydata.io",
        "location": "Portland, OR, USA",
        "industry": "Bakery Supply Chain & Point of Sale",
        "description": "Inventory management and point-of-sale telemetry software for artisan bakeries."
    }),
]

LEGAL_SUFFIXES = [
    "Inc.", "Inc", "Incorporated", "LLC", "L.L.C.", "Corp.", "Corporation",
    "GmbH", "Co.", "Company", "Ltd", "Ltd.", "Holdings", "Technologies", "Tech Inc."
]

LOCATIONS_MUTATIONS = {
    "San Francisco, CA, USA": [
        "SF, California", "San Francisco, CA", "San Francisco Bay Area",
        "SF", "San Francisco, United States", "San Francisco, California, US"
    ],
    "New York, NY, USA": [
        "NYC", "New York, NY", "New York City", "Manhattan, NY", "New York, USA"
    ],
    "Berlin, Germany": [
        "Berlin, DE", "Berlin", "Berlin, Deutschland", "Berlin, Germany (Remote)"
    ],
    "London, UK": [
        "London, United Kingdom", "London, Greater London", "London, GB", "Central London"
    ],
    "Seattle, WA, USA": [
        "Seattle, WA", "Seattle, Washington", "Seattle Metro", "Seattle, US"
    ],
}


class B2BDataGenerator:
    """Generates synthetic, highly realistic noisy records for B2B Entity Resolution."""

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)
        self.rng = random.Random(seed)

    def _corrupt_name(self, name: str) -> str:
        choice = self.rng.random()
        if choice < 0.35:
            # Add or swap legal suffix
            suffix = self.rng.choice(LEGAL_SUFFIXES)
            return f"{name} {suffix}"
        elif choice < 0.55:
            # Lowercase or titlecase
            return name.lower() if self.rng.random() < 0.5 else name.upper()
        elif choice < 0.75:
            # Introduce a minor realistic typo
            if len(name) > 4:
                idx = self.rng.randint(1, len(name) - 2)
                # Swap adjacent characters or drop a letter
                if self.rng.random() < 0.5:
                    chars = list(name)
                    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                    return "".join(chars)
                else:
                    return name[:idx] + name[idx + 1:]
            return name
        elif choice < 0.90:
            # Add descriptive token
            token = self.rng.choice(["HQ", "Cloud", "App", "Platform", "Software"])
            return f"{name} {token}"
        return name

    def _corrupt_domain(self, domain: Optional[str]) -> Optional[str]:
        if domain is None or self.rng.random() < 0.12:
            return None  # Missing domain
        choice = self.rng.random()
        if choice < 0.30:
            return f"https://www.{domain}/"
        elif choice < 0.50:
            return f"http://{domain}/about"
        elif choice < 0.70:
            sub = self.rng.choice(["app", "api", "auth", "docs", "portal"])
            return f"{sub}.{domain}"
        elif choice < 0.85:
            # Strip TLD and append alternative
            base = domain.split(".")[0]
            alt_tld = self.rng.choice([".co", ".io", ".dev", ".ai", ".com"])
            return f"{base}{alt_tld}"
        return domain

    def _corrupt_location(self, loc: str) -> str:
        if loc in LOCATIONS_MUTATIONS and self.rng.random() < 0.70:
            return self.rng.choice(LOCATIONS_MUTATIONS[loc])
        if self.rng.random() < 0.15:
            return "Remote"
        return loc

    def _corrupt_industry(self, industry: str) -> str:
        parts = industry.split("&")
        if len(parts) > 1 and self.rng.random() < 0.5:
            return parts[0].strip()
        return industry

    def generate_record_from_canonical(
        self, canonical: Dict[str, Any], canonical_id: str
    ) -> CompanyRecord:
        """Create a noisy variation of a canonical company."""
        return CompanyRecord(
            id=str(uuid.uuid4())[:8],
            canonical_id=canonical_id,
            name=self._corrupt_name(canonical["name"]),
            domain=self._corrupt_domain(canonical.get("domain")),
            location=self._corrupt_location(canonical["location"]),
            industry=self._corrupt_industry(canonical["industry"]),
            description=canonical.get("description", ""),
        )

    def generate_corpus(
        self,
        num_canonical: int = 100,
        duplicates_per_entity: int = 4,
        noise_ratio: float = 0.15,
    ) -> Tuple[List[CompanyRecord], Dict[str, List[str]]]:
        """Generate a complete enterprise corpus of records with known ground-truth clusters."""
        records: List[CompanyRecord] = []
        clusters: Dict[str, List[str]] = {}

        # Base pool of canonical templates
        pool = list(CANONICAL_COMPANIES)
        while len(pool) < num_canonical:
            idx = len(pool)
            base = self.rng.choice(CANONICAL_COMPANIES)
            mutated_name = f"{base['name']}_{idx}"
            mutated_domain = f"{mutated_name.lower().replace('_', '')}.io"
            pool.append({
                "name": mutated_name,
                "domain": mutated_domain,
                "location": base["location"],
                "industry": base["industry"],
                "description": base["description"],
            })

        for i, can in enumerate(pool[:num_canonical]):
            can_id = f"CANONICAL_{i:04d}"
            clusters[can_id] = []

            # Always add one canonical pristine record
            pristine = CompanyRecord(
                id=f"rec_{can_id}_0",
                canonical_id=can_id,
                name=can["name"],
                domain=can["domain"],
                location=can["location"],
                industry=can["industry"],
                description=can["description"],
            )
            records.append(pristine)
            clusters[can_id].append(pristine.id)

            # Generate duplicates
            n_dupes = self.rng.randint(1, duplicates_per_entity * 2)
            for d in range(n_dupes):
                noisy = self.generate_record_from_canonical(can, can_id)
                records.append(noisy)
                clusters[can_id].append(noisy.id)

        # Shuffle records to simulate unindexed raw incoming ingestion stream
        self.rng.shuffle(records)
        return records, clusters

    def generate_triplets(
        self,
        records: List[CompanyRecord],
        num_triplets: int = 2000,
        hard_negative_prob: float = 0.50,
    ) -> List[Tuple[CompanyRecord, CompanyRecord, CompanyRecord]]:
        """Mine and generate (Anchor, Positive, Negative) training triplets."""
        # Index records by canonical ID
        canonical_map: Dict[str, List[CompanyRecord]] = {}
        for r in records:
            canonical_map.setdefault(r.canonical_id, []).append(r)

        triplets = []
        canonical_ids = [cid for cid, group in canonical_map.items() if len(group) >= 2]

        # Build lookup for hard lookalikes
        lookalike_dict = dict(LOOKALIKE_PAIRS)

        for _ in range(num_triplets):
            cid = self.rng.choice(canonical_ids)
            anchor, positive = self.rng.sample(canonical_map[cid], 2)

            # Negative mining strategy
            use_hard_negative = (self.rng.random() < hard_negative_prob)
            negative = None

            if use_hard_negative and anchor.name in lookalike_dict:
                # Use crafted domain lookalike
                raw_neg = lookalike_dict[anchor.name]
                negative = CompanyRecord(
                    id=str(uuid.uuid4())[:8],
                    canonical_id=f"LOOKALIKE_{anchor.name}",
                    name=raw_neg["name"],
                    domain=raw_neg["domain"],
                    location=raw_neg["location"],
                    industry=raw_neg["industry"],
                    description=raw_neg["description"],
                )
            else:
                # Random or semi-hard negative from different canonical cluster
                other_cid = self.rng.choice([c for c in canonical_map.keys() if c != cid])
                negative = self.rng.choice(canonical_map[other_cid])

            triplets.append((anchor, positive, negative))

        return triplets
