"""Multilingual Dataset Generator & Hard Negative Miner for Job Title Normalization.

Generates realistic, noisy raw job titles across 4 languages (EN, FR, DE, ES)
mapped to canonical roles, departments, and seniority levels, with synthetic noise:
- Executive prefixes ("Interim", "Fractional", "Acting", "Global Head of")
- Seniority qualifiers ("Junior", "Senior", "Lead", "Staff", "Principal", "VP")
- Niche startup variations ("Founding AI Engineer", "Head of RevOps & GTM Systems")
- Hard negative triplets distinguishing IC vs Manager vs Executive.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .ontology import (
    CANONICAL_ONTOLOGY,
    CanonicalRole,
    DEP_TO_IDX,
    DEPARTMENTS,
    SEN_TO_IDX,
    SENIORITIES,
)


@dataclass(frozen=True)
class TitleSample:
    raw_title: str
    canonical_title: str
    department: str
    seniority: str
    department_idx: int
    seniority_idx: int
    language: str


# Hard negative confusion groups: titles that look lexically identical but have different seniority or function
CONFUSION_CLUSTERS = [
    # IC vs Management vs Executive in Engineering
    [
        ("Software Engineer", "Engineering", "Junior_IC"),
        ("Senior Software Engineer", "Engineering", "Senior_IC"),
        ("Staff Software Engineer", "Engineering", "Lead_Staff_Principal"),
        ("Engineering Manager", "Engineering", "Manager"),
        ("Director of Engineering", "Engineering", "Director_VP"),
        ("VP of Engineering", "Engineering", "Director_VP"),
        ("Chief Technology Officer", "Executive_Leadership", "CXO_Founder"),
    ],
    # RevOps vs Sales vs Marketing
    [
        ("Revenue Operations Analyst", "Revenue_Operations", "Junior_IC"),
        ("Revenue Operations Director", "Revenue_Operations", "Director_VP"),
        ("Account Executive", "Sales", "Junior_IC"),
        ("Chief Revenue Officer", "Executive_Leadership", "CXO_Founder"),
    ],
    # Product vs Engineering
    [
        ("Product Manager", "Product_Management", "Junior_IC"),
        ("VP of Product", "Product_Management", "Director_VP"),
        ("Technical Product Manager", "Product_Management", "Junior_IC"),
        ("Software Engineer", "Engineering", "Junior_IC"),
    ],
    # Data & AI
    [
        ("Machine Learning Engineer", "Data_AI", "Junior_IC"),
        ("Head of AI", "Data_AI", "Director_VP"),
        ("Software Engineer", "Engineering", "Junior_IC"),
    ],
]


PREFIX_MUTATIONS = [
    "Interim ", "Fractional ", "Acting ", "Global ", "Head of ",
    "Vice President, ", "Principal ", "Associate ", "Lead ", "Senior ", "Group "
]

SUFFIX_MUTATIONS = [
    " (Remote)", " - EMEA", " - Americas", " (Contractor)", " & Team Lead",
    " | GTM Systems", " - Platform Team", " (Individual Contributor)"
]


class TitleDataGenerator:
    """Generates synthetic multilingual title dataset with hard negative mining."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def _corrupt_raw_title(self, base_title: str) -> str:
        res = base_title
        if self.rng.random() < 0.25:
            # Add prefix if not already present
            p = self.rng.choice(PREFIX_MUTATIONS)
            if not any(res.startswith(prefix.strip()) for prefix in PREFIX_MUTATIONS):
                res = f"{p}{res}"
        if self.rng.random() < 0.25:
            s = self.rng.choice(SUFFIX_MUTATIONS)
            res = f"{res}{s}"
        if self.rng.random() < 0.15:
            res = res.lower() if self.rng.random() < 0.5 else res.upper()
        return res.strip()

    def generate_corpus(
        self, samples_per_role: int = 40
    ) -> List[TitleSample]:
        """Generate balanced multilingual corpus across all ontology roles."""
        samples: List[TitleSample] = []

        for role in CANONICAL_ONTOLOGY:
            dep_idx = DEP_TO_IDX[role.department]
            sen_idx = SEN_TO_IDX[role.seniority]

            # Collect alias pools per language
            lang_pools = [
                ("en", role.aliases_en),
                ("fr", role.aliases_fr),
                ("de", role.aliases_de),
                ("es", role.aliases_es),
            ]

            for _ in range(samples_per_role):
                lang, aliases = self.rng.choice(lang_pools)
                base = self.rng.choice(aliases)
                noisy_title = self._corrupt_raw_title(base)

                sample = TitleSample(
                    raw_title=noisy_title,
                    canonical_title=role.canonical_title,
                    department=role.department,
                    seniority=role.seniority,
                    department_idx=dep_idx,
                    seniority_idx=sen_idx,
                    language=lang,
                )
                samples.append(sample)

        self.rng.shuffle(samples)
        return samples

    def mine_hard_negatives(
        self, sample: TitleSample
    ) -> List[str]:
        """Mine confusing job titles that share keywords but differ in seniority or function."""
        hard_negs: List[str] = []

        for cluster in CONFUSION_CLUSTERS:
            # Check if this sample matches any item in the cluster
            matching_items = [
                item for item in cluster
                if item[0] == sample.canonical_title
            ]
            if matching_items:
                # Find contradictory items in same cluster
                confusions = [
                    item[0] for item in cluster
                    if item[0] != sample.canonical_title
                ]
                hard_negs.extend(confusions)

        if not hard_negs:
            # Fallback to other canonical roles
            other_roles = [
                r.canonical_title for r in CANONICAL_ONTOLOGY
                if r.canonical_title != sample.canonical_title
            ]
            hard_negs = self.rng.sample(other_roles, min(3, len(other_roles)))

        return list(set(hard_negs))
