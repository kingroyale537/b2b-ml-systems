"""Multilingual Job Title Normalization & Hierarchy Classifier."""

from .ontology import (
    DEPARTMENTS,
    SENIORITIES,
    DEP_TO_IDX,
    SEN_TO_IDX,
    IDX_TO_DEP,
    IDX_TO_SEN,
    CANONICAL_ONTOLOGY,
    CanonicalRole,
)
from .data import TitleDataGenerator, TitleSample
from .loss import MultipleNegativesRankingLoss, HierarchicalMultiTaskLoss
from .model import HierarchicalTitleNormalizer
from .inference import TitleNormalizerInference
from .evaluate import TitleEvaluator

__all__ = [
    "DEPARTMENTS",
    "SENIORITIES",
    "DEP_TO_IDX",
    "SEN_TO_IDX",
    "IDX_TO_DEP",
    "IDX_TO_SEN",
    "CANONICAL_ONTOLOGY",
    "CanonicalRole",
    "TitleDataGenerator",
    "TitleSample",
    "MultipleNegativesRankingLoss",
    "HierarchicalMultiTaskLoss",
    "HierarchicalTitleNormalizer",
    "TitleNormalizerInference",
    "TitleEvaluator",
]
