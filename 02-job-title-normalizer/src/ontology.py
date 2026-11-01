"""Taxonomy and Ontology Schema for Job Title Normalization.

Defines:
  1. 12 Canonical Functional Departments
  2. 7 Hierarchical Seniority Tiers
  3. Canonical Role Entities with ground truth mappings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


DEPARTMENTS = [
    "Engineering",
    "Product_Management",
    "Revenue_Operations",
    "Sales",
    "Marketing",
    "Finance_Accounting",
    "People_Talent",
    "Legal_Compliance",
    "Data_AI",
    "Customer_Success",
    "Design_UX",
    "Executive_Leadership",
]

SENIORITIES = [
    "Intern",
    "Junior_IC",
    "Senior_IC",
    "Lead_Staff_Principal",
    "Manager",
    "Director_VP",
    "CXO_Founder",
]

DEP_TO_IDX = {dep: i for i, dep in enumerate(DEPARTMENTS)}
IDX_TO_DEP = {i: dep for i, dep in enumerate(DEPARTMENTS)}

SEN_TO_IDX = {sen: i for i, sen in enumerate(SENIORITIES)}
IDX_TO_SEN = {i: sen for i, sen in enumerate(SENIORITIES)}


@dataclass(frozen=True)
class CanonicalRole:
    canonical_title: str
    department: str
    seniority: str
    aliases_en: List[str]
    aliases_fr: List[str]
    aliases_de: List[str]
    aliases_es: List[str]


CANONICAL_ONTOLOGY: List[CanonicalRole] = [
    # --- Revenue Operations ---
    CanonicalRole(
        canonical_title="Revenue Operations Director",
        department="Revenue_Operations",
        seniority="Director_VP",
        aliases_en=["Director of RevOps", "Head of RevOps", "VP of Revenue Operations", "Director, Global Rev Ops"],
        aliases_fr=["Directeur des opérations de revenus", "Responsable des opérations de revenus", "Directeur RevOps"],
        aliases_de=["Leiter Ertragssteuerung", "Direktor Revenue Operations", "Bereichsleiter RevOps"],
        aliases_es=["Director de Operaciones de Ingresos", "Jefe de RevOps", "Director de Operaciones Comerciales"],
    ),
    CanonicalRole(
        canonical_title="Revenue Operations Analyst",
        department="Revenue_Operations",
        seniority="Junior_IC",
        aliases_en=["RevOps Analyst", "Sales Operations Analyst", "Junior RevOps Specialist", "Associate, Revenue Operations"],
        aliases_fr=["Analyste des opérations de revenus", "Analyste RevOps", "Chargé d'opérations commerciales"],
        aliases_de=["RevOps Analyst", "Junior Revenue Operations Spezialist", "Analyst Vertriebssteuerung"],
        aliases_es=["Analista de Operaciones de Ingresos", "Analista RevOps Junior"],
    ),
    # --- Engineering ---
    CanonicalRole(
        canonical_title="Software Engineer",
        department="Engineering",
        seniority="Junior_IC",
        aliases_en=["Software Developer", "SWE", "Fullstack Developer", "Backend Engineer", "Programmer"],
        aliases_fr=["Ingénieur logiciel", "Développeur fullstack", "Développeur backend", "Ingénieur d'études"],
        aliases_de=["Softwareentwickler", "Fullstack Entwickler", "Software-Ingenieur", "Backend Programmierer"],
        aliases_es=["Ingeniero de Software", "Desarrollador Fullstack", "Programador Backend"],
    ),
    CanonicalRole(
        canonical_title="Senior Software Engineer",
        department="Engineering",
        seniority="Senior_IC",
        aliases_en=["Sr. Software Engineer", "Senior Developer", "Senior Backend Engineer", "Lead Developer (Individual Contributor)"],
        aliases_fr=["Ingénieur logiciel senior", "Développeur senior", "Développeur full-stack confirmé"],
        aliases_de=["Senior Softwareentwickler", "Senior Software-Ingenieur", "Erfahrener Softwareentwickler"],
        aliases_es=["Ingeniero de Software Senior", "Desarrollador Senior", "Programador Senior"],
    ),
    CanonicalRole(
        canonical_title="Staff Software Engineer",
        department="Engineering",
        seniority="Lead_Staff_Principal",
        aliases_en=["Principal Engineer", "Staff Engineer", "Distinguished Engineer", "Lead Systems Architect"],
        aliases_fr=["Ingénieur logiciel principal", "Staff Engineer", "Architecte logiciel senior"],
        aliases_de=["Principal Software Engineer", "Leitender Systemarchitekt", "Staff Engineer"],
        aliases_es=["Ingeniero Principal de Software", "Arquitecto de Sistemas Principal"],
    ),
    CanonicalRole(
        canonical_title="Engineering Manager",
        department="Engineering",
        seniority="Manager",
        aliases_en=["Software Engineering Manager", "EM", "Team Lead - Engineering", "Manager, Software Development"],
        aliases_fr=["Responsable ingénierie logicielle", "Manager d'équipe technique", "Chef de groupe développement"],
        aliases_de=["Engineering Manager", "Teamleiter Softwareentwicklung", "Leiter Entwicklungsteam"],
        aliases_es=["Gerente de Ingeniería de Software", "Jefe de Equipo de Desarrollo", "Manager Técnico"],
    ),
    CanonicalRole(
        canonical_title="VP of Engineering",
        department="Engineering",
        seniority="Director_VP",
        aliases_en=["Vice President of Engineering", "Head of Engineering", "Director of Software Engineering"],
        aliases_fr=["Vice-président ingénierie", "Directeur de l'ingénierie", "Directeur technique adjoint"],
        aliases_de=["Leiter der Softwareentwicklung", "Direktor Entwicklung", "VP Engineering"],
        aliases_es=["Vicepresidente de Ingeniería", "Director de Ingeniería de Software", "Jefe de Tecnología"],
    ),
    CanonicalRole(
        canonical_title="Chief Technology Officer",
        department="Executive_Leadership",
        seniority="CXO_Founder",
        aliases_en=["CTO", "Co-Founder & CTO", "Chief Architect & Executive VP"],
        aliases_fr=["Directeur Général Technique", "CTO & Co-fondateur", "Directeur des Nouvelles Technologies"],
        aliases_de=["Technischer Vorstand", "Cheftechnologe", "CTO & Mitgründer"],
        aliases_es=["Director de Tecnología", "CTO y Cofundador", "Director Técnico Global"],
    ),
    # --- Product Management ---
    CanonicalRole(
        canonical_title="Product Manager",
        department="Product_Management",
        seniority="Junior_IC",
        aliases_en=["PM", "Associate Product Manager", "Technical Product Manager (IC)"],
        aliases_fr=["Chef de produit", "Product Manager", "Responsable produit junior"],
        aliases_de=["Produktmanager", "Junior Product Manager"],
        aliases_es=["Gerente de Producto", "Product Manager Junior"],
    ),
    CanonicalRole(
        canonical_title="VP of Product",
        department="Product_Management",
        seniority="Director_VP",
        aliases_en=["Vice President of Product", "Head of Product", "Chief Product Officer (Group)", "Director of Product"],
        aliases_fr=["Directeur de produit", "Vice-président produit", "Responsable pôle produit"],
        aliases_de=["Leiter Produktmanagement", "VP Produkt", "Direktor Produktstrategie"],
        aliases_es=["Vicepresidente de Producto", "Director de Producto"],
    ),
    # --- Data & AI ---
    CanonicalRole(
        canonical_title="Machine Learning Engineer",
        department="Data_AI",
        seniority="Junior_IC",
        aliases_en=["ML Engineer", "AI Engineer", "Deep Learning Practitioner", "NLP Engineer"],
        aliases_fr=["Ingénieur apprentissage automatique", "Ingénieur IA", "Ingénieur Machine Learning"],
        aliases_de=["Machine Learning Ingenieur", "KI Entwickler", "Ingenieur für Maschinelles Lernen"],
        aliases_es=["Ingeniero de Aprendizaje Automático", "Ingeniero de IA"],
    ),
    CanonicalRole(
        canonical_title="Head of AI",
        department="Data_AI",
        seniority="Director_VP",
        aliases_en=["Director of AI & ML", "VP of Artificial Intelligence", "Head of Machine Learning Research"],
        aliases_fr=["Directeur de l'Intelligence Artificielle", "Responsable pôle IA", "Directeur R&D IA"],
        aliases_de=["Leiter Künstliche Intelligenz", "Direktor KI Forschung", "VP Machine Learning"],
        aliases_es=["Director de Inteligencia Artificial", "Jefe de Machine Learning"],
    ),
    # --- Sales & Marketing ---
    CanonicalRole(
        canonical_title="Account Executive",
        department="Sales",
        seniority="Junior_IC",
        aliases_en=["AE", "Enterprise Account Executive", "Sales Representative", "Corporate Sales Specialist"],
        aliases_fr=["Chargé de clientèle entreprises", "Commercial B2B", "Account Executive"],
        aliases_de=["Vertriebsbeauftragter", "Vertriebsmanager B2B", "Account Executive"],
        aliases_es=["Ejecutivo de Cuentas", "Comercial B2B", "Ejecutivo de Ventas"],
    ),
    CanonicalRole(
        canonical_title="Chief Revenue Officer",
        department="Executive_Leadership",
        seniority="CXO_Founder",
        aliases_en=["CRO", "Chief Commercial Officer", "Executive VP of Worldwide Sales"],
        aliases_fr=["Directeur Commercial & Revenus", "Directeur Général des Revenus", "CRO"],
        aliases_de=["Vorstand Vertrieb und Ertrag", "Chief Revenue Officer", "Vertriebsvorstand"],
        aliases_es=["Director General Comercial", "Chief Revenue Officer"],
    ),
]
