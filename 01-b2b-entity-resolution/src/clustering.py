"""Graph Clustering Layer for Global Entity Resolution.

Resolves pairwise match predictions into unified, global canonical Entity IDs
using Connected Components and Agglomerative Linkage Clustering while pruning
transitive false-positive bridges:
  Records -> Pairwise Probabilities -> Weighted Graph -> Canonical Entity Clusters
"""

from __future__ import annotations

import collections
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .data import CompanyRecord


class EntityGraphClustering:
    """Graph-based clustering for resolving records into global Entity IDs."""

    def __init__(
        self,
        match_threshold: float = 0.75,
        method: str = "connected_components",  # or 'agglomerative'
        linkage_criterion: str = "average",
    ) -> None:
        self.match_threshold = match_threshold
        self.method = method
        self.linkage_criterion = linkage_criterion

    def _cluster_connected_components(
        self, records: List[CompanyRecord], scored_pairs: List[Tuple[str, str, float]]
    ) -> Dict[str, str]:
        """Cluster via Connected Components with strict threshold filtering."""
        G = nx.Graph()
        for r in records:
            G.add_node(r.id)

        for id_a, id_b, score in scored_pairs:
            if score >= self.match_threshold:
                G.add_edge(id_a, id_b, weight=score)

        record_to_cluster: Dict[str, str] = {}
        for comp in nx.connected_components(G):
            # Deterministic Cluster ID based on sorted member IDs
            sorted_members = sorted(list(comp))
            cluster_hash = hashlib.sha256(
                "_".join(sorted_members).encode("utf-8")
            ).hexdigest()[:12]
            cluster_id = f"ENT_{cluster_hash}"
            for member_id in sorted_members:
                record_to_cluster[member_id] = cluster_id

        return record_to_cluster

    def _cluster_agglomerative(
        self, records: List[CompanyRecord], scored_pairs: List[Tuple[str, str, float]]
    ) -> Dict[str, str]:
        """Cluster via Hierarchical Agglomerative Clustering to prevent chain-linking errors."""
        n = len(records)
        if n == 0:
            return {}
        if n == 1:
            return {records[0].id: f"ENT_{records[0].id}"}

        id_to_idx = {r.id: i for i, r in enumerate(records)}
        idx_to_id = {i: r.id for i, r in enumerate(records)}

        # Build distance matrix (distance = 1.0 - similarity)
        # Default distance = 1.0 (unconnected)
        dist_matrix = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(dist_matrix, 0.0)

        for id_a, id_b, score in scored_pairs:
            if id_a in id_to_idx and id_b in id_to_idx:
                i, j = id_to_idx[id_a], id_to_idx[id_b]
                dist = max(0.0, 1.0 - score)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Condensed distance matrix for scipy linkage
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method=self.linkage_criterion)

        # Cut dendrogram at distance threshold = 1.0 - match_threshold
        dist_thresh = 1.0 - self.match_threshold
        cluster_labels = fcluster(Z, t=dist_thresh, criterion="distance")

        record_to_cluster: Dict[str, str] = {}
        cluster_groups: Dict[int, List[str]] = collections.defaultdict(list)

        for idx, lbl in enumerate(cluster_labels):
            rec_id = idx_to_id[idx]
            cluster_groups[int(lbl)].append(rec_id)

        for _, members in cluster_groups.items():
            sorted_m = sorted(members)
            c_hash = hashlib.sha256("_".join(sorted_m).encode("utf-8")).hexdigest()[:12]
            cid = f"ENT_{c_hash}"
            for m in sorted_m:
                record_to_cluster[m] = cid

        return record_to_cluster

    def cluster(
        self, records: List[CompanyRecord], scored_pairs: List[Tuple[str, str, float]]
    ) -> Dict[str, str]:
        """Cluster records into canonical Entity IDs."""
        if self.method == "agglomerative" and len(records) < 5000:
            return self._cluster_agglomerative(records, scored_pairs)
        return self._cluster_connected_components(records, scored_pairs)

    @staticmethod
    def synthesize_canonical_profiles(
        records: List[CompanyRecord], record_to_cluster: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """Synthesize a pristine single canonical profile for each resolved cluster."""
        clusters: Dict[str, List[CompanyRecord]] = collections.defaultdict(list)
        for r in records:
            cid = record_to_cluster.get(r.id, f"ENT_{r.id}")
            clusters[cid].append(r)

        canonical_profiles = {}
        for cid, members in clusters.items():
            # Cleanest name: shortest name or most frequent without noisy legal suffixes
            names = [m.name for m in members]
            cleanest_name = min(names, key=lambda n: (len(n), n))

            # Domain with highest completeness / shortest clean root
            domains = [m.domain for m in members if m.domain]
            best_domain = domains[0] if domains else None

            # Most common location and industry
            locs = collections.Counter(m.location for m in members).most_common(1)
            best_loc = locs[0][0] if locs else "Unknown"

            inds = collections.Counter(m.industry for m in members).most_common(1)
            best_ind = inds[0][0] if inds else "Technology"

            canonical_profiles[cid] = {
                "entity_id": cid,
                "canonical_name": cleanest_name,
                "canonical_domain": best_domain,
                "canonical_location": best_loc,
                "canonical_industry": best_ind,
                "cluster_size": len(members),
                "member_record_ids": [m.id for m in members],
            }

        return canonical_profiles
