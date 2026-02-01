"""Taxonomic consensus, vector search, and novelty discovery."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import lancedb
import numpy as np
import pandas as pd

from src.config import LANCEDB_PENDRIVE_PATH, LANCEDB_TABLE_SEQUENCES
from src.edge.discovery import cluster_novelty_pool

logger = logging.getLogger(__name__)


DEFAULT_TABLE_NAME = "obis_reference_index"
DEFAULT_NOVELTY_THRESHOLD = 0.85


def _is_colab() -> bool:
    """Return True if running in Google Colab."""
    return "google.colab" in sys.modules


def _cosine_similarity_from_distance(distance: float) -> float:
    """Convert cosine distance to similarity.

    LanceDB cosine distance: 0 = identical, 2 = opposite.
    We map to similarity in [0, 1] by sim = 1 - (distance / 2).
    """
    return max(0.0, min(1.0, 1.0 - (distance / 2.0)))


def _normalize_lineage_string(lineage: str) -> str:
    """Normalize lineage string formatting."""
    parts = [p.strip() for p in lineage.split(";") if p.strip()]
    return ";".join(parts)


def _extract_terminal_name(lineage: str) -> str:
    """Extract the terminal taxon name from a lineage string."""
    if not lineage:
        return ""
    parts = [p.strip() for p in lineage.split(";") if p.strip()]
    return parts[-1] if parts else lineage


def _taxonkit_reformat_name(taxon_name: str) -> Optional[str]:
    """Use TaxonKit to standardize a taxon name into a 7-level lineage string.

    This runs:
      echo "name" | taxonkit name2taxid
      echo "taxid" | taxonkit reformat -f "{k};{p};{c};{o};{f};{g};{s}" -t

    Returns None on failure.
    """
    if not taxon_name:
        return None

    try:
        name_cmd = f'echo "{taxon_name}" | taxonkit name2taxid'
        name_result = subprocess.run(
            name_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if name_result.returncode != 0 or not name_result.stdout.strip():
            return None

        taxid = name_result.stdout.strip().split("\t")[1]
        reformat_cmd = (
            f'echo {taxid} | taxonkit reformat -f "{{k}};{{p}};{{c}};{{o}};{{f}};{{g}};{{s}}" -t'
        )
        reformatted = subprocess.run(
            reformat_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if reformatted.returncode != 0 or not reformatted.stdout.strip():
            return None

        # Output format: taxid\tlineage\tname
        parts = reformatted.stdout.strip().split("\t")
        if len(parts) >= 2:
            return _normalize_lineage_string(parts[1])
        return None
    except Exception as exc:
        logger.warning(f"TaxonKit reformat failed for {taxon_name}: {exc}")
        return None


@dataclass
class PredictionResult:
    lineage: str
    confidence: float
    status: str


class TaxonomyPredictor:
    """Predict taxonomy using LanceDB vector search and weighted consensus."""

    def __init__(
        self,
        db_path: str = LANCEDB_PENDRIVE_PATH,
        table_name: str = DEFAULT_TABLE_NAME,
        enable_taxonkit: bool = True,
        novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
    ) -> None:
        self.db_path = db_path
        self.table_name = table_name
        self.enable_taxonkit = enable_taxonkit
        self.novelty_threshold = novelty_threshold
        self.db = lancedb.connect(db_path)
        logger.info(f"Connected to LanceDB at {db_path}")

    def _standardize_lineage(self, lineage: str) -> str:
        """Standardize lineage via TaxonKit; fallback to raw lineage."""
        if not lineage:
            return ""
        lineage = _normalize_lineage_string(lineage)

        if not self.enable_taxonkit:
            return lineage

        terminal_name = _extract_terminal_name(lineage)
        standardized = _taxonkit_reformat_name(terminal_name)
        return standardized if standardized else lineage

    def search_neighbors(self, query_vector: list[float], k: int = 10) -> pd.DataFrame:
        """Search LanceDB and return top-K neighbors with taxonomy and similarity.

        Args:
            query_vector: 768-dim query embedding
            k: Number of neighbors to retrieve

        Returns:
            DataFrame with columns: sequence_id, taxonomy, distance, similarity
        """
        table = self.db.open_table(self.table_name)
        search_query = table.search(query_vector)
        try:
            search_query = search_query.metric("cosine")
        except Exception:
            logger.debug("Cosine metric not available; using default LanceDB metric.")

        results = (
            search_query.limit(k)
            .select(["sequence_id", "taxonomy", "vector"])
            .to_pandas()
        )

        if results.empty:
            return pd.DataFrame(columns=["sequence_id", "taxonomy", "distance", "similarity"])

        # LanceDB returns _distance column for vector search
        if "_distance" in results.columns:
            results.rename(columns={"_distance": "distance"}, inplace=True)
        else:
            results["distance"] = 0.0

        results["similarity"] = results["distance"].apply(_cosine_similarity_from_distance)
        return results[["sequence_id", "taxonomy", "distance", "similarity"]]

    def predict_lineage(self, neighbor_df: pd.DataFrame) -> PredictionResult:
        """Predict lineage using weighted voting over neighbors.

        Consensus Logic:
            - Each neighbor votes for its lineage
            - Vote weight = similarity score
            - TaxonKit standardizes lineages before voting

        Confidence = (winning votes / total votes) * 100
        """
        if neighbor_df.empty:
            return PredictionResult(lineage="Unknown", confidence=0.0, status="UNKNOWN")

        # Standardize lineages for top candidates
        standardized = []
        for _, row in neighbor_df.iterrows():
            lineage = row.get("taxonomy") or ""
            similarity = float(row.get("similarity", 0.0))
            standardized_lineage = self._standardize_lineage(lineage)
            if standardized_lineage:
                standardized.append((standardized_lineage, similarity))

        if not standardized:
            return PredictionResult(lineage="Unknown", confidence=0.0, status="UNKNOWN")

        # Weighted voting
        vote_weights: dict[str, float] = {}
        total_weight = 0.0
        for lineage, weight in standardized:
            vote_weights[lineage] = vote_weights.get(lineage, 0.0) + weight
            total_weight += weight

        best_lineage = max(vote_weights, key=vote_weights.get)
        confidence = (vote_weights[best_lineage] / total_weight) * 100 if total_weight > 0 else 0.0

        status = "KNOWN" if confidence >= (self.novelty_threshold * 100) else "LOW_CONFIDENCE"
        return PredictionResult(lineage=best_lineage, confidence=confidence, status=status)


class NoveltyDetector:
    """Detect novelty and cluster unknown sequences using HDBSCAN."""

    def __init__(self, novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD) -> None:
        self.novelty_threshold = novelty_threshold
        self.colab_mode = _is_colab()

    def is_novel(self, top_similarity: float) -> bool:
        """Return True if top neighbor similarity is below threshold."""
        return top_similarity < self.novelty_threshold

    def cluster_novelty_pool(self, embeddings_list: list[list[float]]) -> dict:
        """Cluster novelty pool embeddings and label candidate NTUs.

        Args:
            embeddings_list: list of embedding vectors

        Returns:
            Dictionary with labels, centroids, and candidate names
        """
        if not embeddings_list:
            return {"labels": [], "centroids": [], "candidates": {}}

        # Hybrid environment support
        if self.colab_mode:
            min_cluster_size = 10
            min_samples = 5
        else:
            min_cluster_size = 5
            min_samples = 3

        clustering_result = cluster_novelty_pool(
            embeddings_list,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        # Assign candidate labels
        labels = clustering_result["labels"]
        centroids = clustering_result["centroids"]
        candidates = {}
        cluster_ids = sorted(set([l for l in labels if l >= 0]))
        for idx, cluster_id in enumerate(cluster_ids, start=1):
            candidates[cluster_id] = f"Candidate_NTU_{idx}"

        return {
            "labels": labels,
            "centroids": centroids,
            "candidates": candidates,
        }

    def generate_discovery_summary(
        self,
        total_sequences: int,
        assigned_known: int,
        novel_clusters: int,
    ) -> str:
        """Generate a discovery summary string."""
        novel_sequences = max(0, total_sequences - assigned_known)
        return (
            f"Found {total_sequences} sequences. "
            f"{assigned_known} assigned to known taxa. "
            f"{novel_sequences} sequences formed {novel_clusters} novel clusters (potential new species)."
        )
