"""Tests for taxonomy prediction and novelty detection."""

import numpy as np
import pandas as pd

from src.edge.discovery import cluster_novelty_pool
from src.edge.taxonomy import NoveltyDetector, TaxonomyPredictor


def test_consensus_voting_cnidaria():
    """If 8/10 neighbors are Cnidaria, consensus should be Cnidaria."""
    data = {
        "taxonomy": [
            "Eukaryota;Animalia;Cnidaria;Anthozoa;Scleractinia;Acropora;Acropora cervicornis",
        ]
        * 8
        + [
            "Eukaryota;Animalia;Arthropoda;Insecta;Diptera;Drosophila;Drosophila melanogaster",
        ]
        * 2,
        "similarity": [0.95] * 8 + [0.80] * 2,
    }
    neighbor_df = pd.DataFrame(data)

    predictor = TaxonomyPredictor(enable_taxonkit=False)
    result = predictor.predict_lineage(neighbor_df)

    assert "Cnidaria" in result.lineage
    assert result.confidence > 70.0


def test_hdbscan_identical_vectors_one_cluster():
    """Three identical vectors should form a single cluster."""
    v = [0.1] * 768
    embeddings = [v, v, v]

    result = cluster_novelty_pool(embeddings, min_cluster_size=2, min_samples=1)
    labels = result["labels"]

    # All labels should be the same non-negative cluster
    unique_labels = set(labels)
    assert len(unique_labels) == 1
    assert list(unique_labels)[0] >= 0


def test_novelty_threshold():
    """Top similarity below threshold should be novel."""
    detector = NoveltyDetector(novelty_threshold=0.85)
    assert detector.is_novel(0.84) is True
    assert detector.is_novel(0.90) is False
