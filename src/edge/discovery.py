"""Novelty discovery utilities using HDBSCAN."""

from __future__ import annotations

import logging
from typing import List

import hdbscan
import numpy as np

logger = logging.getLogger(__name__)


def _compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> list[list[float]]:
    """Compute representative centroids for each cluster.

    The centroid is the vector closest to the cluster mean.
    """
    centroids: list[list[float]] = []
    unique_labels = sorted(set([l for l in labels if l >= 0]))
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_vectors = embeddings[cluster_indices]
        mean_vector = cluster_vectors.mean(axis=0)
        distances = np.linalg.norm(cluster_vectors - mean_vector, axis=1)
        closest_idx = cluster_indices[int(np.argmin(distances))]
        centroids.append(embeddings[closest_idx].tolist())
    return centroids


def cluster_novelty_pool(
    embeddings_list: List[List[float]],
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
) -> dict:
    """Cluster novel embeddings using HDBSCAN.

    Args:
        embeddings_list: List of embedding vectors
        min_cluster_size: HDBSCAN min cluster size
        min_samples: HDBSCAN min samples
        metric: Distance metric

    Returns:
        Dictionary with labels and centroids
    """
    if not embeddings_list:
        return {"labels": [], "centroids": []}

    embeddings = np.asarray(embeddings_list, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    logger.info(
        "Clustering novelty pool with HDBSCAN (min_cluster_size=%s, min_samples=%s)",
        min_cluster_size,
        min_samples,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(embeddings)

    centroids = _compute_centroids(embeddings, labels)

    return {"labels": labels.tolist(), "centroids": centroids}
