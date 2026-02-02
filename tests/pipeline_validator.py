"""End-to-end pipeline validation using mock communities.

This module runs the full pipeline against known lineages and measures
precision, recall, F1-score, taxonomic depth accuracy, and robustness.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path so 'src' module can be imported
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ============================================================================
# WINDOWS COMPATIBILITY PATCHES (Must be before transformers/torch imports!)
# ============================================================================
# Mock triton and its submodules to prevent ImportError on Windows
if sys.platform == "win32":
    triton_mock = MagicMock()
    sys.modules["triton"] = triton_mock
    sys.modules["triton.language"] = triton_mock
    sys.modules["triton.ops"] = triton_mock
    sys.modules["triton.backends"] = triton_mock
    sys.modules["triton.backends.compiler"] = triton_mock
    sys.modules["triton.runtime"] = triton_mock

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.metrics import confusion_matrix

from src.config import EMBEDDING_DIM, LANCEDB_PENDRIVE_PATH, LANCEDB_TABLE_SEQUENCES

# Conditional imports to avoid loading heavy dependencies when not needed
if TYPE_CHECKING:
    from src.edge.embedder import EmbeddingEngine
    from src.edge.taxonomy import TaxonomyPredictor

from src.benchmarks.mock_community import (
    aggregate_depth_accuracy,
    build_mock_embeddings,
    build_mock_lancedb,
    deterministic_vector,
    generate_random_dna,
    lineage_depth_accuracy,
    make_chimera,
    mutate_point,
    parse_mock_community_fasta,
    split_reference_query,
    truncate_sequence,
)

logger = logging.getLogger(__name__)


try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


@dataclass
class PipelineValidationResult:
    metrics: Dict[str, Any]
    per_rank_accuracy: Dict[str, float]
    confusion: Optional[np.ndarray]
    labels: List[str]
    mutation_results: Dict[str, Any]
    fdr_result: Dict[str, Any]
    performance_profile: Dict[str, Any]
    vector_consistency: Dict[str, Any]


def _estimate_embedding_memory(num_vectors: int, dim: int = EMBEDDING_DIM) -> float:
    """Estimate embedding memory usage in MB."""
    bytes_used = num_vectors * dim * 4  # float32
    return bytes_used / (1024 * 1024)


def _process_memory_mb() -> Optional[float]:
    """Return current process memory usage in MB (if psutil available)."""
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def _drive_letter(path: str) -> Optional[str]:
    if ":" in path:
        return path.split(":", 1)[0].upper()
    return None


def _disk_io_snapshot(path: str) -> Optional[Dict[str, float]]:
    if psutil is None:
        return None
    counters = psutil.disk_io_counters(perdisk=True)
    drive = _drive_letter(path)
    if not drive:
        return None
    device = f"{drive}:"
    if device not in counters:
        return None
    stats = counters[device]
    return {
        "read_bytes": float(stats.read_bytes),
        "write_bytes": float(stats.write_bytes),
    }


def run_mock_community_validation(
    fasta_path: Path,
    db_path: Path,
    table_name: str = LANCEDB_TABLE_SEQUENCES,
    use_real_embeddings: bool = False,
    seed: int = 13,
) -> Tuple[Dict[str, Any], Dict[str, float], np.ndarray, List[str]]:
    """Run full pipeline against mock community sequences.

    Returns metrics, per-rank accuracy, confusion matrix, and labels.
    """
    entries = parse_mock_community_fasta(fasta_path)
    reference_entries, query_entries = split_reference_query(entries, seed=seed)

    if use_real_embeddings:
        # Lazy import to avoid loading transformers/torch when not needed
        from src.edge.embedder import EmbeddingEngine
        engine = EmbeddingEngine(use_gpu=False)
        ref_embeddings = engine.get_embeddings([e.sequence for e in reference_entries])
        query_embeddings = engine.get_embeddings([e.sequence for e in query_entries])
    else:
        ref_embeddings = build_mock_embeddings(reference_entries)
        query_embeddings = build_mock_embeddings(query_entries)

    build_mock_lancedb(db_path, reference_entries, ref_embeddings, table_name)

    # Lazy import, bypassing __init__.py to avoid loading embedder
    from src.edge.taxonomy import TaxonomyPredictor
    predictor = TaxonomyPredictor(
        db_path=str(db_path),
        table_name=table_name,
        enable_taxonkit=False,
        novelty_threshold=0.85,
    )

    y_true = []
    y_pred = []
    depth_scores = []

    for entry, vector in zip(query_entries, query_embeddings):
        neighbors = predictor.search_neighbors(vector.tolist(), k=5)
        prediction = predictor.predict_lineage(neighbors)
        y_true.append(entry.lineage)
        y_pred.append(prediction.lineage)
        depth_scores.append(lineage_depth_accuracy(prediction.lineage, entry.lineage))

    per_rank_accuracy = aggregate_depth_accuracy(depth_scores)
    cm_labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    from src.benchmarks.mock_community import compute_precision_recall_f1

    metrics: Dict[str, Any] = compute_precision_recall_f1(y_true, y_pred)
    metrics["taxonomic_depth_accuracy"] = per_rank_accuracy
    metrics["num_reference"] = len(reference_entries)
    metrics["num_query"] = len(query_entries)

    return metrics, per_rank_accuracy, cm, cm_labels


def run_mutation_stress_test(
    fasta_path: Path,
    db_path: Path,
    table_name: str = LANCEDB_TABLE_SEQUENCES,
    use_real_embeddings: bool = False,
) -> Dict[str, Any]:
    """Run noise/mutation stress test on a known sequence."""
    entries = parse_mock_community_fasta(fasta_path)
    if not entries:
        return {}

    reference_entries, _ = split_reference_query(entries, seed=11)
    target = reference_entries[0]

    if use_real_embeddings:
        from src.edge.embedder import EmbeddingEngine
        engine = EmbeddingEngine(use_gpu=False)
        ref_embeddings = engine.get_embeddings([e.sequence for e in reference_entries])
    else:
        ref_embeddings = build_mock_embeddings(reference_entries)

    build_mock_lancedb(db_path, reference_entries, ref_embeddings, table_name)

    from src.edge.taxonomy import TaxonomyPredictor
    predictor = TaxonomyPredictor(
        db_path=str(db_path),
        table_name=table_name,
        enable_taxonkit=False,
        novelty_threshold=0.85,
    )

    stress_results: Dict[str, Any] = {}
    variants = {
        "mutation_1%": mutate_point(target.sequence, 0.01, seed=3),
        "mutation_5%": mutate_point(target.sequence, 0.05, seed=5),
        "mutation_10%": mutate_point(target.sequence, 0.10, seed=7),
        "truncation_70%": truncate_sequence(target.sequence, 0.70),
        "truncation_50%": truncate_sequence(target.sequence, 0.50),
        "chimera": make_chimera(target.sequence, reference_entries[1].sequence, 0.5),
    }

    if use_real_embeddings:
        from src.edge.embedder import EmbeddingEngine
        engine = EmbeddingEngine(use_gpu=False)
    else:
        engine = None

    for name, seq in variants.items():
        if use_real_embeddings and engine is not None:
            vector = engine.get_embedding_single(seq)
        else:
            vector = deterministic_vector(seq)

        if vector is None:
            continue

        neighbors = predictor.search_neighbors(vector.tolist(), k=5)
        prediction = predictor.predict_lineage(neighbors)
        stress_results[name] = {
            "prediction": prediction.lineage,
            "confidence": prediction.confidence,
            "status": prediction.status,
            "depth_accuracy": lineage_depth_accuracy(prediction.lineage, target.lineage),
        }

    return stress_results


def run_vector_consistency_check(
    sequence: str,
    colab_vector_path: Optional[Path],
    similarity_threshold: float = 0.999,
) -> Dict[str, Any]:
    """Compare local vs Colab embeddings for consistency."""
    result = {
        "colab_vector_loaded": False,
        "cosine_similarity": None,
        "passes_threshold": False,
    }

    if colab_vector_path is None or not colab_vector_path.exists():
        return result

    if colab_vector_path.suffix == ".npy":
        colab_vector = np.load(colab_vector_path)
    elif colab_vector_path.suffix == ".json":
        colab_vector = np.array(json.loads(colab_vector_path.read_text()))
    else:
        raw = colab_vector_path.read_text().strip().split(",")
        colab_vector = np.array([float(v) for v in raw if v.strip()])

    from src.edge.embedder import EmbeddingEngine
    engine = EmbeddingEngine(use_gpu=False)
    local_vector = engine.get_embedding_single(sequence)

    if local_vector is None:
        return result

    colab_vector = colab_vector.astype(np.float32)
    local_vector = local_vector.astype(np.float32)
    similarity = float(np.dot(colab_vector, local_vector) / (
        np.linalg.norm(colab_vector) * np.linalg.norm(local_vector) + 1e-9
    ))

    result.update(
        {
            "colab_vector_loaded": True,
            "cosine_similarity": similarity,
            "passes_threshold": similarity >= similarity_threshold,
        }
    )
    return result


def run_false_discovery_rate_test(
    db_path: Path,
    table_name: str,
    num_sequences: int = 50,
    sequence_length: int = 200,
    seed: int = 101,
    use_real_embeddings: bool = False,
) -> Dict[str, Any]:
    """Feed random DNA and ensure low-confidence classification."""
    from src.edge.taxonomy import TaxonomyPredictor
    predictor = TaxonomyPredictor(
        db_path=str(db_path),
        table_name=table_name,
        enable_taxonkit=False,
        novelty_threshold=0.85,
    )

    false_discoveries = 0
    statuses = []
    
    if use_real_embeddings:
        from src.edge.embedder import EmbeddingEngine
        engine = EmbeddingEngine(use_gpu=False)
    else:
        engine = None
        
    for idx in range(num_sequences):
        seq = generate_random_dna(sequence_length, seed + idx)
        if use_real_embeddings:
            vector = engine.get_embedding_single(seq) if engine else None
            if vector is None:
                continue
        else:
            vector = deterministic_vector(seq)
        neighbors = predictor.search_neighbors(vector.tolist(), k=5)
        prediction = predictor.predict_lineage(neighbors)
        statuses.append(prediction.status)
        if prediction.status == "KNOWN":
            false_discoveries += 1

    fdr = false_discoveries / num_sequences if num_sequences else 0.0
    return {
        "num_sequences": num_sequences,
        "false_discoveries": false_discoveries,
        "false_discovery_rate": fdr,
        "status_breakdown": {"KNOWN": statuses.count("KNOWN"), "LOW_CONFIDENCE": statuses.count("LOW_CONFIDENCE")},
    }


def run_lancedb_performance_profile(
    db_path: Path,
    table_name: str,
    sizes: List[int],
) -> Dict[str, Any]:
    """Profile LanceDB search latency and disk I/O on pendrive."""
    import lancedb

    profile = {
        "sizes": [],
        "search_latency_ms": [],
        "index_time_s": [],
        "disk_io": [],
        "disk_throughput_mb_s": [],
        "memory_mb": [],
    }
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))

    for size in sizes:
        vectors = np.random.default_rng(42).normal(0, 1, size=(size, EMBEDDING_DIM)).astype(np.float32)
        records = [
            {
                "sequence_id": f"perf_{i}",
                "vector": vectors[i].tolist(),
                "dna_sequence": "ATGC" * 50,
                "taxonomy": "Eukaryota;Chordata;Mammalia;Primates;Hominidae;Homo;Homo sapiens",
                "marker_gene": "COI",
                "species_name": "Homo sapiens",
            }
            for i in range(size)
        ]

        try:
            db.drop_table(table_name)
        except Exception:
            pass

        io_before_index = _disk_io_snapshot(str(db_path))
        index_start = time.perf_counter()
        db.create_table(table_name, data=records)
        index_elapsed = time.perf_counter() - index_start
        io_after_index = _disk_io_snapshot(str(db_path))

        query_vec = vectors[0].tolist()

        io_before = _disk_io_snapshot(str(db_path))
        mem_before = _process_memory_mb()

        start = time.perf_counter()
        db.open_table(table_name).search(query_vec).limit(5).to_pandas()
        elapsed = (time.perf_counter() - start) * 1000

        io_after = _disk_io_snapshot(str(db_path))
        mem_after = _process_memory_mb()

        profile["sizes"].append(size)
        profile["search_latency_ms"].append(elapsed)
        profile["index_time_s"].append(index_elapsed)
        profile["memory_mb"].append(mem_after if mem_after is not None else None)

        if io_before and io_after:
            profile["disk_io"].append(
                {
                    "read_bytes": io_after["read_bytes"] - io_before["read_bytes"],
                    "write_bytes": io_after["write_bytes"] - io_before["write_bytes"],
                }
            )
        else:
            profile["disk_io"].append(None)

        if io_before_index and io_after_index and index_elapsed > 0:
            write_bytes = io_after_index["write_bytes"] - io_before_index["write_bytes"]
            profile["disk_throughput_mb_s"].append(
                (write_bytes / (1024 * 1024)) / index_elapsed
            )
        else:
            profile["disk_throughput_mb_s"].append(None)

    return profile


def run_pipeline_validation(
    fasta_path: Path,
    output_dir: Path,
    db_path: Optional[Path] = None,
    use_real_embeddings: bool = False,
    colab_vector_path: Optional[Path] = None,
) -> PipelineValidationResult:
    """Run full validation suite and return results."""
    db_path = db_path or Path(LANCEDB_PENDRIVE_PATH)
    db_path.mkdir(parents=True, exist_ok=True)

    metrics, per_rank_accuracy, cm, cm_labels = run_mock_community_validation(
        fasta_path, db_path, LANCEDB_TABLE_SEQUENCES, use_real_embeddings
    )

    dim_override = os.getenv("BIOSCAN_EMBEDDING_DIM")
    embedding_dim = int(dim_override) if dim_override and dim_override.isdigit() else EMBEDDING_DIM
    memory_estimate_mb = _estimate_embedding_memory(
        metrics.get("num_reference", 0) + metrics.get("num_query", 0), embedding_dim
    )
    metrics["embedding_memory_estimate_mb"] = memory_estimate_mb
    metrics["process_memory_mb"] = _process_memory_mb()

    mutation_results = run_mutation_stress_test(
        fasta_path, db_path, LANCEDB_TABLE_SEQUENCES, use_real_embeddings
    )

    vector_consistency = run_vector_consistency_check(
        parse_mock_community_fasta(fasta_path)[0].sequence,
        colab_vector_path,
    )

    fdr_result = run_false_discovery_rate_test(
        db_path,
        LANCEDB_TABLE_SEQUENCES,
        num_sequences=50,
        use_real_embeddings=use_real_embeddings,
    )

    profile_sizes = [1000, 5000, 10000, 50000, 100000]
    max_profile = os.getenv("BIOSCAN_PROFILE_MAX_ROWS")
    if max_profile and max_profile.isdigit():
        cap = int(max_profile)
        profile_sizes = [s for s in profile_sizes if s <= cap]

    performance_profile = run_lancedb_performance_profile(
        db_path, LANCEDB_TABLE_SEQUENCES, sizes=profile_sizes
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "per_rank_accuracy": per_rank_accuracy,
                "confusion": cm.tolist() if cm is not None else [],
                "labels": cm_labels,
                "mutation_results": mutation_results,
                "fdr_result": fdr_result,
                "performance_profile": performance_profile,
                "vector_consistency": vector_consistency,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return PipelineValidationResult(
        metrics=metrics,
        per_rank_accuracy=per_rank_accuracy,
        confusion=cm,
        labels=cm_labels,
        mutation_results=mutation_results,
        fdr_result=fdr_result,
        performance_profile=performance_profile,
        vector_consistency=vector_consistency,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run mock community pipeline validation")
    parser.add_argument(
        "--fasta",
        default=str(Path("data/test/mock_community.fasta")),
        help="Path to mock community FASTA",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("data/test/validation_reports")),
        help="Output directory",
    )
    parser.add_argument(
        "--db-path",
        default=str(Path(LANCEDB_PENDRIVE_PATH)),
        help="LanceDB path (pendrive)",
    )
    parser.add_argument(
        "--use-real-embeddings",
        action="store_true",
        help="Use EmbeddingEngine instead of deterministic mock embeddings",
    )
    parser.add_argument(
        "--colab-vector",
        default=None,
        help="Optional .npy/.json vector exported from Colab",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    result = run_pipeline_validation(
        fasta_path=Path(args.fasta),
        output_dir=Path(args.output_dir),
        db_path=Path(args.db_path),
        use_real_embeddings=args.use_real_embeddings,
        colab_vector_path=Path(args.colab_vector) if args.colab_vector else None,
    )

    logger.info("Validation complete")
    logger.info("Precision: %.3f", result.metrics.get("precision", 0.0))
    logger.info("Recall: %.3f", result.metrics.get("recall", 0.0))
    logger.info("F1: %.3f", result.metrics.get("f1_score", 0.0))
    logger.info("Depth accuracy: %.3f", result.per_rank_accuracy.get("depth_score", 0.0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
