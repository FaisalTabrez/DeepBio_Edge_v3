"""Mock community utilities for pipeline validation.

Provides deterministic mock community parsing, mutation stress tests,
lineage comparison, and LanceDB setup for benchmarking.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import lancedb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import EMBEDDING_DIM, LANCEDB_TABLE_SEQUENCES

logger = logging.getLogger(__name__)


RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


@dataclass(frozen=True)
class MockCommunityEntry:
    sequence_id: str
    sequence: str
    lineage: str
    group: str

    @property
    def lineage_levels(self) -> List[str]:
        parts = [p.strip() for p in self.lineage.split(";") if p.strip()]
        if len(parts) < 7:
            parts = parts + ["Unknown"] * (7 - len(parts))
        return parts[:7]


def _parse_fasta_header(header: str) -> Tuple[str, str, str]:
    """Parse mock community FASTA header.

    Expected format:
        >ID|group=Group|lineage=K;P;C;O;F;G;S
    """
    header = header.lstrip(">")
    parts = header.split("|")
    seq_id = parts[0].strip()
    group = "Unknown"
    lineage = "Unknown"
    for part in parts[1:]:
        if part.startswith("group="):
            group = part.split("=", 1)[1].strip()
        if part.startswith("lineage="):
            lineage = part.split("=", 1)[1].strip()
    return seq_id, group, lineage


def parse_mock_community_fasta(path: Path) -> List[MockCommunityEntry]:
    """Parse mock community FASTA into entries."""
    entries: List[MockCommunityEntry] = []
    sequence_id = ""
    group = ""
    lineage = ""
    seq_chunks: List[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sequence_id and seq_chunks:
                    sequence = "".join(seq_chunks).upper()
                    entries.append(
                        MockCommunityEntry(
                            sequence_id=sequence_id,
                            sequence=sequence,
                            lineage=lineage,
                            group=group,
                        )
                    )
                sequence_id, group, lineage = _parse_fasta_header(line)
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if sequence_id and seq_chunks:
        sequence = "".join(seq_chunks).upper()
        entries.append(
            MockCommunityEntry(
                sequence_id=sequence_id,
                sequence=sequence,
                lineage=lineage,
                group=group,
            )
        )

    return entries


def split_reference_query(
    entries: List[MockCommunityEntry],
    reference_fraction: float = 0.8,
    seed: int = 13,
) -> Tuple[List[MockCommunityEntry], List[MockCommunityEntry]]:
    """Split entries into reference and query sets."""
    rng = random.Random(seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * reference_fraction)
    return shuffled[:split_idx], shuffled[split_idx:]


def lineage_depth_accuracy(predicted: str, truth: str) -> Dict[str, float]:
    """Compute per-rank accuracy and depth score for a lineage prediction."""
    pred_parts = [p.strip() for p in predicted.split(";") if p.strip()]
    true_parts = [p.strip() for p in truth.split(";") if p.strip()]
    if len(pred_parts) < 7:
        pred_parts = pred_parts + ["Unknown"] * (7 - len(pred_parts))
    if len(true_parts) < 7:
        true_parts = true_parts + ["Unknown"] * (7 - len(true_parts))

    per_rank = {}
    depth_correct = 0
    for idx, rank in enumerate(RANKS):
        match = pred_parts[idx] == true_parts[idx]
        per_rank[rank] = 1.0 if match else 0.0
        if match and depth_correct == idx:
            depth_correct += 1

    depth_score = depth_correct / len(RANKS)
    per_rank["depth_score"] = depth_score
    return per_rank


def aggregate_depth_accuracy(results: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-rank accuracies across many sequences."""
    totals: Dict[str, float] = {rank: 0.0 for rank in RANKS}
    totals["depth_score"] = 0.0
    count = 0
    for res in results:
        for rank in RANKS:
            totals[rank] += res.get(rank, 0.0)
        totals["depth_score"] += res.get("depth_score", 0.0)
        count += 1
    if count == 0:
        return totals
    return {k: v / count for k, v in totals.items()}


def compute_precision_recall_f1(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    """Compute precision, recall, F1, and accuracy for taxonomy labels."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(acc),
    }


def mutate_point(sequence: str, rate: float, seed: int = 7) -> str:
    """Introduce point mutations at a given rate (0-1)."""
    rng = random.Random(seed)
    bases = ["A", "T", "G", "C"]
    seq_list = list(sequence.upper())
    for i, base in enumerate(seq_list):
        if rng.random() < rate:
            alternatives = [b for b in bases if b != base]
            seq_list[i] = rng.choice(alternatives)
    return "".join(seq_list)


def truncate_sequence(sequence: str, fraction: float) -> str:
    """Truncate a sequence to a fraction of its length."""
    if fraction <= 0 or fraction >= 1:
        return sequence
    new_len = max(50, int(len(sequence) * fraction))
    return sequence[:new_len]


def make_chimera(seq_a: str, seq_b: str, split: float = 0.5) -> str:
    """Create a chimera by joining two sequences at a split ratio."""
    split_a = int(len(seq_a) * split)
    split_b = len(seq_b) - split_a
    return seq_a[:split_a] + seq_b[-split_b:]


def generate_random_dna(length: int, seed: int) -> str:
    """Generate random DNA sequence."""
    rng = random.Random(seed)
    return "".join(rng.choice(["A", "T", "G", "C"]) for _ in range(length))


def build_mock_lancedb(
    db_path: Path,
    entries: List[MockCommunityEntry],
    embeddings: np.ndarray,
    table_name: str = LANCEDB_TABLE_SEQUENCES,
) -> None:
    """Create a LanceDB table for mock community sequences."""
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))

    records = []
    for entry, vector in zip(entries, embeddings):
        records.append(
            {
                "sequence_id": entry.sequence_id,
                "vector": vector.tolist(),
                "dna_sequence": entry.sequence,
                "taxonomy": entry.lineage,
                "marker_gene": "COI",
                "species_name": entry.lineage_levels[-1],
            }
        )

    try:
        db.drop_table(table_name)
    except Exception:
        pass

    db.create_table(table_name, data=records)


def deterministic_vector(sequence: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Generate a deterministic vector from a sequence for quick tests."""
    digest = hashlib.sha256(sequence.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vec = rng.normal(0, 1, size=(dim,)).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_mock_embeddings(entries: List[MockCommunityEntry]) -> np.ndarray:
    """Build deterministic mock embeddings for a list of entries."""
    vectors = [deterministic_vector(entry.sequence) for entry in entries]
    return np.vstack(vectors).astype(np.float32)
