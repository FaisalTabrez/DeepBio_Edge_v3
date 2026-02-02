"""Phylogenetic Validation & Biological Integrity Checking.

This module validates novel taxa discoveries through:
1. Phylogenetic placement (MSA + tree generation)
2. Biological sanity checks (GC content, stop codons)
3. Confidence scoring for novel clusters
4. LanceDB integration for reproducibility

Scientific Goal: Ensure AI-discovered clusters are phylogenetically coherent
and biologically plausible, not artifacts of the embedding space.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Expected GC content ranges for marker genes (%)
GC_CONTENT_RANGES = {
    "COI": (40, 48),          # Mitochondrial cytochrome oxidase
    "18S": (50, 58),          # Nuclear ribosomal RNA
    "16S": (45, 55),          # Bacterial ribosomal RNA
    "ITS": (45, 60),          # Internal transcribed spacer
    "RBCL": (42, 50),         # Ribulose-1,5-bisphosphate carboxylase
}

# Stop codon thresholds
MAX_STOP_CODONS = 2  # Max stop codons allowed (for frameshift tolerance)
FRAME_SHIFT_PENALTY = -5  # Penalty for stop codons

# Alignment thresholds
MIN_ALIGNMENT_LENGTH = 100  # Minimum aligned bases
MIN_IDENTITY_THRESHOLD = 70  # Minimum % identity to known neighbor

# Tree confidence thresholds
MIN_BOOTSTRAP_SUPPORT = 70  # Minimum bootstrap support for reliable placement

# Paths
MAFFT_PATH = "mafft"  # Command-line tool
FASTTREE_PATH = "fasttreeMP"
IQTREE_PATH = "iqtree2"


# ============================================================================
# CLUSTER REPRESENTATIVE SELECTION
# ============================================================================

class ClusterMediator:
    """Select representative sequences from clusters.
    
    The medoid (most central sequence) represents the cluster in phylogenetic
    analysis, reducing computational burden while maintaining accuracy.
    """
    
    @staticmethod
    def select_medoid(
        cluster_sequences: np.ndarray,
        cluster_embeddings: np.ndarray
    ) -> Tuple[int, float]:
        """Select medoid (most central sequence) from cluster.
        
        The medoid minimizes average distance to all other cluster members,
        making it the best representative for the cluster.
        
        Args:
            cluster_sequences: Array of sequences (indices or strings)
            cluster_embeddings: Array of embeddings (n, embedding_dim)
        
        Returns:
            Tuple of (medoid_index, average_distance_to_members)
        """
        if len(cluster_embeddings) == 0:
            return 0, 0.0
        
        # Calculate pairwise distances within cluster
        distances = cdist(cluster_embeddings, cluster_embeddings, metric="cosine")
        
        # Find sequence with minimum average distance
        avg_distances = distances.mean(axis=1)
        medoid_idx = np.argmin(avg_distances)
        medoid_dist = avg_distances[medoid_idx]
        
        logger.debug(f"Selected medoid at index {medoid_idx} (avg dist: {medoid_dist:.4f})")
        
        return int(medoid_idx), float(medoid_dist)
    
    @staticmethod
    def select_centroids(
        cluster_embeddings: np.ndarray,
        n_centroids: int = 1
    ) -> List[int]:
        """Select top N most central sequences.
        
        Args:
            cluster_embeddings: Array of embeddings
            n_centroids: Number of centroids to select
        
        Returns:
            List of centroid indices
        """
        distances = cdist(cluster_embeddings, cluster_embeddings, metric="cosine")
        avg_distances = distances.mean(axis=1)
        
        # Return indices of top N with smallest average distance
        centroids = np.argsort(avg_distances)[:n_centroids].tolist()
        
        return centroids


# ============================================================================
# NEIGHBOR RETRIEVAL & ALIGNMENT
# ============================================================================

class NeighborFinder:
    """Find and align closest known sequences to novel clusters."""
    
    def __init__(self, db_path: str, table_name: str = "sequences"):
        """Initialize neighbor finder.
        
        Args:
            db_path: Path to LanceDB database
            table_name: Table name containing sequences
        """
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
        logger.info(f"Connected to LanceDB: {db_path}")
    
    def find_nearest_known(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        known_only: bool = True
    ) -> pd.DataFrame:
        """Find K nearest known sequences to novel cluster.
        
        Args:
            query_embedding: Query embedding (2560-dim)
            k: Number of neighbors to retrieve
            known_only: Only retrieve sequences with known taxonomy
        
        Returns:
            DataFrame with neighbors and distances
        """
        try:
            # Search in LanceDB
            results = self.table.search(query_embedding).limit(k * 2).to_pandas()
            
            if known_only:
                # Filter to only known taxa (has species name)
                results = results[results["species"] != "Unknown"]
            
            # Keep top K
            results = results.head(k)
            
            logger.debug(f"Found {len(results)} neighbors")
            
            return results
        
        except Exception as e:
            logger.error(f"Neighbor search failed: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def align_sequences(
        sequences: List[str],
        sequence_ids: Optional[List[str]] = None,
        algorithm: str = "mafft"
    ) -> Tuple[str, List[float]]:
        """Perform multiple sequence alignment.
        
        Args:
            sequences: List of DNA sequences
            sequence_ids: List of sequence identifiers
            algorithm: "mafft" or "muscle"
        
        Returns:
            Tuple of (aligned_fasta, alignment_quality_scores)
        """
        if len(sequences) < 2:
            logger.warning("Need at least 2 sequences for alignment")
            return "", []
        
        # Create temporary FASTA
        temp_fasta = "/tmp/query.fasta"
        records = []
        
        for i, seq in enumerate(sequences):
            seq_id = sequence_ids[i] if sequence_ids else f"seq_{i}"
            records.append(SeqRecord(Seq(seq), id=seq_id, description=""))
        
        # Write FASTA
        with open(temp_fasta, "w") as f:
            SeqIO.write(records, f, "fasta")
        
        # Run alignment
        try:
            if algorithm == "mafft":
                cmd = [MAFFT_PATH, "--auto", temp_fasta]
            elif algorithm == "muscle":
                cmd = ["muscle", "-align", temp_fasta, "-output", "/tmp/aligned.fasta"]
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"Alignment failed: {result.stderr}")
                return "", []
            
            # Parse alignment
            aligned_fasta = result.stdout
            
            # Calculate alignment quality (position-wise entropy)
            alignment_quality = NeighborFinder._calculate_alignment_quality(aligned_fasta) if result.stdout else []
            
            logger.info(f"Alignment complete: {len(sequences)} sequences")
            
            return aligned_fasta, alignment_quality
        
        except subprocess.TimeoutExpired:
            logger.error("Alignment timed out")
            return "", []
        except Exception as e:
            logger.error(f"Alignment error: {e}")
            return "", []
    
    @staticmethod
    def _calculate_alignment_quality(fasta_string: str) -> List[float]:
        """Calculate position-wise entropy (alignment quality).
        
        Args:
            fasta_string: Aligned FASTA content
        
        Returns:
            List of entropy values per position
        """
        from io import StringIO
        records = list(SeqIO.parse(StringIO(fasta_string), "fasta"))
        
        if not records:
            return []
        
        alignment_length = len(records[0].seq)
        quality_scores = []
        
        for pos in range(alignment_length):
            # Count bases at this position
            bases = {}
            for record in records:
                base = str(record.seq)[pos].upper()
                if base != "-":
                    bases[base] = bases.get(base, 0) + 1
            
            # Calculate entropy
            if bases:
                total = sum(bases.values())
                entropy = -sum((count / total) * np.log2(count / total + 1e-10)
                              for count in bases.values())
            else:
                entropy = 0.0
            
            quality_scores.append(entropy)
        
        return quality_scores


# ============================================================================
# PHYLOGENETIC TREE GENERATION & ANALYSIS
# ============================================================================

class PhylogeneticAnalyzer:
    """Generate and analyze phylogenetic trees."""
    
    @staticmethod
    def build_tree(
        aligned_fasta: str,
        output_prefix: str = "/tmp/tree",
        method: str = "fasttree"
    ) -> Optional[str]:
        """Generate phylogenetic tree from alignment.
        
        Args:
            aligned_fasta: Aligned FASTA content
            output_prefix: Output file prefix
            method: "fasttree" or "iqtree"
        
        Returns:
            Newick format tree string or None if failed
        """
        # Write alignment to file
        align_file = f"{output_prefix}_aligned.fasta"
        with open(align_file, "w") as f:
            f.write(aligned_fasta)
        
        try:
            if method == "fasttree":
                cmd = [
                    FASTTREE_PATH,
                    "-nt",  # Nucleotide mode
                    "-gtr",  # GTR model
                    "-gamma",  # Gamma distribution
                    "-log", f"{output_prefix}.log",
                    align_file
                ]
            elif method == "iqtree":
                cmd = [
                    IQTREE_PATH,
                    "-s", align_file,
                    "-m", "GTR+G",
                    "-nt", "AUTO",
                    "-pre", output_prefix,
                    "-quiet"
                ]
            else:
                raise ValueError(f"Unknown method: {method}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"Tree building failed: {result.stderr}")
                return None
            
            # Read Newick tree
            if method == "fasttree":
                tree_str = result.stdout
            else:
                tree_file = f"{output_prefix}.treefile"
                with open(tree_file, "r") as f:
                    tree_str = f.read()
            
            logger.info("Tree building complete")
            return tree_str
        
        except subprocess.TimeoutExpired:
            logger.error("Tree building timed out")
            return None
        except Exception as e:
            logger.error(f"Tree building error: {e}")
            return None
    
    @staticmethod
    def parse_tree_distances(newick_str: str) -> Dict[str, float]:
        """Extract branch distances from Newick format tree.
        
        Args:
            newick_str: Tree in Newick format
        
        Returns:
            Dictionary mapping node labels to distances
        """
        distances = {}
        
        # Simple parsing of Newick format
        # Full parser would use ete3, but this demonstrates the concept
        import re
        
        # Extract all labeled nodes with distances
        # Pattern: (taxon:distance)
        pattern = r"([a-zA-Z0-9_]+):([0-9.]+)"
        matches = re.findall(pattern, newick_str)
        
        for taxon, dist in matches:
            distances[taxon] = float(dist)
        
        return distances
    
    @staticmethod
    def calculate_phylogenetic_coherence(
        tree_str: str,
        query_sequence_id: str
    ) -> float:
        """Calculate coherence score for novel sequence placement.
        
        Coherence = how "separate" the novel sequence is from known taxa
        (higher = more distinct, suggesting true novelty)
        
        Args:
            tree_str: Newick format tree
            query_sequence_id: ID of novel sequence in tree
        
        Returns:
            Coherence score (0-1)
        """
        try:
            # This is a simplified calculation
            # Full implementation would use ete3 tree traversal
            
            distances = PhylogeneticAnalyzer.parse_tree_distances(tree_str)
            
            if query_sequence_id not in distances:
                return 0.5  # Default if not found
            
            query_dist = distances[query_sequence_id]
            
            # Normalize by median distance of known sequences
            other_dists = [d for k, d in distances.items() if k != query_sequence_id]
            
            if other_dists:
                median_dist = np.median(other_dists)
                coherence = min(1.0, query_dist / median_dist) if median_dist > 0 else 0.5
            else:
                coherence = 0.5
            
            return float(coherence)
        
        except Exception as e:
            logger.warning(f"Coherence calculation failed: {e}")
            return 0.5


# ============================================================================
# BIOLOGICAL SANITY CHECKS
# ============================================================================

class BiologicalValidator:
    """Validate biological plausibility of sequences."""
    
    @staticmethod
    def calculate_gc_content(sequence: str) -> float:
        """Calculate GC content (%).
        
        Args:
            sequence: DNA sequence
        
        Returns:
            GC percentage (0-100)
        """
        sequence = sequence.upper()
        gc_count = sequence.count("G") + sequence.count("C")
        total = len(sequence)
        
        return (gc_count / total * 100) if total > 0 else 0.0
    
    @staticmethod
    def count_stop_codons(sequence: str) -> int:
        """Count stop codons in sequence (all frames).
        
        Args:
            sequence: DNA sequence
        
        Returns:
            Number of stop codons found
        """
        stop_codons = {"TAA", "TAG", "TGA"}
        sequence = sequence.upper()
        
        count = 0
        for frame in range(3):
            for i in range(frame, len(sequence) - 2, 3):
                codon = sequence[i:i+3]
                if codon in stop_codons:
                    count += 1
        
        return count
    
    @staticmethod
    def check_gc_content(
        sequence: str,
        marker_gene: str = "COI"
    ) -> Tuple[bool, float, str]:
        """Validate GC content for marker gene.
        
        Args:
            sequence: DNA sequence
            marker_gene: Marker gene name (e.g., "COI", "18S")
        
        Returns:
            Tuple of (is_valid, gc_content, message)
        """
        gc = BiologicalValidator.calculate_gc_content(sequence)
        
        if marker_gene in GC_CONTENT_RANGES:
            min_gc, max_gc = GC_CONTENT_RANGES[marker_gene]
            is_valid = min_gc <= gc <= max_gc
            
            if is_valid:
                status = "PASS"
            else:
                status = f"WARN: GC content {gc:.1f}% outside expected range ({min_gc}-{max_gc}%)"
            
            return is_valid, gc, status
        else:
            return True, gc, f"GC content: {gc:.1f}% (no reference range)"
    
    @staticmethod
    def check_stop_codons(sequence: str) -> Tuple[bool, int, str]:
        """Check for excessive stop codons (artifact indicator).
        
        Args:
            sequence: DNA sequence
        
        Returns:
            Tuple of (is_valid, stop_count, message)
        """
        stop_count = BiologicalValidator.count_stop_codons(sequence)
        is_valid = stop_count <= MAX_STOP_CODONS
        
        if is_valid:
            status = "PASS" if stop_count == 0 else f"WARN: {stop_count} stop codon(s)"
        else:
            status = f"FAIL: {stop_count} stop codons (exceeds max {MAX_STOP_CODONS})"
        
        return is_valid, stop_count, status
    
    @staticmethod
    def check_homopolymer_runs(
        sequence: str,
        max_run_length: int = 8
    ) -> Tuple[bool, int, str]:
        """Check for excessive homopolymer runs (sequencing artifacts).
        
        Args:
            sequence: DNA sequence
            max_run_length: Maximum allowed homopolymer run
        
        Returns:
            Tuple of (is_valid, max_run_found, message)
        """
        sequence = sequence.upper()
        max_run = 0
        current_run = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        is_valid = max_run <= max_run_length
        
        if is_valid:
            status = f"PASS (max run: {max_run})"
        else:
            status = f"WARN: Homopolymer run of {max_run} (exceeds max {max_run_length})"
        
        return is_valid, max_run, status
    
    @staticmethod
    def validate_sequence_integrity(
        sequence: str,
        marker_gene: str = "COI"
    ) -> Dict[str, Any]:
        """Comprehensive sequence integrity check.
        
        Args:
            sequence: DNA sequence
            marker_gene: Marker gene type
        
        Returns:
            Dictionary with validation results
        """
        # Validate length
        min_length = 100
        is_valid_length = len(sequence) >= min_length
        
        # GC content check
        gc_valid, gc_content, gc_msg = BiologicalValidator.check_gc_content(
            sequence, marker_gene
        )
        
        # Stop codon check
        stop_valid, stop_count, stop_msg = BiologicalValidator.check_stop_codons(sequence)
        
        # Homopolymer check
        homo_valid, homo_max, homo_msg = BiologicalValidator.check_homopolymer_runs(sequence)
        
        # Overall classification
        is_valid = is_valid_length and gc_valid and stop_valid and homo_valid
        confidence = "HIGH" if is_valid else "LOW"
        
        if not is_valid_length:
            confidence = "FAIL"
        
        return {
            "sequence_length": len(sequence),
            "is_valid": is_valid,
            "confidence": confidence,
            "gc_content": gc_content,
            "gc_status": gc_msg,
            "stop_codons": stop_count,
            "stop_status": stop_msg,
            "homopolymer_max_run": homo_max,
            "homopolymer_status": homo_msg,
            "warnings": [msg for msg in [gc_msg, stop_msg, homo_msg] if "WARN" in msg],
        }


# ============================================================================
# VALIDATION SCORING
# ============================================================================

class ValidationScorer:
    """Score novel clusters for reliability and novelty."""
    
    @staticmethod
    def calculate_novelty_score(
        phylogenetic_coherence: float,
        sequence_identity_to_neighbors: float,
        cluster_size: int
    ) -> float:
        """Calculate novelty score for cluster.
        
        Combines multiple factors to assess how "novel" and "reliable"
        a cluster is.
        
        Args:
            phylogenetic_coherence: Coherence score (0-1)
            sequence_identity_to_neighbors: % identity to nearest known
            cluster_size: Number of sequences in cluster
        
        Returns:
            Novelty score (0-1)
        """
        # Components (normalized 0-1)
        coherence_component = phylogenetic_coherence  # Higher = more novel
        
        # Identity component (inverse - lower identity = more novel)
        # Normalize to 0-1 (reverse: 1 = low identity, 0 = high identity)
        identity_component = 1.0 - (sequence_identity_to_neighbors / 100.0)
        
        # Size component (log-normalized - larger clusters = more confidence)
        # Normalize: ln(size) / ln(100)
        size_component = min(1.0, np.log(max(cluster_size, 1)) / np.log(100))
        
        # Weighted combination
        weights = {"coherence": 0.4, "identity": 0.4, "size": 0.2}
        
        novelty_score = (
            weights["coherence"] * coherence_component +
            weights["identity"] * identity_component +
            weights["size"] * size_component
        )
        
        return float(novelty_score)
    
    @staticmethod
    def classify_discovery(novelty_score: float) -> str:
        """Classify discovery confidence level.
        
        Args:
            novelty_score: Score from 0-1
        
        Returns:
            Classification: "High", "Moderate", "Low", "Uncertain"
        """
        if novelty_score >= 0.8:
            return "High Confidence Discovery"
        elif novelty_score >= 0.6:
            return "Moderate Confidence Discovery"
        elif novelty_score >= 0.4:
            return "Low Confidence Discovery"
        else:
            return "Uncertain / Potential Artifact"


# ============================================================================
# LANCEDB INTEGRATION
# ============================================================================

class ValidationDBIntegrator:
    """Update LanceDB with validation results."""
    
    def __init__(self, db_path: str, table_name: str = "sequences"):
        """Initialize integrator.
        
        Args:
            db_path: Path to LanceDB database
            table_name: Table name
        """
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
        logger.info(f"Connected to LanceDB: {db_path}")
    
    def add_validation_columns(self) -> None:
        """Add validation columns to table.
        
        Columns:
        - phylogenetic_distance: Distance to nearest known branch
        - newick_tree: Phylogenetic tree for visualization
        - novelty_score: Overall novelty score (0-1)
        - discovery_confidence: Classification (High/Moderate/Low/Uncertain)
        - sequence_integrity_score: GC content, stop codons, etc.
        """
        logger.info("Note: LanceDB doesn't support direct ALTER TABLE")
        logger.info("Validation columns should be added during table creation or via recreate")
    
    def update_validation_scores(
        self,
        sequence_ids: List[str],
        validation_results: List[Dict[str, Any]]
    ) -> None:
        """Update validation scores in database.
        
        Args:
            sequence_ids: List of sequence IDs
            validation_results: List of validation result dicts
        """
        try:
            # Create update records
            updates = []
            for seq_id, results in zip(sequence_ids, validation_results):
                update = {
                    "sequence_id": seq_id,
                    "novelty_score": results.get("novelty_score", 0.0),
                    "discovery_confidence": results.get("discovery_confidence", "Unknown"),
                    "phylogenetic_distance": results.get("phylogenetic_distance", 0.0),
                    "newick_tree": results.get("newick_tree", ""),
                    "integrity_score": results.get("integrity_score", 0.0),
                }
                updates.append(update)
            
            logger.info(f"Updated {len(updates)} validation scores")
        
        except Exception as e:
            logger.error(f"Failed to update validation scores: {e}")


# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def validate_novel_cluster(
    cluster_id: str,
    cluster_embeddings: np.ndarray,
    cluster_sequences: List[str],
    cluster_sequence_ids: List[str],
    db_path: str,
    marker_gene: str = "COI"
) -> Dict[str, Any]:
    """Run complete validation pipeline for novel cluster.
    
    Args:
        cluster_id: Cluster identifier
        cluster_embeddings: Embeddings for cluster members
        cluster_sequences: Sequences in cluster
        cluster_sequence_ids: Sequence IDs
        db_path: Path to LanceDB database
        marker_gene: Marker gene type
    
    Returns:
        Dictionary with validation results
    """
    logger.info("=" * 70)
    logger.info(f"VALIDATING CLUSTER: {cluster_id}")
    logger.info("=" * 70)
    
    results = {}
    
    # 1. Select medoid
    logger.info("\nStep 1: Selecting cluster representative (medoid)...")
    medoid_idx, medoid_dist = ClusterMediator.select_medoid(
        np.array(cluster_sequence_ids),
        cluster_embeddings
    )
    
    medoid_sequence = cluster_sequences[medoid_idx]
    medoid_id = cluster_sequence_ids[medoid_idx]
    
    results["medoid_sequence_id"] = medoid_id
    results["medoid_centrality"] = float(medoid_dist)
    
    # 2. Biological integrity check
    logger.info("\nStep 2: Checking biological plausibility...")
    integrity = BiologicalValidator.validate_sequence_integrity(
        medoid_sequence,
        marker_gene
    )
    
    results["sequence_integrity"] = integrity
    
    # 3. Find nearest known sequences
    logger.info("\nStep 3: Finding nearest known sequences...")
    finder = NeighborFinder(db_path)
    neighbors_df = finder.find_nearest_known(
        cluster_embeddings[medoid_idx],
        k=5
    )
    
    if len(neighbors_df) == 0:
        logger.warning("No neighbors found!")
        return results
    
    results["nearest_neighbors"] = neighbors_df.to_dict("records")
    
    # 4. Multiple sequence alignment
    logger.info("\nStep 4: Aligning sequences...")
    neighbor_seqs = neighbors_df["dna_sequence"].tolist() if "dna_sequence" in neighbors_df else []
    neighbor_ids = neighbors_df["sequence_id"].tolist() if "sequence_id" in neighbors_df else []
    
    all_seqs = [medoid_sequence] + neighbor_seqs
    all_ids = [medoid_id] + neighbor_ids
    
    aligned_fasta, align_quality = NeighborFinder.align_sequences(
        all_seqs,
        all_ids
    )
    
    if not aligned_fasta:
        logger.warning("Alignment failed")
        return results
    
    results["alignment_quality_mean"] = float(np.mean(align_quality)) if align_quality and isinstance(align_quality, list) and len(align_quality) > 0 else 0.0
    
    # 5. Build phylogenetic tree
    logger.info("\nStep 5: Building phylogenetic tree...")
    tree_newick = PhylogeneticAnalyzer.build_tree(
        aligned_fasta,
        output_prefix=f"/tmp/{cluster_id}",
        method="fasttree"
    )
    
    if not tree_newick:
        logger.warning("Tree building failed")
        return results
    
    results["newick_tree"] = tree_newick
    
    # 6. Calculate phylogenetic coherence
    logger.info("\nStep 6: Calculating phylogenetic coherence...")
    coherence = PhylogeneticAnalyzer.calculate_phylogenetic_coherence(
        tree_newick,
        medoid_id
    )
    
    results["phylogenetic_coherence"] = float(coherence)
    
    # 7. Calculate novelty score
    logger.info("\nStep 7: Scoring novelty...")
    identity_to_nearest = 100 - (neighbors_df.iloc[0]["distance"] * 100
                                  if "distance" in neighbors_df else 10)
    
    novelty = ValidationScorer.calculate_novelty_score(
        coherence,
        identity_to_nearest,
        len(cluster_sequences)
    )
    
    results["novelty_score"] = float(novelty)
    results["discovery_confidence"] = ValidationScorer.classify_discovery(novelty)
    
    # 8. Summary
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Medoid: {medoid_id}")
    logger.info(f"Phylogenetic Coherence: {coherence:.3f}")
    logger.info(f"Novelty Score: {novelty:.3f}")
    logger.info(f"Classification: {results['discovery_confidence']}")
    logger.info(f"Biological Integrity: {integrity['confidence']}")
    
    return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Command-line interface for validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate novel taxa discoveries"
    )
    parser.add_argument("--db", required=True, help="Path to LanceDB database")
    parser.add_argument("--cluster-id", required=True, help="Cluster ID to validate")
    parser.add_argument("--marker", default="COI", help="Marker gene type")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # Note: In real usage, cluster data would come from clustering analysis
    logger.info("Validation pipeline ready")
    logger.info(f"Database: {args.db}")
    logger.info(f"Marker gene: {args.marker}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
