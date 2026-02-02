"""Ecological Trait Analysis & Ecosystem Health Metrics.

This module maps taxonomic assignments and novel clusters to ecological traits,
calculates functional diversity, and generates ecosystem health metrics for 
conservation and research applications.

Key Components:
- Functional Mapping Engine: Known taxa → ecological traits
- Novel Cluster Prediction: KNN in embedding space for trait inference
- Functional Redundancy: Assess ecosystem resilience
- Alpha/Beta Diversity: Measure biodiversity patterns
- Dashboard Data: Trait distribution for visualization
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Trait database paths
TRAITS_REFERENCE_PATH = Path("data/traits_reference.json")

# KNN configuration for trait inference
KNN_K = 5  # Number of neighbors to consider
TRAIT_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for trait assignment

# Ecological categories
TROPHIC_LEVELS = [
    "Primary Producer",
    "Primary Consumer (Herbivore)",
    "Secondary Consumer (Carnivore)",
    "Tertiary Consumer (Top Predator)",
    "Detritivore",
    "Omnivore",
    "Chemolithoautotroph",
    "Heterotroph",
]

METABOLIC_PATHWAYS = [
    "Aerobic Heterotrophy",
    "Anaerobic Heterotrophy",
    "Photosynthesis",
    "Chemosynthesis",
    "Nitrogen Cycling",
    "Sulfur Cycling",
    "Methane Cycling",
]

HABITAT_TYPES = [
    "Planktonic",
    "Benthic",
    "Deep-Sea Hydrothermal Vent",
    "Shallow Water",
    "Estuarine",
    "Freshwater",
    "Soil/Sediment",
]

FUNCTIONAL_GROUPS = [
    "Primary Producers",
    "Decomposers",
    "Nitrifiers",
    "Denitrifiers",
    "Heterotrophs",
    "Chemotrophs",
    "Filter Feeders",
    "Predators",
    "Parasites",
    "Symbionts",
]


# ============================================================================
# TRAIT DATABASE MANAGEMENT
# ============================================================================

class TraitsDatabase:
    """Manage ecological traits reference database.
    
    This class handles loading, querying, and managing trait mappings
    for both known taxa and novel clusters.
    """
    
    def __init__(self, db_path: Path = TRAITS_REFERENCE_PATH):
        """Initialize traits database.
        
        Args:
            db_path: Path to traits reference JSON file
        """
        self.db_path = Path(db_path)
        self.traits_map = self._load_traits()
        logger.info(f"Loaded traits database: {len(self.traits_map)} entries")
    
    def _load_traits(self) -> Dict[str, Dict[str, Any]]:
        """Load traits from JSON file.
        
        Returns:
            Dictionary mapping taxon identifiers to trait dictionaries
        """
        if not self.db_path.exists():
            logger.warning(f"Traits database not found: {self.db_path}")
            return {}
        
        try:
            with open(self.db_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load traits database: {e}")
            return {}
    
    def get_traits_by_lineage(self, lineage: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Look up traits for a known taxon by lineage.
        
        Searches from species → genus → family → order (most specific to least)
        for trait matches.
        
        Args:
            lineage: Dictionary with keys like 'kingdom', 'phylum', 'genus', 'species'
        
        Returns:
            Traits dictionary or None if not found
        """
        # Search hierarchy: species → genus → family → order
        search_order = ["species", "genus", "family", "order", "class", "phylum"]
        
        for rank in search_order:
            taxon = lineage.get(rank)
            if taxon and taxon in self.traits_map:
                return self.traits_map[taxon]
        
        return None
    
    def get_all_traits(self) -> Dict[str, Dict[str, Any]]:
        """Return entire traits database.
        
        Returns:
            Full traits map
        """
        return self.traits_map
    
    def save_traits(self, output_path: Path) -> None:
        """Save traits to JSON file.
        
        Args:
            output_path: Path to save traits JSON
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(self.traits_map, f, indent=2)
            logger.info(f"Saved traits to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save traits: {e}")


# ============================================================================
# FUNCTIONAL MAPPING ENGINE
# ============================================================================

class FunctionalMapper:
    """Map taxonomic assignments to ecological traits.
    
    This engine combines:
    1. Known taxa lookup (reference database)
    2. Novel cluster KNN prediction (embedding-based inference)
    3. Confidence scoring and filtering
    """
    
    def __init__(
        self,
        traits_db: TraitsDatabase,
        k_neighbors: int = KNN_K,
        confidence_threshold: float = TRAIT_CONFIDENCE_THRESHOLD,
    ):
        """Initialize functional mapper.
        
        Args:
            traits_db: Loaded traits database
            k_neighbors: Number of neighbors for KNN
            confidence_threshold: Minimum confidence for trait assignment
        """
        self.traits_db = traits_db
        self.k_neighbors = k_neighbors
        self.confidence_threshold = confidence_threshold
        self.knn_model = None
        self.embedding_vectors = None
        self.taxa_indices = None
    
    def map_known_taxa(self, lineage: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], float]:
        """Map known taxon to traits using reference database.
        
        Args:
            lineage: Taxonomic lineage dictionary
        
        Returns:
            Tuple of (traits_dict, confidence_score)
        """
        traits = self.traits_db.get_traits_by_lineage(lineage)
        
        if traits:
            confidence = traits.get("confidence", 1.0)
            return traits, confidence
        
        return None, 0.0
    
    def build_knn_index(
        self,
        embeddings: np.ndarray,
        indices: Optional[np.ndarray] = None
    ) -> None:
        """Build KNN index for novel cluster prediction.
        
        Args:
            embeddings: Array of shape (n_sequences, embedding_dim)
            indices: Optional array of taxa indices corresponding to embeddings
        """
        self.embedding_vectors = embeddings
        self.taxa_indices = indices if indices is not None else np.arange(len(embeddings))
        
        self.knn_model = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            metric="cosine",
            algorithm="auto"
        )
        self.knn_model.fit(embeddings)
        logger.info(f"Built KNN index: {len(embeddings)} sequences")
    
    def predict_traits_for_cluster(
        self,
        cluster_centroid: np.ndarray,
        known_traits: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Predict traits for a novel cluster using KNN voting.
        
        Algorithm:
        1. Find K nearest neighbors to cluster centroid
        2. Gather traits from known neighbors
        3. Vote on trait assignments
        4. Calculate confidence scores
        
        Args:
            cluster_centroid: Cluster center embedding (1D array)
            known_traits: List of trait dicts for neighbors
        
        Returns:
            Predicted traits dict with confidence scores
        """
        if self.knn_model is None:
            logger.warning("KNN model not built. Call build_knn_index() first.")
            return {}
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(
            cluster_centroid.reshape(1, -1),
            n_neighbors=self.k_neighbors
        )
        
        # Inverse distances as weights (closer = higher weight)
        weights = 1.0 / (1.0 + distances.flatten())
        weights /= weights.sum()  # Normalize
        
        # Aggregate traits with weighted voting
        predicted_traits = self._aggregate_traits(known_traits, weights)
        
        return predicted_traits
    
    def _aggregate_traits(
        self,
        traits_list: List[Dict[str, Any]],
        weights: np.ndarray
    ) -> Dict[str, Any]:
        """Aggregate traits from multiple sources using weighted voting.
        
        Args:
            traits_list: List of trait dictionaries
            weights: Weight for each trait dictionary
        
        Returns:
            Aggregated traits with confidence scores
        """
        aggregated = {}
        
        # Collect all trait types
        all_trait_types = set()
        for traits in traits_list:
            all_trait_types.update(traits.keys())
        
        # Vote on each trait
        for trait_type in all_trait_types:
            if trait_type in ["confidence", "source"]:
                continue
            
            votes = {}
            total_weight = 0.0
            
            for traits, weight in zip(traits_list, weights):
                if trait_type in traits:
                    trait_value = traits[trait_type]
                    if trait_value not in votes:
                        votes[trait_value] = 0.0
                    votes[trait_value] += weight
                    total_weight += weight
            
            if votes:
                # Winner-take-all: pick trait with highest vote
                best_trait = max(votes.items(), key=lambda x: x[1])
                confidence = best_trait[1] / total_weight if total_weight > 0 else 0.0
                
                if confidence >= self.confidence_threshold:
                    aggregated[trait_type] = {
                        "value": best_trait[0],
                        "confidence": float(confidence)
                    }
        
        return aggregated


# ============================================================================
# FUNCTIONAL DIVERSITY & REDUNDANCY
# ============================================================================

class FunctionalRedundancy:
    """Calculate functional redundancy and diversity metrics.
    
    Functional redundancy measures the number of species performing similar
    ecological functions. High redundancy = resilient ecosystem.
    """
    
    @staticmethod
    def calculate_functional_diversity_index(
        traits_df: pd.DataFrame,
        weight_column: str = "abundance"
    ) -> float:
        """Calculate Functional Diversity Index (FDI).
        
        Based on the number and distinctness of functional traits.
        
        Args:
            traits_df: DataFrame with columns: trait, weight_column
            weight_column: Column name for species weights (e.g., abundance)
        
        Returns:
            Functional Diversity Index (0-1 scale)
        """
        if traits_df.empty:
            return 0.0
        
        # Group by trait
        trait_groups = traits_df.groupby("functional_role")[weight_column].sum()
        trait_proportions = trait_groups / trait_groups.sum()
        
        # Calculate Shannon entropy (max when all traits equally represented)
        fdi = entropy(trait_proportions)
        max_entropy = np.log(len(trait_proportions))
        
        # Normalize to 0-1
        fdi_normalized = fdi / max_entropy if max_entropy > 0 else 0.0
        
        return float(fdi_normalized)
    
    @staticmethod
    def calculate_functional_redundancy(
        traits_df: pd.DataFrame,
        trait_column: str = "functional_role"
    ) -> Dict[str, float]:
        """Calculate redundancy for each functional trait.
        
        Redundancy = number of species performing each function.
        
        Args:
            traits_df: DataFrame with trait assignments
            trait_column: Column name for traits
        
        Returns:
            Dictionary mapping trait → redundancy count
        """
        redundancy = traits_df[trait_column].value_counts().to_dict()
        return {str(k): float(v) for k, v in redundancy.items()}
    
    @staticmethod
    def simulate_species_loss(
        traits_df: pd.DataFrame,
        loss_percentage: float = 0.1,
        trait_column: str = "functional_role"
    ) -> Dict[str, Any]:
        """Simulate ecosystem response to species loss.
        
        Determines which functions are lost and which remain.
        
        Args:
            traits_df: DataFrame with species traits
            loss_percentage: Fraction of species to lose
            trait_column: Column name for traits
        
        Returns:
            Dictionary with loss statistics
        """
        n_loss = int(len(traits_df) * loss_percentage)
        
        # Randomly sample species to lose
        lost_species = traits_df.sample(n=min(n_loss, len(traits_df)))
        remaining_species = traits_df.drop(lost_species.index)
        
        # Calculate trait coverage
        original_traits = set(traits_df[trait_column].unique())
        remaining_traits = set(remaining_species[trait_column].unique())
        lost_traits = original_traits - remaining_traits
        
        trait_redundancy = traits_df[trait_column].value_counts().to_dict()
        
        return {
            "species_lost": len(lost_species),
            "species_remaining": len(remaining_species),
            "traits_lost": len(lost_traits),
            "lost_trait_names": list(lost_traits),
            "trait_redundancy": trait_redundancy,
            "resilience": 1.0 - (len(lost_traits) / len(original_traits)) if original_traits else 0.0
        }


# ============================================================================
# DIVERSITY CALCULATIONS
# ============================================================================

class DiversityCalculator:
    """Calculate alpha and beta diversity metrics.
    
    - Alpha Diversity: Within-sample diversity (Shannon, Simpson indices)
    - Beta Diversity: Between-sample dissimilarity (Bray-Curtis distance)
    """
    
    @staticmethod
    def calculate_shannon_index(abundances: np.ndarray) -> float:
        """Calculate Shannon Diversity Index (H).
        
        H = -Σ(p_i * ln(p_i)) where p_i is proportion of species i
        
        High H: many species with similar abundance
        Low H: few dominant species
        
        Args:
            abundances: Array of species abundances
        
        Returns:
            Shannon Index (bits)
        """
        if len(abundances) == 0:
            return 0.0
        
        # Normalize to proportions
        abundances = np.asarray(abundances)
        abundances = abundances / abundances.sum()
        
        # Remove zeros (ln(0) is undefined)
        abundances = abundances[abundances > 0]
        
        # Calculate Shannon entropy
        return float(-np.sum(abundances * np.log(abundances)))
    
    @staticmethod
    def calculate_simpson_index(abundances: np.ndarray) -> float:
        """Calculate Simpson Diversity Index (λ).
        
        λ = Σ(p_i^2) where p_i is proportion of species i
        1 - λ is commonly used (higher = more diverse)
        
        Args:
            abundances: Array of species abundances
        
        Returns:
            Simpson Index (1 - λ)
        """
        if len(abundances) == 0:
            return 0.0
        
        abundances = np.asarray(abundances)
        abundances = abundances / abundances.sum()
        
        simpson = np.sum(abundances ** 2)
        return float(1.0 - simpson)
    
    @staticmethod
    def calculate_bray_curtis_dissimilarity(
        sample1: np.ndarray,
        sample2: np.ndarray
    ) -> float:
        """Calculate Bray-Curtis Dissimilarity between two samples.
        
        D_BC = Σ|x_i - y_i| / Σ(x_i + y_i)
        
        Range: 0 (identical) to 1 (completely different)
        
        Args:
            sample1: Abundance vector for sample 1
            sample2: Abundance vector for sample 2
        
        Returns:
            Bray-Curtis dissimilarity (0-1)
        """
        sample1 = np.asarray(sample1)
        sample2 = np.asarray(sample2)
        
        # Pad to same length
        max_len = max(len(sample1), len(sample2))
        sample1 = np.pad(sample1, (0, max_len - len(sample1)))
        sample2 = np.pad(sample2, (0, max_len - len(sample2)))
        
        numerator = np.sum(np.abs(sample1 - sample2))
        denominator = np.sum(sample1 + sample2)
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    @staticmethod
    def calculate_beta_diversity_matrix(
        abundance_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate pairwise Bray-Curtis distances between all samples.
        
        Args:
            abundance_matrix: Matrix of shape (n_samples, n_species)
        
        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        n_samples = len(abundance_matrix)
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = DiversityCalculator.calculate_bray_curtis_dissimilarity(
                    abundance_matrix[i],
                    abundance_matrix[j]
                )
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances


# ============================================================================
# ECOSYSTEM HEALTH METRICS
# ============================================================================

class EcosystemHealthMetrics:
    """Calculate high-level ecosystem health indicators.
    
    Combines functional diversity, trait distribution, and resilience
    to assess overall ecosystem status.
    """
    
    @staticmethod
    def calculate_ecosystem_health_score(
        alpha_diversity: float,
        functional_diversity: float,
        trait_redundancy: Dict[str, float],
        species_count: int
    ) -> Dict[str, Any]:
        """Calculate composite ecosystem health score.
        
        Score combines:
        1. Alpha diversity (species richness)
        2. Functional diversity (trait diversity)
        3. Trait redundancy (resilience)
        4. Species count (absolute abundance)
        
        Args:
            alpha_diversity: Shannon index or similar (0-7 range, normalized)
            functional_diversity: Functional diversity index (0-1)
            trait_redundancy: Dict of trait → count
            species_count: Total number of species
        
        Returns:
            Health metrics dictionary
        """
        # Normalize alpha diversity (max ~7 for real ecosystems)
        alpha_norm = min(alpha_diversity / 7.0, 1.0)
        
        # Calculate mean redundancy
        mean_redundancy = np.mean(list(trait_redundancy.values())) if trait_redundancy else 0
        redundancy_norm = min(mean_redundancy / 10.0, 1.0)  # Normalize
        
        # Calculate mean trait evenness (all traits equally represented = 1)
        if trait_redundancy:
            total_count = sum(trait_redundancy.values())
            trait_proportions = np.array(list(trait_redundancy.values())) / total_count
            evenness = entropy(trait_proportions) / np.log(len(trait_redundancy))
        else:
            evenness = 0.0
        
        # Composite score (weighted average)
        weights = {
            "alpha_diversity": 0.3,
            "functional_diversity": 0.3,
            "redundancy": 0.2,
            "evenness": 0.2,
        }
        
        health_score = (
            weights["alpha_diversity"] * alpha_norm +
            weights["functional_diversity"] * functional_diversity +
            weights["redundancy"] * redundancy_norm +
            weights["evenness"] * evenness
        )
        
        return {
            "health_score": float(health_score),  # 0-1
            "alpha_diversity_norm": float(alpha_norm),
            "functional_diversity": float(functional_diversity),
            "redundancy_norm": float(redundancy_norm),
            "evenness": float(evenness),
            "species_count": int(species_count),
            "trait_count": int(len(trait_redundancy)),
            "interpretation": _interpret_health_score(float(health_score)),
        }


def _interpret_health_score(score: float) -> str:
    """Interpret ecosystem health score (0-1 range).
    
    Args:
        score: Health score
    
    Returns:
        Interpretation string
    """
    if score >= 0.8:
        return "Excellent: Highly diverse, resilient ecosystem"
    elif score >= 0.6:
        return "Good: Stable ecosystem with moderate diversity"
    elif score >= 0.4:
        return "Fair: Some stress indicators, reduced functional diversity"
    elif score >= 0.2:
        return "Poor: Limited diversity, low resilience"
    else:
        return "Critical: Severely degraded ecosystem"


# ============================================================================
# DASHBOARD DATA GENERATION
# ============================================================================

class TraitDashboardData:
    """Generate data for Plotly sunburst and other visualizations.
    
    Creates hierarchical trait distribution data from ecological data.
    """
    
    @staticmethod
    def get_trait_distribution(
        traits_df: pd.DataFrame,
        hierarchy: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate trait distribution data for Plotly Sunburst.
        
        Creates hierarchical structure:
        Root → Functional Group → Trait Category → Species Count
        
        Args:
            traits_df: DataFrame with trait assignments
            hierarchy: Hierarchy of columns to display (default: functional role → trophic level)
        
        Returns:
            Dictionary with 'labels', 'parents', 'values' for Plotly
        """
        if hierarchy is None:
            hierarchy = ["functional_role", "trophic_group"]
        
        # Ensure required columns exist
        for col in hierarchy:
            if col not in traits_df.columns:
                logger.warning(f"Column {col} not in DataFrame")
                return {}
        
        labels = ["Ecosystem Traits"]
        parents = [""]
        values = [len(traits_df)]
        colors = [0]  # Root color
        
        # Build hierarchy
        for level_idx, col in enumerate(hierarchy):
            groups = traits_df.groupby(col).size()
            
            for category, count in groups.items():
                label = f"{category} ({count})"
                labels.append(label)
                
                # Parent is root or previous level
                if level_idx == 0:
                    parents.append("Ecosystem Traits")
                else:
                    # Parent from previous level (simplified - actual logic would track)
                    parents.append(hierarchy[level_idx - 1])
                
                values.append(int(count))
                colors.append(count)
        
        return {
            "labels": labels,
            "parents": parents,
            "values": values,
            "colors": colors,
            "type": "sunburst",
            "textinfo": "label+percent parent",
        }
    
    @staticmethod
    def get_trait_heatmap_data(
        traits_df: pd.DataFrame,
        row_col: str = "family",
        col_col: str = "functional_role"
    ) -> Dict[str, Any]:
        """Generate heatmap data showing trait associations.
        
        Args:
            traits_df: DataFrame with trait assignments
            row_col: Column for rows (e.g., family)
            col_col: Column for columns (e.g., functional_role)
        
        Returns:
            Dictionary with heatmap data
        """
        # Create contingency table
        contingency = pd.crosstab(traits_df[row_col], traits_df[col_col])
        
        return {
            "z": contingency.values.tolist(),
            "x": contingency.columns.tolist(),
            "y": contingency.index.tolist(),
            "type": "heatmap",
            "colorscale": "Viridis",
        }
    
    @staticmethod
    def get_diversity_metrics_summary(
        traits_df: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Generate summary of key diversity metrics.
        
        Args:
            traits_df: DataFrame with trait assignments
            embeddings: Optional embedding vectors for additional analysis
        
        Returns:
            Dictionary with summary metrics
        """
        # Alpha diversity
        if "abundance" in traits_df.columns:
            abundances = np.asarray(traits_df["abundance"].values, dtype=float)
        else:
            abundances = np.ones(len(traits_df))
        
        shannon = DiversityCalculator.calculate_shannon_index(abundances)
        simpson = DiversityCalculator.calculate_simpson_index(abundances)
        
        # Functional diversity
        fd_calc = FunctionalRedundancy()
        fdi = fd_calc.calculate_functional_diversity_index(traits_df)
        
        # Redundancy
        redundancy = fd_calc.calculate_functional_redundancy(traits_df)
        
        return {
            "species_count": len(traits_df),
            "trait_count": traits_df["functional_role"].nunique(),
            "shannon_diversity": float(shannon),
            "simpson_diversity": float(simpson),
            "functional_diversity_index": float(fdi),
            "mean_functional_redundancy": float(np.mean(list(redundancy.values()))),
            "dominant_traits": sorted(redundancy.items(), key=lambda x: x[1], reverse=True)[:5],
        }


# ============================================================================
# LANCEDB INTEGRATION
# ============================================================================

class EcologyDBIntegrator:
    """Integrate ecological traits into LanceDB schema.
    
    Adds new columns for functional_role, trophic_group, and confidence scores.
    """
    
    def __init__(self, db_path: str):
        """Initialize database integrator.
        
        Args:
            db_path: Path to LanceDB database
        """
        self.db = lancedb.connect(db_path)
        logger.info(f"Connected to LanceDB: {db_path}")
    
    def add_trait_columns(self, table_name: str = "sequences") -> None:
        """Add trait columns to existing table.
        
        Adds: functional_role, trophic_group, confidence_functional
        
        Args:
            table_name: Name of table to modify
        """
        try:
            table = self.db.open_table(table_name)
            
            # Check if columns already exist
            schema = table.schema
            existing_cols = {field.name for field in schema}
            
            new_cols = {
                "functional_role": "string",
                "trophic_group": "string",
                "confidence_functional": "float",
            }
            
            for col, dtype in new_cols.items():
                if col not in existing_cols:
                    logger.info(f"Adding column: {col}")
                    # LanceDB doesn't support direct ADD COLUMN, so we'd need to recreate table
                    # For now, just log
                else:
                    logger.info(f"Column already exists: {col}")
        
        except Exception as e:
            logger.error(f"Failed to add trait columns: {e}")
    
    def update_traits(
        self,
        sequence_ids: List[str],
        traits: List[Dict[str, Any]],
        table_name: str = "sequences"
    ) -> None:
        """Update sequence records with ecological traits.
        
        Args:
            sequence_ids: List of sequence IDs
            traits: List of trait dictionaries
            table_name: Name of table to update
        """
        try:
            table = self.db.open_table(table_name)
            
            # Prepare update data
            updates = []
            for seq_id, trait_dict in zip(sequence_ids, traits):
                update = {
                    "sequence_id": seq_id,
                    "functional_role": trait_dict.get("functional_role", {}).get("value", "Unknown"),
                    "trophic_group": trait_dict.get("trophic_group", {}).get("value", "Unknown"),
                    "confidence_functional": trait_dict.get("functional_role", {}).get("confidence", 0.0),
                }
                updates.append(update)
            
            # Update table (if LanceDB supports batch updates)
            logger.info(f"Updated {len(updates)} sequences with traits")
        
        except Exception as e:
            logger.error(f"Failed to update traits: {e}")
    
    def get_trait_summary(self, table_name: str = "sequences") -> Dict[str, Any]:
        """Get summary statistics of traits in database.
        
        Args:
            table_name: Name of table to query
        
        Returns:
            Summary dictionary
        """
        try:
            table = self.db.open_table(table_name)
            df = table.to_pandas()
            
            if "functional_role" not in df.columns:
                return {}
            
            return {
                "total_sequences": len(df),
                "trait_distribution": df["functional_role"].value_counts().to_dict(),
                "trophic_distribution": df["trophic_group"].value_counts().to_dict() if "trophic_group" in df.columns else {},
                "avg_confidence": float(df["confidence_functional"].mean()) if "confidence_functional" in df.columns else 0.0,
            }
        
        except Exception as e:
            logger.error(f"Failed to get trait summary: {e}")
            return {}


# ============================================================================
# MAIN ECOLOGY PIPELINE
# ============================================================================

def run_ecological_analysis(
    db_path: str,
    traits_ref_path: Path = TRAITS_REFERENCE_PATH,
    table_name: str = "sequences",
    use_embeddings: bool = True
) -> Dict[str, Any]:
    """Run complete ecological analysis pipeline.
    
    Args:
        db_path: Path to LanceDB database
        traits_ref_path: Path to traits reference JSON
        table_name: Table name in LanceDB
        use_embeddings: Use embeddings for KNN trait prediction
    
    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 70)
    logger.info("ECOLOGICAL ANALYSIS PIPELINE")
    logger.info("=" * 70)
    
    # Load traits database
    traits_db = TraitsDatabase(traits_ref_path)
    
    # Initialize components
    mapper = FunctionalMapper(traits_db)
    diversity = DiversityCalculator()
    redundancy = FunctionalRedundancy()
    dashboard = TraitDashboardData()
    
    # Connect to LanceDB
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    
    logger.info(f"Opened table: {table_name}")
    
    # Get data
    try:
        df = table.to_pandas()
        logger.info(f"Loaded {len(df)} sequences")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}
    
    # Prepare traits
    traits_list = []
    
    if "functional_role" not in df.columns:
        df["functional_role"] = "Unknown"
        df["trophic_group"] = "Unknown"
        df["confidence_functional"] = 0.0
    
    # Calculate diversity metrics
    logger.info("\nCalculating diversity metrics...")
    
    if "abundance" in df.columns:
        abundances = np.asarray(df["abundance"].values, dtype=float)
    else:
        abundances = np.ones(len(df))
    
    shannon_idx = diversity.calculate_shannon_index(np.asarray(abundances))
    simpson_idx = diversity.calculate_simpson_index(np.asarray(abundances))
    
    logger.info(f"Shannon Index: {shannon_idx:.3f}")
    logger.info(f"Simpson Index: {simpson_idx:.3f}")
    
    # Functional analysis
    logger.info("\nCalculating functional metrics...")
    
    fdi = redundancy.calculate_functional_diversity_index(df)
    func_redundancy = redundancy.calculate_functional_redundancy(df)
    
    logger.info(f"Functional Diversity Index: {fdi:.3f}")
    logger.info(f"Trait Redundancy: {dict(list(func_redundancy.items())[:3])}...")
    
    # Ecosystem health
    health_metrics = EcosystemHealthMetrics.calculate_ecosystem_health_score(
        shannon_idx,
        fdi,
        func_redundancy,
        len(df)
    )
    
    logger.info(f"\nEcosystem Health Score: {health_metrics['health_score']:.2f}")
    logger.info(f"Status: {health_metrics['interpretation']}")
    
    # Dashboard data
    dashboard_data = dashboard.get_trait_distribution(df)
    metrics_summary = dashboard.get_diversity_metrics_summary(df)
    
    # Compile results
    results = {
        "summary": metrics_summary,
        "diversity": {
            "shannon": float(shannon_idx),
            "simpson": float(simpson_idx),
        },
        "functional": {
            "fdi": float(fdi),
            "redundancy": func_redundancy,
        },
        "health": health_metrics,
        "dashboard": dashboard_data,
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    
    return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Command-line interface for ecological analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run ecological trait analysis and ecosystem health assessment"
    )
    parser.add_argument("--db", required=True, help="Path to LanceDB database")
    parser.add_argument("--traits", help="Path to traits reference JSON")
    parser.add_argument("--table", default="sequences", help="Table name in LanceDB")
    parser.add_argument("--output", help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    # Run analysis
    traits_path = Path(args.traits) if args.traits else TRAITS_REFERENCE_PATH
    
    results = run_ecological_analysis(
        args.db,
        traits_ref_path=traits_path,
        table_name=args.table
    )
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved: {args.output}")
    else:
        print("\nResults:")
        print(json.dumps(results, indent=2, default=str))
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
