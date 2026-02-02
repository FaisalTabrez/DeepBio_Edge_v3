# 🌍 Ecological Traits & Functional Diversity Analysis

**Integrating Functional Ecology into GlobalBioScan Research**

---

## Table of Contents

1. [Overview](#overview)
2. [Methodology](#methodology)
3. [Trait Mapping System](#trait-mapping-system)
4. [Novel Cluster Prediction](#novel-cluster-prediction)
5. [Diversity Calculations](#diversity-calculations)
6. [Functional Redundancy](#functional-redundancy)
7. [Ecosystem Health Metrics](#ecosystem-health-metrics)
8. [Data Integration](#data-integration)
9. [Usage Examples](#usage-examples)
10. [Citation & References](#citation--references)

---

## Overview

### Purpose

Add **Functional Depth** to eDNA metabarcoding research by mapping taxonomic assignments to ecological traits. This enables:

- **Functional Redundancy Assessment**: Understand ecosystem resilience
- **Trait-Based Diversity**: Move beyond species lists to functional capabilities
- **Novel Cluster Characterization**: Infer functions of unclassified organisms
- **Ecosystem Health Scoring**: Composite metric for conservation prioritization

### Key Innovations

1. **Known Taxa Mapping**: Hierarchical lookup system (species → genus → family)
2. **KNN-Based Inference**: Predict traits of novel clusters from embedding neighbors
3. **Hierarchical Diversity**: Combine alpha, beta, and functional metrics
4. **Trait Sunburst Visualization**: Interactive functional hierarchy display

---

## Methodology

### Data Sources

#### FAPROTAX (Functional Annotation of Prokaryotic Taxonomy)
- **Coverage**: Bacterial and Archaeal traits
- **Categories**: 
  - Trophic level (autotroph, heterotroph, mixotroph)
  - Metabolic pathways (N-cycling, S-cycling, methanogenesis)
  - Habitat preferences
  - Lifestyle (planktonic, biofilm, etc.)
- **Integration**: JSON lookup table (example: `data/traits_reference.json`)

#### WoRMS Traits (World Register of Marine Species)
- **Coverage**: Marine eukaryotic organisms
- **Categories**:
  - Feeding mode (filter feeder, predator, herbivore, etc.)
  - Motility
  - Depth range
  - Salinity tolerance
  - Thermal range

#### Custom Databases
- Domain-specific traits (deep-sea hydrothermal vent specialists)
- Metabolic capabilities (chemosynthetic vs photosynthetic)
- Biogeographic distributions

### Workflow

```
Sequences with Taxonomy (from Phase 4)
    ↓
Known Taxa Lookup (TaxonKit lineage)
    ├─ Hit: Direct trait assignment ✓
    └─ Miss: Novel cluster candidate
    ↓
Embedding Generation (2560-dim, NT-2.5B)
    ↓
KNN Search in Embedding Space
    ├─ Find K=5 nearest neighbors
    ├─ Gather neighbor traits
    └─ Predict via weighted voting
    ↓
Trait Assignment (with confidence scores)
    ├─ Functional role
    ├─ Trophic group
    └─ Metabolic pathway
    ↓
LanceDB Update (new columns: functional_role, trophic_group, confidence_functional)
    ↓
Diversity Calculations
    ├─ Alpha diversity (Shannon, Simpson)
    ├─ Functional diversity (FDI)
    ├─ Beta diversity (Bray-Curtis)
    └─ Functional redundancy
    ↓
Ecosystem Health Score
    └─ Output: Health metrics, interpretation, dashboard data
```

---

## Trait Mapping System

### Known Taxa Lookup

**Algorithm**: Hierarchical search from most to least specific

```python
def map_known_taxa(lineage: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Search order: species → genus → family → order → class
    
    Example:
        Input: {"kingdom": "Bacteria", "phylum": "Proteobacteria", 
                "genus": "Vibrio", "species": "V. parahaemolyticus"}
        Output: {"functional_role": "Heterotroph", 
                 "trophic_group": "Primary Consumer",
                 "confidence": 0.92, ...}
    """
```

**Confidence Scoring**:
- Species-level match: confidence = 0.95+ (high certainty)
- Genus-level match: confidence = 0.85-0.90 (moderate)
- Family-level match: confidence = 0.75-0.80 (lower)
- No match: confidence = 0.0 (unknown)

### Traits Reference Structure

```json
{
  "Bacteria|Proteobacteria|Gammaproteobacteria|Vibronales|Vibrionaceae|Vibrio|Vibrio parahaemolyticus": {
    "source": "WoRMS",
    "confidence": 0.92,
    "functional_role": "Heterotroph",
    "trophic_group": "Primary Consumer",
    "metabolic_pathway": "Aerobic Heterotrophy",
    "habitat": "Shallow Water",
    "traits": ["Marine pathogen", "Motile", "Salt-tolerant"]
  }
}
```

### Trait Categories

| Category | Options | Source |
|----------|---------|--------|
| **Functional Role** | Primary Producer, Decomposer, Nitrifier, Heterotroph, Predator, Filter Feeder, Parasite | FAPROTAX, WoRMS |
| **Trophic Level** | Primary Producer, Primary Consumer, Secondary Consumer, Tertiary Consumer, Detritivore, Chemolithoautotroph | Ecological theory |
| **Metabolic Pathway** | Aerobic Heterotrophy, Photosynthesis, Chemosynthesis, Nitrogen Cycling, Sulfur Cycling, Methane Cycling | FAPROTAX |
| **Habitat** | Planktonic, Benthic, Deep-Sea, Freshwater, Estuarine, Soil | WoRMS, domain knowledge |

---

## Novel Cluster Prediction

### Problem Statement

For novel/unclassified clusters (no matches in reference taxonomy):
- Cannot do direct trait lookup
- Need inference from known neighbors

### K-Nearest Neighbors (KNN) Solution

**Algorithm**:

1. **Build Index**: 
   - Use 2560-dimensional embeddings from NT-2.5B model
   - Build KNN tree (cosine distance, 5 neighbors)

2. **Find Neighbors**:
   ```
   Query: cluster_centroid (novel cluster's mean embedding)
   K = 5 (number of nearest neighbors)
   Distance metric: Cosine similarity
   ```

3. **Gather Neighbor Traits**:
   ```
   For each of K=5 neighbors:
       Get their known traits (from taxonomy lookup)
   ```

4. **Weighted Voting**:
   ```
   Weight_i = 1 / (1 + distance_i)  # Closer neighbors = higher weight
   
   For each trait type (e.g., "trophic_group"):
       votes = {trait_value: Σweight_i for all neighbors with that trait}
       predicted_trait = argmax(votes)
       confidence = max(votes) / Σweight_i
   ```

5. **Confidence Filtering**:
   - Only assign if confidence ≥ 0.6
   - Otherwise: mark as "Low Confidence" or "Unknown"

### Example

```
Novel cluster centroid: embedding_vector (2560-dim)
    ↓
KNN search with K=5
    ↓
Neighbors found:
  1. Vibrio parahaemolyticus (distance=0.05, confidence=0.92, trophic="Primary Consumer")
  2. Photobacterium phosphoreum (distance=0.08, confidence=0.90, trophic="Primary Consumer")
  3. Alteromonas macleodii (distance=0.12, confidence=0.87, trophic="Heterotroph")
  4. Vibrio vulnificus (distance=0.15, confidence=0.91, trophic="Primary Consumer")
  5. Colwellia psychrerythraea (distance=0.18, confidence=0.85, trophic="Heterotroph")
    ↓
Weighted voting:
  weights = [0.95, 0.93, 0.89, 0.87, 0.85]
  "Primary Consumer": 0.95 + 0.93 + 0.87 = 2.75 (highest)
  "Heterotroph": 0.89 + 0.85 = 1.74
    ↓
Prediction:
  trophic_group = "Primary Consumer"
  confidence = 2.75 / 4.39 = 0.63 ✓ (≥ 0.6 threshold)
```

### Confidence Interpretation

- **≥ 0.80**: High confidence (reliable for analysis)
- **0.60-0.79**: Moderate confidence (use with caution)
- **< 0.60**: Low confidence (mark as uncertain)

---

## Diversity Calculations

### Alpha Diversity (Within-Sample)

#### Shannon Diversity Index (H)

```
H = -Σ(p_i * ln(p_i))

where p_i = abundance of species i / total abundance

Interpretation:
  - H = 0: Only one species present
  - H = 2-3: Moderate diversity (typical for eDNA samples)
  - H > 4: High diversity (many species with even abundance)
  - Max = ln(S) where S = number of species
```

**In GlobalBioScan**: Calculate Shannon Index for species (traditional) AND for functional groups (novel).

**Functional Shannon**:
```
H_func = -Σ(p_functional * ln(p_functional))

where p_functional = total abundance of species with that function

Example:
  If "Primary Producer" functions represent 20% of reads
  and "Decomposer" functions represent 80%
  → Lower H_func = less functionally diverse ecosystem
```

#### Simpson Diversity Index (D)

```
D_Simpson = Σ(p_i^2)
D_Simpson_adjusted = 1 - Σ(p_i^2)  # Commonly used

Interpretation:
  - D_adj = 0: Only one species
  - D_adj = 0.5-0.8: Moderate diversity
  - D_adj > 0.9: High diversity

Advantage over Shannon: Less sensitive to rare species
```

### Beta Diversity (Between-Sample)

#### Bray-Curtis Dissimilarity

```
D_BC = Σ|x_i - y_i| / Σ(x_i + y_i)

where x_i, y_i = abundances of species i in samples X and Y

Range: 0 (identical) to 1 (completely different)

Interpretation:
  - D_BC = 0: Identical species composition
  - D_BC < 0.5: Similar ecosystems (might share dominant species)
  - D_BC > 0.8: Very different ecosystems
```

**Use Case**: Compare trait diversity between:
- Geographic regions (coastal vs open ocean)
- Temporal samples (before/after disturbance)
- Ecological zones (shallow vs deep sea)

### Functional Diversity Index (FDI)

Custom metric combining trait diversity and abundance:

```
FDI = H_func / ln(K_func)

where:
  H_func = Shannon index of functional groups
  K_func = number of functional groups observed

Range: 0 (all same function) to 1 (maximum functional diversity)

Advantages:
  - Incorporates both richness (K) and evenness (H)
  - Normalized to 0-1 scale
  - Comparable across studies
```

---

## Functional Redundancy

### Definition

**Functional Redundancy**: Number of species performing the same ecological function.

High redundancy = ecosystem resilience (loss of one species doesn't eliminate that function).

### Calculation

```python
redundancy = {
    "Primary Producer": 15,    # 15 species produce primary biomass
    "Decomposer": 45,          # 45 species decompose organic matter
    "Nitrifier": 8,            # 8 species cycle nitrogen
    "Predator": 12,            # 12 species are predators
}

mean_redundancy = mean([15, 45, 8, 12]) = 20.0
```

### Ecosystem Resilience Scenario

**Question**: What happens if we lose 10% of species?

```python
species_before = 100
species_after_loss = 90

functions_before = ["Primary Producer", "Decomposer", "Nitrifier", "Predator"]
# Simulate random loss
functions_after_loss = ["Primary Producer", "Decomposer", "Predator"]
# Lost "Nitrifier" function!

resilience = (len(functions_after) / len(functions_before)) × 100
           = (3 / 4) × 100
           = 75%

Interpretation: Losing 10% of species resulted in losing 25% of functions.
High redundancy species (e.g., Decomposers with 45 members) can absorb losses.
Low redundancy species (e.g., Nitrifiers with 8 members) are vulnerable.
```

---

## Ecosystem Health Metrics

### Composite Health Score

```
Health Score = 0.3 × Alpha_Norm + 0.3 × FDI + 0.2 × Redundancy + 0.2 × Evenness

Range: 0 (critical) to 1 (excellent)

Interpretation:
  ≥ 0.80: Excellent   (highly diverse, resilient, stable)
  0.60-0.79: Good     (stable, moderate stress tolerant)
  0.40-0.59: Fair     (declining diversity, increasing stress)
  0.20-0.39: Poor     (limited diversity, low resilience)
  < 0.20: Critical    (severely degraded, vulnerable to disturbance)
```

### Calculation Steps

1. **Normalize Alpha Diversity**:
   ```
   Alpha_Norm = min(Shannon_Index / 7.0, 1.0)
   # Max Shannon ≈ 7 for diverse natural ecosystems
   ```

2. **Calculate FDI**: Already normalized (0-1)

3. **Normalize Redundancy**:
   ```
   Redundancy_Norm = min(mean_redundancy / 10.0, 1.0)
   # Assuming mean redundancy ~10 is "healthy"
   ```

4. **Calculate Evenness**:
   ```
   Evenness = Shannon_Index / ln(Species_Count)
   # 0 (uneven, few dominants) to 1 (even distribution)
   ```

5. **Weight and Sum**:
   ```
   Health = 0.3×Alpha + 0.3×FDI + 0.2×Redundancy + 0.2×Evenness
   ```

### Conservation Applications

**Priority Ranking**:
- Health < 0.3: **URGENT** intervention needed
- Health 0.3-0.5: **HIGH** priority
- Health 0.5-0.7: **MEDIUM** priority
- Health > 0.7: **MONITOR** (low threat)

**Example Report for CMLRE**:
```
Ecosystem Health Assessment - Arctic Ocean eDNA Survey
Date: 2026-02-01

Site 1 (Coastal):
  Health Score: 0.72 (GOOD)
  Status: Stable, moderate functional diversity
  Recommendation: Continue monitoring

Site 2 (Deep Sea):
  Health Score: 0.45 (FAIR)
  Status: Declining diversity, emerging stress signals
  Recommendation: Investigate stressors (pollution, temperature, etc.)

Site 3 (Hydrothermal Vent):
  Health Score: 0.55 (FAIR)
  Status: Specialized community, moderate redundancy risk
  Recommendation: Protect from mining/extraction activities
```

---

## Data Integration

### LanceDB Schema Update

**New columns added to `sequences` table**:

```sql
ALTER TABLE sequences ADD COLUMN functional_role VARCHAR;
ALTER TABLE sequences ADD COLUMN trophic_group VARCHAR;
ALTER TABLE sequences ADD COLUMN confidence_functional FLOAT;
```

**Example records**:

| sequence_id | dna_sequence | vector (2560-dim) | kingdom | phylum | genus | species | functional_role | trophic_group | confidence_functional |
|---|---|---|---|---|---|---|---|---|---|
| seq_001 | ATGC... | [0.1, 0.2, ...] | Bacteria | Proteobacteria | Vibrio | V. parahaemolyticus | Heterotroph | Primary Consumer | 0.92 |
| seq_042 | ATGC... | [0.15, 0.18, ...] | Novel | Novel | Novel | Novel | Heterotroph (KNN) | Primary Consumer | 0.63 |
| seq_100 | ATGC... | [0.08, 0.25, ...] | Bacteria | Bacteroidota | Alistipes | A. finegoldii | Decomposer | Heterotroph | 0.95 |

### Query Examples

**Find all primary producers**:
```python
table = db.open_table("sequences")
producers = table.search("functional_role = 'Primary Producer'").limit(100).to_pandas()
```

**Get low-confidence predictions**:
```python
uncertain = table.search("confidence_functional < 0.6").limit(50).to_pandas()
print(f"Uncertain trait assignments: {len(uncertain)}")
```

**Calculate functional diversity by region**:
```python
for region in regions:
    region_data = table.search(f"region = '{region}'").to_pandas()
    fdi = calculate_functional_diversity_index(region_data)
    print(f"{region}: FDI = {fdi:.3f}")
```

---

## Usage Examples

### 1. Load and Map Traits

```python
from src.edge.ecology import TraitsDatabase, FunctionalMapper

# Load reference database
traits_db = TraitsDatabase("data/traits_reference.json")

# Initialize mapper
mapper = FunctionalMapper(traits_db)

# Map known taxa
lineage = {
    "kingdom": "Bacteria",
    "phylum": "Proteobacteria",
    "genus": "Vibrio",
    "species": "V. parahaemolyticus"
}

traits, confidence = mapper.map_known_taxa(lineage)
print(f"Traits: {traits}")
print(f"Confidence: {confidence:.2f}")
```

**Output**:
```
Traits: {
    'functional_role': 'Heterotroph',
    'trophic_group': 'Primary Consumer',
    'metabolic_pathway': 'Aerobic Heterotrophy',
    'habitat': 'Shallow Water',
    'traits': ['Marine pathogen', 'Motile', 'Salt-tolerant']
}
Confidence: 0.92
```

### 2. Predict Traits for Novel Clusters

```python
# Build KNN index
embeddings = get_embeddings_from_lancedb()  # Shape: (n_sequences, 2560)
mapper.build_knn_index(embeddings)

# Predict trait for novel cluster
novel_centroid = embeddings[42]  # Example novel cluster
predicted_traits = mapper.predict_traits_for_cluster(
    novel_centroid,
    known_traits=[...]
)

print(f"Predicted: {predicted_traits}")
```

### 3. Calculate Diversity Metrics

```python
from src.edge.ecology import DiversityCalculator

diversity = DiversityCalculator()

# Alpha diversity
abundances = [10, 20, 15, 5, 3]  # Species counts
shannon = diversity.calculate_shannon_index(abundances)
simpson = diversity.calculate_simpson_index(abundances)

print(f"Shannon: {shannon:.3f}")
print(f"Simpson: {simpson:.3f}")

# Beta diversity
sample1 = np.array([10, 20, 15])
sample2 = np.array([15, 18, 20])
bc_dist = diversity.calculate_bray_curtis_dissimilarity(sample1, sample2)

print(f"Bray-Curtis: {bc_dist:.3f}")
```

### 4. Assess Ecosystem Health

```python
from src.edge.ecology import run_ecological_analysis

# Run complete pipeline
results = run_ecological_analysis(
    db_path="data/vectors/lancedb.lance",
    table_name="sequences"
)

# Print results
print(f"Health Score: {results['health']['health_score']:.2f}")
print(f"Status: {results['health']['interpretation']}")
print(f"Shannon Diversity: {results['diversity']['shannon']:.3f}")
print(f"Functional Diversity: {results['functional']['fdi']:.3f}")
```

**Output**:
```
Health Score: 0.72
Status: Good: Stable ecosystem with moderate diversity
Shannon Diversity: 3.45
Functional Diversity: 0.68
```

### 5. Generate Dashboard Data

```python
from src.edge.ecology import TraitDashboardData

dashboard = TraitDashboardData()

# Get trait distribution for Sunburst chart
traits_df = get_traits_from_lancedb()  # Get DataFrame

sunburst_data = dashboard.get_trait_distribution(
    traits_df,
    hierarchy=["functional_role", "trophic_group"]
)

# Use with Plotly
import plotly.graph_objects as go

fig = go.Figure(go.Sunburst(**sunburst_data))
fig.show()
```

---

## Citation & References

### Primary Literature

1. **Functional Annotation of Prokaryotic Taxonomy (FAPROTAX)**
   - Louca, S., Parfrey, L. W., & Doebeli, M. (2016). "Decoupling function and taxonomy in the global ocean microbiome." *Science*, 353(6305), aaf4507.
   - Database: http://faprotax.readthedocs.io/

2. **World Register of Marine Species (WoRMS)**
   - Horton, T., et al. (2021). *World Register of Marine Species*.
   - Accessed via: https://www.marinespecies.org/

3. **Ecological Diversity Indices**
   - Hill, M. O. (1973). "Diversity and evenness: a unifying notation and its consequences." *Ecology*, 54(2), 427-432.
   - Shannon, C. E. (1948). "A mathematical theory of communication." *Bell System Technical Journal*, 27(3), 379-423.

4. **Functional Ecology**
   - Villéger, S., Mason, N. W., & Mouillot, D. (2008). "New multidimensional functional diversity indices for a multifaceted framework in functional ecology." *Ecology*, 89(8), 2290-2301.

5. **Ecosystem Resilience**
   - Holling, C. S. (1973). "Resilience and stability of ecological systems." *Annual Review of Ecology and Systematics*, 4(1), 1-23.

### Bioinformatics Tools

- **scikit-learn**: KNN implementation
- **scipy**: Distance calculations, diversity indices
- **pandas**: Data manipulation
- **lancedb**: Vector storage and search

### Conservation Applications

- **CMLRE**: Convention on Migratory Species - Conservation of Migratory Species of Wild Animals
- **CBD**: Convention on Biological Diversity - Global Biodiversity Framework
- **IUCN**: Red List of Threatened Species classifications

---

## Appendix: Trait Category Reference

### Functional Roles

| Role | Definition | Examples |
|------|-----------|----------|
| Primary Producer | Fix carbon (autotrophic) | Photosynthesizers, chemosynthesizers |
| Decomposer | Break down organic matter | Saprophytic bacteria, fungi |
| Nitrifier | Oxidize ammonia → nitrate | Nitrosomonas, Nitrobacter |
| Denitrifier | Reduce nitrate → N₂ | Pseudomonas, Paracoccus |
| Heterotroph | Consume organic matter | Bacteria, protists, animals |
| Predator | Hunt other organisms | Fish, copepods, ciliates |
| Filter Feeder | Extract particles from water | Copepods, bivalves, baleen whales |
| Parasite | Live on/in host | Tapeworms, protist parasites |
| Symbiont | Mutually beneficial partner | Zooxanthellae, nitrogen-fixing bacteria |

### Trophic Levels

| Level | Trophic Position | Energy Source |
|-------|------------------|----------------|
| Primary Producer | 1 | Sunlight or chemicals |
| Primary Consumer | 2 | Plants/producers |
| Secondary Consumer | 3 | Primary consumers |
| Tertiary Consumer | 4+ | Secondary consumers |
| Detritivore | 2-3 | Dead organic matter |
| Omnivore | 2-4 | Mixed diet |
| Chemolithoautotroph | 1 | Inorganic chemicals |

### Metabolic Pathways

| Pathway | Input | Output | Key Organisms |
|---------|-------|--------|----------------|
| Aerobic Heterotrophy | O₂, organic C | CO₂, H₂O, energy | Most animals, aerobic bacteria |
| Anaerobic Heterotrophy | Organic C, alternative e⁻ acceptor | Fermentation products | Fermentative bacteria |
| Photosynthesis | Light, H₂O, CO₂ | Glucose, O₂ | Algae, cyanobacteria, plants |
| Chemosynthesis | Inorganic chemicals | Glucose, energy | Deep-sea vent bacteria |
| Nitrogen Cycling | NH₃, NO₃⁻ | NO₂⁻, NO₃⁻, N₂ | Nitrifiers, denitrifiers |
| Sulfur Cycling | H₂S, SO₄²⁻ | SO₄²⁻, H₂S | Thiobacillus, purple sulfur bacteria |
| Methane Cycling | CH₄, CO₂ | CH₄, CO₂ | Methanogens, methanotrophs |

---

**🌍 Ecological Traits Analysis - Advancing Functional Understanding of Global Biodiversity**

*Developed for GlobalBioScan v2.0 - Conservation through eDNA Intelligence*
