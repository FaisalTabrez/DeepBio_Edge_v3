# Global-BioScan: End-to-End Integration Guide

## Project Phases

### Phase 1: Data Ingestion ✅
**Status:** Complete  
**Script:** `run_ingestion.py`  
**Output:** LanceDB with DNA sequences + metadata + placeholder vectors

### Phase 2: Embedding Generation ✅ (JUST COMPLETED)
**Status:** Complete  
**Script:** `run_embeddings.py`  
**Output:** LanceDB updated with 768-dim biological embeddings

### Phase 3: Novelty Detection 🔄 (NEXT)
**Status:** In Progress  
**Script:** `src/edge/taxonomy.py` (to be implemented)  
**Tasks:**
- Run HDBSCAN clustering on embeddings
- Compute distances to cluster centroids
- Assign novelty scores
- Infer novel taxonomic units

### Phase 4: Visualization 🔄 (NEXT)
**Status:** Ready for Implementation  
**Script:** `src/interface/app.py`  
**Pages:**
- Dashboard: Overview + statistics
- Embedding Explorer: UMAP + vector search
- Novelty Detection: Novel taxa results
- Diversity Metrics: Alpha/Beta diversity
- Configuration: Settings panel

---

## Complete Workflow

### Prerequisites
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment**
   ```bash
   # .env file
   NCBI_EMAIL=your-email@example.com
   NCBI_API_KEY=your-api-key  # Optional but recommended
   BIOSCANSCAN_DB_DRIVE=E:\GlobalBioScan_DB  # Pendrive location
   ```

3. **Prepare Hardware**
   - Ensure 32GB pendrive mounted at `E:\` (or custom path)
   - Check internet connection for API access
   - GPU optional but recommended

### Full Pipeline

```bash
# 1. INGEST DATA (fetch OBIS + NCBI sequences)
# Time: ~5-10 minutes for 100 species
# Output: 100 sequences in LanceDB with metadata
python run_ingestion.py --max-species 100 --db-drive "E:\GlobalBioScan_DB"

# 2. GENERATE EMBEDDINGS (NT-500M transformations)
# Time: ~30 min (CPU), ~5 min (GPU)
# Output: All sequences now have 768-dim vectors
python run_embeddings.py --batch-size 8

# 3. DETECT NOVELTY (HDBSCAN clustering + scoring)
# Time: ~1 minute
# Output: Novelty scores, cluster assignments, inferred taxonomy
python run_novelty.py  # TODO: create this script

# 4. VISUALIZE (Streamlit dashboard)
# Output: Interactive exploration interface
streamlit run src/interface/app.py
```

---

## Phase 1: Data Ingestion (COMPLETED)

### What Happens
```
OBIS API
  ↓ (1000s occurrences at depth > 1000m)
  ↓
Species Deduplication
  ↓ (50 unique species)
  ↓
NCBI Entrez (fetch sequences)
  ↓ (45 species have COI/18S)
  ↓
TaxonKit (normalize taxonomy)
  ↓ (7-level lineage)
  ↓
LanceDB Storage
  ├─ sequence_id
  ├─ dna_sequence (raw nucleotides)
  ├─ vector (placeholder: zeros)  ← Will be filled by Phase 2
  ├─ taxonomy (NCBI lineage)
  ├─ depth, lat/lon
  └─ metadata
```

### Files
- **[src/edge/init_db.py](src/edge/init_db.py)** - Implementation
- **[run_ingestion.py](run_ingestion.py)** - Entry point
- **[tests/test_init_db.py](tests/test_init_db.py)** - Tests
- **[docs/DATA_INGESTION.md](docs/DATA_INGESTION.md)** - Detailed docs

### Usage
```bash
# Full workflow
python run_ingestion.py

# Test components
python tests/test_init_db.py obis      # Test OBIS fetching
python tests/test_init_db.py ncbi      # Test NCBI Entrez
python tests/test_init_db.py full      # Full pipeline (5 species)
```

### Output Example
```
LanceDB Table: sequences
┌─────────────────────────────────────────────────────────────────┐
│ sequence_id     │ dna_sequence    │ vector              │ depth │
├─────────────────────────────────────────────────────────────────┤
│ OBIS_COI_AB1234 │ ATGCATGC...     │ [0.0, 0.0, ...]    │ 2500  │
│ OBIS_18S_CD5678 │ GCTAGCTA...     │ [0.0, 0.0, ...]    │ 3200  │
│ ...             │ ...             │ ...                 │ ...   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: Embedding Generation (🆕 COMPLETE)

### What Happens
```
LanceDB Sequences
  ↓ (fetch rows with zero vectors)
  ↓
Tokenization (CharacterTokenizer)
  ├─ DNA → token IDs
  ├─ Padding to 1000bp
  └─ Attention masks
  ↓
NT-500M Forward Pass
  ├─ GPU (FP16) or CPU (FP32)
  ├─ Extract hidden states (last layer)
  └─ 768-dimensional representation
  ↓
Mean Pooling
  ├─ Average over sequence dimension
  ├─ Respects attention mask (ignore padding)
  └─ Final embedding: 768-dim vector
  ↓
LanceDB Update
  ├─ Write embedding to vector column
  ├─ Batch insert (10 sequences/batch)
  ├─ Save checkpoint (resume on interrupt)
  └─ Progress bar (tqdm)
```

### Files
- **[src/edge/embedder.py](src/edge/embedder.py)** - Implementation (500+ lines)
- **[run_embeddings.py](run_embeddings.py)** - Entry point
- **[tests/test_embeddings.py](tests/test_embeddings.py)** - Test suite
- **[docs/EMBEDDING_ENGINE.md](docs/EMBEDDING_ENGINE.md)** - Detailed docs

### Key Features

✅ **Windows Compatibility**
- Mocks Triton (CUDA kernels) at script top
- Mocks FlashAttention (optimized attention)
- Works on Windows 11 laptop with no special setup

✅ **GPU/CPU Auto-Detection**
- Detects NVIDIA GPU if available
- Uses FP16 precision on GPU (faster)
- Falls back to FP32 on CPU (compatible)

✅ **Memory Management**
- Configurable batch size (adjust for RAM)
- Default: 8 sequences/batch (handles 16GB laptops)
- Reduce to 4 if OOM errors

✅ **Checkpoint/Resume**
- Saves progress after each batch
- Auto-resumes from last checkpoint on restart
- No data loss if interrupted

✅ **Validation**
- Embeddings semantically meaningful
- Similar sequences (same genus) → high similarity
- Different sequences (different markers) → lower similarity

### Usage

```bash
# Full pipeline (auto-detect GPU/CPU)
python run_embeddings.py

# With custom batch size
python run_embeddings.py --batch-size 4   # For low-RAM systems

# Force CPU
python run_embeddings.py --cpu

# Force GPU
python run_embeddings.py --gpu

# Limit sequences (for testing)
python run_embeddings.py --max-sequences 10

# Validation test only (no LanceDB update)
python run_embeddings.py --validate-only
```

### Python API

```python
from src.edge.embedder import EmbeddingEngine

# Initialize
engine = EmbeddingEngine(batch_size=8)

# Single sequence
embedding = engine.get_embedding_single("ATGCATGC")
# Returns: np.ndarray shape (768,)

# Batch
embeddings = engine.get_embeddings(["ATGC...", "GCTA..."])
# Returns: np.ndarray shape (3, 768)

# Full LanceDB update
stats = engine.embed_and_update_lancedb(
    db_path="E:\\GlobalBioScan_DB\\lancedb",
    max_sequences=100,
    resume=True
)

# Validation
engine.validate_embeddings()
```

### Testing

```bash
# Test all
python tests/test_embeddings.py all

# Test individual components
python tests/test_embeddings.py model      # Model loading
python tests/test_embeddings.py single     # Single embedding
python tests/test_embeddings.py batch      # Batch processing
python tests/test_embeddings.py validation # Cosine similarity
python tests/test_embeddings.py invalid    # Error handling
```

### Performance

| Hardware | Speed | Memory |
|----------|-------|--------|
| CPU (i7) | 5-10 seq/min | 12-14 GB |
| RTX 3060 | 50-100 seq/min | 4-6 GB |

### Output Example

```
LanceDB Table: sequences (UPDATED)
┌─────────────────────────────────────────────────────────┐
│ sequence_id     │ vector (768-dim)      │ dna_sequence   │
├─────────────────────────────────────────────────────────┤
│ OBIS_COI_AB1234 │ [0.234, -0.156, ...] │ ATGCATGC...    │
│ OBIS_18S_CD5678 │ [-0.087, 0.923, ...] │ GCTAGCTA...    │
│ ...             │ ...                   │ ...            │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 3: Novelty Detection (NEXT)

### What Will Happen
```
LanceDB Embeddings
  ↓
HDBSCAN Clustering
  ├─ Unsupervised grouping
  ├─ Density-based clusters
  └─ Noise points identified
  ↓
Novelty Scoring
  ├─ Distance to cluster centroid
  ├─ Percentile-based threshold
  └─ Assign novelty score (0-1)
  ↓
Taxonomy Assignment
  ├─ Consensus from cluster members
  ├─ Known species → reference taxonomy
  ├─ Novel clusters → inferred genus/species
  └─ Confidence scoring
  ↓
Diversity Metrics
  ├─ Alpha diversity (Shannon, Simpson)
  ├─ Beta diversity (clustering coefficient)
  └─ Geographic distribution heatmaps
  ↓
LanceDB Storage
  ├─ novelty_score
  ├─ cluster_id
  ├─ proposed_taxonomy
  └─ confidence
```

### Script to Create
- **[src/edge/taxonomy.py](src/edge/taxonomy.py)** (needs completion)

---

## Phase 4: Visualization (NEXT)

### Dashboard Pages

1. **Dashboard**
   - Total sequences ingested
   - Species diversity
   - Novel taxa discovered
   - Geographic heatmap

2. **Embedding Explorer**
   - UMAP projection of 768-dim vectors
   - Interactive scatter plot
   - Color by: species, novelty, marker gene
   - Hover: sequence info

3. **Vector Search**
   - Upload query sequence
   - Find similar sequences
   - Return top-K matches
   - Display similarity scores

4. **Novelty Detection**
   - Table of novel taxa
   - Proposed taxonomy
   - Confidence scores
   - Phylogenetic tree

5. **Diversity Metrics**
   - Alpha diversity chart (Shannon/Simpson)
   - Beta diversity heatmap
   - Geographic distribution map
   - Depth distribution

6. **Configuration**
   - Model settings
   - Database path
   - API credentials
   - Export options

### Script Location
- **[src/interface/app.py](src/interface/app.py)** (needs completion)

---

## Troubleshooting

### "CUDA out of memory"
```bash
python run_embeddings.py --batch-size 4 --cpu
```

### "No module named 'triton'"
This is expected on Windows. The script automatically mocks it. If you see an import error instead of just a warning, check that the mocking code is at the **very top** of `embedder.py`.

### "LanceDB table not found"
```bash
python run_ingestion.py  # Run Phase 1 first
```

### "Model download fails"
```bash
# Manual pre-download
python -c "
from transformers import AutoModel, AutoTokenizer
model_name = 'InstaDeepAI/nucleotide-transformer-500m-1000-multi-species'
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
AutoModel.from_pretrained(model_name, trust_remote_code=True)
"
```

---

## Timeline

**Phase 1 (Complete):** Data Ingestion
- Fetch OBIS occurrences
- Retrieve NCBI sequences
- Normalize taxonomy with TaxonKit
- Store in LanceDB

**Phase 2 (Complete):** Embedding Generation
- Load NT-500M model
- Generate 768-dim vectors
- Update LanceDB with embeddings
- Validate semantic similarity

**Phase 3 (Next):** Novelty Detection
- HDBSCAN clustering
- Novelty scoring
- Taxonomy assignment
- Diversity metrics

**Phase 4 (Next):** Dashboard & Visualization
- Streamlit interface
- UMAP exploration
- Vector search
- Real-time results

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   Global-BioScan Pipeline                    │
└──────────────────────────────────────────────────────────────┘

┌─ Phase 1: Data Ingestion ──────────────────────┐
│                                                │
│  OBIS API ──> NCBI Entrez ──> TaxonKit        │
│         ↓              ↓            ↓          │
│    (occurrences)  (sequences)   (taxonomy)    │
│         └──────────────┬──────────────┘        │
│                        ↓                       │
│                  LanceDB Table                │
│   (sequences + metadata + zero vectors)      │
└────────────────────────┬──────────────────────┘
                         │
┌────────────────────────▼──────────────────────┐
│ Phase 2: Embedding Generation              │
│                                             │
│  Tokenization ──> NT-500M ──> Mean Pooling │
│  (CharToken)     (Forward Pass)  (768-dim) │
│      ↓               ↓              ↓       │
│   [A,T,G,C]    [GPU/CPU]      [embedding] │
│      └──────────────┬──────────────┘        │
│                     ↓                       │
│              LanceDB Update                │
│      (sequences + embeddings)              │
└────────────────────────┬──────────────────┘
                         │
┌────────────────────────▼──────────────────────┐
│ Phase 3: Novelty Detection (NEXT)           │
│                                             │
│  HDBSCAN ──> Novelty ──> Taxonomy          │
│  (clustering) (scoring)   (assignment)     │
│      ↓          ↓          ↓               │
│   clusters  scores     lineages            │
│      └──────────┬──────────┘               │
│                 ↓                          │
│          LanceDB Update                   │
│    (novelty_score, cluster_id, etc.)      │
└────────────────────────┬───────────────────┘
                         │
┌────────────────────────▼──────────────────────┐
│ Phase 4: Visualization (NEXT)               │
│                                             │
│  Streamlit Dashboard                       │
│  ├─ Dashboard (summary stats)             │
│  ├─ Embedding Explorer (UMAP)             │
│  ├─ Vector Search                         │
│  ├─ Novelty Detection                     │
│  ├─ Diversity Metrics                     │
│  └─ Configuration                         │
└─────────────────────────────────────────────┘
```

---

## Next: Taxonomy & Novelty Detection

Ready to implement Phase 3?

Key components to add to `src/edge/taxonomy.py`:
1. `TaxonomyEngine.cluster_embeddings()` - HDBSCAN
2. `TaxonomyEngine.compute_novelty_scores()` - Distance-based
3. `TaxonomyEngine.assign_taxonomy()` - Consensus voting
4. `TaxonomyEngine.compute_diversity()` - Shannon/Simpson
5. LanceDB update logic for novel taxa

See [agents.md](agents.md) for role assignments.
