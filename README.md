# Global-BioScan: Deep-Sea Biodiversity Discovery

An AI-driven pipeline for identifying eukaryotic taxa and assessing biodiversity from deep-sea eDNA datasets using Nucleotide Transformers, LanceDB vector search, and HDBSCAN clustering.

## Project Structure

```
.
├── src/
│   ├── config.py              # Global configuration (paths, models, constants)
│   ├── schemas/               # Pydantic data models
│   │   ├── sequence.py        # DNA sequence and batch schemas
│   │   ├── taxonomy.py        # Taxonomic lineage and novelty schemas
│   │   └── vector.py          # Embedding and search result schemas
│   ├── edge/                  # Core bioinformatics pipeline
│   │   ├── init_db.py         # Data ingestion from OBIS/NCBI
│   │   ├── embedder.py        # NT-2.5B embedding generation
│   │   └── taxonomy.py        # TaxonKit-based consensus & novelty detection
│   └── interface/             # Web dashboard
│       └── app.py             # Streamlit visualization
├── data/
│   ├── raw/                   # Input FASTA/FASTQ files
│   ├── processed/             # Cleaned sequences
│   └── vectors/               # LanceDB vector store (disk-native)
├── tests/                     # Unit and integration tests
├── requirements.txt           # Python dependencies
├── .env.template              # Environment variables template
└── README.md                  # This file
```

## Quick Start

### 1. Environment Setup

```bash
# Clone/navigate to project
cd c:\Volume D\DeepBio_Edge_v3

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies (CPU-only for Windows)
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
copy .env.template .env

# Edit .env with your settings
# - NCBI_EMAIL (for Entrez API)
# - NCBI_API_KEY (optional, for faster API access)
# - DEBUG=true (for development)
```

### 3. Run Dashboard

```bash
streamlit run src/interface/app.py
```

## Core Components

### Data Ingestion (`src/edge/init_db.py`)
- **OBIS Integration**: Fetch occurrence records from Ocean Biodiversity Information System
- **NCBI Entrez**: Download reference sequences from GenBank
- **FASTA/FASTQ Parsing**: Standardize raw genomic data
- **LanceDB Storage**: Disk-native vector database

### Embedding Generation (`src/edge/embedder.py`)
- **Model**: NT-500M (768-dim embeddings, optimized for laptop inference)
- **Precision**: FP16 on GPU, FP32 on CPU (auto-detected)
- **Windows Compatibility**: Automatic Triton/FlashAttention mocking
- **Batch Processing**: Memory-efficient with checkpoint/resume capability
- **LanceDB Integration**: Direct vector updates in database
- **Validation**: Cosine similarity test for biological understanding

### Taxonomy & Novelty (`src/edge/taxonomy.py`)
- **TaxonKit**: NCBI taxonomy standardization and lineage resolution
- **HDBSCAN Clustering**: Unsupervised discovery of novel taxa
- **Novelty Scoring**: Distance-based classification (known vs. novel)
- **Diversity Metrics**: Alpha/Beta diversity computation

### Web Dashboard (`src/interface/app.py`)
- **Pages**:
  - Dashboard: Summary statistics and overview
  - Data Ingestion: Upload FASTA or fetch from APIs
  - Embedding Explorer: Interactive UMAP visualization
  - Novelty Detection: Discover novel taxonomic units
  - Diversity Metrics: Alpha/Beta diversity and geographic distribution
  - Configuration: System settings

## Data Schemas

### DNASequence
```python
{
  "sequence_id": "OBIS_COI_1",
  "sequence": "ATGCATGCATGC...",
  "marker_gene": "COI",
  "species": "Unknown",
  "latitude": -60.5,
  "longitude": 45.2,
  "depth_m": 3000,
  "source": "OBIS",
  "length_bp": 658
}
```

### EmbeddingRecord
```python
{
  "sequence_id": "OBIS_COI_1",
  "embedding": [0.1, -0.2, 0.5, ...],  # 256-dim vector
  "marker_gene": "COI",
  "batch_id": "BATCH_20260201_001"
}
```

### NoveltyResult
```python
{
  "sequence_id": "OBIS_COI_1",
  "novelty_score": 0.85,  # 0=known, 1=novel
  "is_novel": true,
  "proposed_genus": "Novelus",
  "proposed_species": "abyssalis",
  "confidence": 0.78
}
```

## Configuration

See [src/config.py](src/config.py) for all options:

- **Model**: NT-2.5B (256-dim embeddings)
- **Clustering**: HDBSCAN (min_cluster_size=10)
- **Novelty Threshold**: 0.7 (distance percentile-based)
- **Marker Genes**: COI, 18S, 16S
- **Vector DB**: LanceDB (IVF-PQ indexing)
- **Resource Limits**: 16GB RAM, 25GB disk (Windows laptop)

## Workflow Examples

### 1. Ingest Data
```python
from src.edge import DataIngestionEngine

engine = DataIngestionEngine()
batch = engine.ingest_fasta("data/raw/deep_sea_coi.fasta", "COI")
engine.store_batch(batch)
```

### 2. Generate Embeddings
```python
from src.edge import EmbeddingEngine

embedder = EmbeddingEngine()
embeddings = embedder.embed_batch(sequence_ids, sequences)
```

### 3. Detect Novelty
```python
from src.edge import TaxonomyEngine

taxonomy = TaxonomyEngine()
novelty = taxonomy.detect_novelty("OBIS_COI_1", embedding_vector)
print(f"Novel? {novelty.is_novel} (Score: {novelty.novelty_score})")
```

## System Requirements

- **OS**: Windows 10+ / Linux / macOS
- **RAM**: 16GB (minimum)
- **Storage**: 32GB+ for vector database
- **Python**: 3.10+
## Workflow: Data → Embeddings

### Step 1: Ingest Data
```bash
# Fetch OBIS deep-sea species and NCBI sequences
python run_ingestion.py --max-species 100

# Or fetch from local FASTA
python -c "
from src.edge.init_db import DataIngestionEngine
engine = DataIngestionEngine()
batch = engine.ingest_fasta('data/raw/sequences.fasta', 'COI')
engine.store_batch(batch)
"
```

### Step 2: Generate Embeddings
```bash
# Full pipeline (all sequences from LanceDB)
python run_embeddings.py

# With custom batch size (for memory constraints)
python run_embeddings.py --batch-size 4

# Force CPU
python run_embeddings.py --cpu

# Test only (validate model works)
python run_embeddings.py --validate-only
```

**What Happens:**
1. Loads NT-500M model from HuggingFace (~2GB)
2. Detects GPU (FP16) or uses CPU (FP32)
3. Fetches sequences with zero-vectors from LanceDB
4. Tokenizes each sequence (max 1000bp)
5. Runs through last transformer layer
6. Mean-pools to get 768-dim vector
7. Updates LanceDB in batches (checkpoint every 10 seqs)
8. Validates with cosine similarity test

### Step 3: Detect Novelty
```bash
# Run HDBSCAN clustering on embeddings
python -c "
from src.edge.taxonomy import TaxonomyEngine
taxonomy = TaxonomyEngine()
# TODO: implement novelty detection
"
```

### Step 4: Explore Dashboard
```bash
streamlit run src/interface/app.py
```

## Testing

```bash
# Test embedding engine
python tests/test_embeddings.py all

# Test individual components
python tests/test_embeddings.py model      # Load model
python tests/test_embeddings.py single     # Single sequence
python tests/test_embeddings.py batch      # Batch processing
python tests/test_embeddings.py validation # Similarity test
python tests/test_embeddings.py invalid    # Error handling

# Test data ingestion
python tests/test_init_db.py obis
python tests/test_init_db.py ncbi
python tests/test_init_db.py full
```

## System Requirements

- **OS**: Windows 10+ / Linux / macOS
- **RAM**: 16GB minimum (8GB CPU-only, 16GB+ for GPU)
- **Storage**: 32GB+ for vector database
- **GPU**: Optional (NVIDIA/CUDA recommended)
- **Python**: 3.10+

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/

# Type checking
mypy src/
```

## Funding Demo Environment

For stakeholder demos, set environment variables:

```bash
# Enable demo mode (simulated 1000-sequence subset)
export DEMO_MODE=true
export DEMO_SAMPLE_SIZE=1000

# Run dashboard
streamlit run src/interface/app.py
```

## Troubleshooting

### Windows-Specific Issues

1. **Triton/FlashAttention Not Available**
   - Automatically mocked with CPU fallback (see `config.MOCK_TRITON`)
   
2. **LanceDB Path Issues**
   - Use `Path()` objects, not raw strings
   - Example: `VECTORS_DB_DIR / "lancedb"`

3. **GPU Out of Memory**
   - Reduce `MODEL_BATCH_SIZE` in config
   - Enable `DEBUG=true` for diagnostics

### Data Ingestion

- **NCBI Rate Limiting**: Set `NCBI_API_KEY` in .env
- **OBIS Timeout**: Increase `OBIS_TIMEOUT` in config
- **Memory Error on Large Batches**: Reduce `BATCH_SIZE`

## References

- [Nucleotide Transformers](https://github.com/instadeepai/nucleotide-transformer)
- [LanceDB Documentation](https://lancedb.com/)
- [HDBSCAN](https://hdbscan.readthedocs.io/)
- [TaxonKit](https://bioinf.shenwei.me/taxonkit/)
- [OBIS API](https://api.obis.org/)
- [NCBI Entrez](https://www.ncbi.nlm.nih.gov/books/NBK25499/)

## Team

See [agents.md](agents.md) for role descriptions and team structure.

## License

[Your License Here]

## Support

For issues or questions, open an issue on the repository.
