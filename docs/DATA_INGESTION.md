# Data Ingestion Pipeline Documentation

## Overview

The `DataIngestionEngine` in [src/edge/init_db.py](../src/edge/init_db.py) orchestrates real-world deep-sea biodiversity data collection from OBIS and NCBI, with taxonomic standardization via TaxonKit and storage in LanceDB.

## Architecture

```
┌─────────────────────┐
│  OBIS API           │  Fetch occurrence records at depth > 1000m
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ NCBI Entrez         │  Retrieve COI/18S sequences for species
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ TaxonKit            │  Standardize taxonomy to 7-level lineage
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ LanceDB             │  Store embeddings + metadata on pendrive
└─────────────────────┘
```

## Configuration

### Database Path (Pendrive)

Default: `E:\GlobalBioScan_DB` (Windows 32GB pendrive)

**Override via environment variable:**
```bash
set BIOSCANSCAN_DB_DRIVE=D:\CustomPath
```

Or via command-line argument:
```bash
python run_ingestion.py --db-drive "D:\CustomPath"
```

### Key Settings (src/config.py)

| Setting | Default | Purpose |
|---------|---------|---------|
| `NCBI_EMAIL` | From `.env` | Required for NCBI Entrez API |
| `NCBI_API_KEY` | Optional | Speeds up API calls (from `.env`) |
| `OBIS_TIMEOUT` | 30s | Request timeout for OBIS API |
| `FASTA_MIN_LENGTH` | 300bp | Minimum sequence length |
| `FASTA_MAX_LENGTH` | 100,000bp | Maximum sequence length |
| `MARKER_GENES` | COI, 18S, 16S | Supported marker genes |

## API Integration

### 1. OBIS (Ocean Biodiversity Information System)

**Method:** `fetch_deep_sea_species()`

- **Endpoint:** `https://api.obis.org/v3/occurrence`
- **Query:** Marine species at depths > 1000m (abyssal/hadal zones)
- **Geographic Focus:** Central Indian Ridge (configurable)
- **Rate Limiting:** 0.5s delay between requests
- **Returns:** List of species with scientificName, taxonID, depth, lat/lon

**Example:**
```python
engine = DataIngestionEngine()
species = engine.fetch_deep_sea_species(min_depth=1000, limit=100)
# [
#   {
#     "scientific_name": "Serranus cabrilla",
#     "taxon_id": 12345,
#     "depth": 2500.0,
#     "latitude": -10.5,
#     "longitude": 70.2,
#     ...
#   }
# ]
```

### 2. NCBI Entrez (GenBank)

**Method:** `fetch_ncbi_sequence()`

- **Database:** Nucleotide (GenBank)
- **Query Pattern:** `"{species}"[ORGN] AND {marker}[GENE]`
- **Markers:** COI, 18S, 16S
- **Rate Limiting:** 
  - Without API key: 3 requests/second
  - With API key: 10 requests/second
- **Backoff:** 1s delay on errors
- **Validation:** 
  - Sequence length: 300-100,000 bp
  - Only ATGCN characters allowed
- **Returns:** (sequence, marker_gene, accession_id) or None

**Graceful Failure:**
- If no sequence found for a species, logs warning and continues
- If API rate limit hit, automatic backoff applied
- Invalid sequences are skipped

**Example:**
```python
result = engine.fetch_ncbi_sequence("Serranus cabrilla", marker_genes=["COI"])
# Returns: ("ATGCATGC...", "COI", "AB123456")
```

### 3. TaxonKit (NCBI Taxonomy)

**Method:** `normalize_taxonomy_taxonkit()`

- **Subprocess Calls:**
  1. `taxonkit name2taxid` - Convert species name → NCBI taxon ID
  2. `taxonkit reformat` - Expand taxon ID → full 7-level lineage
- **Output Format:** `Kingdom;Phylum;Class;Order;Family;Genus;Species`
- **Requirements:**
  - TaxonKit installed (`conda install taxonkit`)
  - TaxonKit database synced (`taxonkit update`)
- **Timeout:** 10 seconds per call
- **Fallback:** Returns None if TaxonKit unavailable (graceful degradation)

**Example:**
```python
lineage = engine.normalize_taxonomy_taxonkit("Serranus cabrilla")
# Returns: "Eukaryota;Chordata;Actinopterygii;Perciformes;Serranidae;Serranus;cabrilla"
```

## LanceDB Schema

### Table: `sequences`

| Column | Type | Description |
|--------|------|-------------|
| `sequence_id` | string | Unique ID (e.g., `OBIS_COI_AB123456`) |
| `vector` | float32[768] | Embedding placeholder (filled by embedder.py) |
| `dna_sequence` | string | Full DNA sequence |
| `taxonomy` | string | 7-level lineage (from TaxonKit) |
| `obis_id` | string | OBIS occurrence ID |
| `marker_gene` | string | COI, 18S, or 16S |
| `depth` | float | Sampling depth (meters) |
| `latitude` | float | Sampling latitude |
| `longitude` | float | Sampling longitude |
| `species_name` | string | Scientific name |
| `accession_id` | string | NCBI GenBank accession |
| `timestamp` | string | ISO8601 creation timestamp |

**Schema Inference:**
```python
# LanceDB auto-infers schema from first record
sample_record = {
    "sequence_id": "OBIS_COI_AB123456",
    "vector": [0.0] * 768,  # float32
    "dna_sequence": "ATGCATGC...",
    "taxonomy": "Eukaryota;...",
    "depth": 2500.0,  # float
    "latitude": -10.5,
    ...
}
```

## End-to-End Workflow

### Step 1: Initialize LanceDB
```python
engine = DataIngestionEngine()
engine.initialize_lancedb_schema()
```
Creates `sequences` table with schema.

### Step 2: Fetch OBIS Species
```python
species_list = engine.fetch_deep_sea_species(
    min_depth=1000,
    limit=500,
)
```
Returns ~50-100 unique deep-sea species.

### Step 3: Process Each Species
For each species:
1. Fetch NCBI sequence (COI or 18S)
2. Normalize taxonomy with TaxonKit
3. Create record dict
4. Store in LanceDB (batch insert every 10 records)

### Step 4: Return Statistics
```python
stats = engine.run_full_pipeline(
    min_depth=1000,
    max_species=100,
    skip_embedding=True,  # Embedder.py will fill vectors later
)
# {
#   "timestamp": "2026-02-01T10:30:00",
#   "species_fetched": 50,
#   "species_with_sequences": 45,
#   "sequences_fetched": 45,
#   "sequences_stored": 45,
#   "errors": [...]
# }
```

## Usage

### Command Line

**Run full pipeline (default: 50 species):**
```bash
python run_ingestion.py
```

**With custom parameters:**
```bash
python run_ingestion.py \
  --min-depth 2000 \
  --max-species 200 \
  --skip-embedding \
  --db-drive "E:\GlobalBioScan_DB"
```

### Python API

```python
from src.edge.init_db import DataIngestionEngine

# Initialize
engine = DataIngestionEngine()

# Run full pipeline
stats = engine.run_full_pipeline(
    min_depth=1000,
    max_species=50,
)

# Or run components separately
species = engine.fetch_deep_sea_species(limit=20)
for sp in species:
    seq_data = engine.fetch_ncbi_sequence(sp["scientific_name"])
    if seq_data:
        lineage = engine.normalize_taxonomy_taxonkit(sp["scientific_name"])
```

### Testing

```bash
# Test OBIS fetching
python tests/test_init_db.py obis

# Test NCBI fetching
python tests/test_init_db.py ncbi

# Test TaxonKit
python tests/test_init_db.py taxonkit

# Full pipeline test (5 species)
python tests/test_init_db.py full
```

## Error Handling

### Graceful Failures

| Error | Handling |
|-------|----------|
| Species not found in NCBI | Log warning, skip species, continue |
| TaxonKit unavailable | Return None, allow manual curation |
| OBIS API timeout | Retry with backoff, log error |
| LanceDB storage error | Log error, report in stats |
| Invalid sequence length | Skip sequence, log info |

### Logging

All operations logged to:
- **Console:** `INFO` level by default
- **File:** Optional (configurable via logging config)

**Example log output:**
```
2026-02-01 10:30:00 - src.edge.init_db - INFO - Connected to LanceDB at E:\GlobalBioScan_DB\lancedb
2026-02-01 10:30:01 - src.edge.init_db - INFO - Fetching OBIS occurrences at depth > 1000m...
2026-02-01 10:30:05 - src.edge.init_db - INFO - Found 50 unique deep-sea species from 1250 occurrences
2026-02-01 10:30:06 - src.edge.init_db - INFO - [1/50] Processing: Serranus cabrilla
2026-02-01 10:30:08 - src.edge.init_db - INFO - ✓ Found COI for Serranus cabrilla (658bp, ID: AB123456)
2026-02-01 10:30:10 - src.edge.init_db - DEBUG - Normalized Serranus cabrilla: Eukaryota;...
```

## Performance

### Benchmarks (Windows 11, 16GB RAM)

| Operation | Time | Species |
|-----------|------|---------|
| OBIS fetch | ~5s | 50 unique |
| NCBI per-species | ~2s | 1 sequence |
| TaxonKit per-species | ~1s | 1 lineage |
| LanceDB store (batch) | ~0.1s | 10 sequences |

**Total for 50 species:** ~3-4 minutes

### Optimization Tips

1. **Set `NCBI_API_KEY`** in `.env` (10x faster Entrez)
2. **Increase `max_species`** (batch processing is efficient)
3. **Pre-download TaxonKit DB** (`taxonkit update`)
4. **Use local LanceDB path** (SSD recommended)

## Troubleshooting

### NCBI Rate Limiting

**Issue:** "HTTP 429 Too Many Requests"

**Solution:**
1. Set `NCBI_API_KEY` in `.env` (requires registration)
2. Increase `NCBI_TIMEOUT` in config
3. Reduce `MODEL_BATCH_SIZE`

### TaxonKit Not Found

**Issue:** "TaxonKit command not found"

**Solution:**
```bash
# Install via conda
conda install taxonkit

# Download NCBI taxonomy database
taxonkit update
```

### LanceDB Path Issues

**Issue:** "Path not found" on Windows

**Solution:** Use forward slashes in Python paths:
```python
os.environ["BIOSCANSCAN_DB_DRIVE"] = "E:/GlobalBioScan_DB"
```

### OBIS API Timeouts

**Issue:** "Connection timeout" from OBIS

**Solution:**
1. Increase `OBIS_TIMEOUT` in config
2. Reduce `limit` parameter
3. Check internet connection

## Next Steps

After data ingestion completes:
1. **Embeddings:** Run `src/edge/embedder.py` to generate NT-2.5B embeddings
2. **Novelty Detection:** Run `src/edge/taxonomy.py` for HDBSCAN clustering
3. **Visualization:** Launch `src/interface/app.py` Streamlit dashboard

## References

- [OBIS API Documentation](https://api.obis.org/)
- [NCBI Entrez Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25499/)
- [TaxonKit GitHub](https://github.com/shenwei356/taxonkit)
- [LanceDB Documentation](https://lancedb.com/)
- [BioPython Entrez](https://biopython.org/wiki/Documentation#Entrez)
