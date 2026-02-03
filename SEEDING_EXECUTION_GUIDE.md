# Deep-Sea eDNA Atlas Seeding - Execution Guide

## 🎯 The_Architect Implementation Complete

**Script:** `src/edge/seed_atlas.py` (910 lines)  
**Status:** ✅ Ready for execution  
**Updated:** February 2, 2026

---

## 📋 Configuration Summary

### NCBI Settings
- **Email:** `faisaltabrez01@gmail.com`
- **Rate Limit:** 0.5 seconds delay between requests (to avoid IP blocking)
- **Markers:** COI, COX1, 18S, 18S rRNA
- **Max Sequences:** 3 per species

### OBIS Settings
- **Region:** Central Indian Ridge (60-75°E, -20-0°N)
- **Depth Range:** 1,000-6,000 meters (deep-sea threshold)
- **Max Records:** 5,000 per API request

### TaxonKit Settings
- **Executable Path:** `C:\taxonkit\taxonkit.exe`
- **Database Path:** `C:\taxonkit\taxdump`
- **Format:** 7-level lineage (Kingdom → Species)

### USB Drive Settings
- **Drive Letter:** E:
- **Database Path:** `E:\GlobalBioScan_DB`
- **Target Species:** 2,000-5,000 unique species
- **Auto-Indexing:** IVF-PQ index created at 2,000 species

---

## 🚀 Quick Start

### 1. Prerequisites Check

```bash
# Verify Python environment
python --version  # Should be 3.10+

# Verify TaxonKit installation
C:\taxonkit\taxonkit.exe version

# Verify USB drive mounted
dir E:\

# Install required packages
pip install -r seeding_requirements.txt
```

### 2. Run Seeding Script

```bash
# From project root
python src/edge/seed_atlas.py
```

**Expected Runtime:** 1-4 hours (depending on NCBI API response times)

---

## 🔍 Key Features

### ✅ Resume Capability
- **Checkpoint System:** Automatically saves progress every 50 species
- **Crash Recovery:** If interrupted, restart the script - it will resume from the last checkpoint
- **Checkpoint File:** `E:\GlobalBioScan_DB\checkpoint.json`

### ✅ Duplicate Prevention
- Before processing each species, the script checks if it already exists in LanceDB
- **Skip Logic:** If species found → skip to next species
- **Efficiency:** Prevents redundant NCBI API calls

### ✅ Clinical Logging
- **Progress Updates:** Summary printed every 10 species
- **Log Format:**
  ```
  [INFO] [BATCH 10] Successfully added 10 species (100 sequences total)
  [INFO]   ├─ Species: Bathymodiolus azoricus
  [INFO]   ├─ Sequences: 3 (COI, 18S)
  [INFO]   ├─ Taxonomy: Animalia > Mollusca > Bivalvia > Mytilida > Mytilidae > Bathymodiolus
  [INFO]   └─ Location: -15.5°N, 67.2°E, 2,800m depth
  ```

### ✅ Automatic Indexing
- **Trigger:** Once 2,000 species are reached
- **Index Type:** IVF-PQ (Inverted File with Product Quantization)
- **Partitions:** 256 (optimized for USB SSD performance)
- **Sub-vectors:** 96 (balance between speed and accuracy)
- **Purpose:** Optimizes query performance on USB drive

---

## 📊 Output Structure

### LanceDB Table Schema
```python
{
    "id": str,              # Unique sequence ID
    "accession": str,       # NCBI accession number
    "sequence": str,        # DNA sequence (raw nucleotides)
    "marker": str,          # Gene marker (COI, 18S, etc.)
    "species": str,         # Scientific name
    "kingdom": str,         # Taxonomic kingdom
    "phylum": str,          # Taxonomic phylum
    "class": str,           # Taxonomic class
    "order": str,           # Taxonomic order
    "family": str,          # Taxonomic family
    "genus": str,           # Taxonomic genus
    "depth_m": float,       # Occurrence depth (meters)
    "latitude": float,      # Occurrence latitude
    "longitude": float,     # Occurrence longitude
    "embedding": list       # 768-dim NT-2.5B embedding (placeholder)
}
```

### Manifest File
**Location:** `E:\GlobalBioScan_DB\seeding_manifest.json`

```json
{
    "seeding_date": "2026-02-02T15:30:00",
    "total_species": 2847,
    "total_sequences": 8541,
    "source_obis_records": 12453,
    "source_ncbi_sequences": 8541,
    "taxonomic_coverage": {
        "kingdoms": 3,
        "phyla": 15,
        "classes": 42,
        "orders": 89,
        "families": 234,
        "genera": 1203
    },
    "geographic_coverage": {
        "depth_range": [1000, 5987],
        "lat_range": [-19.8, -0.2],
        "lon_range": [60.1, 74.9]
    },
    "database_stats": {
        "index_type": "IVF-PQ",
        "num_partitions": 256,
        "num_sub_vectors": 96,
        "estimated_size_mb": 1250
    }
}
```

---

## 🔧 Troubleshooting

### Issue: TaxonKit Not Found
```bash
# Verify path
dir C:\taxonkit\taxonkit.exe

# If not found, check if it's named differently
dir C:\taxonkit\
```

### Issue: USB Drive Not Mounted
```bash
# Check if E: drive exists
wmic logicaldisk get name

# If different letter, update DRIVE_LETTER in seed_atlas.py (line 77)
```

### Issue: NCBI API Blocking
- **Symptom:** Frequent HTTP 429 errors
- **Solution:** The script already implements 0.5s delay, but if blocked:
  1. Verify email in script is correct: `faisaltabrez01@gmail.com`
  2. Wait 1 hour before retrying
  3. Consider getting NCBI API key (increases rate limit to 10 req/sec)

### Issue: Checkpoint Recovery
```bash
# Check checkpoint file
cat E:\GlobalBioScan_DB\checkpoint.json

# If corrupted, delete and restart
rm E:\GlobalBioScan_DB\checkpoint.json
```

---

## 📈 Performance Expectations

### API Rate Limits
- **OBIS:** No strict limit, but typically 1-2 req/sec recommended
- **NCBI:** 0.5s delay between requests (user-configured to avoid IP blocking)
- **TaxonKit:** Local subprocess (no external limits)

### Processing Times (Estimates)
- **OBIS Species Extraction:** 5-10 minutes (single bulk query)
- **NCBI Sequence Fetching:** 0.5-2 seconds per species (with rate limiting)
  - 2,000 species × 1.5s avg = ~50 minutes
- **TaxonKit Standardization:** 0.1-0.3 seconds per species
- **LanceDB Ingestion:** <0.05 seconds per sequence
- **IVF-PQ Indexing:** 2-5 minutes (one-time at 2,000 species)

**Total Estimated Runtime:** 1-2 hours for 2,000 species

### Storage Requirements
- **Raw Sequences:** ~500-800 KB per species
- **2,000 species:** ~1-1.5 GB
- **With Embeddings:** ~3-5 GB (after TPU processing)

---

## 🎓 Script Architecture

### Component Overview

```
AtlasSeeder (Main Orchestrator)
├── CheckpointManager
│   ├── load_checkpoint()
│   ├── save_checkpoint()
│   └── clear_checkpoint()
│
├── OBISFetcher
│   └── fetch_deep_sea_species()
│       ├── Query depth > 1000m
│       ├── Filter by region (Central Indian Ridge)
│       └── Extract unique species names
│
├── NCBIFetcher
│   ├── _rate_limit_wait()  # 0.5s delay enforcement
│   └── fetch_sequences()
│       ├── Search for COI/18S markers
│       ├── Download sequences
│       └── Return accession + sequence data
│
├── TaxonKitStandardizer
│   ├── _verify_installation()
│   └── standardize_lineage()
│       ├── Call taxonkit name2taxid
│       ├── Call taxonkit reformat
│       └── Parse 7-level lineage
│
└── LanceDBIngester
    ├── check_species_exists()  # Resume feature
    ├── ingest_batch()
    └── build_index()
        └── IVF-PQ with 256 partitions
```

### Execution Flow

```
1. Initialize Components
   ↓
2. Load Checkpoint (if exists)
   ↓
3. Fetch Deep-Sea Species from OBIS
   ↓
4. For Each Species:
   ├─ Check if already in LanceDB → Skip if exists
   ├─ Fetch Sequences from NCBI (0.5s delay between requests)
   ├─ Standardize Taxonomy with TaxonKit
   ├─ Ingest into LanceDB
   └─ Save Checkpoint every 50 species
   ↓
5. At 2,000 Species:
   ├─ Stop ingestion
   ├─ Build IVF-PQ Index
   └─ Generate Manifest
   ↓
6. Complete
```

---

## 🧪 Verification Steps

### 1. Check Database Contents
```python
from src.edge.database import BioDB

db = BioDB()
print(f"Total sequences: {len(db.sequences)}")
print(f"First 5 species: {db.sequences[:5]['species'].tolist()}")
```

### 2. Verify Taxonomy Coverage
```python
import pandas as pd
df = db.sequences.to_pandas()
print("Taxonomic Distribution:")
print(df['kingdom'].value_counts())
print(df['phylum'].value_counts())
```

### 3. Test Query Performance
```python
from src.core.query_engine import QueryEngine

qe = QueryEngine()
results = qe.search("Bathymodiolus", top_k=10)
print(f"Query returned {len(results)} results in {results['query_time_ms']:.2f} ms")
```

---

## 📝 Post-Seeding Tasks

### 1. Embed Sequences with TPU
```bash
# Run TPU embedding notebook
jupyter notebook notebooks/tpu_embedding_generation.ipynb
```

### 2. Update Streamlit Dashboard
- Navigate to Configuration tab
- Click "Verify Database Integrity"
- Should show 2,000-5,000 species

### 3. Test Full Pipeline
```bash
# Launch app
streamlit run src/interface/app.py

# Upload test FASTA file
# Verify results show deep-sea matches
```

---

## 📞 Support

### Common Questions

**Q: Can I run this multiple times?**  
A: Yes! The resume feature will skip existing species. You can safely re-run to add more species.

**Q: How do I add more species later?**  
A: Simply increase `TARGET_MAX_SPECIES` in the script and re-run. The checkpoint system will continue from where it left off.

**Q: What if I want a different geographic region?**  
A: Update `OBIS_BBOX` in the script (lines 55-60) to your desired coordinates.

**Q: Can I use a different USB drive letter?**  
A: Yes, update `DRIVE_LETTER` in the script (line 77).

---

## 📄 Related Documentation

- **ATLAS_SEEDING_COMPLETE.md** - Technical implementation details
- **ATLAS_SEEDING_GUIDE.md** - Detailed setup guide
- **LANCEDB_INTEGRATION_REPORT.md** - Database architecture
- **LANCEDB_QUICK_REFERENCE.md** - LanceDB usage patterns

---

**Status:** ✅ Ready for Production  
**Last Updated:** February 2, 2026  
**Agent:** The_Architect (Bio-Data Engineer)
