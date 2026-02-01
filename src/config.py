"""Global configuration for Global-BioScan."""

import os
from pathlib import Path
from typing import Literal

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent  # c:\Volume D\DeepBio_Edge_v3
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORS_DB_DIR = DATA_DIR / "vectors"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORS_DB_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# LanceDB vector store location (disk-native, Windows-compatible)
LANCEDB_URI = str(VECTORS_DB_DIR / "lancedb")
LANCEDB_PENDRIVE_PATH = os.getenv("BIOSCANSCAN_DB_DRIVE", "E:\\GlobalBioScan_DB")
LANCEDB_PENDRIVE_PATH = str(Path(LANCEDB_PENDRIVE_PATH) / "lancedb")
LANCEDB_TABLE_EMBEDDINGS = "embeddings"
LANCEDB_TABLE_SEQUENCES = "sequences"
LANCEDB_TABLE_TAXONOMY = "taxonomy"

# Vector embedding dimensions
EMBEDDING_DIM = 768  # NT-500M output dimension

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Nucleotide Transformer model (Hugging Face)
# Using 500M model for 768-dim embeddings
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-1000-multi-species"
MODEL_REVISION = "main"
MODEL_BATCH_SIZE = 8  # Adjust down for laptops with < 16GB RAM
MODEL_MAX_LENGTH = 1000  # Max sequence length for tokenization (1000 bp)
MODEL_EMBEDDING_DIM = 768  # NT-500M output dimension

# Marker genes
MARKER_GENES = ["COI", "18S", "16S"]
PRIMARY_MARKERS = ["COI", "18S"]

# ============================================================================
# CLUSTERING & NOVELTY DETECTION
# ============================================================================

# HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 5
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0

# Novelty thresholds
NOVELTY_THRESHOLD = 0.7  # Distance-based novelty score [0-1]
DISTANCE_PERCENTILE_NOVELTY = 90  # Top X percentile = novel

# ============================================================================
# TAXONOMIC STANDARDIZATION
# ============================================================================

# TaxonKit database path (auto-downloaded on first run)
TAXONKIT_DB_DIR = DATA_DIR / "taxonkit_db"
TAXONKIT_NAMES_PATH = TAXONKIT_DB_DIR / "names.dmp"
TAXONKIT_NODES_PATH = TAXONKIT_DB_DIR / "nodes.dmp"

# NCBI Entrez API (for sequence retrieval)
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "your-email@example.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_MAX_RETRIES = 3
NCBI_TIMEOUT = 10

# ============================================================================
# DATA INGESTION
# ============================================================================

# FASTA/FASTQ processing
FASTQ_QUALITY_THRESHOLD = 20  # Phred score cutoff
FASTA_MIN_LENGTH = 300  # Minimum sequence length (bp)
FASTA_MAX_LENGTH = 100000  # Maximum sequence length (bp)

# Batch processing
BATCH_SIZE = 100  # Sequences per batch
BATCH_MAX_SIZE_MB = 10  # Max batch file size

# ============================================================================
# WINDOWS COMPATIBILITY
# ============================================================================

# Mock Linux-only libraries
MOCK_TRITON = True  # Use CPU fallback for Triton CUDA kernels
MOCK_FLASHATTENTION = True  # Use standard attention instead of FlashAttention

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================

STREAMLIT_PAGE_TITLE = "Global-BioScan: Deep-Sea Biodiversity Explorer"
STREAMLIT_PAGE_ICON = "🧬"
STREAMLIT_LAYOUT = "wide"

# UMAP visualization parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"

# ============================================================================
# API & EXTERNAL SERVICES
# ============================================================================

# OBIS API for occurrence data
OBIS_API_BASE = "https://api.obis.org/v3"
OBIS_TIMEOUT = 30
OBIS_MAX_RETRIES = 3

# NCBI Entrez API
ENTREZ_DB = "nucleotide"
ENTREZ_RETTYPE = "fasta"
ENTREZ_RETMODE = "text"

# NCBI Entrez Direct (if available)
ENTREZ_DIRECT_ENABLED = False  # Set True if EDirect is installed

# ============================================================================
# RESOURCE CONSTRAINTS (Windows 11 Laptop)
# ============================================================================

MAX_MEMORY_GB = 16  # RAM cap for model inference
MAX_DISK_GB = 25  # Vector DB size cap (~32GB pendrive)
GPU_ENABLED = False  # Assume no dedicated GPU on laptop
NUM_WORKERS = 2  # Thread pool for I/O

# ============================================================================
# DEBUG & DEMO MODE
# ============================================================================

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"  # Simulated scale demo
DEMO_SAMPLE_SIZE = 1000  # Subset of data for demo


def get_config_summary() -> dict:
    """Return active configuration summary."""
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "lancedb_uri": LANCEDB_URI,
        "model_name": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "marker_genes": MARKER_GENES,
        "novelty_threshold": NOVELTY_THRESHOLD,
        "demo_mode": DEMO_MODE,
        "gpu_enabled": GPU_ENABLED,
    }
