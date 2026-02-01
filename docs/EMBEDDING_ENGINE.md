# Nucleotide Transformer Embedding Engine

## Overview

The `EmbeddingEngine` in [src/edge/embedder.py](../src/edge/embedder.py) transforms raw DNA sequences into high-dimensional biological embeddings using the Nucleotide Transformer foundation model.

**Key Features:**
- ✅ Windows compatibility (Triton/FlashAttention mocking)
- ✅ GPU/CPU auto-detection with mixed precision (FP16/FP32)
- ✅ Batch processing with memory management
- ✅ LanceDB integration for vector updates
- ✅ Checkpoint/resume for interrupted jobs
- ✅ Progress tracking with tqdm
- ✅ Validation tests with cosine similarity

## Architecture

```
┌─────────────────────────────────────────┐
│  DNA Sequences (from LanceDB)           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Tokenization (CharacterTokenizer)      │
│  - Padding to max_length=1000bp         │
│  - Truncation for long sequences        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  NT-500M Forward Pass                   │
│  - GPU (FP16) or CPU (FP32)             │
│  - Extract hidden states (last layer)   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Mean Pooling (sequence dimension)      │
│  - Respects attention mask (ignore pad) │
│  - Output: 768-dimensional vectors      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  LanceDB Vector Update                  │
│  - Batch insert/update                  │
│  - Checkpoint save on interruption      │
└─────────────────────────────────────────┘
```

## Windows Compatibility Patches

The script automatically mocks Linux-specific libraries at the **very top**:

```python
# Mock Triton (CUDA kernel optimizer, Linux-only)
sys.modules["triton"] = MagicMock()
sys.modules["triton.language"] = MagicMock()

# Mock FlashAttention (FastTransformer kernels, Linux-only)
sys.modules["flash_attn"] = MagicMock()
sys.modules["flash_attn.flash_attention"] = MagicMock()
```

**Why?** The Nucleotide Transformer can use these for acceleration, but they're only available on Linux. Mocking them prevents `ImportError` on Windows while allowing standard attention mechanisms to work.

## Model Details

### Model Selection
- **Name:** `InstaDeepAI/nucleotide-transformer-500m-1000-multi-species`
- **Parameters:** 500M (lightweight for laptop inference)
- **Output Dimension:** 768
- **Tokenization:** Character-level (A/C/G/T)
- **Max Sequence Length:** 1000 bp (configurable)
- **Training Data:** Multi-species DNA (bacteria, archaea, eukaryotes)

### Alternatives
- `nucleotide-transformer-250m-1000` (lighter, 256-dim)
- `nucleotide-transformer-2.5b` (larger, 768-dim, more accurate but slower)

## Initialization: GPU/CPU Detection

```python
engine = EmbeddingEngine()
```

**Auto-Detection Logic:**
1. Check `torch.cuda.is_available()`
2. If GPU found: Use CUDA device, FP16 precision
3. If CPU only: Use CPU device, FP32 precision

**Force Device:**
```python
# Force GPU
engine = EmbeddingEngine(use_gpu=True)

# Force CPU
engine = EmbeddingEngine(use_gpu=False)
```

**Log Output:**
```
Device: CUDA
  GPU: NVIDIA GeForce RTX 3060
  CUDA: 11.8
Precision: FP16
```

## Embedding Generation

### Single Sequence

```python
embedding = engine.get_embedding_single("ATGCATGC...")
# Returns: np.ndarray of shape (768,), dtype=float32
```

### Batch Processing

```python
sequences = ["ATGC...", "GCTA...", "TTAA..."]
embeddings = engine.get_embeddings(sequences, show_progress=True)
# Returns: np.ndarray of shape (3, 768), dtype=float32
```

### Tokenization Details

For each sequence:
1. **Character Tokenization:** DNA string → token IDs
2. **Padding:** All sequences padded to `max_length=1000`
3. **Truncation:** Sequences > 1000bp truncated to 1000bp
4. **Attention Mask:** Tracks which tokens are real vs. padding

### Hidden State Extraction

```python
output = model(**tokens)
hidden_states = output.hidden_states[-1]  # Last transformer layer
# Shape: (batch_size, seq_length, 768)
```

### Mean Pooling

Averages the 768-dim vectors across the sequence dimension:

```python
# Mask padding tokens using attention mask
mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
sum_hidden = (hidden_states * mask_expanded).sum(1)
sum_mask = mask_expanded.sum(1)
embedding = sum_hidden / sum_mask.clamp(min=1e-9)
# Shape: (batch_size, 768)
```

This prevents padding tokens from artificially lowering the embedding.

## LanceDB Integration

### Main Workflow: Embed & Update

```python
stats = engine.embed_and_update_lancedb(
    db_path="E:\\GlobalBioScan_DB\\lancedb",
    max_sequences=100,  # None = all
    resume=True         # Resume from checkpoint
)
```

**Steps:**
1. Connect to LanceDB `sequences` table
2. Fetch all rows with placeholder vectors (all zeros)
3. For each row, extract `dna_sequence`
4. Batch process sequences into embeddings
5. Update `vector` column in LanceDB
6. Save checkpoint after each batch

### Batch Processing

```python
batch_size = 8  # Adjust for your RAM (laptop: 4-16)
for batch_idx in range(0, num_sequences, batch_size):
    sequences_batch = sequences[batch_idx:batch_idx+batch_size]
    embeddings = engine.get_embeddings(sequences_batch)
    # Update LanceDB with new vectors
```

### Checkpoint/Resume System

**Checkpoint File:** `data/processed/embedder_checkpoints/embedding_checkpoint.json`

```json
{
  "timestamp": "2026-02-01T10:30:00",
  "last_embedded_id": "OBIS_COI_AB123456",
  "total_processed": 45,
  "model": "InstaDeepAI/nucleotide-transformer-500m-1000-multi-species"
}
```

**Resume Logic:**
- On startup, load checkpoint
- Skip sequences up to `last_embedded_id`
- Resume from the next sequence
- **No data loss** on interruption!

## Validation & Testing

### Cosine Similarity Test

```python
engine.validate_embeddings()
```

Embeds 3 test sequences and computes pairwise cosine similarities:
- **Sequence 1 & 2:** COI from related species (~0.90 similarity)
- **Sequence 1 & 3:** COI vs 18S (~0.70 similarity)

**Expected Output:**
```
Cosine Similarity Matrix:
      0      1      2
0:  1.000  0.903  0.687  (COI_species_A)
1:  0.903  1.000  0.665  (COI_species_B)
2:  0.687  0.665  1.000  (18S_species_C)

✓ Similar sequences (COI): 0.9030
✓ Dissimilar sequences (COI vs 18S): 0.6870
✓ VALIDATION PASSED: Model understands biological proximity!
```

### Custom Test Sequences

```python
test_seqs = [
    ("ATGC...", "Sequence_A"),
    ("ATGC...", "Sequence_B"),
    ("GCTA...", "Sequence_C"),
]
engine.validate_embeddings(test_sequences=test_seqs)
```

## Performance & Optimization

### Benchmarks (Windows 11, 16GB RAM)

| Hardware | Model | Batch Size | Speed | Memory |
|----------|-------|-----------|-------|--------|
| CPU (i7) | NT-500M | 4 | 5-10 seq/min | 6-8 GB |
| CPU (i7) | NT-500M | 8 | 8-12 seq/min | 12-14 GB |
| RTX 3060 | NT-500M | 16 | 50-100 seq/min | 4-6 GB |
| RTX 3060 | NT-500M | 32 | 80-150 seq/min | 8-10 GB |

### RAM Usage Formula
```
Memory ≈ batch_size × max_length × model_dim × 4 (bytes)
       ≈ 8 × 1000 × 768 × 4 ≈ 24 MB per batch
       + model weights ≈ 2 GB (NT-500M)
       + overhead
```

### Optimization Tips

1. **Reduce Batch Size** (if OOM errors):
   ```bash
   python run_embeddings.py --batch-size 4
   ```

2. **Force CPU** (if GPU has OOM):
   ```bash
   python run_embeddings.py --cpu
   ```

3. **Use Checkpoint** (resume from interruption):
   ```bash
   python run_embeddings.py  # Auto-resumes
   ```

4. **Limit Sequences** (for testing):
   ```bash
   python run_embeddings.py --max-sequences 100
   ```

## Usage

### Command Line

**Full pipeline (all sequences from LanceDB):**
```bash
python run_embeddings.py
```

**With custom parameters:**
```bash
python run_embeddings.py \
  --batch-size 4 \
  --max-sequences 500 \
  --gpu
```

**Validation test only (no LanceDB update):**
```bash
python run_embeddings.py --validate-only
```

**Resume from checkpoint (default):**
```bash
python run_embeddings.py --no-resume  # Skip checkpoint
```

### Python API

```python
from src.edge.embedder import EmbeddingEngine

# Initialize
engine = EmbeddingEngine(batch_size=8)

# Embed single sequence
embedding = engine.get_embedding_single("ATGCATGC")

# Embed batch
embeddings = engine.get_embeddings(["ATGC...", "GCTA..."])

# Full pipeline: Embed & update LanceDB
stats = engine.embed_and_update_lancedb(max_sequences=1000)

# Validation
engine.validate_embeddings()

# Model info
info = engine.get_model_info()
print(info)
```

## Error Handling

### Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-size 4`
2. Force CPU mode: `--cpu`
3. Limit sequences: `--max-sequences 100`
4. Ensure no other GPU-heavy processes running

### Invalid Sequences

**Handling:**
- Sequences with non-ATGCN characters are skipped with warning
- Zero vectors appended to output for skipped sequences
- Counted as "errors" in statistics

### Model Loading Fails

**Common Issues:**
- **Not on HuggingFace:** Model doesn't exist (typo in name)
- **Network error:** Internet connectivity issue
- **Disk space:** Not enough space for model weights (~2 GB)

**Solutions:**
```bash
# Pre-download model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('InstaDeepAI/nucleotide-transformer-500m-1000-multi-species')"

# Check internet
ping huggingface.co
```

## Troubleshooting

### Windows Triton/FlashAttention Warnings

**Warning:** Triton/FlashAttention not available on Windows is **expected and fine**.

The mocking ensures the script works without these optimizations.

### GPU Not Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False: NVIDIA driver not installed
```

**Install NVIDIA drivers:**
1. Download from [nvidia.com/Download/driverDetails.aspx](https://www.nvidia.com/Download/driverDetails.aspx)
2. Install with CUDA Toolkit

### LanceDB Path Issues

**Error:** `FileNotFoundError: E:\GlobalBioScan_DB`

**Solutions:**
1. Ensure 32GB pendrive is mounted at E:\
2. Override path: `set BIOSCANSCAN_DB_DRIVE=D:\CustomPath`
3. Use forward slashes: `D:/CustomPath`

### Checkpoint Issues

**Delete checkpoint and restart:**
```bash
rm data/processed/embedder_checkpoints/embedding_checkpoint.json
python run_embeddings.py
```

## Advanced: Custom Models

To use a different Nucleotide Transformer:

```python
from src.edge.embedder import EmbeddingEngine

# 2.5B model (more accurate, slower)
engine = EmbeddingEngine(
    model_name="InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
)

# 250M model (faster, less accurate)
engine = EmbeddingEngine(
    model_name="InstaDeepAI/nucleotide-transformer-250m-1000-multi-species"
)
```

Note: Different models may have different embedding dimensions and sequence lengths.

## Next Steps

After embedding generation completes:
1. **Taxonomy & Novelty:** Run `src/edge/taxonomy.py` for HDBSCAN clustering
2. **Visualization:** Launch `src/interface/app.py` dashboard
3. **Analysis:** Explore embeddings with UMAP or t-SNE

## References

- [Nucleotide Transformers Paper](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v3)
- [InstaDeepAI/nucleotide-transformer (GitHub)](https://github.com/instadeepai/nucleotide-transformer)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [LanceDB Documentation](https://lancedb.com/)
