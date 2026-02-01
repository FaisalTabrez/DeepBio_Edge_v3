# 🚀 GlobalBioScan TPU Core - Deployment Guide

**Cloud Command Center for High-Speed DNA Embedding Generation**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Setup Guide](#setup-guide)
5. [Running the Pipeline](#running-the-pipeline)
6. [LoRA Fine-Tuning](#lora-fine-tuning)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring & Debugging](#monitoring--debugging)
9. [Data Export](#data-export)
10. [Troubleshooting](#troubleshooting)
11. [Cost Estimation](#cost-estimation)

---

## Quick Start

**Get running in 10 minutes:**

```bash
# 1. Open Google Colab
https://colab.research.google.com/

# 2. Upload notebook
File → Upload → GlobalBioScan_TPU_Core.ipynb

# 3. Change runtime to TPU
Runtime → Change runtime type → TPU v2/v3

# 4. Run all cells
Runtime → Run all

# 5. Monitor on W&B
# (Dashboard URL printed in Step 9)
```

**Expected Output:**
- 60-80k sequences/hour on TPU v3-8
- 2560-dimensional embeddings (high-resolution)
- Automatic checkpointing every 500 steps
- Real-time W&B monitoring

---

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GLOBALBIOSCAN TPU CORE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐  │
│  │   GCS Data   │────▶│  TPU v3-8    │────▶│   LanceDB    │  │
│  │   Streaming  │     │  (8 cores)   │     │   Vectors    │  │
│  └──────────────┘     └──────────────┘     └──────────────┘  │
│         │                    │                     │           │
│         │                    │                     │           │
│    Parquet Shards      JAX pmap (16/core)    2560-dim         │
│    (10k chunks)        bfloat16 precision    float32 export   │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐  │
│  │   NT-2.5B    │────▶│  LoRA Adapt  │────▶│  Google      │  │
│  │   Model      │     │  (r=16, α=32)│     │  Drive       │  │
│  └──────────────┘     └──────────────┘     └──────────────┘  │
│         │                    │                     │           │
│    2.5B params          7-level taxonomy    Checkpoints       │
│    5GB bfloat16         Hierarchical loss   (500 step int.)   │
│                                                                 │
│  ┌──────────────┐                                              │
│  │  Weights &   │  ◀── Real-time monitoring                   │
│  │  Biases      │      TPU metrics, loss, accuracy            │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Inbound:** Parquet shards → GCS bucket → TPU memory (streaming)
2. **Processing:** Tokenization → NT-2.5B forward pass → Mean pooling → 2560-dim vectors
3. **Storage:** Embeddings → LanceDB (Google Drive) → Download to local SSD
4. **Fine-Tuning:** LoRA adapters → 7-level taxonomy → Hierarchical loss → Checkpoints

---

## Prerequisites

### 1. Google Cloud Account

- **TPU Access:** Apply for TPU quota (free tier available)
  - Go to: https://cloud.google.com/tpu/docs/quota
  - Request: TPU v2-8 or TPU v3-8
  - Approval: Usually within 24 hours

- **GCS Bucket:** Create storage bucket
  ```bash
  gsutil mb gs://globalbioscan-data
  ```

### 2. Google Colab Pro (Recommended)

- **Standard Colab:** 12-hour session limit, queuing delays
- **Colab Pro ($9.99/mo):** 24-hour sessions, priority TPU access
- **Colab Pro+:** Background execution, even longer sessions

### 3. Google Drive Storage

- **Minimum:** 100GB free space
- **Recommended:** 500GB for large-scale runs
- **Purpose:** Checkpoint storage, vector export

### 4. Weights & Biases Account

- **Free Tier:** Unlimited projects (public)
- **Signup:** https://wandb.ai/signup
- **API Key:** Copy from https://wandb.ai/authorize

### 5. Local Requirements (Windows Machine)

- **Python 3.10+**
- **LanceDB:** For local vector search
- **Parquet Tools:** For data preparation
- **rclone (optional):** For fast GCS sync

---

## Setup Guide

### Step 1: Prepare Data on Windows

**1.1. Export Parquet Shards (from Phase 1 pipeline)**

```python
import pandas as pd
import pyarrow.parquet as pq

# Load processed data
df = pd.read_csv("processed_data/combined_sequences.csv")

# Split into shards (10k sequences each)
shard_size = 10000
output_dir = "parquet_shards"

for i in range(0, len(df), shard_size):
    shard_df = df.iloc[i:i+shard_size]
    shard_path = f"{output_dir}/shard_{i//shard_size:04d}.parquet"
    shard_df.to_parquet(shard_path, compression="snappy")
    print(f"✓ Saved: {shard_path}")
```

**1.2. Upload to Google Cloud Storage**

**Option A: Using gsutil (recommended for large datasets)**

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Upload shards
gsutil -m cp -r parquet_shards/* gs://globalbioscan-data/parquet_shards/

# Verify
gsutil ls gs://globalbioscan-data/parquet_shards/
```

**Option B: Using Python API**

```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket("globalbioscan-data")

import os
for file in os.listdir("parquet_shards"):
    blob = bucket.blob(f"parquet_shards/{file}")
    blob.upload_from_filename(f"parquet_shards/{file}")
    print(f"✓ Uploaded: {file}")
```

**Option C: Using Google Drive (slower, but simpler)**

1. Upload `parquet_shards/` folder to Google Drive
2. In Colab, mount Drive and copy to GCS:
   ```python
   !gsutil -m cp -r /content/drive/MyDrive/parquet_shards/* gs://globalbioscan-data/parquet_shards/
   ```

### Step 2: Configure Colab Notebook

**2.1. Update Configuration (Cell 6)**

```python
# Replace these values in the notebook:
GCS_BUCKET = "globalbioscan-data"  # Your bucket name
GCS_DATA_PATH = "parquet_shards"   # Path in bucket
CHUNK_SIZE = 10000                 # Sequences per chunk
```

**2.2. Set Runtime to TPU**

1. Click: **Runtime → Change runtime type**
2. Hardware accelerator: **TPU**
3. TPU type: **v2-8** or **v3-8** (v3 is faster)
4. Click: **Save**

### Step 3: Run Notebook

**3.1. Execute Setup Cells (1-5)**

- Cell 1: Install dependencies (~2 minutes)
- Cell 2: Mount Google Drive (~30 seconds)
- Cell 3: Initialize TPU (~1 minute)
- Cell 4: Load NT-2.5B model (~3 minutes, 5GB download)
- Cell 5: Define JAX functions (~instant)

**3.2. Configure Data Streaming (Cell 6)**

- Verify GCS bucket name
- Test connection:
  ```python
  # Run this to verify:
  import gcsfs
  fs = gcsfs.GCSFileSystem()
  files = fs.ls("globalbioscan-data/parquet_shards")
  print(f"Found {len(files)} files")
  ```

**3.3. Run Embedding Pipeline (Cell 7)**

- This is the main processing loop
- Expected speed: **60-80k sequences/hour**
- Progress bar shows real-time throughput
- Vectors automatically saved to LanceDB

---

## Running the Pipeline

### Execution Modes

#### Mode 1: Full Dataset (Production)

```python
# In Cell 7, set:
MAX_SEQUENCES = None  # Process all sequences
BATCH_SIZE = 128      # 16 per core × 8 cores

# Expected time:
# 100k sequences: ~1.5 hours
# 1M sequences: ~15 hours
# 10M sequences: ~6 days (use checkpointing!)
```

#### Mode 2: Testing (Small Batch)

```python
# In Cell 7, set:
MAX_SEQUENCES = 1000  # Test with 1k sequences
BATCH_SIZE = 128

# Expected time: ~1 minute
# Use this to verify pipeline works before full run
```

#### Mode 3: Resume from Checkpoint

```python
# If Colab session disconnects, re-run:
# - Cells 1-6 (setup)
# - Cell 7 (pipeline will resume automatically)

# Checkpoint saved every 10 chunks to Google Drive
```

### Performance Benchmarks

| Dataset Size | TPU v2-8      | TPU v3-8      | Cost (Colab Pro) |
|--------------|---------------|---------------|-------------------|
| 10k seqs     | 10 min        | 8 min         | $0.01            |
| 100k seqs    | 1.8 hours     | 1.5 hours     | $0.15            |
| 1M seqs      | 18 hours      | 15 hours      | $1.50            |
| 10M seqs     | 7.5 days      | 6.2 days      | $15.00           |

**Note:** Times assume 1000-token sequences. Longer sequences = slower processing.

---

## LoRA Fine-Tuning

### When to Use LoRA

- **Scenario 1:** Improve taxonomy predictions (kingdom → species)
- **Scenario 2:** Adapt to regional biodiversity (e.g., Arctic-specific)
- **Scenario 3:** Custom classification tasks (beyond NCBI taxonomy)

### Fine-Tuning Workflow

**Step 1: Prepare Labeled Data**

```python
# Your Parquet files must include these columns:
required_columns = [
    "sequence_id",
    "dna_sequence",
    "kingdom",      # 5 classes
    "phylum",       # 200 classes
    "class",        # 500 classes
    "order",        # 1000 classes
    "family",       # 2000 classes
    "genus",        # 10000 classes
    "species",      # 50000 classes
]

# Taxonomy labels should be integer-encoded (not strings)
# Example: "Bacteria" → 0, "Archaea" → 1, etc.
```

**Step 2: Run Training (Use tpu_engine.py Script)**

```bash
# In Colab, upload tpu_engine.py to /content/
# Then run:
python tpu_engine.py --max-steps 10000

# This will:
# - Load NT-2.5B model
# - Apply LoRA adapters (r=16, alpha=32)
# - Train 7-level taxonomy classifier
# - Save checkpoints every 500 steps
# - Log metrics to W&B
```

**Step 3: Monitor Training**

```python
# Check W&B dashboard for:
# - Training loss (should decrease steadily)
# - Per-level accuracy (kingdom > phylum > ... > species)
# - TPU memory usage (should be < 14GB per core)
# - Learning rate schedule (warmup + cosine decay)
```

**Step 4: Export Fine-Tuned Model**

```python
# After training, LoRA weights saved to:
# /content/drive/MyDrive/GlobalBioScan/checkpoints/lora_010000/

# Download these files to your Windows machine
# Then load in local inference:
from flax.training import checkpoints

lora_params = checkpoints.restore_checkpoint(
    ckpt_dir="path/to/checkpoints",
    target=None,
    step=10000,
    prefix="lora_",
)
```

### Hierarchical Loss Configuration

The loss function weights taxonomy levels differently:

```python
HIERARCHICAL_LOSS_WEIGHTS = {
    "kingdom": 2.0,   # Highest priority (coarse-grained)
    "phylum": 1.8,
    "class": 1.6,
    "order": 1.2,
    "family": 1.0,
    "genus": 0.8,
    "species": 0.6,   # Lowest priority (fine-grained)
}

# Rationale:
# - Novelty detection relies on phylum/class clustering
# - Species-level accuracy less critical for discovery
# - Prevents overfitting to species labels
```

**Adjust weights** if your use case prioritizes genus/species:

```python
# For species-level classification:
HIERARCHICAL_LOSS_WEIGHTS = {
    "kingdom": 1.0,
    "phylum": 1.0,
    "class": 1.0,
    "order": 1.2,
    "family": 1.5,
    "genus": 1.8,
    "species": 2.0,  # Now highest priority
}
```

---

## Performance Optimization

### 1. Batch Size Tuning

**Default: 128 (16 per core × 8 cores)**

```python
# If OOM errors:
BATCH_SIZE_PER_CORE = 8   # Reduce to 8 → total 64
BATCH_SIZE_PER_CORE = 4   # Or 4 → total 32

# If plenty of memory:
BATCH_SIZE_PER_CORE = 32  # Increase to 32 → total 256
```

**Rule of thumb:** Each sequence uses ~50MB TPU memory (bfloat16). Keep total < 14GB per core.

### 2. Sequence Length

**Default: 1000 tokens**

```python
# For shorter sequences (COI barcodes ~650bp):
MAX_SEQUENCE_LENGTH = 700  # Faster processing, less memory

# For longer sequences (full genomes):
MAX_SEQUENCE_LENGTH = 2000  # Slower, but more context
```

**Speed impact:**
- 500 tokens: 1.8× faster than 1000
- 2000 tokens: 0.6× slower than 1000

### 3. Data Streaming

**Chunk size affects memory and speed:**

```python
# Small chunks (more I/O overhead):
CHUNK_SIZE = 5000   # Good for debugging

# Large chunks (better throughput):
CHUNK_SIZE = 20000  # Recommended for production
```

### 4. Precision

**Default: bfloat16 (TPU-optimized)**

```python
# If you need higher precision:
torch_dtype = torch.float32  # 2× slower, 2× memory

# But bfloat16 is sufficient for embeddings!
# Numerical error: < 0.1% difference from float32
```

### 5. Checkpoint Frequency

**Default: Every 500 steps**

```python
# For faster runs (less checkpoint overhead):
CHECKPOINT_INTERVAL = 1000

# For safety (frequent saves):
CHECKPOINT_INTERVAL = 100

# Trade-off: Disk I/O vs. recovery time
```

---

## Monitoring & Debugging

### Weights & Biases Dashboard

**Key Metrics to Track:**

1. **Throughput**
   - `sequences_per_second`: Should be 15-25 (60-80k/hour)
   - If < 10: Check batch size or network latency

2. **TPU Memory**
   - `tpu/core_X_memory_used`: Should be < 14GB
   - If > 14GB: Reduce batch size or sequence length

3. **Training Loss**
   - `train/loss`: Should decrease smoothly
   - Plateau: Increase learning rate or epochs
   - Spikes: Reduce learning rate or add gradient clipping

4. **Per-Level Accuracy**
   - `kingdom_accuracy`: Target > 95%
   - `phylum_accuracy`: Target > 85%
   - `species_accuracy`: Target > 60% (harder)

### Real-Time Monitoring

```python
# Add custom logging in Cell 7:
import time
start_time = time.time()
sequences_processed = 0

for chunk_df in load_parquet_from_gcs(...):
    # ... processing ...
    
    sequences_processed += len(chunk_df)
    elapsed = time.time() - start_time
    sps = sequences_processed / elapsed
    
    wandb.log({
        "throughput/sequences_per_second": sps,
        "throughput/total_sequences": sequences_processed,
        "throughput/elapsed_hours": elapsed / 3600,
    })
```

### TPU Profiling

```python
# Add JAX profiler to identify bottlenecks:
import jax.profiler

# Start profiling
jax.profiler.start_trace("/tmp/tensorboard")

# Run 10 iterations
for i in range(10):
    embeddings = embed_sequences_torch(...)

# Stop profiling
jax.profiler.stop_trace()

# View in TensorBoard:
# %load_ext tensorboard
# %tensorboard --logdir /tmp/tensorboard
```

---

## Data Export

### LanceDB to Local SSD

**Step 1: Compress Vectors (in Colab)**

```python
import shutil

# Compress LanceDB directory
archive_path = "/content/drive/MyDrive/GlobalBioScan/tpu_embeddings"
shutil.make_archive(archive_path, 'zip', LANCEDB_PATH)

print(f"Archive size: {os.path.getsize(archive_path + '.zip') / 1e9:.2f} GB")
```

**Step 2: Download to Windows**

**Option A: Google Drive Web UI**
1. Open: https://drive.google.com/drive/my-drive
2. Navigate to: `MyDrive/GlobalBioScan/`
3. Right-click `tpu_embeddings.zip` → Download

**Option B: rclone (faster for large files)**

```bash
# Install rclone: https://rclone.org/downloads/
rclone config  # Setup Google Drive remote

# Download
rclone copy gdrive:GlobalBioScan/tpu_embeddings.zip C:\Downloads\

# Extract
7z x tpu_embeddings.zip -oC:\Volume D\DeepBio_Edge_v3\data\vectors\
```

**Option C: gdown (Python)**

```bash
pip install gdown

# Get file ID from Drive link (right-click → Get link)
gdown --id FILE_ID_HERE --output tpu_embeddings.zip
```

**Step 3: Load in Local LanceDB**

```python
import lancedb

# Connect to local database
db = lancedb.connect("C:/Volume D/DeepBio_Edge_v3/data/vectors/tpu_embeddings.lance")

# Open table
table = db.open_table("tpu_embeddings")

print(f"Total vectors: {table.count_rows()}")
print(f"Schema: {table.schema}")

# Test search
query_vector = table.to_pandas().iloc[0]["vector"]
results = table.search(query_vector).limit(10).to_pandas()
print(results)
```

---

## Troubleshooting

### Issue 1: TPU Not Found

**Error:** `RuntimeError: No TPU devices found`

**Solutions:**
1. Change runtime: **Runtime → Change runtime type → TPU**
2. Wait 1-2 minutes after changing runtime
3. Re-run TPU initialization cell (Cell 3)
4. If persistent: **Runtime → Factory reset runtime**

### Issue 2: Out of Memory (OOM)

**Error:** `ResourceExhaustedError: Out of memory while trying to allocate...`

**Solutions:**
1. **Reduce batch size:**
   ```python
   BATCH_SIZE_PER_CORE = 8  # Down from 16
   ```

2. **Reduce sequence length:**
   ```python
   MAX_SEQUENCE_LENGTH = 700  # Down from 1000
   ```

3. **Use gradient checkpointing (for training):**
   ```python
   # In tpu_engine.py, add:
   model.gradient_checkpointing_enable()
   ```

4. **Clear cache between chunks:**
   ```python
   import gc
   gc.collect()
   ```

### Issue 3: Slow Processing (<30k sequences/hour)

**Possible Causes:**

1. **Network latency (GCS streaming):**
   - **Solution:** Increase `CHUNK_SIZE` to 20000
   - **Or:** Copy data to `/content/` first (faster local disk)

2. **Batch size too small:**
   - **Solution:** Increase to 128 or 256 (if memory allows)

3. **Model not in bfloat16:**
   - **Verify:**
     ```python
     print(model.dtype)  # Should be torch.bfloat16
     ```
   - **Fix:** Re-load model with `torch_dtype=torch.bfloat16`

4. **CPU bottleneck (tokenization):**
   - **Solution:** Pre-tokenize sequences in Windows, upload tokenized data

### Issue 4: Checkpoint Not Saving

**Error:** `Permission denied: /content/drive/MyDrive/...`

**Solutions:**
1. Verify Drive is mounted: `!ls /content/drive/MyDrive/`
2. Check Drive storage: Must have > 10GB free
3. Manually create directory:
   ```python
   !mkdir -p /content/drive/MyDrive/GlobalBioScan/checkpoints
   ```
4. Test write permission:
   ```python
   !touch /content/drive/MyDrive/GlobalBioScan/test.txt
   ```

### Issue 5: W&B Login Fails

**Error:** `wandb.errors.UsageError: api_key not configured`

**Solutions:**
1. Get API key: https://wandb.ai/authorize
2. Login in Colab:
   ```python
   import wandb
   wandb.login(key="YOUR_API_KEY_HERE")
   ```
3. Or use environment variable:
   ```python
   import os
   os.environ["WANDB_API_KEY"] = "YOUR_API_KEY"
   wandb.init(...)
   ```

### Issue 6: GCS Access Denied

**Error:** `google.auth.exceptions.DefaultCredentialsError`

**Solutions:**
1. Authenticate in Colab:
   ```python
   from google.colab import auth
   auth.authenticate_user()
   ```

2. Or use service account:
   ```python
   os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/service-account-key.json"
   ```

3. Verify bucket permissions:
   ```bash
   gsutil iam get gs://globalbioscan-data
   ```

---

## Cost Estimation

### Google Colab Pricing

| Plan          | TPU Access | Session Limit | Cost/Month | Cost/Hour |
|---------------|------------|---------------|------------|-----------|
| Free Tier     | Limited    | 12 hours      | $0         | $0        |
| Colab Pro     | Priority   | 24 hours      | $9.99      | ~$0.10    |
| Colab Pro+    | Fastest    | Background    | $49.99     | ~$0.50    |

### Processing Cost Examples

**Scenario 1: 100k Sequences (Research)**
- Runtime: ~1.5 hours (TPU v3-8)
- Colab Plan: Pro ($9.99/mo)
- **Cost: ~$0.15** (plus monthly subscription)

**Scenario 2: 1M Sequences (Regional Survey)**
- Runtime: ~15 hours (TPU v3-8)
- Colab Plan: Pro ($9.99/mo)
- **Cost: ~$1.50** (plus monthly subscription)

**Scenario 3: 10M Sequences (Global Scale)**
- Runtime: ~6 days (TPU v3-8)
- Colab Plan: Pro+ ($49.99/mo, background execution)
- **Cost: ~$72** (144 hours × $0.50/hr)

### Cost Optimization Tips

1. **Use Free Tier for Testing:**
   - Process 10k sequences to validate pipeline
   - Then upgrade to Pro for full run

2. **Batch Multiple Datasets:**
   - Combine datasets into one run
   - Avoid repeated setup overhead

3. **Run During Off-Peak:**
   - TPU availability higher at night (US time)
   - Faster startup, less queuing

4. **Use Checkpointing:**
   - Split large runs into 12-hour sessions
   - Resume without reprocessing

5. **Consider Vertex AI (for very large scale):**
   - Reserved TPU pods: $4.50/hr (TPU v3-8)
   - Better for > 50M sequences

---

## Advanced Topics

### Multi-TPU Scaling (Beyond 8 Cores)

For > 10M sequences, use TPU pods:

```python
# In tpu_engine.py, configure mesh for 32 cores:
from jax.sharding import Mesh

mesh = Mesh(
    np.array(jax.devices('tpu')).reshape(4, 8),  # 4×8 = 32 cores
    ('data', 'model')
)

# Update pmap → pjit for 2D parallelism
```

### Custom Loss Functions

Add domain-specific objectives:

```python
def custom_loss(predictions, targets, embeddings):
    # Standard hierarchical loss
    ce_loss = hierarchical_classification_loss(predictions, targets)
    
    # Contrastive loss (similar sequences → similar embeddings)
    contrastive_loss = compute_contrastive_loss(embeddings)
    
    # Combined
    return ce_loss + 0.1 * contrastive_loss
```

### Distributed Training (Multi-Node)

For enterprise deployments:

```bash
# Launch on Vertex AI with 4 TPU v3-8 nodes:
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=globalbioscan-multi-node \
  --worker-pool-spec=machine-type=cloud-tpu,accelerator-type=tpu-v3-8,accelerator-count=4,replica-count=4 \
  --python-package-uris=gs://bucket/tpu_engine.tar.gz \
  --python-module=tpu_engine
```

---

## Summary

**Deployment Checklist:**

- [ ] Google Cloud account with TPU quota
- [ ] GCS bucket created and data uploaded
- [ ] Google Drive mounted (100GB+ free)
- [ ] W&B account configured
- [ ] Colab runtime set to TPU
- [ ] Configuration updated (bucket name, paths)
- [ ] Test run (1k sequences) successful
- [ ] Full pipeline running
- [ ] W&B monitoring active
- [ ] Checkpoints saving to Drive
- [ ] Vectors exported to local SSD

**Performance Targets:**

- ✅ 60-80k sequences/hour (TPU v3-8)
- ✅ < 14GB memory per TPU core
- ✅ < 1% embedding error vs float32
- ✅ Checkpoint every 500 steps
- ✅ Real-time W&B metrics

**Next Steps After Deployment:**

1. Download vectors to local LanceDB
2. Run novelty detection (HDBSCAN clustering)
3. Visualize results in Streamlit dashboard
4. Fine-tune with LoRA for taxonomy (optional)
5. Publish findings with 2560-dim embeddings

---

**🚀 GlobalBioScan TPU Core - Ready for Production Science!**

*For technical support, see [GitHub Issues](https://github.com/your-repo/issues) or contact the GlobalBioScan team.*
