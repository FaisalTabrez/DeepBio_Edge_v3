# GlobalBioScan Cloud Workflow
## End-to-End TPU-Accelerated Embedding & Fine-Tuning Guide

**Version:** 1.0  
**Last Updated:** February 2026  
**Environment:** Google Colab + TPU v3-8 / A100 GPU

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Data Sync: Pendrive ↔ Cloud](#data-sync-pendrive--cloud)
4. [Colab Notebook Execution](#colab-notebook-execution)
5. [TPU Inference Pipeline](#tpu-inference-pipeline)
6. [LoRA Fine-Tuning Workflow](#lora-fine-tuning-workflow)
7. [LanceDB Vector Export](#lancedb-vector-export)
8. [W&B Monitoring Setup](#wb-monitoring-setup)
9. [Checkpoint Management](#checkpoint-management)
10. [Troubleshooting & Performance](#troubleshooting--performance)

---

## Quick Start

### 10-Minute Cloud Setup

```bash
# 1. Upload Parquet data to Google Drive
# Copy from: C:\Volume D\DeepBio_Edge_v3\data\*.parquet
# To: Google Drive/DeepBio_Edge/data/

# 2. Open Colab notebook
# https://colab.research.google.com
# Run: notebooks/GlobalBioScan_Cloud_Engine.ipynb

# 3. After execution, download vectors
# From: Google Drive/DeepBio_Edge/outputs/*.lance
# To: C:\Volume D\DeepBio_Edge_v3\data\

# 4. Import into local LanceDB
# Run: python src/cloud/lancedb_import.py
```

---

## Prerequisites

### Google Colab Resources

- **TPU v3-8 Cluster** (via Google TPU Research Cloud)
  - OR **A100 GPU** (80GB VRAM)
- **Google Drive** (100GB+ available space)
- **Weights & Biases** (W&B) account for monitoring
- **GitHub** token (for pulling code)

### Local Machine Requirements

- **Python 3.10+**
- **32GB Pendrive** for storing vector outputs
- **Git** for version control
- **gsutil** for GCS access (optional)

### API Keys & Credentials

```bash
# Store in Colab environment
GOOGLE_API_KEY = "your-google-api-key"
WANDB_API_KEY = "your-wandb-api-key"
GITHUB_TOKEN = "your-github-token"
GCS_PROJECT_ID = "your-gcp-project-id"
```

---

## Data Sync: Pendrive ↔ Cloud

### Upload Data to Google Drive

#### Option A: Web Browser Upload

1. Navigate to **Google Drive** → **My Drive**
2. Create folder: `DeepBio_Edge/data/`
3. Upload Parquet files:
   ```
   data/
   ├── sequences.parquet (Phase 1 output)
   ├── embeddings.parquet (Phase 2, optional)
   └── taxonomy.parquet (Phase 3 metadata)
   ```
4. Note the folder IDs for Colab access

#### Option B: `rclone` Command-Line Sync

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive
rclone config

# Sync to Drive
rclone sync "C:\Volume D\DeepBio_Edge_v3\data\" \
  gdrive:DeepBio_Edge/data/ \
  --progress

# Verify
rclone ls gdrive:DeepBio_Edge/data/
```

#### Option C: Python Script

```python
# pip install google-colab google-auth google-auth-oauthlib google-auth-httplib2

from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

auth.authenticate_user()
drive_service = build('drive', 'v3')

# Upload file
file_path = "C:\\Volume D\\DeepBio_Edge_v3\\data\\sequences.parquet"
file_metadata = {
    'name': 'sequences.parquet',
    'parents': ['your-folder-id']
}

media = MediaFileUpload(file_path, resumable=True)
request = drive_service.files().create(
    body=file_metadata,
    media_body=media,
    fields='id',
    resumable=True
)
response = request.execute()
print(f"Uploaded with ID: {response['id']}")
```

### Download Vectors from Google Drive

#### After Colab Execution:

```bash
# Option 1: Browser download
# Visit Google Drive → DeepBio_Edge/outputs/
# Download *.lance files manually

# Option 2: rclone download
rclone sync gdrive:DeepBio_Edge/outputs/ \
  "C:\Volume D\DeepBio_Edge_v3\outputs\" \
  --progress

# Option 3: Python script
from google.colab import drive
drive.mount("/content/drive")

import shutil
shutil.copy(
    "/content/drive/MyDrive/DeepBio_Edge/outputs/sequences.lance",
    "C:\\Volume D\\DeepBio_Edge_v3\\outputs\\"
)
```

---

## Colab Notebook Execution

### Step 1: Open Notebook

```bash
# Method A: Direct link
# https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID

# Method B: Upload to Google Drive
# 1. Save notebook: notebooks/GlobalBioScan_Cloud_Engine.ipynb
# 2. Upload to Google Drive
# 3. Right-click → Open with → Google Colaboratory

# Method C: From GitHub
# !git clone https://github.com/FaisalTabrez/DeepBio_Edge_v3.git
# Then navigate to notebooks/GlobalBioScan_Cloud_Engine.ipynb
```

### Step 2: Runtime Configuration

```python
# Cell 1: Configure Colab Runtime
# ✓ Runtime type: TPU v3-8 or GPU (A100)
# ✓ GPU memory: High RAM
# ✓ Disk size: 100GB

# In Colab menu:
# Runtime → Change runtime type
# Hardware accelerator: TPU v3-8 or GPU
# ✓ High RAM
# ✓ Storage: 100GB
```

### Step 3: Run Notebook Sequentially

**Cell 1:** Install dependencies
```python
!pip install -q torch transformers jax[tpu] flax optax peft wandb
!pip install -q pandas numpy pyarrow lancedb duckdb
```

**Cell 2:** TPU initialization
```python
import jax
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
devices = jax.devices()
print(f"✓ TPU cores: {len(devices)}")
```

**Cell 3:** Load model & tokenizer
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
model = AutoModelForMaskedLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    torch_dtype=torch.bfloat16
)
```

**Cell 4-6:** Stream data → Generate embeddings → Export vectors

---

## TPU Inference Pipeline

### Architecture Diagram

```
Google Drive (Parquet)
        ↓
Streaming Loader
        ↓
Batch Processor (16 sequences/core × 8 cores = 128 batch)
        ↓
Model Forward Pass (bfloat16)
        ↓
Mean Pooling (2560-dim vectors)
        ↓
LanceDB Writer
        ↓
Google Drive (Lance format)
```

### Performance Characteristics

| Component | Expected Time | Throughput |
|-----------|----------------|-----------|
| Data Loading | 2-3s per batch | ~1GB/min |
| Model Forward Pass | 4-6s per batch | 10-20k vectors/min |
| Vector Storage | 1-2s per batch | Full throughput |
| **Total** | **7-11s per 128 sequences** | **~40-50k vectors/hour** |

### Code Snippet: Batch Processing

```python
# From src/cloud/tpu_worker.py
def generate_embeddings_batched(
    sequences: List[str],
    model,
    tokenizer,
    batch_size: int = 128,
    device: str = "tpu"
) -> np.ndarray:
    """Generate 2560-dim embeddings in batches."""
    
    embeddings = []
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(sequences))
        batch_seqs = sequences[start:end]
        
        # Tokenize
        tokens = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024
        ).to(device)
        
        # Forward pass (jitted for TPU efficiency)
        with torch.no_grad():
            output = model(**tokens)
            hidden = output.hidden_states[-1]  # (batch, seq_len, 2560)
            
            # Mean pooling
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        
        embeddings.append(pooled.cpu().numpy().astype(np.float32))
    
    return np.vstack(embeddings)
```

---

## LoRA Fine-Tuning Workflow

### Configuration

```python
# LoRA Hyperparameters (from src/cloud/fine_tune_lora.py)
LORA_R = 16              # Rank (lower = fewer params)
LORA_ALPHA = 32          # Scaling factor
LORA_DROPOUT = 0.1       # Dropout rate
TARGET_MODULES = ["query", "value"]  # Only Q,V in attention
LEARNING_RATE = 2e-4     # Base learning rate
BATCH_SIZE = 32          # Per-GPU batch
EPOCHS = 10              # Training epochs
WARMUP_STEPS = 500       # Learning rate warmup
```

### Training Loop

```python
# Execute in Colab:
!python src/cloud/fine_tune_lora.py \
  --train-data "/content/drive/MyDrive/DeepBio_Edge/data/taxonomy.parquet" \
  --output-model "/content/outputs/model-lora-finetuned" \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --wandb-project "global-bioscan-lora"
```

### Hierarchical Loss Function

```python
# 7-level taxonomy loss (from fine_tune_lora.py)
def hierarchical_classification_loss(logits, labels, level_weights=None):
    """
    Penalize errors at coarser levels more heavily:
    - kingdom: 2.0x
    - phylum: 1.8x
    - class: 1.5x
    - order: 1.2x
    - family: 1.0x
    - genus: 0.8x
    - species: 0.6x
    """
    
    level_weights = {
        "kingdom": 2.0,
        "phylum": 1.8,
        "class": 1.5,
        "order": 1.2,
        "family": 1.0,
        "genus": 0.8,
        "species": 0.6
    }
    
    total_loss = 0.0
    for level_name, weight in level_weights.items():
        level_loss = cross_entropy(logits[level_name], labels[level_name])
        total_loss += weight * level_loss
    
    return total_loss
```

### Training Time Estimates

| Dataset Size | Batch Size | Epochs | TPU v3-8 Time |
|--------------|-----------|--------|---------------|
| 10k sequences | 32 | 10 | ~15 minutes |
| 100k sequences | 32 | 10 | ~2 hours |
| 1M sequences | 32 | 10 | ~20 hours |

---

## LanceDB Vector Export

### Convert Embeddings to Lance Format

```python
# From src/cloud/tpu_worker.py
import lancedb
import pyarrow as pa
import duckdb

def write_vectors_to_lance(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: str
):
    """Convert embeddings + metadata to Lance format."""
    
    # Attach vectors to metadata
    df_with_vectors = metadata.copy()
    df_with_vectors["vector"] = [emb.tolist() for emb in embeddings]
    
    # Convert to PyArrow table
    table = pa.Table.from_pandas(df_with_vectors)
    
    # Write to LanceDB
    db = lancedb.connect(output_path)
    db.create_table("sequences", data=table, mode="overwrite")
    
    # Verify
    result = db.open_table("sequences").search().limit(10).to_list()
    print(f"✓ Exported {len(result)} vectors to {output_path}")
```

### Join Vectors with Metadata (DuckDB)

```python
import duckdb

# SQL query to join vectors with metadata
query = """
SELECT 
    sequence_id,
    taxonomy,
    depth,
    latitude,
    longitude,
    vector  -- 2560-dim embedding
FROM sequences
WHERE depth BETWEEN 0 AND 3000
ORDER BY sequence_id
"""

# Execute with DuckDB for efficient processing
con = duckdb.connect()
result = con.execute(query).fetch_all()

print(f"✓ Retrieved {len(result)} sequences with metadata")
```

### Download to Local Pendrive

```bash
# After Colab execution:
rclone sync gdrive:DeepBio_Edge/outputs/ \
  "G:\DeepBio_Edge_Vectors\" \  # 32GB Pendrive
  --progress \
  --transfers=4 \
  --checkers=4

# Verify integrity
ls -lh "G:\DeepBio_Edge_Vectors\*.lance"
```

---

## W&B Monitoring Setup

### Initialize W&B in Colab

```python
import wandb

# Cell in notebook:
wandb.login()  # Paste API key when prompted

wandb.init(
    project="global-bioscan-lora",
    config={
        "model": "nucleotide-transformer-2.5b",
        "lora_r": 16,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "epochs": 10,
    }
)
```

### Log Metrics During Training

```python
# In training loop:
for step, (loss, metrics) in enumerate(training_loop):
    wandb.log({
        "train/loss": loss,
        "train/learning_rate": current_lr,
        "train/epoch": epoch,
        "metrics/embedding_norm": embedding_norm,
    }, step=step)
    
    # Also log hardware metrics
    wandb.log({
        "hardware/tpu_utilization": tpu_util,
        "hardware/memory_used_gb": memory_used / 1e9,
        "hardware/throughput_seqs_per_sec": throughput,
    })
```

### Access W&B Dashboard

```bash
# After training starts:
# https://wandb.ai/YOUR_USERNAME/global-bioscan-lora

# Key metrics to monitor:
# - Loss curves (training & validation)
# - Learning rate schedule
# - TPU utilization %
# - Embedding statistics (mean, std)
# - Per-level classification accuracy
```

---

## Checkpoint Management

### Automatic Checkpointing (30-min intervals)

```python
# From src/cloud/tpu_worker.py
class CheckpointManager:
    def __init__(self, checkpoint_dir="/content/drive/MyDrive/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, embeddings, metadata, step):
        """Save checkpoint every 30 minutes."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.npz"
        
        np.savez_compressed(
            checkpoint_path,
            embeddings=embeddings,
            metadata=metadata.to_json(orient="records"),
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def load_latest_checkpoint(self):
        """Resume from latest checkpoint on reconnection."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.npz"))
        if checkpoints:
            latest = checkpoints[-1]
            data = np.load(latest, allow_pickle=True)
            logger.info(f"✓ Resumed from {latest.name}")
            return data["embeddings"], pd.read_json(data["metadata"])
        return None, None
```

### Checkpoint Directory Structure

```
Google Drive/DeepBio_Edge/checkpoints/
├── checkpoint_step_0.npz       (0 sequences)
├── checkpoint_step_1000.npz    (1,000 sequences)
├── checkpoint_step_2000.npz    (2,000 sequences)
└── checkpoint_final.npz        (all sequences)

Each checkpoint contains:
- embeddings.npy (2560-dim vectors)
- metadata.json (sequence_id, taxonomy, depth, lat/lon)
- timestamp (ISO 8601)
```

### Resume from Checkpoint

```python
# In notebook, after reconnection:
checkpoint_mgr = CheckpointManager()
embeddings, metadata = checkpoint_mgr.load_latest_checkpoint()

if embeddings is not None:
    logger.info(f"Resuming with {len(embeddings)} embeddings")
    # Continue from where left off
else:
    logger.info("Starting fresh (no checkpoints found)")
```

---

## Troubleshooting & Performance

### Common Issues & Solutions

#### Issue 1: "TPU Not Found"

```python
# Error: RuntimeError: TPU not found
# Solution:
import jax
devices = jax.devices()
if not devices or 'tpu' not in str(devices[0]):
    logger.warning("TPU not available, using GPU/CPU")
    # Code will fall back to GPU/CPU automatically
```

#### Issue 2: Out of Memory (OOM)

```python
# Error: RuntimeError: Memory allocation failed
# Solutions:
# 1. Reduce batch size
BATCH_SIZE = 16  # Was 128

# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 3. Use bfloat16 instead of float32
model = model.to(torch.bfloat16)

# 4. Chunk data more aggressively
CHUNK_SIZE = 50  # Was 100
```

#### Issue 3: Slow Data Loading

```python
# Problem: Parquet loading is bottleneck
# Solution: Use multiple workers
import concurrent.futures

def load_parquet_parallel(paths, num_workers=4):
    with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
        results = executor.map(pd.read_parquet, paths)
    return pd.concat(results, ignore_index=True)
```

#### Issue 4: Colab Connection Timeout

```python
# Problem: Notebook times out after 12 hours
# Solution: Use automatic checkpoint recovery
if os.path.exists(LATEST_CHECKPOINT):
    logger.info("Recovering from checkpoint...")
    embeddings, metadata = checkpoint_mgr.load_latest_checkpoint()
    start_from_step = len(embeddings)
else:
    start_from_step = 0

for step in range(start_from_step, total_steps):
    # Continue training from last checkpoint
    pass
```

### Performance Optimization Checklist

- [ ] Use **bfloat16** precision (TPU-optimized)
- [ ] Enable **gradient checkpointing** to reduce memory
- [ ] Use **jit compilation** for model forward pass
- [ ] Set **batch_size = 128** (8 cores × 16 per core)
- [ ] Enable **pmap** for cross-core parallelization
- [ ] Use **streaming data loader** (not full dataset in RAM)
- [ ] Set **max_seq_length = 1024** (trade-off: speed vs accuracy)
- [ ] Use **LoRA ranks** r=16 (smaller r = faster training)

### Performance Benchmarks

**Hardware:** TPU v3-8  
**Model:** NT-2.5B (bfloat16)  
**Batch:** 128 sequences × 1024 bp

| Task | Time | Throughput |
|------|------|-----------|
| Model Load | 3-5 min | - |
| Tokenization | 0.5s/128 | 256k tokens/sec |
| Forward Pass | 5s/128 | 25.6k vectors/sec |
| Mean Pooling | 0.5s/128 | 256k ops/sec |
| LanceDB Write | 1s/128 | 128 vectors/sec |
| **Total** | **7-8s/128** | **~45k vectors/hour** |

### Expected Memory Usage

```
Model: NT-2.5B (bfloat16)  = ~5GB
Batch (128×1024): tokens   = ~1GB
Batch: embeddings          = ~1.3GB
Optimizer states           = ~2GB
Total: ~9-10GB per TPU core

TPU v3-8 total: ~80GB (8 cores × 10GB)
```

---

## Advanced Topics

### Custom Hierarchical Loss

```python
# Fine-tune the loss weights for your taxonomy
level_weights = {
    "kingdom": 2.5,      # Highest penalty
    "phylum": 2.0,
    "class": 1.5,
    "order": 1.0,
    "family": 0.8,
    "genus": 0.5,
    "species": 0.2       # Lowest penalty
}

# Rationale: Misclassifying kingdom is worst,
# species-level errors are more tolerable
```

### Multi-Modal Training

```python
# Combine embedding + metadata features
def combined_loss(embedding_logits, metadata_logits, taxonomies):
    """
    embedding_logits: from NT embeddings
    metadata_logits: from depth, lat/lon
    """
    loss1 = hierarchical_loss(embedding_logits, taxonomies, w=0.7)
    loss2 = spatial_loss(metadata_logits, taxonomies, w=0.3)
    return loss1 + loss2
```

### Export for Edge Deployment

```python
# Save LoRA adapters for edge use
adapter_config = lora_model.get_adapter_config()
lora_model.save_pretrained("/content/outputs/lora-adapters")

# Then on edge device:
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b",
    adapter_path="/path/to/lora-adapters"
)
```

---

## FAQ

**Q: How do I use GPU instead of TPU?**  
A: Change runtime to GPU (A100 recommended). Code will auto-detect and use CUDA.

**Q: Can I run fine-tuning without TPU?**  
A: Yes, use A100 GPU. Expect 2-3x slower but fully functional.

**Q: How large can my Parquet file be?**  
A: Streaming loader handles GBs. Limited only by Google Drive storage (100GB+ recommended).

**Q: How do I monitor training remotely?**  
A: Use W&B dashboard (wandb.ai) accessible from any browser on Windows laptop.

**Q: Can I interrupt and resume?**  
A: Yes! Checkpoint system auto-saves every 30 min. Reconnect and notebook resumes.

---

## Contact & Support

**Repository:** https://github.com/FaisalTabrez/DeepBio_Edge_v3  
**Issues:** https://github.com/FaisalTabrez/DeepBio_Edge_v3/issues  
**Documentation:** [README.md](../README.md)

