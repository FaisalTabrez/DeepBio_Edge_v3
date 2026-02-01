"""High-Performance TPU Engine for DNA Embeddings using JAX/XLA.

This module implements the Cloud Command Center for processing massive-scale 
eDNA datasets using Google Cloud TPU v3-8 with the Nucleotide Transformer 2.5B model.

Architecture:
    - JAX Data Parallelism: pmap across 8 TPU cores
    - Model: InstaDeepAI/nucleotide-transformer-2.5b-multi-species (bfloat16)
    - Output: 2560-dimensional embeddings (high-resolution vectors)
    - Storage: GCS-backed streaming with LanceDB export
    - LoRA: Parameter-efficient fine-tuning for 7-level taxonomy

Performance Targets:
    - 60-80k sequences/hour on TPU v3-8
    - <10GB TPU memory per core (bfloat16 precision)
    - Real-time W&B monitoring for production runs
"""

# ============================================================================
# IMPORTS
# ============================================================================

import json
import logging
import os
import pickle
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import device_put, jit, pmap, vmap  # type: ignore
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # type: ignore

import flax  # type: ignore
import flax.linen as nn  # type: ignore
from flax.training import checkpoints, train_state  # type: ignore
from flax.core import frozen_dict  # type: ignore

import optax  # type: ignore

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# PyTorch for model loading (HuggingFace integration)
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Weights & Biases for monitoring
import wandb  # type: ignore

# Google Cloud Storage
from google.cloud import storage  # type: ignore
import gcsfs  # type: ignore

# LanceDB for vector storage
import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Model Configuration
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
EMBEDDING_DIM = 2560  # NT-2.5B output dimension (high-res)
MAX_SEQUENCE_LENGTH = 1000
BATCH_SIZE_PER_CORE = 16  # 16 seqs/core × 8 cores = 128 total batch
TPU_CORES = 8

# LoRA Configuration
LORA_RANK = 16  # Low-rank adaptation dimension
LORA_ALPHA = 32  # Scaling factor
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]  # Which attention layers to adapt

# Training Configuration
LEARNING_RATE = 2e-4
WARMUP_STEPS = 500
CHECKPOINT_INTERVAL = 500  # Save every 500 steps
MAX_TRAIN_STEPS = 10000

# Taxonomy Configuration (7-level NCBI hierarchy)
TAXONOMY_LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
HIERARCHICAL_LOSS_WEIGHTS = {
    "kingdom": 2.0,   # Highest priority
    "phylum": 1.8,
    "class": 1.6,
    "order": 1.2,
    "family": 1.0,
    "genus": 0.8,
    "species": 0.6,   # Lowest priority (most granular)
}

# Paths
GCS_BUCKET = "globalbioscan-data"  # Replace with your bucket
GCS_DATA_PATH = "parquet_shards"
DRIVE_CHECKPOINT_PATH = "/content/drive/MyDrive/GlobalBioScan/checkpoints"
DRIVE_VECTORS_PATH = "/content/drive/MyDrive/GlobalBioScan/vectors"


# ============================================================================
# JAX DEVICE MESH & TPU INITIALIZATION
# ============================================================================

def setup_tpu() -> Tuple[Any, int]:
    """Initialize TPU and return device mesh configuration.
    
    Returns:
        Tuple of (devices, num_devices)
    
    Raises:
        RuntimeError: If TPU not available
    """
    try:
        # Check TPU availability
        devices = jax.devices("tpu")
        if not devices:
            raise RuntimeError("No TPU devices found")
        
        num_devices = len(devices)
        logger.info("=" * 70)
        logger.info("TPU INITIALIZATION")
        logger.info("=" * 70)
        logger.info(f"TPU Devices: {num_devices}")
        for i, device in enumerate(devices):
            logger.info(f"  Device {i}: {device}")
        
        # Verify expected device count
        if num_devices != TPU_CORES:
            logger.warning(f"Expected {TPU_CORES} cores, got {num_devices}")
        
        return devices, num_devices
    
    except Exception as e:
        logger.error(f"TPU setup failed: {e}")
        raise RuntimeError(f"Failed to initialize TPU: {e}")


def create_device_mesh(devices: List[Any]) -> Mesh:
    """Create JAX device mesh for data parallelism.
    
    Args:
        devices: List of JAX devices (TPU cores)
    
    Returns:
        JAX Mesh object for pmap operations
    """
    num_devices = len(devices)
    mesh = Mesh(np.array(devices).reshape(num_devices), ("batch",))
    logger.info(f"Device Mesh: {mesh}")
    return mesh


# ============================================================================
# MODEL LOADING (PyTorch → JAX Bridge)
# ============================================================================

def load_nt_model_torch(model_name: str = MODEL_NAME) -> Tuple[Any, Any]:
    """Load Nucleotide Transformer model using HuggingFace/PyTorch.
    
    Note: We load in PyTorch first, then convert weights to JAX for TPU.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("=" * 70)
    logger.info("MODEL LOADING")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("✓ Tokenizer loaded")
        
        # Load model (bfloat16 for TPU optimization)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,  # TPU-optimized precision
        )
        logger.info("✓ Model loaded (bfloat16)")
        
        # Get model size
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {num_params:,} ({num_params/1e9:.2f}B)")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise


def convert_torch_to_jax(torch_model: Any) -> Dict[str, jnp.ndarray]:
    """Convert PyTorch model weights to JAX arrays.
    
    Args:
        torch_model: PyTorch model instance
    
    Returns:
        Dictionary of JAX arrays (model parameters)
    """
    logger.info("Converting PyTorch weights → JAX...")
    
    jax_params = {}
    for name, param in torch_model.named_parameters():
        # Convert to numpy, then to JAX
        jax_params[name] = jnp.array(param.detach().cpu().numpy())
    
    logger.info(f"✓ Converted {len(jax_params)} parameter tensors")
    return jax_params


# ============================================================================
# JAX EMBEDDING FUNCTIONS (JIT-COMPILED)
# ============================================================================

@jit
def mean_pooling_jax(
    hidden_states: jnp.ndarray,
    attention_mask: jnp.ndarray
) -> jnp.ndarray:
    """Perform mean pooling over sequence dimension (JAX-optimized).
    
    This function is JIT-compiled for maximum TPU performance.
    
    Args:
        hidden_states: Token embeddings (batch, seq_len, hidden_dim)
        attention_mask: Attention mask (batch, seq_len)
    
    Returns:
        Pooled embeddings (batch, hidden_dim)
    """
    # Expand mask to match hidden_states dimensions
    mask_expanded = jnp.expand_dims(attention_mask, axis=-1)  # (batch, seq_len, 1)
    mask_expanded = jnp.broadcast_to(mask_expanded, hidden_states.shape)
    
    # Masked sum
    sum_hidden = jnp.sum(hidden_states * mask_expanded, axis=1)  # (batch, hidden_dim)
    sum_mask = jnp.sum(mask_expanded, axis=1)  # (batch, hidden_dim)
    
    # Mean (avoid division by zero)
    mean_pooled = sum_hidden / jnp.maximum(sum_mask, 1e-9)
    
    return mean_pooled


@partial(pmap, axis_name="batch")
def embed_sequences_pmap(
    model_fn: Any,
    params: Dict[str, jnp.ndarray],
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray
) -> jnp.ndarray:
    """Parallel embedding generation across TPU cores (pmap).
    
    This function distributes batches across 8 TPU cores using JAX's pmap.
    Each core processes BATCH_SIZE_PER_CORE sequences independently.
    
    Args:
        model_fn: Model forward function
        params: Model parameters (replicated across cores)
        input_ids: Tokenized sequences (num_cores, batch_per_core, seq_len)
        attention_mask: Attention masks (num_cores, batch_per_core, seq_len)
    
    Returns:
        Embeddings (num_cores, batch_per_core, embedding_dim)
    """
    # Forward pass (each core processes its batch)
    hidden_states = model_fn(params, input_ids, attention_mask)
    
    # Mean pooling
    embeddings = mean_pooling_jax(hidden_states, attention_mask)
    
    return embeddings


def generate_embeddings_batched(
    sequences: List[str],
    tokenizer: Any,
    model_fn: Any,
    params: Dict[str, jnp.ndarray],
    batch_size: int = BATCH_SIZE_PER_CORE * TPU_CORES
) -> np.ndarray:
    """Generate embeddings for a batch of sequences using TPU parallelism.
    
    Args:
        sequences: List of DNA sequences
        tokenizer: HuggingFace tokenizer
        model_fn: JAX model function
        params: Model parameters
        batch_size: Total batch size (will be split across cores)
    
    Returns:
        Numpy array of embeddings (num_sequences, embedding_dim)
    """
    all_embeddings = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        
        # Tokenize
        tokens = tokenizer(
            batch,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        
        # Reshape for pmap: (num_cores, batch_per_core, seq_len)
        num_cores = TPU_CORES
        batch_per_core = len(batch) // num_cores
        
        input_ids = input_ids[:num_cores * batch_per_core].reshape(
            num_cores, batch_per_core, -1
        )
        attention_mask = attention_mask[:num_cores * batch_per_core].reshape(
            num_cores, batch_per_core, -1
        )
        
        # Convert to JAX arrays and distribute to devices
        input_ids = device_put(input_ids, NamedSharding(create_device_mesh(jax.devices("tpu")), P("batch")))
        attention_mask = device_put(attention_mask, NamedSharding(create_device_mesh(jax.devices("tpu")), P("batch")))
        
        # Run parallel embedding
        embeddings = embed_sequences_pmap(model_fn, params, input_ids, attention_mask)
        
        # Reshape back: (num_cores, batch_per_core, embedding_dim) → (total_batch, embedding_dim)
        embeddings = embeddings.reshape(-1, EMBEDDING_DIM)
        
        # Convert to numpy
        all_embeddings.append(np.array(embeddings))
    
    return np.concatenate(all_embeddings, axis=0)


# ============================================================================
# LORA IMPLEMENTATION (Flax/Optax)
# ============================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for parameter-efficient fine-tuning.
    
    LoRA decomposes weight updates into two low-rank matrices:
        ΔW = B @ A
    where A ∈ R^(r×d) and B ∈ R^(d×r), with r << d.
    
    Attributes:
        original_dim: Original layer dimension
        rank: Low-rank dimension (r)
        alpha: Scaling factor
        dropout_rate: Dropout probability
    """
    original_dim: int
    rank: int = LORA_RANK
    alpha: float = LORA_ALPHA
    dropout_rate: float = LORA_DROPOUT
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply LoRA transformation.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            training: Training mode flag
        
        Returns:
            Adapted output (batch, seq_len, hidden_dim)
        """
        # Low-rank matrices
        lora_A = self.param(
            "lora_A",
            nn.initializers.normal(stddev=0.01),
            (self.original_dim, self.rank)
        )
        lora_B = self.param(
            "lora_B",
            nn.initializers.zeros,
            (self.rank, self.original_dim)
        )
        
        # Scaling factor
        scale = self.alpha / self.rank
        
        # LoRA forward: x @ A @ B * scale
        lora_output = x @ lora_A @ lora_B * scale
        
        # Dropout during training
        if training:
            lora_output = nn.Dropout(rate=self.dropout_rate)(
                lora_output,
                deterministic=not training
            )
        
        return lora_output


class TaxonomyClassificationHead(nn.Module):
    """7-level hierarchical taxonomy classifier.
    
    This head predicts NCBI lineage from kingdom to species level.
    Each level has its own classification layer with appropriate output size.
    
    Attributes:
        embedding_dim: Input embedding dimension
        num_classes_per_level: Dictionary mapping level → num_classes
    """
    embedding_dim: int = EMBEDDING_DIM
    num_classes_per_level: Optional[Dict[str, int]] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Predict taxonomy at all 7 levels.
        
        Args:
            x: Sequence embeddings (batch, embedding_dim)
        
        Returns:
            Dictionary of logits per taxonomy level
        """
        if self.num_classes_per_level is None:
            # Default class counts (adjust based on your dataset)
            self.num_classes_per_level = {
                "kingdom": 5,      # Bacteria, Archaea, Eukaryota, Viruses, Unclassified
                "phylum": 200,     # ~200 major phyla
                "class": 500,      # ~500 classes
                "order": 1000,     # ~1000 orders
                "family": 2000,    # ~2000 families
                "genus": 10000,    # ~10k genera
                "species": 50000,  # ~50k species (expandable)
            }
        
        # Shared trunk
        hidden = nn.Dense(features=1024)(x)
        hidden = nn.relu(hidden)
        hidden = nn.Dropout(rate=0.2)(hidden, deterministic=False)
        
        # Per-level classification heads
        outputs = {}
        for level in TAXONOMY_LEVELS:
            num_classes = self.num_classes_per_level[level]
            logits = nn.Dense(features=num_classes)(hidden)
            outputs[level] = logits
        
        return outputs


def hierarchical_classification_loss(
    predictions: Dict[str, jnp.ndarray],
    targets: Dict[str, jnp.ndarray],
    weights: Dict[str, float] = HIERARCHICAL_LOSS_WEIGHTS
) -> jnp.ndarray:
    """Compute weighted hierarchical cross-entropy loss.
    
    Higher taxonomic levels (kingdom, phylum) are weighted more heavily
    to ensure the novelty detection has a solid taxonomic foundation.
    
    Args:
        predictions: Dictionary of logits per level
        targets: Dictionary of ground-truth labels per level
        weights: Loss weights per taxonomy level
    
    Returns:
        Scalar loss value
    """
    total_loss = 0.0
    
    for level in TAXONOMY_LEVELS:
        # Cross-entropy loss for this level
        logits = predictions[level]
        labels = targets[level]
        
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(loss)
        
        # Apply hierarchical weight
        weight = weights.get(level, 1.0)
        total_loss += weight * loss
    
    return total_loss


# ============================================================================
# TRAINING STATE & OPTIMIZATION
# ============================================================================

class TrainState(train_state.TrainState):
    """Extended training state with LoRA parameters.
    
    Attributes:
        apply_fn: Model forward function
        params: Frozen model parameters (not trained)
        lora_params: LoRA adapter parameters (trainable)
        tx: Optimizer
        opt_state: Optimizer state
        step: Training step counter
    """
    lora_params: frozen_dict.FrozenDict


def create_train_state(
    model: nn.Module,
    params: Dict[str, jnp.ndarray],
    lora_params: Dict[str, jnp.ndarray],
    learning_rate: float = LEARNING_RATE
) -> TrainState:
    """Initialize training state with optimizer.
    
    Args:
        model: Flax model
        params: Frozen model parameters
        lora_params: LoRA adapter parameters
        learning_rate: Initial learning rate
    
    Returns:
        TrainState object
    """
    # Optimizer: AdamW with learning rate warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_TRAIN_STEPS,
        end_value=learning_rate * 0.1,
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=frozen_dict.freeze(params),
        lora_params=frozen_dict.freeze(lora_params),
        tx=tx,
    )


@jit
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray]
) -> Tuple[TrainState, Dict[str, float]]:
    """Single training step with gradient update.
    
    Args:
        state: Current training state
        batch: Dictionary with input_ids, attention_mask, taxonomy_labels
    
    Returns:
        Tuple of (updated_state, metrics)
    """
    def loss_fn(lora_params):
        # Forward pass with LoRA
        predictions = state.apply_fn(
            {"params": state.params, "lora": lora_params},
            batch["input_ids"],
            batch["attention_mask"]
        )
        
        # Hierarchical loss
        loss = hierarchical_classification_loss(
            predictions,
            batch["taxonomy_labels"]
        )
        
        return loss, predictions
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, predictions), grads = grad_fn(state.lora_params)
    
    # Update LoRA parameters only
    state = state.apply_gradients(grads=grads)
    
    # Compute accuracy metrics
    metrics = {}
    for level in TAXONOMY_LEVELS:
        pred_labels = jnp.argmax(predictions[level], axis=-1)
        true_labels = batch["taxonomy_labels"][level]
        accuracy = jnp.mean(pred_labels == true_labels)
        metrics[f"{level}_accuracy"] = float(accuracy)
    
    metrics["loss"] = float(loss)
    
    return state, metrics


# ============================================================================
# GCS DATA STREAMING
# ============================================================================

def load_parquet_from_gcs(
    bucket_name: str = GCS_BUCKET,
    blob_path: str = GCS_DATA_PATH,
    chunk_size: int = 10000
):
    """Stream Parquet shards from Google Cloud Storage.
    
    This function yields chunks of data without loading the entire dataset
    into memory, enabling processing of TB-scale datasets.
    
    Args:
        bucket_name: GCS bucket name
        blob_path: Path to Parquet files in bucket
        chunk_size: Number of rows per chunk
    
    Yields:
        DataFrame chunks
    """
    logger.info("=" * 70)
    logger.info("GCS DATA STREAMING")
    logger.info("=" * 70)
    logger.info(f"Bucket: {bucket_name}")
    logger.info(f"Path: {blob_path}")
    
    # Initialize GCS filesystem
    fs = gcsfs.GCSFileSystem()
    
    # List all Parquet files
    pattern = f"{bucket_name}/{blob_path}/*.parquet"
    parquet_files = fs.glob(pattern)
    logger.info(f"Found {len(parquet_files)} Parquet files")
    
    for file_path in parquet_files:
        logger.info(f"Processing: {file_path}")
        
        # Open Parquet file with GCS
        with fs.open(file_path, "rb") as f:
            parquet_file = pq.ParquetFile(f)
            
            # Stream in chunks
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                yield df


# ============================================================================
# CHECKPOINTING & PERSISTENCE
# ============================================================================

class CheckpointManager:
    """Manage model checkpoints and LoRA weights.
    
    Saves checkpoints to Google Drive for persistence across Colab sessions.
    """
    
    def __init__(self, checkpoint_dir: str = DRIVE_CHECKPOINT_PATH):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints (Google Drive)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        state: TrainState,
        step: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save training checkpoint.
        
        Args:
            state: Training state
            step: Current training step
            metrics: Training metrics
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:06d}"
        
        try:
            # Save LoRA parameters
            checkpoints.save_checkpoint(
                ckpt_dir=str(self.checkpoint_dir),
                target=state.lora_params,
                step=step,
                prefix="lora_",
                keep=5,  # Keep last 5 checkpoints
            )
            
            # Save metrics
            metrics_path = checkpoint_path.with_suffix(".json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"✓ Checkpoint saved: step {step}")
        
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint.
        
        Returns:
            Dictionary with lora_params and step, or None if no checkpoint
        """
        try:
            latest_step = checkpoints.latest_checkpoint(str(self.checkpoint_dir), prefix="lora_")
            if latest_step:
                lora_params = checkpoints.restore_checkpoint(
                    ckpt_dir=str(self.checkpoint_dir),
                    target=None,
                    step=int(latest_step.split("_")[-1]),
                    prefix="lora_",
                )
                logger.info(f"✓ Loaded checkpoint: {latest_step}")
                return {"lora_params": lora_params, "step": int(latest_step.split("_")[-1])}
        
        except Exception as e:
            logger.warning(f"No checkpoint found: {e}")
        
        return None


# ============================================================================
# LANCEDB VECTOR EXPORT
# ============================================================================

class LanceDBExporter:
    """Export embeddings to LanceDB for local vector search.
    
    Vectors are saved to Google Drive, then downloaded to your local SSD.
    """
    
    def __init__(self, db_path: str = DRIVE_VECTORS_PATH):
        """Initialize LanceDB exporter.
        
        Args:
            db_path: Path to LanceDB database (Google Drive)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db = lancedb.connect(str(self.db_path))
        logger.info(f"LanceDB connected: {self.db_path}")
    
    def write_vectors(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        table_name: str = "tpu_embeddings"
    ) -> None:
        """Write embeddings and metadata to LanceDB.
        
        Args:
            embeddings: Embedding vectors (num_sequences, 2560)
            metadata: DataFrame with sequence_id, dna_sequence, etc.
            table_name: LanceDB table name
        """
        # Prepare PyArrow table
        data = {
            "sequence_id": metadata["sequence_id"].tolist(),
            "vector": embeddings.tolist(),
            "dna_sequence": metadata["dna_sequence"].tolist(),
        }
        
        # Add taxonomy columns if available
        for level in TAXONOMY_LEVELS:
            if level in metadata.columns:
                data[level] = metadata[level].tolist()
        
        table = pa.Table.from_pydict(data)
        
        # Write to LanceDB
        try:
            if table_name in self.db.table_names():
                # Append to existing table
                existing_table = self.db.open_table(table_name)
                existing_table.add(table)
                logger.info(f"✓ Appended {len(embeddings)} vectors to {table_name}")
            else:
                # Create new table
                self.db.create_table(table_name, table)
                logger.info(f"✓ Created table {table_name} with {len(embeddings)} vectors")
        
        except Exception as e:
            logger.error(f"LanceDB write failed: {e}")
            raise


# ============================================================================
# W&B MONITORING
# ============================================================================

def init_wandb(project_name: str = "GlobalBioScan-TPU", config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize Weights & Biases monitoring.
    
    Args:
        project_name: W&B project name
        config: Training configuration dictionary
    """
    if config is None:
        config = {
            "model": MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "tpu_cores": TPU_CORES,
            "batch_size": BATCH_SIZE_PER_CORE * TPU_CORES,
            "learning_rate": LEARNING_RATE,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
        }
    
    wandb.init(project=project_name, config=config)
    logger.info("✓ W&B initialized")


def log_tpu_metrics(step: int) -> None:
    """Log TPU temperature and memory usage to W&B.
    
    Args:
        step: Current training step
    """
    try:
        # Get TPU metrics (JAX backend)
        devices = jax.devices("tpu")
        
        for i, device in enumerate(devices):
            # Memory usage (if available)
            memory_stats = device.memory_stats() if hasattr(device, "memory_stats") else {}
            
            metrics = {
                f"tpu/core_{i}_memory_used": memory_stats.get("bytes_in_use", 0) / 1e9,  # GB
                f"tpu/core_{i}_memory_limit": memory_stats.get("bytes_limit", 0) / 1e9,  # GB
            }
            
            wandb.log(metrics, step=step)
    
    except Exception as e:
        logger.debug(f"TPU metrics logging failed: {e}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def run_tpu_pipeline(
    max_steps: int = MAX_TRAIN_STEPS,
    resume_from_checkpoint: bool = True
) -> None:
    """Execute complete TPU training pipeline.
    
    This is the main entry point for the Cloud Command Center.
    
    Args:
        max_steps: Maximum training steps
        resume_from_checkpoint: Resume from last checkpoint if available
    """
    logger.info("=" * 70)
    logger.info("GLOBALBIOSCAN TPU PIPELINE")
    logger.info("=" * 70)
    
    # 1. Initialize TPU
    devices, num_devices = setup_tpu()
    mesh = create_device_mesh(devices)
    
    # 2. Load model
    torch_model, tokenizer = load_nt_model_torch()
    jax_params = convert_torch_to_jax(torch_model)
    
    # 3. Initialize LoRA
    lora_model = TaxonomyClassificationHead()
    # Initialize LoRA params (dummy forward pass)
    dummy_input = jnp.ones((1, EMBEDDING_DIM))
    lora_params = lora_model.init(jax.random.PRNGKey(0), dummy_input)
    
    # 4. Create training state
    state = create_train_state(lora_model, jax_params, lora_params)
    
    # 5. Load checkpoint if resuming
    checkpoint_mgr = CheckpointManager()
    if resume_from_checkpoint:
        checkpoint = checkpoint_mgr.load_latest_checkpoint()
        if checkpoint:
            state = state.replace(
                lora_params=checkpoint["lora_params"],
                step=checkpoint["step"]
            )
    
    # 6. Initialize W&B
    init_wandb()
    
    # 7. Initialize LanceDB exporter
    lance_exporter = LanceDBExporter()
    
    # 8. Training loop
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING STARTED")
    logger.info("=" * 70)
    
    step = int(state.step)
    
    for chunk_df in load_parquet_from_gcs():
        if step >= max_steps:
            break
        
        # Prepare batch
        sequences = chunk_df["dna_sequence"].tolist()
        
        # Generate embeddings (pmap across TPU cores)
        embeddings = generate_embeddings_batched(
            sequences,
            tokenizer,
            lora_model.apply,
            state.params,
        )
        
        # Training step (if taxonomy labels available)
        if "kingdom" in chunk_df.columns:
            batch = {
                "input_ids": embeddings,  # Use pre-computed embeddings
                "attention_mask": jnp.ones(embeddings.shape[0]),
                "taxonomy_labels": {
                    level: jnp.array(chunk_df[level].values)
                    for level in TAXONOMY_LEVELS
                    if level in chunk_df.columns
                }
            }
            
            state, metrics = train_step(state, batch)
            
            # Log to W&B
            wandb.log(metrics, step=step)
            log_tpu_metrics(step)
            
            # Checkpoint
            if step % CHECKPOINT_INTERVAL == 0:
                checkpoint_mgr.save_checkpoint(state, step, metrics)
        
        # Export embeddings to LanceDB
        lance_exporter.write_vectors(embeddings, chunk_df)
        
        step += 1
        
        logger.info(f"Step {step}/{max_steps} | Loss: {metrics.get('loss', 0.0):.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    # Final checkpoint
    checkpoint_mgr.save_checkpoint(state, step, metrics)
    
    wandb.finish()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Command-line interface for TPU engine."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GlobalBioScan TPU Engine - High-Performance DNA Embeddings"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_TRAIN_STEPS,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="GlobalBioScan-TPU",
        help="W&B project name"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    try:
        run_tpu_pipeline(
            max_steps=args.max_steps,
            resume_from_checkpoint=not args.no_resume
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
