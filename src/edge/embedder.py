"""AI-powered DNA embedding using Nucleotide Transformers."""

# ============================================================================
# WINDOWS COMPATIBILITY PATCHES (Must be at top!)
# ============================================================================
# Mock Triton and FlashAttention to prevent ImportError on Windows
import sys
from unittest.mock import MagicMock

# Mock triton (CUDA kernel optimizer, Linux-only)
if sys.platform == "win32" or True:  # Force mock for safety
    sys.modules["triton"] = MagicMock()
    sys.modules["triton.language"] = MagicMock()
    sys.modules["triton.ops"] = MagicMock()

# Mock flash_attn (FastTransformer kernels, Linux-only)
sys.modules["flash_attn"] = MagicMock()
sys.modules["flash_attn.flash_attention"] = MagicMock()
sys.modules["flash_attn.ops"] = MagicMock()

# ============================================================================
# IMPORTS
# ============================================================================

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import lancedb
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.config import (
    DEBUG,
    LANCEDB_PENDRIVE_PATH,
    LANCEDB_TABLE_SEQUENCES,
    MODEL_BATCH_SIZE,
    MODEL_MAX_LENGTH,
    MODEL_NAME,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

CHECKPOINT_DIR = PROCESSED_DATA_DIR / "embedder_checkpoints"
EMBEDDING_DIM = 768  # NT-500M output dimension
VALID_DNA_CHARS = set("ATGCN")


class EmbeddingEngine:
    """Generate embeddings using Nucleotide Transformer foundation model.
    
    This class handles:
    - Model loading with GPU/CPU detection
    - Batch inference with memory management
    - LanceDB integration for vector updates
    - Checkpointing for resume capability
    - Progress tracking and statistics
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = MODEL_BATCH_SIZE,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        use_gpu: Optional[bool] = None,
    ):
        """Initialize embedding engine with pretrained model.

        Args:
            model_name: Hugging Face model identifier
            batch_size: Batch size for inference (adjust for RAM)
            checkpoint_dir: Directory for saving checkpoints
            use_gpu: Force GPU (True) or CPU (False). Auto-detect if None.

        Raises:
            RuntimeError: If model loading fails
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("NUCLEOTIDE TRANSFORMER EMBEDDING ENGINE")
        logger.info("=" * 70)

        # Detect GPU availability
        self.gpu_available = torch.cuda.is_available()
        if use_gpu is not None:
            self.gpu_available = use_gpu and self.gpu_available
        
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
        logger.info(f"Device: {self.device.type.upper()}")
        if self.gpu_available:
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            torch_version = getattr(torch, "version", None)
            cuda_version = getattr(torch_version, "cuda", None)
            logger.info(f"  CUDA: {cuda_version}")

        # Determine precision
        self.use_fp16 = self.gpu_available  # Use FP16 on GPU, FP32 on CPU
        dtype_str = "FP16" if self.use_fp16 else "FP32"
        logger.info(f"Precision: {dtype_str}")

        # Load model and tokenizer
        logger.info(f"\nLoading model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            logger.info("  ✓ Tokenizer loaded")

            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True, output_hidden_states=True
            )
            logger.info("  ✓ Model loaded")

            # Set precision
            if self.use_fp16:
                self.model = self.model.half()
                logger.info("  ✓ Converted to FP16")

            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Evaluation mode
            logger.info(f"  ✓ Moved to {self.device.type.upper()}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

        # Statistics
        self.stats = {
            "sequences_processed": 0,
            "sequences_embedded": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

    # ========================================================================
    # EMBEDDING GENERATION
    # ========================================================================

    def validate_sequence(self, sequence: str) -> bool:
        """Validate DNA sequence format.

        Args:
            sequence: DNA sequence string

        Returns:
            True if valid (contains only ATGCN), False otherwise
        """
        if not sequence or not isinstance(sequence, str):
            return False
        return set(sequence.upper()).issubset(VALID_DNA_CHARS)

    def get_embeddings(
        self, sequences: list[str], show_progress: bool = False
    ) -> np.ndarray:
        """Generate embeddings for multiple DNA sequences.

        Args:
            sequences: List of DNA sequences (each <= max_length)
            show_progress: Show progress bar if True

        Returns:
            Numpy array of shape (num_sequences, 768)

        Raises:
            ValueError: If sequences list is empty or invalid
        """
        if not sequences:
            raise ValueError("Sequences list is empty")

        valid_sequences = []
        for seq in sequences:
            if self.validate_sequence(seq):
                valid_sequences.append(seq.upper())
            else:
                logger.warning(f"Invalid sequence (skipped): {seq[:50]}...")

        if not valid_sequences:
            raise ValueError("No valid sequences in list")

        embeddings = []
        iterator = (
            tqdm(valid_sequences, desc="Embedding", disable=not show_progress)
            if show_progress
            else valid_sequences
        )

        with torch.no_grad():
            for sequence in iterator:
                try:
                    # Tokenize
                    tokens = self.tokenizer(
                        sequence,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=MODEL_MAX_LENGTH,
                    )

                    # Move to device
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}

                    # Forward pass
                    output = self.model(**tokens)

                    # Extract hidden states from last layer
                    hidden_states = output.hidden_states[-1]  # (batch, seq_len, 768)

                    # Mean pooling over sequence dimension
                    # Ignore padding tokens using attention mask
                    attention_mask = tokens["attention_mask"]
                    mask_expanded = (
                        attention_mask.unsqueeze(-1)
                        .expand(hidden_states.size())
                        .float()
                    )
                    sum_hidden = (hidden_states * mask_expanded).sum(1)
                    sum_mask = mask_expanded.sum(1)
                    embedding = sum_hidden / sum_mask.clamp(min=1e-9)

                    # Convert to numpy
                    embedding = embedding.squeeze().cpu().numpy().astype(np.float32)
                    embeddings.append(embedding)
                    self.stats["sequences_embedded"] += 1

                except Exception as e:
                    logger.error(f"Embedding error: {e}")
                    self.stats["errors"] += 1
                    # Append zero vector on error
                    embeddings.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))

        return np.array(embeddings, dtype=np.float32)

    def get_embedding_single(self, sequence: str) -> Optional[np.ndarray]:
        """Generate embedding for a single DNA sequence.

        Args:
            sequence: DNA sequence string

        Returns:
            Embedding vector (1D array, 768-dim) or None on error
        """
        if not self.validate_sequence(sequence):
            logger.warning(f"Invalid sequence: {sequence[:50]}...")
            return None

        embeddings = self.get_embeddings([sequence])
        return embeddings[0] if len(embeddings) > 0 else None

    # ========================================================================
    # LANCEDB INTEGRATION
    # ========================================================================

    def connect_lancedb(self, db_path: str = LANCEDB_PENDRIVE_PATH) -> lancedb.db.DBConnection:
        """Connect to LanceDB instance.

        Args:
            db_path: Path to LanceDB database

        Returns:
            LanceDB connection object

        Raises:
            RuntimeError: If connection fails
        """
        try:
            db = lancedb.connect(db_path)
            logger.info(f"Connected to LanceDB: {db_path}")
            return db
        except Exception as e:
            logger.error(f"LanceDB connection failed: {e}")
            raise RuntimeError(f"Failed to connect to LanceDB: {e}")

    def get_checkpoint_data(self) -> dict:
        """Load last checkpoint if it exists.

        Returns:
            Dictionary with checkpoint data or empty dict if none exists
        """
        checkpoint_file = self.checkpoint_dir / "embedding_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r") as f:
                    data = json.load(f)
                logger.info(f"Loaded checkpoint: {data['last_embedded_id']}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return {}

    def save_checkpoint(self, last_embedded_id: str, total_processed: int) -> None:
        """Save checkpoint for resume capability.

        Args:
            last_embedded_id: ID of last embedded sequence
            total_processed: Total sequences processed so far
        """
        checkpoint_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "last_embedded_id": last_embedded_id,
            "total_processed": total_processed,
            "model": self.model_name,
        }
        checkpoint_file = self.checkpoint_dir / "embedding_checkpoint.json"
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.debug(f"Checkpoint saved: {last_embedded_id}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    # ========================================================================
    # BATCH EMBEDDING & DATABASE UPDATE
    # ========================================================================

    def embed_and_update_lancedb(
        self,
        db_path: str = LANCEDB_PENDRIVE_PATH,
        max_sequences: Optional[int] = None,
        resume: bool = True,
    ) -> dict:
        """Main workflow: Embed sequences and update LanceDB vectors.

        This is the "brain surgery" function that:
        1. Connects to LanceDB
        2. Fetches rows with placeholder vectors (all zeros)
        3. Batches sequences and generates real embeddings
        4. Updates vectors in-place
        5. Saves checkpoints for resume

        Args:
            db_path: Path to LanceDB database
            max_sequences: Max sequences to embed (None = all)
            resume: Resume from last checkpoint if True

        Returns:
            Dictionary with processing statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("BATCH EMBEDDING & LANCEDB UPDATE")
        logger.info("=" * 70)

        self.stats["start_time"] = datetime.utcnow().isoformat()

        try:
            # Connect to LanceDB
            db = self.connect_lancedb(db_path)
            table = db.open_table(LANCEDB_TABLE_SEQUENCES)
            logger.info(f"Opened table: {LANCEDB_TABLE_SEQUENCES}")

            # Get checkpoint if resuming
            checkpoint = self.get_checkpoint_data() if resume else {}
            last_embedded_id = checkpoint.get("last_embedded_id")

            # Fetch all rows
            all_rows = table.search().limit(100000).to_list()
            logger.info(f"Total rows in table: {len(all_rows)}")

            if not all_rows:
                logger.warning("Table is empty!")
                return self.stats

            # Filter: rows with placeholder vectors (all zeros)
            rows_to_embed = []
            for row in all_rows:
                # Skip if already embedded
                if last_embedded_id and row.get("sequence_id") == last_embedded_id:
                    logger.info(f"Resuming from: {last_embedded_id}")
                    rows_to_embed = all_rows[all_rows.index(row) + 1 :]
                    break

                # Check if vector is placeholder (all zeros or near-zero)
                vector = row.get("vector", [])
                if isinstance(vector, list) and len(vector) > 0:
                    if not np.allclose(vector, 0.0, atol=1e-6):
                        continue  # Already embedded

                rows_to_embed.append(row)

            if max_sequences:
                rows_to_embed = rows_to_embed[:max_sequences]

            logger.info(f"Rows to embed: {len(rows_to_embed)}")

            if not rows_to_embed:
                logger.info("All sequences already embedded!")
                self.stats["end_time"] = datetime.utcnow().isoformat()
                return self.stats

            # Process in batches
            update_count = 0
            sequences_batch = []
            ids_batch = []

            pbar = tqdm(rows_to_embed, desc="Processing sequences")

            for row in pbar:
                sequence = row.get("dna_sequence", "")
                seq_id = row.get("sequence_id", "unknown")

                if not sequence:
                    logger.warning(f"Empty sequence for {seq_id}")
                    self.stats["errors"] += 1
                    continue

                sequences_batch.append(sequence)
                ids_batch.append((seq_id, row))
                self.stats["sequences_processed"] += 1

                # Process batch
                if len(sequences_batch) >= self.batch_size or row == rows_to_embed[-1]:
                    logger.debug(f"Processing batch of {len(sequences_batch)}")

                    # Generate embeddings
                    embeddings = self.get_embeddings(sequences_batch, show_progress=False)

                    # Update LanceDB with new vectors
                    for (seq_id, original_row), embedding in zip(ids_batch, embeddings):
                        updated_row = original_row.copy()
                        updated_row["vector"] = embedding.tolist()

                        try:
                            table.update(
                                where=f"sequence_id == '{seq_id}'",
                                values={"vector": updated_row["vector"]},
                            )
                            update_count += 1
                            self.save_checkpoint(seq_id, self.stats["sequences_processed"])
                        except Exception as e:
                            logger.error(f"Update failed for {seq_id}: {e}")
                            self.stats["errors"] += 1

                    sequences_batch = []
                    ids_batch = []

                pbar.set_postfix(
                    {
                        "embedded": self.stats["sequences_embedded"],
                        "errors": self.stats["errors"],
                    }
                )

            self.stats["end_time"] = datetime.utcnow().isoformat()

            logger.info("\n" + "=" * 70)
            logger.info("EMBEDDING COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Total processed: {self.stats['sequences_processed']}")
            logger.info(f"Total embedded: {self.stats['sequences_embedded']}")
            logger.info(f"Total updated: {update_count}")
            logger.info(f"Errors: {self.stats['errors']}")

            return self.stats

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}", exc_info=True)
            self.stats["end_time"] = datetime.utcnow().isoformat()
            raise

    # ========================================================================
    # VALIDATION & TESTING
    # ========================================================================

    def validate_embeddings(self, test_sequences: Optional[list[tuple[str, str]]] = None) -> None:
        """Validate embeddings by checking semantic similarity.

        This test embeds similar sequences and verifies that their cosine
        similarity is high, proving the model understands biological proximity.

        Args:
            test_sequences: Optional list of test sequences. If None, uses defaults.
        """
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION: SEMANTIC SIMILARITY TEST")
        logger.info("=" * 70)

        # Default test sequences: COI sequences from related species
        if test_sequences is None:
            test_sequences = [
                # Similar COI sequences (same genus, different species)
                (
                    "ATGATTATCAATACATTAATATTAATCATTAAAGAATTAATGAAATTATCACCACTATATAATATCAACC"
                    "ACGCGCGTTGCATTACATAGTATTCGTAGCCGTATTTATTACAGTAGCACAGATCGCAAATGTAAAAGAG"
                    "ATCGGACAATGACTATTTAACACTATTCGACGAATTAATATACCGGACCCGCACGAATGTTCTTATGCC"
                    "CCAATATATGAAGATGTACTCACAGAGTTACTAGCCGATATTGTTCTATTAACTGCCGTTTTAGCCGGT"
                    "ATGTTAACCGTATCAGAAATACGAAATGCTATTTACGACTCTTACACGGATGAGGAGACCCAGAAGTAC",
                    "COI_species_A",
                ),
                (
                    "ATGATTATCAATACATTAATATTAATCATTAAAGAATTAATGAAATTATCACCACTATATAATATCAACC"
                    "ACGCGCGTTGCATTACATAGTATTCGTAGCCGTATTTATTACAGTAGCACAGATCGCAAATGTAAAAGAG"
                    "ATCGGACAATGACTATTTAACACTATTCGACGAATTAATATACCGGACCCGCACGAATGTTCTTATGCC"
                    "CCAATATATGAAGATGTACTCACAGAGTTACTAGCCGATATTGTTCTATTAACTGCCGTTTTAGCCGGT"
                    "ATGTTAACCGTATCAGAAATACGAAATGCTATTTACGACTCTTACACGGATGAGCAGACCCAGAAGTAC",  # Slightly different
                    "COI_species_B (similar)",
                ),
                (
                    # Dissimilar 18S sequence
                    "AACGAGTGAGCTGCAGTGTGAGTGCAGAGGTGAAATTCTAGATTGAGGTGGGATAGGGCGGAGCGAGAA"
                    "GTGATCAGAGAGGATTAGGCTGGTTTCTTTTGGTGGTGCACTCGATGCCTAGAGGTGAGATTGTTGATT"
                    "CGGATAACGGTGGTCATGCATGGCGGGGTGGTGGTGCATGGCCGTTCTTAGTTGGTGGAGCGATTTGTC"
                    "TGGTTAATTCCGATAACGAACGAGACTCTGGCATGCTAATCGTAGACGAGTCAGCCGTTCGATGGATCA"
                    "GGTAGGAGTGCGTTGCACTGGGAGTGAGTGGTAGTGAGCGAGCAGCGAACGTGGTGAAACTCCGTCTGA",
                    "18S_species_C (dissimilar)",
                ),
            ]

        logger.info(f"Testing {len(test_sequences)} sequences...")

        # Generate embeddings
        sequences = [seq[0] for seq in test_sequences]
        labels = [seq[1] for seq in test_sequences]

        embeddings = self.get_embeddings(sequences, show_progress=False)
        logger.info(f"Generated embeddings: shape {embeddings.shape}")

        # Compute pairwise cosine similarities
        similarities = cosine_similarity(embeddings)

        # Log results
        logger.info("\nCosine Similarity Matrix:")
        logger.info("  " + "  ".join(f"{i:6d}" for i in range(len(labels))))

        for i, label_i in enumerate(labels):
            row_str = f"{i}: "
            for j in range(len(labels)):
                sim = similarities[i][j]
                row_str += f"{sim:6.3f} "
            logger.info(row_str + f"  ({label_i})")

        # Assertions
        similar_sim = similarities[0][1]  # Should be high (same genus)
        dissimilar_sim = similarities[0][2]  # Should be lower (different marker)

        logger.info(f"\n✓ Similar sequences (COI): {similar_sim:.4f}")
        logger.info(f"✓ Dissimilar sequences (COI vs 18S): {dissimilar_sim:.4f}")

        if similar_sim > dissimilar_sim:
            logger.info("✓ VALIDATION PASSED: Model understands biological proximity!")
        else:
            logger.warning(
                "⚠ Similarity inversion detected. This may indicate model issues."
            )

    def get_model_info(self) -> dict:
        """Return model metadata.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": EMBEDDING_DIM,
            "max_length": MODEL_MAX_LENGTH,
            "batch_size": self.batch_size,
            "device": self.device.type,
            "precision": "FP16" if self.use_fp16 else "FP32",
            "gpu_available": self.gpu_available,
            "tokenizer_type": type(self.tokenizer).__name__,
        }


# ============================================================================
# CLI & MAIN EXECUTION
# ============================================================================


def main():
    """Command-line entry point for embedding generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Nucleotide Transformer embeddings for DNA sequences"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MODEL_BATCH_SIZE,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Max sequences to embed (for testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation test only (no LanceDB update)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU usage",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage",
    )

    args = parser.parse_args()

    # Determine device
    use_gpu = None
    if args.gpu:
        use_gpu = True
    elif args.cpu:
        use_gpu = False

    # Initialize engine
    engine = EmbeddingEngine(batch_size=args.batch_size, use_gpu=use_gpu)

    # Print model info
    logger.info("\nModel Information:")
    for key, value in engine.get_model_info().items():
        logger.info(f"  {key}: {value}")

    try:
        if args.validate_only:
            # Run validation only
            engine.validate_embeddings()
        else:
            # Full pipeline: embed sequences and update LanceDB
            stats = engine.embed_and_update_lancedb(
                max_sequences=args.max_sequences,
                resume=not args.no_resume,
            )

            # Validation test
            engine.validate_embeddings()

            logger.info("\nFinal Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
