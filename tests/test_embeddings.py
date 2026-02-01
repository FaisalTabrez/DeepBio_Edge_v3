"""Test suite for Nucleotide Transformer embedding engine."""

import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edge.embedder import EmbeddingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test model and tokenizer loading."""
    logger.info("=" * 70)
    logger.info("TEST 1: Model Loading")
    logger.info("=" * 70)

    try:
        engine = EmbeddingEngine()
        logger.info("✓ Model and tokenizer loaded successfully")
        
        info = engine.get_model_info()
        logger.info("\nModel Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False


def test_single_embedding():
    """Test embedding a single sequence."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Single Sequence Embedding")
    logger.info("=" * 70)

    try:
        engine = EmbeddingEngine()
        
        # Test sequence: COI marker gene snippet
        sequence = (
            "ATGATTATCAATACATTAATATTAATCATTAAAGAATTAATGAAATTATCACCACTATATAATATCAACC"
            "ACGCGCGTTGCATTACATAGTATTCGTAGCCGTATTTATTACAGTAGCACAGATCGCAAATGTAAAAGAG"
        )
        
        logger.info(f"Sequence length: {len(sequence)} bp")
        embedding = engine.get_embedding_single(sequence)
        
        if embedding is not None:
            logger.info(f"✓ Embedding generated: shape {embedding.shape}, dtype {embedding.dtype}")
            logger.info(f"  Min: {embedding.min():.6f}, Max: {embedding.max():.6f}, Mean: {embedding.mean():.6f}")
            return True
        else:
            logger.error("✗ Embedding is None")
            return False
    except Exception as e:
        logger.error(f"✗ Single embedding test failed: {e}", exc_info=True)
        return False


def test_batch_embedding():
    """Test batch embedding."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Batch Embedding")
    logger.info("=" * 70)

    try:
        engine = EmbeddingEngine(batch_size=4)
        
        sequences = [
            "ATGATTATCAATACATTAATATTAATCATTAAAGAATTAATGAAATTATCACCACTATATAATATCAACC",
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT",
            "TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA",
        ]
        
        logger.info(f"Processing {len(sequences)} sequences...")
        embeddings = engine.get_embeddings(sequences, show_progress=True)
        
        if embeddings is not None and len(embeddings) == len(sequences):
            logger.info(f"✓ Batch embeddings generated: shape {embeddings.shape}")
            return True
        else:
            logger.error("✗ Batch embedding shape mismatch")
            return False
    except Exception as e:
        logger.error(f"✗ Batch embedding test failed: {e}", exc_info=True)
        return False


def test_validation():
    """Test embedding validation (semantic similarity)."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Validation (Cosine Similarity)")
    logger.info("=" * 70)

    try:
        engine = EmbeddingEngine()
        engine.validate_embeddings()
        logger.info("✓ Validation test completed")
        return True
    except Exception as e:
        logger.error(f"✗ Validation test failed: {e}", exc_info=True)
        return False


def test_invalid_sequences():
    """Test handling of invalid sequences."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Invalid Sequence Handling")
    logger.info("=" * 70)

    try:
        engine = EmbeddingEngine()
        
        sequences = [
            "ATGCATGC",  # Valid
            "INVALIDXYZ",  # Invalid (has non-DNA chars)
            "ATGCATGC",  # Valid
        ]
        
        logger.info("Processing mix of valid and invalid sequences...")
        embeddings = engine.get_embeddings(sequences, show_progress=False)
        
        if len(embeddings) == len(sequences):
            logger.info(f"✓ Processed {len(sequences)} sequences (invalid ones handled gracefully)")
            logger.info(f"  Statistics: {engine.stats}")
            return True
        else:
            logger.error("✗ Output length mismatch")
            return False
    except Exception as e:
        logger.error(f"✗ Invalid sequence test failed: {e}", exc_info=True)
        return False


def run_all_tests():
    """Run all tests."""
    logger.info("Starting Embedding Engine Test Suite\n")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Single Embedding", test_single_embedding),
        ("Batch Embedding", test_batch_embedding),
        ("Validation", test_validation),
        ("Invalid Sequences", test_invalid_sequences),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return all(results.values())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Embedding Engine")
    parser.add_argument(
        "test",
        nargs="?",
        choices=["model", "single", "batch", "validation", "invalid", "all"],
        default="all",
        help="Test to run",
    )
    
    args = parser.parse_args()
    
    success = False
    if args.test == "model":
        success = test_model_loading()
    elif args.test == "single":
        success = test_single_embedding()
    elif args.test == "batch":
        success = test_batch_embedding()
    elif args.test == "validation":
        success = test_validation()
    elif args.test == "invalid":
        success = test_invalid_sequences()
    elif args.test == "all":
        success = run_all_tests()
    
    sys.exit(0 if success else 1)
