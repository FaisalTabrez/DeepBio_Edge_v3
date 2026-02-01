"""Test and demo script for data ingestion pipeline."""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edge.init_db import DataIngestionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_obis_fetching():
    """Test OBIS API integration."""
    logger.info("Testing OBIS API...")
    engine = DataIngestionEngine()
    
    species_list = engine.fetch_deep_sea_species(
        min_depth=1000,
        limit=10,  # Start small for testing
    )
    
    logger.info(f"Fetched {len(species_list)} species")
    for species in species_list[:3]:
        logger.info(f"  - {species['scientific_name']} (depth: {species['depth']}m)")
    
    return species_list


def test_ncbi_fetching():
    """Test NCBI Entrez integration."""
    logger.info("Testing NCBI Entrez API...")
    engine = DataIngestionEngine()
    
    test_species = ["Serranus cabrilla", "Gadus morhua", "Paraliparis bathypelagicus"]
    
    for species in test_species:
        result = engine.fetch_ncbi_sequence(species)
        if result:
            seq, marker, accession = result
            logger.info(f"✓ {species}: {marker} ({len(seq)}bp, {accession})")
        else:
            logger.warning(f"✗ {species}: No sequences found")


def test_taxonkit():
    """Test TaxonKit integration."""
    logger.info("Testing TaxonKit...")
    engine = DataIngestionEngine()
    
    test_species = ["Serranus cabrilla", "Homo sapiens"]
    
    for species in test_species:
        lineage = engine.normalize_taxonomy_taxonkit(species)
        if lineage:
            logger.info(f"✓ {species}: {lineage}")
        else:
            logger.warning(f"✗ {species}: Normalization failed")


def run_full_pipeline():
    """Run complete data ingestion pipeline."""
    logger.info("\n" + "="*70)
    logger.info("RUNNING FULL DATA INGESTION PIPELINE")
    logger.info("="*70)
    
    engine = DataIngestionEngine()
    stats = engine.run_full_pipeline(
        min_depth=1000,
        max_species=5,  # Small test run
        skip_embedding=True,
    )
    
    logger.info("\nPipeline Statistics:")
    for key, value in stats.items():
        if key != "errors":
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Global-BioScan data ingestion")
    parser.add_argument(
        "test",
        nargs="?",
        choices=["obis", "ncbi", "taxonkit", "full"],
        default="full",
        help="Test to run",
    )
    
    args = parser.parse_args()
    
    try:
        if args.test == "obis":
            test_obis_fetching()
        elif args.test == "ncbi":
            test_ncbi_fetching()
        elif args.test == "taxonkit":
            test_taxonkit()
        elif args.test == "full":
            run_full_pipeline()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
