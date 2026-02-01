#!/usr/bin/env python
"""Main entry point for Global-BioScan data ingestion."""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.edge.init_db import DataIngestionEngine


def main():
    """Run data ingestion pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Global-BioScan: Deep-Sea Data Ingestion Pipeline"
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=1000,
        help="Minimum depth threshold (meters)",
    )
    parser.add_argument(
        "--max-species",
        type=int,
        default=100,
        help="Maximum species to process",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding generation (use placeholder vectors)",
    )
    parser.add_argument(
        "--db-drive",
        type=str,
        default=None,
        help="Custom database drive path (default: E:\\GlobalBioScan_DB)",
    )

    args = parser.parse_args()

    # Override DB drive if specified
    if args.db_drive:
        import os

        os.environ["BIOSCANSCAN_DB_DRIVE"] = args.db_drive

    # Run pipeline
    engine = DataIngestionEngine()

    try:
        stats = engine.run_full_pipeline(
            min_depth=args.min_depth,
            max_species=args.max_species,
            skip_embedding=args.skip_embedding,
        )

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Timestamp: {stats['timestamp']}")
        logger.info(f"Species fetched: {stats['species_fetched']}")
        logger.info(f"Species with sequences: {stats['species_with_sequences']}")
        logger.info(f"Sequences stored: {stats['sequences_stored']}")
        logger.info(f"Errors: {len(stats['errors'])}")

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
