#!/usr/bin/env python
"""Entry point for Nucleotide Transformer embedding generation."""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.edge.embedder import main

if __name__ == "__main__":
    sys.exit(main())
