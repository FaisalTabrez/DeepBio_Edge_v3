"""Edge processing module - core bioinformatics pipeline."""

from typing import TYPE_CHECKING

from .init_db import DataIngestionEngine
from .taxonomy import NoveltyDetector, TaxonomyPredictor

# Conditionally import EmbeddingEngine to avoid transformers/torch loading on Windows
if TYPE_CHECKING:
    from .embedder import EmbeddingEngine

__all__ = [
	"DataIngestionEngine",
	"EmbeddingEngine",
	"TaxonomyPredictor",
	"NoveltyDetector",
]
