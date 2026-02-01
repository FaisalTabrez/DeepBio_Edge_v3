"""Taxonomic schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class TaxonomicLineage(BaseModel):
    """Complete taxonomic lineage from TaxonKit."""

    sequence_id: str = Field(..., description="Reference sequence ID")
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_: Optional[str] = Field(None, alias="class")
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None
    taxid: Optional[int] = Field(None, description="NCBI Taxonomy ID")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        """Pydantic config."""

        populate_by_name = True

    def get_lineage_string(self) -> str:
        """Return full lineage as semicolon-separated string."""
        levels = [
            self.kingdom,
            self.phylum,
            self.class_,
            self.order,
            self.family,
            self.genus,
            self.species,
        ]
        return ";".join(str(l) for l in levels if l)


class NoveltyResult(BaseModel):
    """Novelty detection result."""

    sequence_id: str = Field(..., description="Query sequence ID")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Score [0=known, 1=novel]")
    nearest_cluster_id: Optional[int] = Field(None, description="HDBSCAN cluster ID")
    cluster_distance: Optional[float] = Field(None, description="Distance to cluster centroid")
    proposed_genus: Optional[str] = Field(None, description="Inferred novel genus")
    proposed_species: Optional[str] = Field(None, description="Inferred novel species")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_novel: bool = Field(..., description="Threshold-based novelty classification")
