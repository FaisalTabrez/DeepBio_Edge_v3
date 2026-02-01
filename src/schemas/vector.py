"""Vector storage schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class EmbeddingRecord(BaseModel):
    """Embedding vector with metadata."""

    sequence_id: str = Field(..., description="Reference sequence ID")
    embedding: list[float] = Field(..., description="NT-2.5B embedding (dim=256)")
    embedding_dim: int = Field(default=256, description="Embedding dimensionality")
    marker_gene: str = Field(..., description="COI or 18S")
    species: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth_m: Optional[float] = None
    batch_id: str = Field(..., description="Source batch ID")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "sequence_id": "OBIS_COI_1",
                "embedding": [0.1, -0.2, 0.5],  # truncated for brevity
                "embedding_dim": 256,
                "marker_gene": "COI",
                "species": "Unknown",
                "latitude": -60.5,
                "longitude": 45.2,
                "depth_m": 3000,
                "batch_id": "BATCH_20260201_001",
            }
        }


class SearchResult(BaseModel):
    """Result from LanceDB vector search."""

    query_sequence_id: str
    rank: int = Field(ge=1, description="Ranking in result set (1=closest)")
    match_sequence_id: str = Field(..., description="ID of matched sequence")
    distance: float = Field(..., ge=0.0, description="L2 distance in embedding space")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity [0-1]")
    species: Optional[str] = None
    marker_gene: str = Field(..., description="COI or 18S")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
