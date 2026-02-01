"""DNA sequence schemas."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class DNASequence(BaseModel):
    """Single DNA sequence record."""

    sequence_id: str = Field(..., description="Unique identifier (e.g., OBIS_COI_12345)")
    sequence: str = Field(..., description="DNA sequence (A/C/G/T)")
    marker_gene: Literal["COI", "18S", "16S"] = Field(..., description="Marker gene type")
    species: Optional[str] = Field(None, description="Known species (NCBI taxonomy)")
    latitude: Optional[float] = Field(None, description="Sampling latitude")
    longitude: Optional[float] = Field(None, description="Sampling longitude")
    depth_m: Optional[float] = Field(None, description="Sampling depth in meters")
    source: Literal["OBIS", "NCBI", "Custom"] = Field(default="NCBI")
    length_bp: int = Field(..., description="Sequence length in base pairs")

    @field_validator("sequence")
    @classmethod
    def validate_dna_sequence(cls, v: str) -> str:
        """Ensure valid DNA characters."""
        valid_chars = set("ACGTN")
        if not set(v.upper()).issubset(valid_chars):
            raise ValueError("Sequence contains invalid DNA characters")
        return v.upper()

    @field_validator("length_bp")
    @classmethod
    def validate_length(cls, v: int) -> int:
        """Ensure reasonable sequence length."""
        if v < 100 or v > 100000:
            raise ValueError("Sequence length must be between 100 and 100,000 bp")
        return v


class SequenceBatch(BaseModel):
    """Batch of sequences for processing."""

    batch_id: str = Field(..., description="Unique batch identifier")
    sequences: list[DNASequence] = Field(..., min_items=1)
    timestamp: str = Field(..., description="ISO8601 timestamp")
    processing_stage: Literal["raw", "cleaned", "embedded"] = Field(default="raw")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "batch_id": "BATCH_20260201_001",
                "sequences": [
                    {
                        "sequence_id": "OBIS_COI_1",
                        "sequence": "ATGCATGCATGC",
                        "marker_gene": "COI",
                        "species": "Unknown",
                        "latitude": -60.5,
                        "longitude": 45.2,
                        "depth_m": 3000,
                        "source": "OBIS",
                        "length_bp": 658,
                    }
                ],
                "timestamp": "2026-02-01T10:30:00Z",
                "processing_stage": "raw",
            }
        }
