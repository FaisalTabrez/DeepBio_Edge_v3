"""Real-world data ingestion from OBIS and NCBI."""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, cast
from urllib.parse import urlencode

import lancedb
import requests
from Bio import Entrez, SeqIO

from src.config import (
    FASTA_MAX_LENGTH,
    FASTA_MIN_LENGTH,
    LANCEDB_URI,
    LANCEDB_TABLE_SEQUENCES,
    LANCEDB_TABLE_TAXONOMY,
    NCBI_API_KEY,
    NCBI_EMAIL,
    NCBI_MAX_RETRIES,
    NCBI_TIMEOUT,
    OBIS_API_BASE,
    OBIS_TIMEOUT,
    PROCESSED_DATA_DIR,
    TAXONKIT_DB_DIR,
    TAXONKIT_NAMES_PATH,
    TAXONKIT_NODES_PATH,
)
from src.schemas.sequence import DNASequence, SequenceBatch
from src.schemas.taxonomy import TaxonomicLineage

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Database path on 32GB pendrive (make this a variable)
DB_DRIVE = os.getenv("BIOSCANSCAN_DB_DRIVE", "E:\\GlobalBioScan_DB")
LANCEDB_PENDRIVE_PATH = str(Path(DB_DRIVE) / "lancedb")

# Deep-sea hotspot for OBIS
DEEP_SEA_HOTSPOT = {
    "geometry": {"type": "Polygon", "coordinates": [
        [[70, -10], [75, -10], [75, 0], [70, 0], [70, -10]]  # Central Indian Ridge
    ]},
    "depth_min": 1000,  # Abyssal/hadal zones
}

# NCBI configuration
Entrez.email = NCBI_EMAIL
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY


class DataIngestionEngine:
    """Handle OBIS/NCBI data ingestion and LanceDB initialization."""

    def __init__(self, db_path: str = LANCEDB_PENDRIVE_PATH):
        """Initialize data ingestion engine with LanceDB connection.

        Args:
            db_path: Path to LanceDB database on pendrive
        """
        self.db_path = db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)
        logger.info(f"Connected to LanceDB at {db_path}")
        
        self.session = requests.Session()
        self.default_timeout = OBIS_TIMEOUT
        self.sequences_fetched = 0
        self.sequences_stored = 0

    # ========================================================================
    # OBIS INTEGRATION
    # ========================================================================

    def fetch_deep_sea_species(
        self, 
        min_depth: int = 1000,
        limit: int = 500,
        geometry: Optional[dict] = None
    ) -> list[dict]:
        """Fetch marine species from OBIS at depths > min_depth.

        Args:
            min_depth: Minimum depth in meters (default: 1000 for abyssal)
            limit: Maximum number of occurrences to fetch
            geometry: Optional GeoJSON geometry polygon for spatial filtering

        Returns:
            List of unique species records with scientificName, taxonID, lat, lon, depth
        """
        logger.info(f"Fetching OBIS occurrences at depth > {min_depth}m...")

        # Build OBIS API query
        params: dict[str, str | int | float] = {
            "limit": int(min(limit, 50000)),  # OBIS pagination limit
            "offset": 0,
        }
        
        # Add geometry if provided (Central Indian Ridge by default)
        if geometry:
            params["geometry"] = json.dumps(geometry)

        url = f"{OBIS_API_BASE}/occurrence"
        all_records = []
        unique_species = {}

        try:
            offset = 0
            while offset < limit:
                params["offset"] = offset
                logger.debug(f"Fetching OBIS batch at offset {offset}...")
                response = self.session.get(url, params=params, timeout=self.default_timeout)
                response.raise_for_status()
                data = response.json()

                if not data.get("results"):
                    logger.info(f"No more results. Fetched {len(all_records)} total occurrences.")
                    break

                for record in data["results"]:
                    # Filter by depth
                    depth = record.get("maximumDepthInMeters", 0)
                    if depth and depth < min_depth:
                        continue

                    species_name = record.get("scientificName", "Unknown")
                    taxon_id = record.get("taxonID")
                    
                    # De-duplicate by species name
                    if species_name not in unique_species:
                        unique_species[species_name] = {
                            "scientific_name": species_name,
                            "taxon_id": taxon_id,
                            "depth": depth,
                            "latitude": record.get("decimalLatitude"),
                            "longitude": record.get("decimalLongitude"),
                            "obis_id": record.get("id"),
                            "kingdom": record.get("kingdom"),
                        }

                offset += len(data["results"])
                all_records.extend(data["results"])
                time.sleep(0.5)  # Rate limiting

        except requests.exceptions.RequestException as e:
            logger.error(f"OBIS API error: {e}")

        logger.info(f"Found {len(unique_species)} unique deep-sea species from {len(all_records)} occurrences")
        return list(unique_species.values())

    # ========================================================================
    # NCBI ENTREZ INTEGRATION
    # ========================================================================

    def fetch_ncbi_sequence(
        self,
        species_name: str,
        marker_genes: Optional[list[str]] = None,
        retmax: int = 5
    ) -> Optional[tuple[str, str, str]]:
        """Fetch COI or 18S sequence from NCBI for given species.

        Args:
            species_name: Species name (e.g., "Homo sapiens")
            marker_genes: List of marker genes to search (default: ["COI", "18S"])
            retmax: Maximum number of sequences to fetch per gene

        Returns:
            Tuple of (sequence, marker_gene, accession_id) or None if not found
        """
        if marker_genes is None:
            marker_genes = ["COI", "18S"]

        for marker in marker_genes:
            try:
                # Build Entrez query
                query = f'"{species_name}"[ORGN] AND {marker}[GENE]'
                logger.debug(f"Searching NCBI: {query}")

                # Search
                search_handle = Entrez.esearch(
                    db="nucleotide",
                    term=query,
                    retmax=retmax,
                    timeout=NCBI_TIMEOUT,
                )
                search_results = Entrez.read(search_handle)
                search_handle.close()

                id_list: list[str] = []
                if isinstance(search_results, dict):
                    raw_list = search_results.get("IdList", [])
                    if isinstance(raw_list, list):
                        id_list = [str(v) for v in raw_list]

                if not id_list:
                    logger.debug(f"No {marker} sequences found for {species_name}")
                    continue

                # Fetch first hit
                seq_id = id_list[0]
                fetch_handle = Entrez.efetch(
                    db="nucleotide",
                    id=seq_id,
                    rettype="fasta",
                    retmode="text",
                    timeout=NCBI_TIMEOUT,
                )
                record = SeqIO.read(fetch_handle, "fasta")
                fetch_handle.close()

                sequence = str(record.seq).upper()
                
                # Validate sequence length
                if not (FASTA_MIN_LENGTH <= len(sequence) <= FASTA_MAX_LENGTH):
                    logger.warning(
                        f"Sequence length {len(sequence)} outside bounds "
                        f"[{FASTA_MIN_LENGTH}-{FASTA_MAX_LENGTH}]"
                    )
                    continue

                logger.info(f"✓ Found {marker} for {species_name} ({len(sequence)}bp, ID: {seq_id})")
                self.sequences_fetched += 1
                time.sleep(0.3)  # NCBI rate limiting
                
                return (sequence, marker, str(seq_id))

            except Exception as e:
                logger.warning(f"Error fetching {marker} for {species_name}: {e}")
                time.sleep(1)  # Backoff on error
                continue

        logger.warning(f"✗ No sequences found for {species_name}")
        return None

    # ========================================================================
    # TAXONKIT INTEGRATION
    # ========================================================================

    def normalize_taxonomy_taxonkit(self, species_name: str, taxon_id: Optional[int] = None) -> Optional[str]:
        """Standardize taxonomy using TaxonKit (subprocess).

        Calls `taxonkit name2taxid` and `taxonkit reformat` to get 7-level lineage.

        Args:
            species_name: Scientific name
            taxon_id: Optional NCBI taxon ID (faster if provided)

        Returns:
            Standardized lineage string (semicolon-separated) or None
        """
        try:
            if taxon_id:
                # Direct approach: taxonkit reformat
                cmd_reformat = f'echo {taxon_id} | taxonkit reformat -F 0 -s "/" -t'
            else:
                # Two-step: name2taxid, then reformat
                cmd_name2taxid = f'echo "{species_name}" | taxonkit name2taxid'
                result = subprocess.run(
                    cmd_name2taxid,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                
                if result.returncode != 0 or not result.stdout.strip():
                    logger.warning(f"TaxonKit name2taxid failed for {species_name}")
                    return None

                taxon_id_raw = result.stdout.split("\t")[1].strip()
                taxon_id = int(taxon_id_raw) if taxon_id_raw.isdigit() else None
                cmd_reformat = f'echo {taxon_id} | taxonkit reformat -F 0 -s ";" -t'

            # Run reformat command
            result = subprocess.run(
                cmd_reformat,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.warning(f"TaxonKit reformat failed for {species_name} (ID: {taxon_id})")
                return None

            # Parse output: "taxid\tlineage\tname"
            parts = result.stdout.strip().split("\t")
            if len(parts) >= 2:
                lineage = parts[1]
                logger.debug(f"Normalized {species_name}: {lineage}")
                return lineage

            return None

        except subprocess.TimeoutExpired:
            logger.error(f"TaxonKit timeout for {species_name}")
            return None
        except Exception as e:
            logger.warning(f"TaxonKit error for {species_name}: {e}")
            return None

    def parse_taxonkit_lineage(self, lineage_str: str) -> TaxonomicLineage:
        """Parse TaxonKit lineage string into TaxonomicLineage object.

        Expected format: "kingdom;phylum;class;order;family;genus;species"

        Args:
            lineage_str: Semicolon-separated lineage string

        Returns:
            TaxonomicLineage object
        """
        parts = lineage_str.split(";")
        # Pad with None if fewer than 7 levels
        parts += [None] * (7 - len(parts))
        
        return TaxonomicLineage.model_validate(
            {
                "sequence_id": "",
                "kingdom": parts[0] if parts[0] else None,
                "phylum": parts[1] if parts[1] else None,
                "class": parts[2] if parts[2] else None,
                "order": parts[3] if parts[3] else None,
                "family": parts[4] if parts[4] else None,
                "genus": parts[5] if parts[5] else None,
                "species": parts[6] if parts[6] else None,
            }
        )

    # ========================================================================
    # LANCEDB INITIALIZATION & STORAGE
    # ========================================================================

    def initialize_lancedb_schema(self) -> None:
        """Create LanceDB tables with proper schema for embeddings and metadata."""
        logger.info("Initializing LanceDB schema...")

        # Sample data to infer schema
        sample_record = {
            "sequence_id": "OBIS_COI_001",
            "vector": [0.0] * 768,  # 768-dim embedding (from NT-2.5B)
            "dna_sequence": "ATGCATGC",
            "taxonomy": "Eukaryota;Animalia;Chordata;Actinopterygii;Perciformes;Serranidae;Serranus",
            "obis_id": "OBIS_001",
            "marker_gene": "COI",
            "depth": 2500.0,
            "latitude": -10.5,
            "longitude": 70.2,
            "species_name": "Serranus cabrilla",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Create or overwrite embeddings table
        try:
            try:
                self.db.drop_table(LANCEDB_TABLE_SEQUENCES)
            except Exception:
                pass
            table = self.db.create_table(LANCEDB_TABLE_SEQUENCES, data=[sample_record])
            logger.info(f"✓ Created LanceDB table: {LANCEDB_TABLE_SEQUENCES}")
        except Exception as e:
            logger.error(f"Failed to create LanceDB table: {e}")
            raise

    def store_sequences(self, records: list[dict]) -> int:
        """Store sequence records in LanceDB.

        Args:
            records: List of record dictionaries with embedding, sequence, metadata

        Returns:
            Number of records successfully stored
        """
        if not records:
            logger.warning("No records to store")
            return 0

        try:
            table = self.db.open_table(LANCEDB_TABLE_SEQUENCES)
            table.add(records)
            self.sequences_stored += len(records)
            logger.info(f"✓ Stored {len(records)} records in LanceDB")
            return len(records)
        except Exception as e:
            logger.error(f"Failed to store sequences: {e}")
            return 0

    # ========================================================================
    # END-TO-END WORKFLOW
    # ========================================================================

    def run_full_pipeline(
        self,
        min_depth: int = 1000,
        max_species: int = 50,
        skip_embedding: bool = True,  # Skip embeddings (to be done by embedder.py)
    ) -> dict:
        """Execute complete data ingestion pipeline.

        Args:
            min_depth: Minimum depth threshold for deep-sea species
            max_species: Maximum species to process
            skip_embedding: If True, skip embedding generation (store placeholder vectors)

        Returns:
            Dictionary with pipeline statistics
        """
        logger.info("=" * 70)
        logger.info("GLOBAL-BIOSCAN: DEEP-SEA DATA INGESTION PIPELINE")
        logger.info("=" * 70)

        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "species_fetched": 0,
            "species_with_sequences": 0,
            "sequences_fetched": self.sequences_fetched,
            "sequences_stored": self.sequences_stored,
            "errors": [],
        }

        try:
            # Step 1: Initialize LanceDB
            logger.info("\n[STEP 1] Initializing LanceDB schema...")
            self.initialize_lancedb_schema()

            # Step 2: Fetch OBIS species
            logger.info("\n[STEP 2] Fetching deep-sea species from OBIS...")
            species_list = self.fetch_deep_sea_species(
                min_depth=min_depth,
                limit=max_species,
                geometry=DEEP_SEA_HOTSPOT["geometry"],
            )
            stats["species_fetched"] = len(species_list)
            logger.info(f"Found {len(species_list)} unique species")

            # Step 3: Process each species
            logger.info(f"\n[STEP 3] Fetching sequences and normalizing taxonomy...")
            records_to_store = []

            for i, species_record in enumerate(species_list[:max_species], 1):
                species_name = species_record["scientific_name"]
                logger.info(f"\n[{i}/{len(species_list[:max_species])}] Processing: {species_name}")

                # Fetch sequence from NCBI
                seq_data = self.fetch_ncbi_sequence(species_name)
                if not seq_data:
                    stats["errors"].append(f"No sequence found: {species_name}")
                    continue

                sequence, marker_gene, accession_id = seq_data

                # Normalize taxonomy with TaxonKit
                taxon_id_value = species_record.get("taxon_id")
                if isinstance(taxon_id_value, str) and taxon_id_value.isdigit():
                    taxon_id_value = int(taxon_id_value)
                elif not isinstance(taxon_id_value, int):
                    taxon_id_value = None

                lineage_str = self.normalize_taxonomy_taxonkit(
                    species_name,
                    taxon_id_value,
                )
                
                if not lineage_str:
                    stats["errors"].append(f"TaxonKit failed: {species_name}")
                    continue

                # Create record
                sequence_id = f"OBIS_{marker_gene}_{accession_id[-6:]}"
                
                record = {
                    "sequence_id": sequence_id,
                    "vector": [0.0] * 768,  # Placeholder
                    "dna_sequence": sequence,
                    "taxonomy": lineage_str,
                    "obis_id": species_record.get("obis_id", ""),
                    "marker_gene": marker_gene,
                    "depth": species_record.get("depth", 0.0),
                    "latitude": species_record.get("latitude", 0.0),
                    "longitude": species_record.get("longitude", 0.0),
                    "species_name": species_name,
                    "accession_id": accession_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                records_to_store.append(record)
                stats["species_with_sequences"] += 1

                if len(records_to_store) >= 10:  # Batch store
                    self.store_sequences(records_to_store)
                    records_to_store = []

            # Final batch store
            if records_to_store:
                self.store_sequences(records_to_store)

            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Total species fetched: {stats['species_fetched']}")
            logger.info(f"Species with sequences: {stats['species_with_sequences']}")
            logger.info(f"Total sequences stored: {self.sequences_stored}")
            logger.info(f"Errors encountered: {len(stats['errors'])}")
            if stats['errors']:
                for error in stats['errors'][:5]:  # Show first 5 errors
                    logger.warning(f"  - {error}")

            stats["sequences_fetched"] = self.sequences_fetched
            stats["sequences_stored"] = self.sequences_stored

            return stats

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            stats["errors"].append(str(e))
            raise

    def ingest_fasta(self, fasta_path: Path, marker_gene: str) -> SequenceBatch:
        """Parse FASTA file and create sequence batch.

        Args:
            fasta_path: Path to FASTA file
            marker_gene: Marker gene type (COI, 18S, 16S)

        Returns:
            SequenceBatch with parsed sequences
        """
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        sequences = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_str = str(record.seq).upper()
            if not (FASTA_MIN_LENGTH <= len(seq_str) <= FASTA_MAX_LENGTH):
                continue

            if marker_gene not in {"COI", "18S", "16S"}:
                raise ValueError(f"Unsupported marker gene: {marker_gene}")

            marker_gene_literal = cast(Literal["COI", "18S", "16S"], marker_gene)
            seq_obj = DNASequence(
                sequence_id=record.id,
                sequence=seq_str,
                marker_gene=marker_gene_literal,
                species=None,
                latitude=None,
                longitude=None,
                depth_m=None,
                source="Custom",
                length_bp=len(seq_str),
            )
            sequences.append(seq_obj)

        logger.info(f"Parsed {len(sequences)} sequences from {fasta_path}")
        return SequenceBatch(
            batch_id=f"BATCH_{fasta_path.stem}",
            sequences=sequences,
            timestamp=datetime.utcnow().isoformat(),
            processing_stage="raw",
        )

    def list_batches(self) -> list[str]:
        """List all available batches in LanceDB.

        Returns:
            List of batch IDs
        """
        try:
            table = self.db.open_table(LANCEDB_TABLE_SEQUENCES)
            results = table.search().limit(1000).to_list()
            batch_ids = {str(r.get("batch_id")) for r in results if r.get("batch_id")}
            return sorted(batch_ids)
        except Exception as e:
            logger.warning(f"Failed to list batches: {e}")
            return []
