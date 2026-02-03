"""Deep-Sea eDNA Reference Atlas Seeding Utility.

Populates LanceDB on USB drive with curated deep-sea taxonomic data from:
- OBIS (Ocean Biodiversity Information System): depth > 1000m occurrences
- NCBI GenBank: COI/18S mitochondrial and nuclear markers
- TaxonKit: Clinical-grade 7-level taxonomic standardization

Target: 2,000-5,000 unique species from Central Indian Ridge and Abyssal Plains.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import hashlib
import numpy as np
import pandas as pd

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARN] psutil not installed. Install with: pip install psutil")

# OBIS API client
try:
    import pyobis  # type: ignore
    PYOBIS_AVAILABLE = True
except ImportError:
    PYOBIS_AVAILABLE = False
    print("[WARN] pyobis not installed. Install with: pip install pyobis")

# NCBI Entrez API
try:
    from Bio import Entrez, SeqIO
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("[WARN] Biopython not installed. Install with: pip install biopython")

# Project imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.edge.database import BioDB, DriveNotMountedError, DatabaseIntegrityError
from src.config import LANCEDB_TABLE_SEQUENCES

# ============================================================================
# CONFIGURATION
# ============================================================================

# OBIS Query Parameters
OBIS_DEPTH_MIN = 200  # meters (get more data, especially deep-sea fauna)
OBIS_DEPTH_MAX = 6000  # meters (avoid hadal zone for initial seed)
OBIS_BBOX = {
    "westlng": 60.0,   # Central Indian Ridge
    "eastlng": 75.0,
    "southlat": -20.0,  # Abyssal Plains
    "northlat": 0.0
}
OBIS_MAX_RECORDS = 5000  # API limit per request

# NCBI Configuration
NCBI_EMAIL = "faisaltabrez01@gmail.com"  # Required by NCBI API
NCBI_API_KEY = None  # Optional: Get from NCBI account for higher rate limits
NCBI_RATE_LIMIT = 0.5  # delay in seconds between requests (to avoid IP blocking)
NCBI_MARKERS = ["COI", "COX1", "18S", "18S rRNA"]  # Priority markers

# TaxonKit Configuration
TAXONKIT_EXE = r"C:\taxonkit\taxonkit.exe"  # Executable path
TAXONKIT_DATA = r"C:\taxonkit\data"  # Data directory (new requirement)
TAXONKIT_FORMAT = "{K};{p};{c};{o};{f};{g};{s}"  # K=Kingdom (Metazoa), p=Phylum, c=Class, o=Order, f=Family, g=Genus, s=Species

# Target Data Volume
TARGET_MIN_SPECIES = 2000
TARGET_MAX_SPECIES = 5000

# USB Drive Configuration
DRIVE_LETTER = "E"  # USB drive letter
CHECKPOINT_FILE = "checkpoint.json"
MANIFEST_FILE = "seeding_manifest.json"

# Logging
LOG_LEVEL = logging.INFO

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('seed_atlas.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manages resume capability for interrupted seeding operations."""
    
    def __init__(self, bio_db: BioDB):
        self.bio_db = bio_db
        self.checkpoint_path = bio_db.db_root / CHECKPOINT_FILE
        self.checkpoint_data = self._load()
    
    def _load(self) -> Dict:
        """Load existing checkpoint or create new."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"[RESUME] Loaded checkpoint: {data['processed_species']} species processed")
                return data
            except Exception as e:
                logger.warning(f"[WARN] Could not load checkpoint: {e}. Starting fresh.")
        
        return {
            "processed_species": 0,
            "successful_species": 0,
            "failed_species": 0,
            "processed_names": [],
            "failed_names": [],
            "last_update": None,
            "total_sequences": 0
        }
    
    def save(self):
        """Persist checkpoint to USB drive."""
        try:
            self.checkpoint_data["last_update"] = datetime.now().isoformat()
            with open(self.checkpoint_path, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
            logger.debug(f"[CHECKPOINT] Saved: {self.checkpoint_data['processed_species']} species")
        except Exception as e:
            logger.error(f"[FAIL] Could not save checkpoint: {e}")
    
    def is_processed(self, species_name: str) -> bool:
        """Check if species already processed."""
        return species_name in self.checkpoint_data["processed_names"]
    
    def mark_success(self, species_name: str, seq_count: int = 1):
        """Mark species as successfully processed."""
        if species_name not in self.checkpoint_data["processed_names"]:
            self.checkpoint_data["processed_names"].append(species_name)
            self.checkpoint_data["processed_species"] += 1
            self.checkpoint_data["successful_species"] += 1
            self.checkpoint_data["total_sequences"] += seq_count
    
    def mark_failure(self, species_name: str):
        """Mark species as failed."""
        if species_name not in self.checkpoint_data["processed_names"]:
            self.checkpoint_data["processed_names"].append(species_name)
            self.checkpoint_data["failed_names"].append(species_name)
            self.checkpoint_data["processed_species"] += 1
            self.checkpoint_data["failed_species"] += 1
    
    def get_stats(self) -> Dict:
        """Get checkpoint statistics."""
        return {
            "processed": self.checkpoint_data["processed_species"],
            "successful": self.checkpoint_data["successful_species"],
            "failed": self.checkpoint_data["failed_species"],
            "sequences": self.checkpoint_data["total_sequences"]
        }


# ============================================================================
# OBIS DATA FETCHER
# ============================================================================

class OBISFetcher:
    """Fetch deep-sea species occurrences from OBIS API."""
    
    def __init__(self):
        if not PYOBIS_AVAILABLE:
            raise ImportError("pyobis package required. Install with: pip install pyobis")
    
    def fetch_deep_sea_species(
        self,
        depth_min: int = OBIS_DEPTH_MIN,
        depth_max: int = OBIS_DEPTH_MAX,
        bbox: Dict = OBIS_BBOX,
        max_records: int = OBIS_MAX_RECORDS
    ) -> List[str]:
        """Fetch unique species names from OBIS deep-sea occurrences.
        
        Args:
            depth_min: Minimum depth in meters (default 1000m)
            depth_max: Maximum depth in meters (default 6000m)
            bbox: Bounding box dict with westlng, eastlng, southlat, northlat
            max_records: Maximum records to fetch
        
        Returns:
            List of unique species names (scientificName field)
        """
        logger.info(f"[OBIS] Fetching deep-sea species (depth: {depth_min}-{depth_max}m)")
        logger.info(f"[OBIS] Region: Central Indian Ridge {bbox}")
        
        try:
            # Query OBIS occurrences
            logger.info(f"[OBIS] Fetching deep-sea species (depth: {depth_min}-{depth_max}m)")
            logger.info(f"[OBIS] Region: Central Indian Ridge {bbox}")
            
            # Use depthmin/depthmax parameters instead of geometry filter
            results = pyobis.occurrences.search(
                depthmin=depth_min,
                depthmax=depth_max,
                size=max_records
            )
            
            # Execute the query if needed
            if results.data is None:
                results.execute()
            
            # Convert to DataFrame
            try:
                df = results.to_pandas()
                if df is None or len(df) == 0:
                    logger.warning("[WARN] No OBIS results returned")
                    return []
                logger.info(f"[OBIS] Retrieved {len(df)} raw occurrences")
            except Exception as e:
                logger.warning(f"[WARN] Error parsing OBIS response: {e}")
                return []
            
            # Filter by depth if available
            if 'depth' in df.columns:
                df = df[(df['depth'] >= depth_min) & (df['depth'] <= depth_max)]
                logger.info(f"[OBIS] After depth filter: {len(df)} occurrences")
            
            # Extract unique species names (don't filter by taxonomicStatus - get all records)
            if 'scientificName' not in df.columns:
                logger.error("[FAIL] scientificName field not in OBIS response")
                return []
            
            species_list = df['scientificName'].dropna().unique().tolist()
            
            # Clean species names - keep full names, just remove blank/null values
            cleaned_species = []
            for sp in species_list:
                sp = sp.strip()
                # Keep species with at least one word (genus)
                if sp and len(sp) > 2:
                    cleaned_species.append(sp)
            
            unique_species = list(set(cleaned_species))
            logger.info(f"[PASS] Extracted {len(unique_species)} unique species names")
            
            return unique_species[:TARGET_MAX_SPECIES]  # Cap at target
        
        except Exception as e:
            logger.error(f"[FAIL] OBIS query failed: {e}")
            return []


# ============================================================================
# NCBI SEQUENCE FETCHER
# ============================================================================

class NCBIFetcher:
    """Fetch genetic sequences from NCBI GenBank."""
    
    def __init__(self, email: str, api_key: Optional[str] = None):
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("Biopython required. Install with: pip install biopython")
        
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        
        self.rate_limit_delay = NCBI_RATE_LIMIT  # 0.5 seconds between requests
        self.last_request_time = 0
    
    def _rate_limit_wait(self):
        """Enforce NCBI rate limiting (0.5s delay between requests to avoid IP blocking)."""
        elapsed = time.time() - self.last_request_time
        wait_time = self.rate_limit_delay - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def fetch_sequences(
        self,
        species_name: str,
        markers: List[str] = NCBI_MARKERS,
        max_sequences: int = 3
    ) -> List[Dict]:
        """Fetch genetic sequences for a species.
        
        Args:
            species_name: Scientific name (e.g., "Bathymodiolus azoricus")
            markers: List of marker genes to search for
            max_sequences: Maximum sequences to fetch per species
        
        Returns:
            List of sequence dicts with keys: accession, sequence, marker, organism
        """
        sequences = []
        
        for marker in markers:
            try:
                self._rate_limit_wait()
                
                # Search query
                query = f'{species_name}[Organism] AND ({marker}[Gene] OR {marker}[Title])'
                
                # ESearch to get IDs
                search_handle = Entrez.esearch(
                    db="nucleotide",
                    term=query,
                    retmax=max_sequences
                )
                search_results = Entrez.read(search_handle)  # type: ignore
                search_handle.close()
                
                id_list = search_results.get("IdList", [])  # type: ignore
                
                if not id_list:
                    continue
                
                logger.debug(f"[NCBI] Found {len(id_list)} {marker} sequences for {species_name}")
                
                # EFetch to get sequences
                self._rate_limit_wait()
                fetch_handle = Entrez.efetch(
                    db="nucleotide",
                    id=id_list,
                    rettype="fasta",
                    retmode="text"
                )
                
                # Parse FASTA
                for record in SeqIO.parse(fetch_handle, "fasta"):
                    sequences.append({
                        "accession": record.id,
                        "sequence": str(record.seq),
                        "marker": marker,
                        "organism": species_name,
                        "length": len(record.seq),
                        "description": record.description
                    })
                    
                    if len(sequences) >= max_sequences:
                        break
                
                fetch_handle.close()
                
                # If we found sequences for this marker, stop searching
                if sequences:
                    break
            
            except Exception as e:
                logger.warning(f"[WARN] NCBI fetch failed for {species_name} ({marker}): {e}")
                continue
        
        return sequences


# ============================================================================
# TAXONKIT STANDARDIZER
# ============================================================================

class TaxonKitStandardizer:
    """Standardize taxonomic names using TaxonKit."""
    
    def __init__(self, taxonkit_exe: str = TAXONKIT_EXE, taxonkit_data: str = TAXONKIT_DATA):
        self.taxonkit_exe = taxonkit_exe
        self.taxonkit_data = taxonkit_data
        self._verify_installation()
    
    def _verify_installation(self):
        """Verify TaxonKit is installed and database available."""
        try:
            result = subprocess.run(
                [self.taxonkit_exe, "--data-dir", self.taxonkit_data, "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"[PASS] TaxonKit found: {result.stdout.strip()}")
            else:
                logger.warning("[WARN] TaxonKit not responding correctly")
        except FileNotFoundError:
            logger.error(f"[FAIL] TaxonKit not found at: {self.taxonkit_exe}")
            raise
        except Exception as e:
            logger.error(f"[FAIL] TaxonKit verification failed: {e}")
            raise
        
        # Check database
        data_path = Path(self.taxonkit_data)
        if not data_path.exists():
            logger.warning(f"[WARN] TaxonKit data directory not found at: {self.taxonkit_data}")
            logger.info("[INFO] Download with: mkdir C:\\taxonkit\\data && cd C:\\taxonkit\\data && wget ftp://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz && tar -xzf taxdump.tar.gz")
    
    def standardize_lineage(self, species_name: str) -> Optional[Dict[str, str]]:
        """Get 7-level taxonomic lineage from TaxonKit.
        
        Args:
            species_name: Scientific name (e.g., "Bathymodiolus azoricus")
        
        Returns:
            Dict with keys: kingdom, phylum, class, order, family, genus, species
            or None if name not found
        """
        try:
            # Step 1: name2taxid
            name2taxid_result = subprocess.run(
                [self.taxonkit_exe, "name2taxid", "--data-dir", self.taxonkit_data],
                input=species_name,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if name2taxid_result.returncode != 0:
                logger.debug(f"[WARN] name2taxid failed for: {species_name}")
                return None
            
            # Parse output: "Name\tTaxID"
            lines = name2taxid_result.stdout.strip().split('\n')
            if not lines or '\t' not in lines[0]:
                return None
            
            taxid = lines[0].split('\t')[1].strip()
            logger.debug(f"[TAXONKIT] Got TaxID {taxid} for {species_name}")
            
            # Step 2: lineage to get full taxonomic path
            lineage_result = subprocess.run(
                [self.taxonkit_exe, "lineage", "--data-dir", self.taxonkit_data],
                input=taxid,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if lineage_result.returncode != 0:
                logger.debug(f"[WARN] lineage failed for TaxID: {taxid}")
                return None
            
            # Step 3: reformat to get structured output with {K} = Kingdom (Animalia, not Eukaryota)
            reformat_result = subprocess.run(
                [
                    self.taxonkit_exe, "reformat",
                    "--data-dir", self.taxonkit_data,
                    "--format", "{K};{p};{c};{o};{f};{g};{s}"  # K=kingdom, p=phylum, c=class, o=order, f=family, g=genus, s=species
                ],
                input=lineage_result.stdout,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if reformat_result.returncode != 0:
                logger.debug(f"[WARN] reformat failed for TaxID: {taxid}")
                return None
            
            # Parse output: "TaxID\tRawLineage\tFormattedOutput"
            reformat_lines = reformat_result.stdout.strip().split('\n')
            if not reformat_lines:
                return None
            
            parts = reformat_lines[0].split('\t')
            if len(parts) < 3:
                logger.debug(f"[DEBUG] Expected 3+ fields, got {len(parts)}: {parts}")
                return None
            
            formatted_output = parts[2].strip()  # Get the formatted ranks
            logger.debug(f"[TAXONKIT] Formatted: {formatted_output}")
            ranks = formatted_output.split(';')
            
            if len(ranks) < 7:
                # Pad with empty strings
                ranks.extend([''] * (7 - len(ranks)))
            
            lineage_dict = {
                "kingdom": ranks[0].strip() or "Unknown",
                "phylum": ranks[1].strip() or "Unknown",
                "class": ranks[2].strip() or "Unknown",
                "order": ranks[3].strip() or "Unknown",
                "family": ranks[4].strip() or "Unknown",
                "genus": ranks[5].strip() or "Unknown",
                "species": ranks[6].strip() or species_name
            }
            
            logger.debug(f"[TAXONKIT] Lineage: {' > '.join([lineage_dict[k] for k in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']])}")
            return lineage_dict
        
        except Exception as e:
            logger.warning(f"[WARN] TaxonKit standardization failed for {species_name}: {e}")
            return None


# ============================================================================
# LANCEDB INGESTION
# ============================================================================

class LanceDBIngester:
    """Ingest sequences into LanceDB on USB drive."""
    
    def __init__(self, bio_db: BioDB, table_name: str = LANCEDB_TABLE_SEQUENCES):
        self.bio_db = bio_db
        self.table_name = table_name
        self.records = []
    
    def add_record(
        self,
        species_name: str,
        sequence: str,
        accession: str,
        marker: str,
        lineage: Dict[str, str],
        embedding: Optional[np.ndarray] = None
    ):
        """Add a record to the ingestion buffer.
        
        Args:
            species_name: Scientific name
            sequence: DNA sequence string
            accession: GenBank accession number
            marker: Gene marker (COI, 18S, etc.)
            lineage: 7-level taxonomic dict
            embedding: Optional 768-dim embedding vector
        """
        # Generate mock embedding if not provided
        if embedding is None:
            # Use sequence hash for deterministic mock embedding
            seed = int(hashlib.md5(sequence.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.randn(768).astype(np.float32)
        
        record = {
            "species": species_name,
            "sequence": sequence,
            "accession": accession,
            "marker": marker,
            "kingdom": lineage.get("kingdom", "Unknown"),
            "phylum": lineage.get("phylum", "Unknown"),
            "class": lineage.get("class", "Unknown"),
            "order": lineage.get("order", "Unknown"),
            "family": lineage.get("family", "Unknown"),
            "genus": lineage.get("genus", "Unknown"),
            "taxonomy": f"{lineage.get('kingdom', 'Unknown')};{lineage.get('phylum', 'Unknown')};{lineage.get('class', 'Unknown')};{lineage.get('order', 'Unknown')};{lineage.get('family', 'Unknown')};{lineage.get('genus', 'Unknown')};{species_name}",
            "embedding": embedding.tolist(),
            "sequence_length": len(sequence),
            "timestamp": datetime.now().isoformat()
        }
        
        self.records.append(record)
    
    def flush(self, batch_size: int = 100):
        """Write buffered records to LanceDB.
        
        Args:
            batch_size: Number of records to write at once
        """
        if not self.records:
            logger.debug("[INFO] No records to flush")
            return
        
        try:
            table = self.bio_db.get_table(self.table_name)
            
            if table is None:
                # Create new table
                logger.info(f"[INFO] Creating new table: {self.table_name}")
                db = self.bio_db.connect()
                if db is None:
                    raise DatabaseIntegrityError("Could not connect to LanceDB")
                
                # Create with first batch
                import pyarrow as pa
                df = pd.DataFrame(self.records[:batch_size])
                table = db.create_table(self.table_name, df)
                logger.info(f"[PASS] Created table with {len(df)} records")
                
                # Add remaining records
                if len(self.records) > batch_size:
                    remaining_df = pd.DataFrame(self.records[batch_size:])
                    table.add(remaining_df)
                    logger.info(f"[PASS] Added {len(remaining_df)} more records")
            else:
                # Add to existing table
                df = pd.DataFrame(self.records)
                table.add(df)
                logger.info(f"[PASS] Added {len(df)} records to existing table")
            
            # Clear buffer
            count = len(self.records)
            self.records = []
            
            return count
        
        except Exception as e:
            logger.error(f"[FAIL] LanceDB flush failed: {e}")
            raise
    
    def get_record_count(self) -> int:
        """Get total record count in table."""
        try:
            stats = self.bio_db.get_table_stats(self.table_name)
            return stats.get("row_count", 0)
        except:
            return 0


# ============================================================================
# MAIN SEEDING ORCHESTRATOR
# ============================================================================

class AtlasSeeder:
    """Main orchestrator for seeding the reference atlas."""
    
    def __init__(self, drive_letter: str = DRIVE_LETTER):
        logger.info("[INIT] Initializing Atlas Seeder")
        
        # Initialize BioDB
        self.bio_db = BioDB(drive_letter=drive_letter, enable_auto_init=True)
        
        # Verify USB drive
        is_mounted, msg = self.bio_db.detect_drive()
        if not is_mounted:
            raise DriveNotMountedError(f"USB drive not detected: {msg}")
        logger.info(f"[PASS] {msg}")
        
        # Initialize components
        self.checkpoint = CheckpointManager(self.bio_db)
        self.obis = OBISFetcher()
        self.ncbi = NCBIFetcher(email=NCBI_EMAIL, api_key=NCBI_API_KEY)
        self.taxonkit = TaxonKitStandardizer(taxonkit_exe=TAXONKIT_EXE, taxonkit_data=TAXONKIT_DATA)
        self.ingester = LanceDBIngester(self.bio_db)
        
        # Statistics
        self.stats = {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "obis_species_fetched": 0,
            "sequences_found": 0,
            "sequences_ingested": 0,
            "species_processed": 0,
            "species_successful": 0,
            "species_failed": 0,
            "errors": []
        }
    
    def run(self):
        """Execute the full seeding pipeline."""
        try:
            logger.info("=" * 80)
            logger.info("DEEP-SEA EDNA REFERENCE ATLAS SEEDING")
            logger.info("=" * 80)
            
            # Clean environment: Delete existing database folder for fresh start
            logger.info("\n[CLEANUP] Preparing fresh database environment...")
            db_path = self.bio_db.db_root
            if db_path.exists():
                try:
                    logger.info(f"[CLEANUP] Removing existing database at: {db_path}")
                    shutil.rmtree(db_path)
                    logger.info("[PASS] Database folder removed")
                except Exception as e:
                    logger.error(f"[WARN] Could not remove existing database: {e}")
            
            # Create fresh database directory
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[PASS] Fresh database directory created: {db_path}")
            
            # Pre-flight validation: Test TaxonKit with known species
            logger.info("\n[PREFLIGHT] Running TaxonKit sanity check...")
            test_lineage = self.taxonkit.standardize_lineage("Homo sapiens")
            
            if test_lineage and test_lineage.get("species") == "Homo sapiens":
                logger.info("[PASS] TaxonKit sanity check: Homo sapiens resolved correctly")
                logger.info(f"[TAXONKIT] Lineage: {test_lineage['kingdom']} > {test_lineage['phylum']} > {test_lineage['class']} > {test_lineage['order']} > {test_lineage['family']} > {test_lineage['genus']} > {test_lineage['species']}")
            else:
                logger.error("[FAIL] TaxonKit sanity check failed: Could not resolve Homo sapiens")
                logger.error("[FAIL] Aborting seeding - TaxonKit configuration issue")
                raise RuntimeError("TaxonKit pre-flight validation failed")
            
            # Step 1: Fetch species list from OBIS
            logger.info("\n[STEP 1] Fetching species list from OBIS...")
            species_list = self.obis.fetch_deep_sea_species()
            
            if not species_list:
                logger.error("[FAIL] No species retrieved from OBIS. Aborting.")
                return
            
            self.stats["obis_species_fetched"] = len(species_list)
            logger.info(f"[PASS] Retrieved {len(species_list)} unique species")
            
            # Step 2: Process each species
            logger.info("\n[STEP 2] Fetching sequences and standardizing taxonomy...")
            logger.info(f"[INFO] Target: {TARGET_MIN_SPECIES}-{TARGET_MAX_SPECIES} species")
            
            for idx, species in enumerate(species_list, 1):
                # Check if already processed
                if self.checkpoint.is_processed(species):
                    logger.debug(f"[SKIP] Already processed: {species}")
                    continue
                
                logger.info(f"\n[{idx}/{len(species_list)}] Processing: {species}")
                
                try:
                    # Fetch sequences from NCBI
                    sequences = self.ncbi.fetch_sequences(species)
                    
                    if not sequences:
                        logger.warning(f"[WARN] No sequences found for: {species}")
                        self.checkpoint.mark_failure(species)
                        self.stats["species_failed"] += 1
                        continue
                    
                    logger.info(f"[NCBI] Found {len(sequences)} sequences")
                    self.stats["sequences_found"] += len(sequences)
                    
                    # Standardize taxonomy with TaxonKit
                    lineage = self.taxonkit.standardize_lineage(species)
                    
                    if not lineage:
                        logger.warning(f"[WARN] TaxonKit standardization failed: {species}")
                        # Use fallback lineage
                        lineage = {
                            "kingdom": "Unknown",
                            "phylum": "Unknown",
                            "class": "Unknown",
                            "order": "Unknown",
                            "family": "Unknown",
                            "genus": species.split()[0] if ' ' in species else "Unknown",
                            "species": species
                        }
                    
                    logger.info(f"[TAXONOMY] {lineage['kingdom']} > {lineage['phylum']} > {lineage['class']} > {lineage['order']} > {lineage['family']} > {lineage['genus']}")
                    
                    # Add to ingestion buffer
                    for seq_data in sequences:
                        self.ingester.add_record(
                            species_name=species,
                            sequence=seq_data["sequence"],
                            accession=seq_data["accession"],
                            marker=seq_data["marker"],
                            lineage=lineage
                        )
                    
                    self.stats["sequences_ingested"] += len(sequences)
                    self.checkpoint.mark_success(species, len(sequences))
                    self.stats["species_successful"] += 1
                    
                    # Flush every 50 species
                    if idx % 50 == 0:
                        logger.info(f"\n[FLUSH] Writing batch to LanceDB...")
                        self.ingester.flush()
                        self.checkpoint.save()
                        logger.info(f"[CHECKPOINT] Progress: {self.checkpoint.get_stats()}")
                
                except Exception as e:
                    logger.error(f"[FAIL] Error processing {species}: {e}")
                    self.checkpoint.mark_failure(species)
                    self.stats["species_failed"] += 1
                    self.stats["errors"].append({
                        "species": species,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    continue
            
            # Final flush
            logger.info("\n[STEP 3] Final flush to LanceDB...")
            self.ingester.flush()
            self.checkpoint.save()
            
            # Step 4: Build IVF-PQ Index
            logger.info("\n[STEP 4] Building IVF-PQ index for USB optimization...")
            success, msg = self.bio_db.build_ivf_pq_index()
            
            if success:
                logger.info(f"[PASS] {msg}")
            else:
                logger.error(f"[FAIL] {msg}")
            
            # Step 5: Generate manifest
            logger.info("\n[STEP 5] Generating seeding manifest...")
            self._generate_manifest()
            
            # Final statistics
            self.stats["end_time"] = datetime.now().isoformat()
            self.stats["species_processed"] = self.checkpoint.get_stats()["processed"]
            
            logger.info("\n" + "=" * 80)
            logger.info("SEEDING COMPLETE")
            logger.info("=" * 80)
            logger.info(f"OBIS Species Fetched:    {self.stats['obis_species_fetched']}")
            logger.info(f"Species Processed:       {self.stats['species_processed']}")
            logger.info(f"Species Successful:      {self.stats['species_successful']}")
            logger.info(f"Species Failed:          {self.stats['species_failed']}")
            logger.info(f"Sequences Found:         {self.stats['sequences_found']}")
            logger.info(f"Sequences Ingested:      {self.stats['sequences_ingested']}")
            logger.info(f"Total Records in DB:     {self.ingester.get_record_count()}")
            logger.info("=" * 80)
            
            # Verify integrity
            logger.info("\n[VERIFY] Running database integrity check...")
            is_valid, report = self.bio_db.verify_integrity()
            
            if is_valid:
                logger.info("[PASS] Database integrity verified")
            else:
                logger.warning("[WARN] Some integrity checks failed:")
                for check, result in report.items():
                    status = "[PASS]" if result else "[FAIL]"
                    logger.info(f"  {status} {check}")
        
        except KeyboardInterrupt:
            logger.info("\n[INTERRUPT] Seeding interrupted by user")
            logger.info("[INFO] Progress saved in checkpoint. Re-run to resume.")
            self.checkpoint.save()
            raise
        
        except Exception as e:
            logger.error(f"[FAIL] Fatal error: {e}")
            self.checkpoint.save()
            raise
    
    def _generate_manifest(self):
        """Generate seeding manifest on USB drive."""
        try:
            manifest_path = self.bio_db.db_root / MANIFEST_FILE
            
            # Get database stats
            storage_stats = self.bio_db.get_storage_stats()
            table_stats = self.bio_db.get_table_stats()
            
            # Compute database hash
            db_path = self.bio_db.db_root / "lancedb_store"
            db_hash = "N/A"  # Computing hash of entire directory is expensive
            
            manifest = {
                "seeding_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0",
                    "target_region": "Central Indian Ridge",
                    "depth_range": f"{OBIS_DEPTH_MIN}-{OBIS_DEPTH_MAX}m"
                },
                "statistics": {
                    "obis_species_fetched": self.stats["obis_species_fetched"],
                    "species_processed": self.stats["species_processed"],
                    "species_successful": self.stats["species_successful"],
                    "species_failed": self.stats["species_failed"],
                    "sequences_found": self.stats["sequences_found"],
                    "sequences_ingested": self.stats["sequences_ingested"],
                    "total_records": table_stats.get("row_count", 0),
                    "success_rate": f"{(self.stats['species_successful'] / max(self.stats['species_processed'], 1)) * 100:.1f}%"
                },
                "database": {
                    "table_name": self.ingester.table_name,
                    "record_count": table_stats.get("row_count", 0),
                    "size_mb": table_stats.get("size_mb", 0),
                    "vector_dimension": table_stats.get("vector_dim", 768),
                    "database_hash": db_hash
                },
                "storage": {
                    "drive_letter": self.bio_db.drive_letter,
                    "total_gb": storage_stats.get("total_gb", 0),
                    "used_gb": storage_stats.get("used_gb", 0),
                    "available_gb": storage_stats.get("available_gb", 0),
                    "percent_used": storage_stats.get("percent_used", 0)
                },
                "data_sources": {
                    "obis": {
                        "bbox": OBIS_BBOX,
                        "depth_range": f"{OBIS_DEPTH_MIN}-{OBIS_DEPTH_MAX}m"
                    },
                    "ncbi": {
                        "markers": NCBI_MARKERS,
                        "email": NCBI_EMAIL
                    },
                    "taxonkit": {
                        "exe": str(self.taxonkit.taxonkit_exe),
                        "data_dir": str(self.taxonkit.taxonkit_data),
                        "format": TAXONKIT_FORMAT
                    }
                },
                "errors": self.stats["errors"][:100]  # Keep first 100 errors
            }
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"[PASS] Manifest written to: {manifest_path}")
        
        except Exception as e:
            logger.error(f"[FAIL] Could not generate manifest: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("GLOBAL-BIOSCAN DEEP-SEA EDNA REFERENCE ATLAS SEEDER")
    print("The Architect Agent - Bio-Data Engineer")
    print("=" * 80 + "\n")
    
    # Check dependencies
    if not PYOBIS_AVAILABLE:
        print("[FAIL] pyobis not installed. Install with: pip install pyobis")
        return 1
    
    if not BIOPYTHON_AVAILABLE:
        print("[FAIL] Biopython not installed. Install with: pip install biopython")
        return 1
    
    if not PSUTIL_AVAILABLE:
        print("[FAIL] psutil not installed. Install with: pip install psutil")
        return 1
    
    # NTFS Safety Check
    print(f"\n[PREFLIGHT] Checking filesystem on drive {DRIVE_LETTER}:")
    if PSUTIL_AVAILABLE:
        try:
            partitions = psutil.disk_partitions()
            drive_found = False
            ntfs_verified = False
            
            for partition in partitions:
                if partition.mountpoint.upper().startswith(f"{DRIVE_LETTER}:"):
                    drive_found = True
                    print(f"  Drive: {partition.mountpoint}")
                    print(f"  Filesystem: {partition.fstype}")
                    
                    if partition.fstype.upper() == "NTFS":
                        ntfs_verified = True
                        print(f"[PASS] Drive {DRIVE_LETTER}: is NTFS - compatible with LanceDB")
                    else:
                        print(f"[FATAL] USB drive must be formatted as NTFS")
                        print(f"[INFO] Current filesystem: {partition.fstype}")
                        print(f"[INFO] Please format {DRIVE_LETTER}: as NTFS and retry")
                        return 1
                    break
            
            if not drive_found:
                print(f"[WARN] Could not verify filesystem type for {DRIVE_LETTER}:")
                print(f"[INFO] Proceeding with caution...")
        except Exception as e:
            print(f"[WARN] Could not check filesystem: {e}")
            print(f"[INFO] Proceeding with caution...")
    
    # Verify NCBI email is configured
    if NCBI_EMAIL == "bioscan.demo@example.com":
        print("[WARN] Using default NCBI email. Please update NCBI_EMAIL in script.")
        print("[INFO] NCBI requires a valid email for API access.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    try:
        seeder = AtlasSeeder(drive_letter=DRIVE_LETTER)
        seeder.run()
        
        print("\n[SUCCESS] Seeding complete! Your USB drive now contains a curated")
        print("          deep-sea reference atlas ready for the funding demo.")
        print("\nNext steps:")
        print(f"  1. Launch Streamlit: streamlit run src/interface/app.py --port 8504")
        print(f"  2. Navigate to Configuration tab")
        print(f"  3. Click 'Verify Database Integrity' to confirm seeding")
        print(f"  4. Check 'System Diagnostics' to see sequence count")
        
        return 0
    
    except DriveNotMountedError as e:
        print(f"\n[FAIL] USB drive not detected: {e}")
        print(f"[INFO] Please mount USB drive as {DRIVE_LETTER}:/ and retry")
        return 1
    
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Seeding interrupted. Progress saved.")
        print("[INFO] Re-run this script to resume from checkpoint.")
        return 0
    
    except Exception as e:
        logger.exception("[FAIL] Fatal error during seeding")
        print(f"\n[FAIL] Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
