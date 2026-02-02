"""Robust LanceDB connection logic for USB-based vector storage.

Handles drive detection, path validation, IVF-PQ indexing, and graceful
disconnect protocols for the 32GB Flash storage deployment.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import hashlib

import lancedb

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_DRIVE_LETTER = "E"
DEFAULT_DB_ROOT = "GlobalBioScan_DB"
DEFAULT_TABLE_NAME = "obis_reference_index"
IVF_NUM_PARTITIONS = 256  # IVF clusters for USB storage (balance speed/accuracy)
IVF_NUM_SUB_VECTORS = 96  # PQ sub-vectors (768 / 8 = 96)


class DriveNotMountedError(Exception):
    """Raised when the USB drive is not detected."""
    pass


class DatabaseIntegrityError(Exception):
    """Raised when database integrity check fails."""
    pass


class BioDB:
    """Robust LanceDB wrapper for USB-based vector storage.
    
    Handles:
    - Hardware detection and validation
    - Path creation and initialization
    - IVF-PQ indexing for USB performance
    - Graceful disconnect on hardware removal
    - Integrity checks and recovery
    """
    
    def __init__(
        self,
        drive_letter: str = DEFAULT_DRIVE_LETTER,
        db_root: str = DEFAULT_DB_ROOT,
        enable_auto_init: bool = True,
    ):
        """Initialize BioDB connection manager.
        
        Args:
            drive_letter: USB drive letter (e.g., 'E')
            db_root: Root directory name on USB
            enable_auto_init: Auto-create directories if missing
        """
        self.drive_letter = drive_letter.upper()
        self.db_root_name = db_root
        self.enable_auto_init = enable_auto_init
        
        # Paths
        self.drive_path = Path(f"{self.drive_letter}:/")
        self.db_root = self.drive_path / db_root
        self.db_uri = str(self.db_root / "lancedb_store")
        self.index_dir = self.db_root / "indices"
        self.logs_dir = self.db_root / "logs"
        self.manifest_file = self.db_root / "manifest.md5"
        
        # Connection state
        self._db = None
        self._is_mounted = False
        self._integrity_status = None
        
        # IVF-PQ parameters
        self.ivf_num_partitions = IVF_NUM_PARTITIONS
        self.ivf_num_sub_vectors = IVF_NUM_SUB_VECTORS
        self.nprobes = 10  # Default: search 10 clusters (tunable in UI)
    
    def detect_drive(self) -> Tuple[bool, str]:
        """Detect and validate USB drive.
        
        Returns:
            (is_mounted, status_message)
        """
        logger.info(f"Detecting drive {self.drive_letter}:/")
        
        # Check if drive exists
        if not self.drive_path.exists():
            msg = f"[FAIL] Drive {self.drive_letter}:/ not detected"
            self._is_mounted = False
            return False, msg
        
        # Check if writable
        try:
            test_file = self.drive_path / ".bioscan_test"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()
        except (PermissionError, OSError) as e:
            msg = f"[FAIL] Drive {self.drive_letter}:/ not writable: {e}"
            self._is_mounted = False
            return False, msg
        
        self._is_mounted = True
        msg = f"[PASS] Drive {self.drive_letter}:/ detected and writable"
        return True, msg
    
    def initialize_directories(self) -> Tuple[bool, str]:
        """Create required directory structure.
        
        Returns:
            (success, status_message)
        """
        logger.info(f"Initializing directories at {self.db_root}")
        
        if not self._is_mounted:
            msg = "[FAIL] Drive not mounted. Run detect_drive() first."
            return False, msg
        
        try:
            # Create directories
            for dir_path in [self.db_root, self.index_dir, self.logs_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created/verified directory: {dir_path}")
            
            msg = "[PASS] Directory structure initialized"
            return True, msg
        
        except OSError as e:
            msg = f"[FAIL] Could not create directories: {e}"
            logger.error(msg)
            return False, msg
    
    def connect(self) -> Optional[Any]:
        """Connect to LanceDB on USB drive.
        
        Returns:
            LanceDB connection object or None on failure
        """
        logger.info(f"Connecting to LanceDB at {self.db_uri}")
        
        # Detect drive
        is_mounted, msg = self.detect_drive()
        if not is_mounted:
            logger.error(msg)
            raise DriveNotMountedError(msg)
        
        # Initialize directories
        success, msg = self.initialize_directories()
        if not success:
            logger.error(msg)
            raise DatabaseIntegrityError(msg)
        
        # Connect to LanceDB
        try:
            self._db = lancedb.connect(self.db_uri)
            logger.info(f"[PASS] Connected to LanceDB at {self.db_uri}")
            return self._db
        except Exception as e:
            msg = f"[FAIL] LanceDB connection failed: {e}"
            logger.error(msg)
            raise DatabaseIntegrityError(msg)
    
    def is_connected(self) -> bool:
        """Check if currently connected to database."""
        if self._db is None:
            return False
        try:
            # Test connection by listing tables
            _ = self._db.table_names()
            return True
        except Exception:
            self._db = None
            return False
    
    def get_table(self, table_name: str = DEFAULT_TABLE_NAME) -> Optional[Any]:
        """Get or create table handle.
        
        Args:
            table_name: Name of the table to open
        
        Returns:
            Table object or None if not found
        """
        if not self.is_connected() or self._db is None:
            logger.error("Not connected to database")
            return None
        
        try:
            table = self._db.open_table(table_name)
            logger.info(f"[PASS] Opened table: {table_name}")
            return table
        except Exception as e:
            logger.warning(f"[WARN] Table not found: {table_name}: {e}")
            return None
    
    def build_ivf_pq_index(
        self,
        table_name: str = DEFAULT_TABLE_NAME,
        vector_column: str = "vector",
        metric: str = "cosine",
    ) -> Tuple[bool, str]:
        """Build IVF-PQ index for sub-second searches on USB.
        
        IVF-PQ combines:
        - IVF (Inverted File): Coarse partitioning into clusters
        - PQ (Product Quantization): Vector compression for RAM efficiency
        
        Args:
            table_name: Table to index
            vector_column: Column containing embeddings
            metric: Distance metric ('cosine', 'l2', 'dot')
        
        Returns:
            (success, status_message)
        """
        logger.info(f"Building IVF-PQ index on {table_name}:{vector_column}")
        
        if not self.is_connected():
            return False, "[FAIL] Not connected to database"
        
        try:
            table = self.get_table(table_name)
            if table is None:
                return False, f"[FAIL] Table {table_name} not found"
            
            # Create index with IVF-PQ parameters
            # Note: LanceDB uses create_index() for newer API
            table.create_index(
                column=vector_column,
                index_type="ivf_pq",
                metric=metric,
                replace=True,  # Replace existing index
            )
            
            msg = (
                f"[PASS] IVF-PQ index created: "
                f"metric={metric}, partitions={self.ivf_num_partitions}"
            )
            logger.info(msg)
            return True, msg
        
        except Exception as e:
            msg = f"[FAIL] Index creation failed: {e}"
            logger.error(msg)
            return False, msg
    
    def verify_integrity(
        self,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify database integrity with checksums.
        
        Returns:
            (is_valid, status_dict)
        """
        logger.info("Verifying database integrity")
        
        status = {
            "drive_mounted": False,
            "directories_exist": False,
            "db_connected": False,
            "table_accessible": False,
            "manifest_valid": False,
            "total_size_mb": 0,
            "row_count": 0,
        }
        
        # Check 1: Drive mounted
        status["drive_mounted"] = self._is_mounted
        
        # Check 2: Directories exist
        dirs_ok = all(d.exists() for d in [self.db_root, self.index_dir, self.logs_dir])
        status["directories_exist"] = dirs_ok
        
        # Check 3: Connected
        status["db_connected"] = self.is_connected()
        
        # Check 4: Table accessible
        if status["db_connected"]:
            table = self.get_table(table_name)
            if table is not None:
                status["table_accessible"] = True
                try:
                    status["row_count"] = table.count_rows()
                except Exception:
                    pass
        
        # Check 5: Manifest checksum
        status["manifest_valid"] = self._verify_manifest()
        
        # Storage stats
        status["total_size_mb"] = self._get_dir_size_mb(self.db_root)
        
        is_valid = all([
            status["drive_mounted"],
            status["directories_exist"],
            status["db_connected"],
            status["table_accessible"],
        ])
        
        self._integrity_status = is_valid
        return is_valid, status
    
    def _verify_manifest(self) -> bool:
        """Verify manifest checksum.
        
        Returns:
            True if manifest is valid or doesn't exist yet
        """
        if not self.manifest_file.exists():
            logger.debug("Manifest file doesn't exist (first run)")
            return True
        
        try:
            # Read stored checksum
            with open(self.manifest_file, "r") as f:
                stored_hash = f.read().strip()
            
            # Compute current checksum
            current_hash = self._compute_manifest_hash()
            
            is_valid = stored_hash == current_hash
            if not is_valid:
                logger.warning("Manifest checksum mismatch - potential corruption")
            return is_valid
        except Exception as e:
            logger.warning(f"Could not verify manifest: {e}")
            return False
    
    def _compute_manifest_hash(self) -> str:
        """Compute hash of database manifest files."""
        hasher = hashlib.md5()
        
        # Hash all .json files in lancedb_store (metadata)
        lancedb_store = Path(self.db_uri)
        if lancedb_store.exists():
            for json_file in lancedb_store.glob("**/*.json"):
                with open(json_file, "rb") as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def update_manifest(self) -> bool:
        """Update manifest checksum file.
        
        Returns:
            Success status
        """
        try:
            new_hash = self._compute_manifest_hash()
            with open(self.manifest_file, "w") as f:
                f.write(new_hash)
            logger.info("Manifest updated")
            return True
        except Exception as e:
            logger.error(f"Could not update manifest: {e}")
            return False
    
    def _get_dir_size_mb(self, path: Path) -> float:
        """Get total size of directory in MB.
        
        Args:
            path: Directory path
        
        Returns:
            Size in MB
        """
        if not path.exists():
            return 0.0
        
        total_bytes = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_bytes += os.path.getsize(filepath)
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"Could not calculate directory size: {e}")
        
        return round(total_bytes / (1024 * 1024), 2)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics.
        
        Returns:
            Dict with capacity, used, available in GB
        """
        if not self._is_mounted:
            return {
                "drive_letter": self.drive_letter,
                "total_gb": 0,
                "used_gb": 0,
                "available_gb": 0,
                "percent_used": 0,
            }
        
        try:
            import shutil
            stat = shutil.disk_usage(str(self.drive_path))
            
            total_gb = stat.total / (1024**3)
            used_gb = stat.used / (1024**3)
            available_gb = stat.free / (1024**3)
            percent = (used_gb / total_gb * 100) if total_gb > 0 else 0
            
            return {
                "drive_letter": self.drive_letter,
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "available_gb": round(available_gb, 2),
                "percent_used": round(percent, 1),
            }
        except Exception as e:
            logger.error(f"Could not get storage stats: {e}")
            return {
                "drive_letter": self.drive_letter,
                "total_gb": 0,
                "used_gb": 0,
                "available_gb": 0,
                "percent_used": 0,
            }
    
    def get_table_stats(self, table_name: str = DEFAULT_TABLE_NAME) -> Dict[str, Any]:
        """Get statistics about a table.
        
        Returns:
            Dict with row count, size, vector dimension
        """
        stats = {
            "table_name": table_name,
            "row_count": 0,
            "size_mb": 0,
            "vector_dim": 0,
        }
        
        if not self.is_connected():
            return stats
        
        try:
            table = self.get_table(table_name)
            if table is None:
                return stats
            
            stats["row_count"] = table.count_rows()
            stats["size_mb"] = self._get_dir_size_mb(self.db_root)
            
            # Try to infer vector dimension from schema
            try:
                schema = table.schema
                for field in schema:
                    if "vector" in field.name.lower():
                        # Parse vector dimension from type
                        type_str = str(field.type)
                        if "768" in type_str:
                            stats["vector_dim"] = 768
                        elif "256" in type_str:
                            stats["vector_dim"] = 256
                        break
            except Exception:
                stats["vector_dim"] = 768  # Default
            
            return stats
        except Exception as e:
            logger.error(f"Could not get table stats: {e}")
            return stats
    
    def disconnect(self) -> str:
        """Gracefully disconnect from database.
        
        Returns:
            Status message
        """
        logger.info("Disconnecting from database")
        
        if self._db is not None:
            try:
                # Update manifest before disconnecting
                self.update_manifest()
                self._db = None
                msg = "[PASS] Disconnected from database"
                logger.info(msg)
                return msg
            except Exception as e:
                msg = f"[WARN] Error during disconnect: {e}"
                logger.warning(msg)
                return msg
        
        return "[INFO] Not connected"
    
    def handle_drive_removal(self) -> str:
        """Handle graceful disconnect if USB drive removed.
        
        Returns:
            Error message to display
        """
        logger.critical("USB drive removal detected!")
        
        self._db = None
        self._is_mounted = False
        
        error_msg = (
            "[CRITICAL] DATA_SOURCE_DISCONNECTED\n"
            "USB drive was unexpectedly removed.\n"
            "Please reconnect the drive and restart the analysis."
        )
        return error_msg
