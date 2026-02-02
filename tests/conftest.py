"""PyTest configuration for GlobalBioScan validation suite."""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def mock_community_fasta(project_root: Path) -> Path:
    return project_root / "data" / "test" / "mock_community.fasta"


@pytest.fixture(scope="session")
def validation_output_dir(project_root: Path) -> Path:
    output_dir = project_root / "data" / "test" / "validation_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="session")
def pendrive_db_path() -> Path:
    db_drive = os.getenv("BIOSCANSCAN_DB_DRIVE", "E:\\GlobalBioScan_DB")
    return Path(db_drive) / "lancedb"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks slow tests")
    config.addinivalue_line("markers", "integration: marks integration tests")
