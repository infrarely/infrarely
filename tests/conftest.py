"""Shared test fixtures."""

import sys
import os
import pytest

# Ensure aos is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def cleanup_db_files(tmp_path, monkeypatch):
    """Use temp directories for DB files so tests don't pollute the workspace."""
    db_dir = str(tmp_path)
    monkeypatch.setenv("INFRARELY_DATA_DIR", db_dir)
    # Disable file logging in tests to avoid creating log dirs
    from infrarely.observability.observability import get_logger

    logger = get_logger()
    logger._file_enabled = False
    yield
    # Cleanup happens automatically with tmp_path
