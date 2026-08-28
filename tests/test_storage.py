"""Tests unitaires du DuckDBManager."""

from pathlib import Path

from psg_tracker.storage.duckdb_manager import DuckDBManager


def test_manager_instantiation(tmp_path: Path) -> None:
    manager = DuckDBManager(db_path=tmp_path / "test.duckdb")
    assert manager is not None
