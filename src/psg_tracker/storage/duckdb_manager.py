"""Gestion de la connexion DuckDB et execution des scripts SQL."""

from __future__ import annotations

from pathlib import Path

import duckdb


class DuckDBManager:
    """Encapsule la connexion DuckDB et les operations de stockage."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Ouvre (ou reutilise) la connexion DuckDB."""
        raise NotImplementedError

    def apply_ddl(self, sql_path: Path) -> None:
        """Execute un script SQL (DDL ou transformations) sur la base."""
        raise NotImplementedError

    def insert_shots(self, shots: list[dict]) -> None:
        """Insere une liste de tirs dans la table `shots`."""
        raise NotImplementedError

    def close(self) -> None:
        """Ferme proprement la connexion."""
        raise NotImplementedError
