"""Gestion de la connexion DuckDB et des operations de stockage."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

import duckdb

from psg_tracker.schemas import MatchSummary, ShotEvent

logger = logging.getLogger(__name__)

_INSERT_MATCH_SQL = """
    INSERT OR REPLACE INTO matches
        (match_id, source, match_date, home_team, away_team, competition, season)
    VALUES (?, ?, CAST(? AS DATE), ?, ?, ?, ?)
"""

_INSERT_PLAYER_POSITION_DETAILED_SQL = """
    INSERT OR REPLACE INTO player_positions_detailed (player_name, match_id, position_detailed)
    VALUES (?, ?, ?)
"""

_INSERT_PLAYER_POSITION_SQL = """
    INSERT OR REPLACE INTO player_positions (player_name, season, position_raw)
    VALUES (?, ?, ?)
"""

_INSERT_SHOT_SQL = """
    INSERT OR REPLACE INTO shots
        (event_id, match_id, source, player_id, player_name, team_id, minute, second,
         loc_x, loc_y, body_part, shot_type, outcome, is_goal)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class DuckDBManager:
    """Encapsule la connexion DuckDB et les operations de stockage.

    Utilisable comme context manager :
        with DuckDBManager(path) as db:
            db.apply_ddl(ddl_path)
            db.insert_matches(matches)
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Ouvre (ou reutilise) la connexion DuckDB, en creant le dossier parent."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self._db_path))
            logger.info("Connexion DuckDB ouverte: %s", self._db_path)
        return self._conn

    def apply_ddl(self, sql_path: Path) -> None:
        """Execute un script SQL (DDL ou transformations) sur la base."""
        sql = sql_path.read_text(encoding="utf-8")
        self.connect().execute(sql)
        logger.info("DDL applique: %s", sql_path)

    def get_match_dates(self, source: str) -> set[str]:
        """Renvoie l'ensemble des dates (YYYY-MM-DD) des matches deja en base pour `source`.

        Sert a deduplication inter-sources a l'ingestion (cf. `scripts/ingest.py`
        `ingest_understat`) : eviter de compter deux fois un meme match reel
        quand plusieurs sources le couvrent a la meme date.
        """
        rows = self.connect().execute(
            "SELECT DISTINCT CAST(match_date AS VARCHAR) FROM matches WHERE source = ?",
            [source],
        ).fetchall()
        return {row[0] for row in rows}

    def insert_matches(self, matches: Sequence[MatchSummary]) -> None:
        """Insere (ou remplace) une liste de matches dans la table `matches`."""
        if not matches:
            return
        rows = [
            (m.match_id, m.source, m.match_date, m.home_team, m.away_team, m.competition, m.season)
            for m in matches
        ]
        self.connect().executemany(_INSERT_MATCH_SQL, rows)
        logger.info("insert_matches: %d match(es) inseres", len(matches))

    def insert_player_positions_detailed(self, rows: Sequence[tuple[str, int, str]]) -> None:
        """Insere (ou remplace) des lignes (player_name, match_id, position_detailed).

        Source : `StatsBombClient.get_lineups`. Complement plus fin (24
        postes StatsBomb) a `player_positions` (4 categories Understat),
        mais couverture partielle : uniquement les joueurs ayant dispute au
        moins un match StatsBomb (95 matchs sur 3 saisons), pas les 397
        matchs du corpus complet.
        """
        if not rows:
            return
        self.connect().executemany(_INSERT_PLAYER_POSITION_DETAILED_SQL, list(rows))
        logger.info("insert_player_positions_detailed: %d ligne(s) inseree(s)", len(rows))

    def insert_shots(self, shots: Sequence[ShotEvent]) -> None:
        """Insere (ou remplace) une liste de tirs dans la table `shots`.

        Necessite que les matches correspondants aient deja ete inseres
        (contrainte de cle etrangere composite `(source, match_id)`).
        """
        if not shots:
            return
        rows = [
            (
                s.event_id,
                s.match_id,
                s.source,
                s.player_id,
                s.player_name,
                s.team_id,
                s.minute,
                s.second,
                s.location.x,
                s.location.y,
                s.body_part,
                s.shot_type,
                s.outcome,
                s.is_goal,
            )
            for s in shots
        ]
        self.connect().executemany(_INSERT_SHOT_SQL, rows)
        logger.info("insert_shots: %d tir(s) inseres", len(shots))

    def insert_player_positions(self, rows: Sequence[tuple[str, str, str]]) -> None:
        """Insere (ou remplace) des lignes (player_name, season, position_raw).

        Source : `UnderstatClient.get_team_players`. Sert a filtrer/annoter
        le dashboard par poste (cf. `app/data_loader.py`) - StatsBomb ne
        fournit pas cette info sur les evenements de tir deja ingeres, donc
        Understat est la seule source pour ce champ (mais couvre toutes les
        saisons 2015-2026, y compris celles ou StatsBomb est aussi present).
        """
        if not rows:
            return
        self.connect().executemany(_INSERT_PLAYER_POSITION_SQL, list(rows))
        logger.info("insert_player_positions: %d ligne(s) inseree(s)", len(rows))

    def close(self) -> None:
        """Ferme proprement la connexion (idempotent)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Connexion DuckDB fermee: %s", self._db_path)

    def __enter__(self) -> DuckDBManager:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
