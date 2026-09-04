"""Tests unitaires du DuckDBManager (base DuckDB reelle, fichier temporaire)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from psg_tracker.schemas import Location, MatchSummary, ShotEvent
from psg_tracker.storage.duckdb_manager import DuckDBManager

DDL_PATH = Path(__file__).parent.parent / "src/psg_tracker/storage/sql/ddl.sql"


def _match(match_id: int = 1, source: str = "statsbomb") -> MatchSummary:
    return MatchSummary(
        match_id=match_id,
        match_date="2026-08-30",
        home_team="Paris Saint-Germain",
        away_team="Marseille",
        competition="Ligue 1",
        season="2026/2027",
        source=source,  # type: ignore[arg-type]
    )


def _shot(match_id: int = 1, source: str = "statsbomb", event_id: str = "evt-1") -> ShotEvent:
    return ShotEvent(
        event_id=event_id,
        match_id=match_id,
        player_id=10,
        player_name="Kylian Mbappe",
        team_id=771,
        minute=23,
        second=5,
        location=Location(x=110.5, y=38.0),
        body_part="Right Foot",
        shot_type="Open Play",
        outcome="Goal",
        is_goal=True,
        source=source,  # type: ignore[arg-type]
    )


def test_manager_instantiation(tmp_path: Path) -> None:
    manager = DuckDBManager(db_path=tmp_path / "test.duckdb")
    assert manager is not None


def test_connect_creates_db_file_and_parent_dirs(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "test.duckdb"
    manager = DuckDBManager(db_path=db_path)

    manager.connect()

    assert db_path.exists()
    manager.close()


def test_connect_is_idempotent(tmp_path: Path) -> None:
    manager = DuckDBManager(db_path=tmp_path / "test.duckdb")

    conn1 = manager.connect()
    conn2 = manager.connect()

    assert conn1 is conn2
    manager.close()


def test_apply_ddl_creates_expected_tables(tmp_path: Path) -> None:
    manager = DuckDBManager(db_path=tmp_path / "test.duckdb")
    manager.apply_ddl(DDL_PATH)

    tables = {
        row[0]
        for row in manager.connect()
        .execute("SELECT table_name FROM information_schema.tables")
        .fetchall()
    }

    assert {"matches", "shots"} <= tables
    manager.close()


def test_insert_matches_and_shots_roundtrip(tmp_path: Path) -> None:
    with DuckDBManager(db_path=tmp_path / "test.duckdb") as manager:
        manager.apply_ddl(DDL_PATH)
        manager.insert_matches([_match()])
        manager.insert_shots([_shot()])

        row = manager.connect().execute(
            "SELECT player_name, is_goal, loc_x, loc_y, source FROM shots WHERE event_id = 'evt-1'"
        ).fetchone()

    assert row == ("Kylian Mbappe", True, 110.5, 38.0, "statsbomb")


def test_insert_matches_is_idempotent_on_replay(tmp_path: Path) -> None:
    with DuckDBManager(db_path=tmp_path / "test.duckdb") as manager:
        manager.apply_ddl(DDL_PATH)
        manager.insert_matches([_match()])
        manager.insert_matches([_match()])  # re-ingestion, ne doit pas planter

        count = manager.connect().execute("SELECT COUNT(*) FROM matches").fetchone()

    assert count == (1,)


def test_insert_shots_without_matching_match_raises_fk_violation(tmp_path: Path) -> None:
    with DuckDBManager(db_path=tmp_path / "test.duckdb") as manager:
        manager.apply_ddl(DDL_PATH)
        # aucun match insere -> violation de la FK composite (source, match_id)
        with pytest.raises(duckdb.Error):
            manager.insert_shots([_shot()])


def test_statsbomb_and_understat_matches_do_not_collide(tmp_path: Path) -> None:
    """Deux matches avec le meme match_id numerique mais des sources differentes
    doivent coexister (cle composite (source, match_id))."""
    with DuckDBManager(db_path=tmp_path / "test.duckdb") as manager:
        manager.apply_ddl(DDL_PATH)
        manager.insert_matches(
            [_match(match_id=42, source="statsbomb"), _match(match_id=42, source="understat")]
        )

        count = manager.connect().execute("SELECT COUNT(*) FROM matches").fetchone()

    assert count == (2,)


def test_get_match_dates_returns_dates_for_given_source_only(tmp_path: Path) -> None:
    with DuckDBManager(db_path=tmp_path / "test.duckdb") as manager:
        manager.apply_ddl(DDL_PATH)
        manager.insert_matches(
            [
                _match(match_id=1, source="statsbomb"),  # 2026-08-30
                _match(match_id=2, source="understat"),  # meme date, autre source
            ]
        )

        statsbomb_dates = manager.get_match_dates("statsbomb")
        pl_dates = manager.get_match_dates("understat")
        empty_dates = manager.get_match_dates("nonexistent_source")

    assert statsbomb_dates == {"2026-08-30"}
    assert pl_dates == {"2026-08-30"}
    assert empty_dates == set()


def test_close_is_idempotent(tmp_path: Path) -> None:
    manager = DuckDBManager(db_path=tmp_path / "test.duckdb")
    manager.connect()

    manager.close()
    manager.close()  # ne doit pas lever


def test_context_manager_closes_connection_on_exit(tmp_path: Path) -> None:
    with DuckDBManager(db_path=tmp_path / "test.duckdb") as manager:
        manager.apply_ddl(DDL_PATH)

    assert manager._conn is None
