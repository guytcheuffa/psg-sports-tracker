"""Test de fumee du dashboard Streamlit (via `streamlit.testing.v1.AppTest`).

Construit une base DuckDB et un modele xG synthetiques dans un dossier
temporaire (le dashboard ne doit pas dependre des vraies donnees ingerees,
absentes en CI puisque `*.duckdb` est gitignore), les branche via les
variables d'environnement lues par `Settings` (pydantic-settings), puis
verifie que la page se rend sans exception.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from psg_tracker.schemas import Location, MatchSummary, ShotEvent
from psg_tracker.storage.duckdb_manager import DuckDBManager
from tests.conftest import synthetic_shots
from tests.test_storage import DDL_PATH


def _build_test_database(db_path: Path) -> None:
    """Peuple une base DuckDB temporaire avec quelques matches/tirs synthetiques."""
    with DuckDBManager(db_path=db_path) as manager:
        manager.apply_ddl(DDL_PATH)
        manager.insert_matches(
            [
                MatchSummary(
                    match_id=1,
                    match_date="2026-08-23",
                    home_team="Paris Saint-Germain",
                    away_team="Rennes",
                    competition="Ligue 1",
                    season="2026/2027",
                    source="statsbomb",
                )
            ]
        )
        synthetic = synthetic_shots(n=30)
        shots = [
            ShotEvent(
                event_id=f"test-{i}",
                match_id=1,
                player_id=int(row.player_id),
                player_name=f"Joueur {row.player_id}",
                team_id=1,
                minute=i % 90,
                second=0,
                location=Location(x=row.loc_x, y=row.loc_y),
                body_part=row.body_part,
                shot_type=row.shot_type,
                outcome="Goal" if row.is_goal else "Off T",
                is_goal=bool(row.is_goal),
                source="statsbomb",
            )
            for i, row in enumerate(synthetic.itertuples())
        ]
        manager.insert_shots(shots)


def _build_test_model(model_path: Path) -> None:
    """Entraine et sauvegarde un XGModel jetable sur des tirs synthetiques."""
    from psg_tracker.features.engineering import build_feature_matrix
    from psg_tracker.models.xg_model import XGModel

    shots = synthetic_shots(n=200)
    features = build_feature_matrix(shots)
    model = XGModel()
    model.train(features, features["is_goal"])
    model.save(model_path)


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirige `settings.duckdb_path`/`settings.model_path` vers des fichiers temporaires.

    `Settings` (pydantic-settings) lit ces chemins depuis les variables
    d'environnement `DUCKDB_PATH`/`MODEL_PATH` au moment de l'instanciation :
    on force donc un `importlib.reload` du module `psg_tracker.config` apres
    avoir positionne les variables, pour que le singleton `settings` reflete
    bien les chemins temporaires (et non ceux du vrai projet).
    """
    db_path = tmp_path / "test.duckdb"
    model_path = tmp_path / "test_model.json"

    _build_test_database(db_path)
    _build_test_model(model_path)

    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    import psg_tracker.config

    importlib.reload(psg_tracker.config)


def test_dashboard_renders_without_exception(isolated_settings: None) -> None:
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).parent.parent / "src/psg_tracker/app/main.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=60)

    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.metric) >= 4  # KPIs : tirs, buts reels, xG cumule, buts-xG
