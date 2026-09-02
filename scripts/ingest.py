#!/usr/bin/env python3
"""Script d'orchestration de l'ingestion : StatsBomb et/ou Understat -> DuckDB.

Usage:
    # decouvre automatiquement les competitions/saisons StatsBomb ou le PSG
    # a joue (Ligue 1 + Champions League) et ingere tout
    python scripts/ingest.py statsbomb --discover

    # ingere une seule competition/saison (pratique pour decouper le travail
    # en plusieurs runs courts)
    python scripts/ingest.py statsbomb --competition-id 7 --season-id 27

    # Understat (saison en cours). Necessite un acces reseau a understat.com,
    # qui peut etre bloque selon la politique reseau du poste.
    python scripts/ingest.py understat --season 2026

Idempotent : toutes les insertions passent par INSERT OR REPLACE, on peut
relancer le script sans creer de doublons.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import requests

from psg_tracker.config import current_understat_season, settings
from psg_tracker.ingestion.statsbomb_client import StatsBombClient
from psg_tracker.ingestion.understat_client import UnderstatClient
from psg_tracker.schemas import MatchSummary
from psg_tracker.storage.duckdb_manager import DuckDBManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DDL_PATH = _REPO_ROOT / "src/psg_tracker/storage/sql/ddl.sql"
# resolu par rapport a la racine du repo, independamment du cwd d'execution
_DB_PATH = (
    settings.duckdb_path
    if settings.duckdb_path.is_absolute()
    else _REPO_ROOT / settings.duckdb_path
)
_REQUEST_DELAY_SECONDS = 0.2  # politesse vis-a-vis de raw.githubusercontent.com


def _ingest_statsbomb_competition(
    manager: DuckDBManager,
    client: StatsBombClient,
    competition_id: int,
    season_id: int,
    team_name: str,
) -> tuple[int, int]:
    """Ingere une competition/saison StatsBomb. Renvoie (n_matches, n_shots)."""
    try:
        matches = client.get_matches(competition_id, season_id, team_name=team_name)
    except requests.HTTPError as exc:
        logger.warning(
            "Matches indisponibles pour competition=%d saison=%d: %s",
            competition_id,
            season_id,
            exc,
        )
        return 0, 0

    if not matches:
        return 0, 0

    manager.insert_matches(matches)
    n_shots = 0

    for match in matches:
        try:
            shots = client.get_shot_events(match.match_id, team_name=team_name)
        except requests.HTTPError as exc:
            logger.warning("Events indisponibles pour match_id=%d: %s", match.match_id, exc)
            continue

        if shots:
            manager.insert_shots(shots)
            n_shots += len(shots)

        time.sleep(_REQUEST_DELAY_SECONDS)

    logger.info(
        "StatsBomb competition=%d saison=%d: %d match(es), %d tir(s)",
        competition_id,
        season_id,
        len(matches),
        n_shots,
    )
    return len(matches), n_shots


def ingest_statsbomb(
    manager: DuckDBManager,
    team_name: str = settings.psg_team_name,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> None:
    """Ingere les donnees StatsBomb pour `team_name`.

    Si `competition_id`/`season_id` sont fournis, ne traite que cette
    competition/saison. Sinon, decouvre dynamiquement toutes les
    competitions/saisons ou l'equipe a joue parmi celles configurees
    (Ligue 1 et Champions League) - evite de deviner/figer des season_id
    a la main.
    """
    client = StatsBombClient()

    if competition_id is not None and season_id is not None:
        pairs = [(competition_id, season_id)]
    else:
        competitions = client.get_competitions()
        watched_competition_ids = {
            settings.statsbomb_ligue1_competition_id,
            settings.statsbomb_champions_league_competition_id,
        }
        pairs = [
            (c["competition_id"], c["season_id"])
            for c in competitions
            if c["competition_id"] in watched_competition_ids
        ]
        logger.info("%d competition(s)/saison(s) StatsBomb a explorer", len(pairs))

    total_matches = 0
    total_shots = 0
    for comp_id, seas_id in pairs:
        n_matches, n_shots = _ingest_statsbomb_competition(
            manager, client, comp_id, seas_id, team_name
        )
        total_matches += n_matches
        total_shots += n_shots

    logger.info("StatsBomb TOTAL: %d match(es), %d tir(s) inseres", total_matches, total_shots)


def ingest_understat(
    manager: DuckDBManager,
    team_slug: str = settings.understat_psg_slug,
    team_name: str = settings.psg_team_name,
    season: str | None = None,
) -> None:
    """Ingere les donnees Understat (saison en cours par defaut)."""
    season = season or current_understat_season()
    client = UnderstatClient()

    raw_matches = client.get_team_matches(team_slug, season)
    matches: list[MatchSummary] = []
    for raw in raw_matches:
        if not raw.get("isResult", False):
            continue
        try:
            matches.append(
                MatchSummary(
                    match_id=int(raw["id"]),
                    match_date=str(raw.get("datetime", ""))[:10],
                    home_team=raw["h"]["title"],
                    away_team=raw["a"]["title"],
                    competition="Ligue 1",
                    season=season,
                    source="understat",
                )
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Match Understat ignore (champ inattendu): %s (%s)", raw, exc)

    if matches:
        manager.insert_matches(matches)

    shots = client.get_current_season_shots(team_slug, season, team_name)
    if shots:
        manager.insert_shots(shots)

    logger.info(
        "Understat TOTAL (saison %s): %d match(es), %d tir(s) inseres",
        season,
        len(matches),
        len(shots),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="source", required=True)

    sb_parser = subparsers.add_parser("statsbomb", help="Ingestion StatsBomb Open Data")
    sb_parser.add_argument("--team-name", default=settings.psg_team_name)
    sb_parser.add_argument("--competition-id", type=int, default=None)
    sb_parser.add_argument("--season-id", type=int, default=None)

    us_parser = subparsers.add_parser("understat", help="Ingestion Understat (scraping)")
    us_parser.add_argument("--team-slug", default=settings.understat_psg_slug)
    us_parser.add_argument("--team-name", default=settings.psg_team_name)
    us_parser.add_argument("--season", default=None)

    args = parser.parse_args()

    with DuckDBManager(db_path=_DB_PATH) as manager:
        manager.apply_ddl(_DDL_PATH)

        if args.source == "statsbomb":
            ingest_statsbomb(
                manager,
                team_name=args.team_name,
                competition_id=args.competition_id,
                season_id=args.season_id,
            )
        elif args.source == "understat":
            ingest_understat(
                manager, team_slug=args.team_slug, team_name=args.team_name, season=args.season
            )


if __name__ == "__main__":
    main()
