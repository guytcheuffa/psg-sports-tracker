#!/usr/bin/env python3
"""Script d'orchestration de l'ingestion : StatsBomb et/ou Understat -> DuckDB.

Usage:
    # decouvre automatiquement les competitions/saisons StatsBomb ou le PSG
    # a joue (Ligue 1 + Champions League) et ingere tout
    python scripts/ingest.py statsbomb --discover

    # ingere une seule competition/saison (pratique pour decouper le travail
    # en plusieurs runs courts)
    python scripts/ingest.py statsbomb --competition-id 7 --season-id 27

    # Understat (une seule saison, celle en cours par defaut). Necessite un
    # acces reseau a understat.com, qui peut etre bloque selon la politique
    # reseau du poste.
    python scripts/ingest.py understat --season 2026

    # Understat (backfill historique) : Understat couvre en realite tout
    # l'historique du club depuis 2014/2015 (pas seulement la saison en
    # cours), via le meme endpoint parametre par saison. Ingere toutes les
    # saisons depuis 2015 jusqu'a la saison en cours incluse.
    python scripts/ingest.py understat --from-season 2015

Idempotent : toutes les insertions passent par INSERT OR REPLACE, on peut
relancer le script sans creer de doublons.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from psg_tracker.config import current_understat_season, settings
from psg_tracker.ingestion.statsbomb_client import StatsBombClient
from psg_tracker.ingestion.understat_client import UnderstatClient
from psg_tracker.schemas import MatchSummary, ShotEvent
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

        try:
            lineup = client.get_lineups(match.match_id, team_name=team_name)
        except requests.HTTPError as exc:
            logger.warning("Composition indisponible pour match_id=%d: %s", match.match_id, exc)
            lineup = []

        position_rows = [
            (str(p["player_name"]), match.match_id, str(p["positions"][0]["position"]))
            for p in lineup
            if p.get("positions")
        ]
        manager.insert_player_positions_detailed(position_rows)

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


def _date_matches_within_one_day(candidate: str, reference_dates: set[str]) -> bool:
    """True si `candidate` (YYYY-MM-DD) est dans `reference_dates`, a 1 jour pres.

    Verifie empiriquement : sur les saisons ou StatsBomb et Understat se
    recoupent, ~6 matchs/saison ont une date de coup d'envoi decalee d'un
    jour d'une source a l'autre (vraisemblablement un artefact de fuseau
    horaire sur les matchs en soiree). Une egalite stricte de dates laisse
    donc passer de vrais doublons ; on tolere +/-1 jour.
    """
    try:
        parsed = date.fromisoformat(candidate)
    except ValueError:
        return candidate in reference_dates
    return any(
        (parsed + timedelta(days=offset)).isoformat() in reference_dates for offset in (-1, 0, 1)
    )


def _understat_season_to_range(season: str) -> str:
    """Convertit une saison Understat ("2015", annee de coup d'envoi) vers le
    format "annee_debut/annee_fin" (ex: "2015/2016"), coherent avec le format
    StatsBomb - evite d'avoir deux formats de saison differents pour la meme
    notion selon la source (ce qui casserait tout regroupement/affichage par
    saison dans le dashboard).
    """
    start_year = int(season)
    return f"{start_year}/{start_year + 1}"


def ingest_understat(
    manager: DuckDBManager,
    team_slug: str = settings.understat_psg_slug,
    team_name: str = settings.understat_team_name,
    season: str | None = None,
) -> tuple[int, int]:
    """Ingere les donnees Understat pour une saison (courante par defaut).

    Deduplication inter-sources : 3 saisons se recoupent entre StatsBomb et
    Understat (2015/2016, 2021/2022, 2022/2023 - StatsBomb n'y couvre
    d'ailleurs pas toujours 100% des matchs). Sans filtrage, un meme match
    reel serait ingere sous deux `match_id` differents (un par source) et
    compte deux fois dans le dashboard (buts/tirs doubles sur ces saisons).
    On identifie les doublons par date de match (StatsBomb prioritaire, plus
    riche en evenements) : Understat ne comble que les dates ou StatsBomb
    n'a aucun match, y compris a l'interieur d'une saison qu'il couvre par
    ailleurs partiellement.

    Renvoie (n_matches, n_shots) inseres.
    """
    raw_season = season or current_understat_season()
    formatted_season = _understat_season_to_range(raw_season)
    client = UnderstatClient()

    raw_matches = client.get_team_matches(team_slug, raw_season)
    statsbomb_dates = manager.get_match_dates("statsbomb")

    matches: list[MatchSummary] = []
    match_ids_to_fetch: list[int] = []
    n_skipped = 0
    for raw in raw_matches:
        if not raw.get("isResult", False):
            continue
        match_date = str(raw.get("datetime", ""))[:10]
        if _date_matches_within_one_day(match_date, statsbomb_dates):
            n_skipped += 1
            continue
        try:
            match_id = int(raw["id"])
            matches.append(
                MatchSummary(
                    match_id=match_id,
                    match_date=match_date,
                    home_team=raw["h"]["title"],
                    away_team=raw["a"]["title"],
                    competition="Ligue 1",
                    season=formatted_season,
                    source="understat",
                )
            )
            match_ids_to_fetch.append(match_id)
        except (KeyError, ValueError) as exc:
            logger.warning("Match Understat ignore (champ inattendu): %s (%s)", raw, exc)

    if matches:
        manager.insert_matches(matches)

    shots: list[ShotEvent] = []
    for match_id in match_ids_to_fetch:
        shots.extend(client.get_match_shots(match_id, team_name=team_name))
        time.sleep(_REQUEST_DELAY_SECONDS)
    if shots:
        manager.insert_shots(shots)

    raw_players = client.get_team_players(team_slug, raw_season)
    position_rows = [
        (str(p["player_name"]), formatted_season, str(p.get("position", "")))
        for p in raw_players
        if p.get("player_name")
    ]
    manager.insert_player_positions(position_rows)

    if n_skipped:
        logger.info(
            "Understat saison %s: %d match(es) ignore(s) (deja couverts par StatsBomb "
            "a la meme date)",
            formatted_season,
            n_skipped,
        )
    logger.info(
        "Understat TOTAL (saison %s): %d match(es), %d tir(s) inseres",
        formatted_season,
        len(matches),
        len(shots),
    )
    return len(matches), len(shots)


def ingest_understat_seasons(
    manager: DuckDBManager,
    seasons: list[str],
    team_slug: str = settings.understat_psg_slug,
    team_name: str = settings.understat_team_name,
) -> None:
    """Ingere Understat pour plusieurs saisons d'affilee (backfill historique).

    Understat expose en realite tout l'historique du club depuis 2014/2015
    sur le meme endpoint que la saison en cours (`getTeamData/{team}/{season}`,
    parametre par saison) - contrairement a l'hypothese initiale de ce projet
    qui ne l'utilisait que pour la saison "live". Boucle saison par saison,
    avec une pause polie entre chaque appel `getTeamData`.
    """
    if not seasons:
        # Ex: --from-season poste apres la saison en cours (erreur de saisie).
        # Sans ce garde-fou, `seasons[0]`/`seasons[-1]` plus bas levent IndexError.
        logger.warning("ingest_understat_seasons: liste de saisons vide, rien a faire")
        return

    total_matches = 0
    total_shots = 0
    for season in seasons:
        n_matches, n_shots = ingest_understat(
            manager, team_slug=team_slug, team_name=team_name, season=season
        )
        total_matches += n_matches
        total_shots += n_shots
        time.sleep(_REQUEST_DELAY_SECONDS)

    logger.info(
        "Understat GRAND TOTAL (%d saison(s), %s -> %s): %d match(es), %d tir(s) inseres",
        len(seasons),
        seasons[0],
        seasons[-1],
        total_matches,
        total_shots,
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
    us_parser.add_argument("--team-name", default=settings.understat_team_name)
    us_parser.add_argument("--season", default=None, help="une seule saison (ex: 2022)")
    us_parser.add_argument(
        "--from-season",
        type=int,
        default=None,
        help=(
            "backfill : ingere toutes les saisons depuis cette annee (incluse) jusqu'a "
            "la saison en cours. Prioritaire sur --season si fourni."
        ),
    )

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
            if args.from_season is not None:
                current_year = int(current_understat_season())
                seasons = [str(year) for year in range(args.from_season, current_year + 1)]
                ingest_understat_seasons(
                    manager, seasons, team_slug=args.team_slug, team_name=args.team_name
                )
            else:
                ingest_understat(
                    manager,
                    team_slug=args.team_slug,
                    team_name=args.team_name,
                    season=args.season,
                )


if __name__ == "__main__":
    main()
