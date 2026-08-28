"""Client pour l'API StatsBomb Open Data (github raw JSON, acces public).

Depot source : https://github.com/statsbomb/open-data
Aucune cle API requise. Structure des fichiers distants :
    competitions.json
    matches/{competition_id}/{season_id}.json
    events/{match_id}.json
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from psg_tracker.ingestion.schemas import Location, MatchSummary, ShotEvent

logger = logging.getLogger(__name__)

_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


class StatsBombClient:
    """Recupere competitions, matches et events depuis StatsBomb Open Data."""

    def __init__(self, base_url: str = _BASE_URL, timeout: int = 15) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._session = requests.Session()

    def _get_json(self, path: str) -> Any:
        """GET un fichier JSON du depot StatsBomb Open Data et le parse.

        Leve `requests.HTTPError` si le fichier n'existe pas (ex: match sans
        donnees d'evenements disponibles) ou en cas d'erreur reseau.
        """
        url = f"{self._base_url}/{path}"
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()
        return response.json()

    def get_competitions(self) -> list[dict[str, Any]]:
        """Liste brute des competitions/saisons disponibles."""
        return self._get_json("competitions.json")  # type: ignore[no-any-return]

    def get_matches(
        self,
        competition_id: int,
        season_id: int,
        team_name: str | None = None,
    ) -> list[MatchSummary]:
        """Matches d'une competition/saison, filtres optionnellement par equipe.

        Args:
            competition_id: id StatsBomb de la competition (ex: 7 = Ligue 1).
            season_id: id StatsBomb de la saison (ex: 27 = 2015/2016).
            team_name: si fourni, ne garde que les matches impliquant cette equipe.
        """
        raw_matches = self._get_json(f"matches/{competition_id}/{season_id}.json")
        summaries: list[MatchSummary] = []

        for raw in raw_matches:
            home = raw["home_team"]["home_team_name"]
            away = raw["away_team"]["away_team_name"]
            if team_name is not None and team_name not in (home, away):
                continue

            summaries.append(
                MatchSummary(
                    match_id=raw["match_id"],
                    match_date=raw["match_date"],
                    home_team=home,
                    away_team=away,
                    competition=raw["competition"]["competition_name"],
                    season=raw["season"]["season_name"],
                )
            )

        logger.info(
            "get_matches: %d match(es) trouves (competition=%d, season=%d, team=%s)",
            len(summaries),
            competition_id,
            season_id,
            team_name,
        )
        return summaries

    def get_shot_events(
        self,
        match_id: int,
        team_name: str | None = None,
    ) -> list[ShotEvent]:
        """Extrait les events de type 'Shot' d'un match.

        Args:
            match_id: id StatsBomb du match.
            team_name: si fourni, ne garde que les tirs de cette equipe.
        """
        raw_events = self._get_json(f"events/{match_id}.json")
        shots: list[ShotEvent] = []

        for event in raw_events:
            if event.get("type", {}).get("name") != "Shot":
                continue
            if team_name is not None and event.get("team", {}).get("name") != team_name:
                continue

            shot = event["shot"]
            outcome_name = shot["outcome"]["name"]
            x, y = event["location"]

            shots.append(
                ShotEvent(
                    event_id=event["id"],
                    match_id=match_id,
                    player_id=event["player"]["id"],
                    player_name=event["player"]["name"],
                    team_id=event["team"]["id"],
                    minute=event["minute"],
                    second=event["second"],
                    location=Location(x=x, y=y),
                    body_part=shot["body_part"]["name"],
                    shot_type=shot["type"]["name"],
                    outcome=outcome_name,
                    is_goal=outcome_name == "Goal",
                )
            )

        logger.info("get_shot_events: %d tir(s) extraits (match_id=%d)", len(shots), match_id)
        return shots
