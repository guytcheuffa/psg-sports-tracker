"""Client pour l'API StatsBomb Open Data (github raw JSON, acces public)."""

from __future__ import annotations

import requests

from psg_tracker.ingestion.schemas import MatchSummary, ShotEvent

_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


class StatsBombClient:
    """Recupere competitions, matches et events depuis StatsBomb Open Data."""

    def __init__(self, base_url: str = _BASE_URL, timeout: int = 15) -> None:
        self._base_url = base_url
        self._timeout = timeout

    def get_competitions(self) -> list[dict]:
        """Liste brute des competitions disponibles."""
        raise NotImplementedError

    def get_matches(self, competition_id: int, season_id: int) -> list[MatchSummary]:
        """Matches d'une competition/saison donnee."""
        raise NotImplementedError

    def get_shot_events(self, match_id: int) -> list[ShotEvent]:
        """Extrait uniquement les events de type 'Shot' d'un match."""
        raise NotImplementedError
