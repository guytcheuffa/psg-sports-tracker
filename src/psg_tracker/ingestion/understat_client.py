"""Client de scraping pour understat.com (saison en cours, non couverte par
StatsBomb Open Data).

Understat n'expose pas d'API publique documentee : les donnees de tir
(coordonnees x/y, xG maison, situation, minute...) sont embarquees dans des
balises <script> des pages HTML sous forme de chaines JS echappees
(`var shotsData = JSON.parse('...')`). Ce module extrait et decode ces
variables.

Usage prevu pour ce projet : recuperer les tirs de la saison en cours (non
disponibles via StatsBomb Open Data, qui s'arrete a 2022/2023) afin
d'alimenter le dashboard "live" avec de la donnee fraiche, tout en gardant
StatsBomb comme corpus d'entrainement du modele xG (features plus riches et
qualite validee).

A usage personnel/educatif (portfolio) : respecter une frequence de requetes
raisonnable (voir `request_delay_seconds`) et le robots.txt du site.
"""

from __future__ import annotations

import json
import logging
import re
import time
import zlib
from typing import Any

import requests

from psg_tracker.ingestion.schemas import Location, ShotEvent

logger = logging.getLogger(__name__)

_BASE_URL = "https://understat.com"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_JSON_VAR_PATTERN = "var {name} = JSON.parse\\('(.*?)'\\);"

# Understat normalise x/y entre 0 et 1. On les remet dans le referentiel
# StatsBomb (120 x 80 yards) pour que le feature engineering soit unifie.
_PITCH_LENGTH = 120.0
_PITCH_WIDTH = 80.0

_RESULT_TO_OUTCOME = {
    "Goal": "Goal",
    "MissedShots": "Off T",
    "SavedShot": "Saved",
    "BlockedShot": "Blocked",
    "ShotOnPost": "Post",
    "OwnGoal": "Own Goal",
}


class UnderstatClient:
    """Scrape les pages equipe/match d'understat.com."""

    def __init__(
        self,
        base_url: str = _BASE_URL,
        timeout: int = 15,
        request_delay_seconds: float = 0.5,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._request_delay_seconds = request_delay_seconds
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def _get_html(self, path: str) -> str:
        """GET une page HTML d'understat.com."""
        url = f"{self._base_url}/{path}"
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _extract_json_var(html: str, var_name: str) -> Any:
        """Extrait et decode une variable `JSON.parse('...')` embarquee en JS.

        La chaine est echappee au format JS (ex: \\x3C pour '<'). On tente un
        `json.loads` direct d'abord (au cas ou le format change), puis on
        retombe sur le decodage `unicode_escape` classique utilise par
        understat.
        """
        pattern = _JSON_VAR_PATTERN.format(name=re.escape(var_name))
        match = re.search(pattern, html)
        if match is None:
            raise ValueError(f"Variable '{var_name}' introuvable dans la page")

        raw = match.group(1)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            decoded = (
                raw.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
            )
            return json.loads(decoded)

    def get_team_matches(self, team_slug: str, season: str) -> list[dict[str, Any]]:
        """Liste brute des matches d'une equipe pour une saison (`datesData`).

        Args:
            team_slug: identifiant understat de l'equipe (ex: "Paris_Saint_Germain").
            season: annee de debut de saison (ex: "2026" pour 2026/2027).
        """
        html = self._get_html(f"team/{team_slug}/{season}")
        matches: list[dict[str, Any]] = self._extract_json_var(html, "datesData")
        logger.info(
            "get_team_matches: %d match(es) (team=%s, season=%s)",
            len(matches),
            team_slug,
            season,
        )
        return matches

    def get_match_shots(self, match_id: int, team_name: str | None = None) -> list[ShotEvent]:
        """Tous les tirs d'un match (`shotsData`), cote domicile + exterieur.

        Args:
            match_id: id understat du match.
            team_name: si fourni, ne garde que les tirs de cette equipe.
        """
        html = self._get_html(f"match/{match_id}")
        raw: dict[str, list[dict[str, Any]]] = self._extract_json_var(html, "shotsData")

        shots: list[ShotEvent] = []
        for side_shots in (raw.get("h", []), raw.get("a", [])):
            for shot in side_shots:
                shooter_team = shot["h_team"] if shot["h_a"] == "h" else shot["a_team"]
                if team_name is not None and shooter_team != team_name:
                    continue

                outcome = _RESULT_TO_OUTCOME.get(shot["result"], shot["result"])
                shots.append(
                    ShotEvent(
                        event_id=f"understat-{shot['id']}",
                        match_id=match_id,
                        player_id=int(shot["player_id"]),
                        player_name=shot["player"],
                        team_id=zlib.crc32(shooter_team.encode('utf-8')),
                        minute=int(shot["minute"]),
                        second=0,
                        location=Location(
                            x=float(shot["X"]) * _PITCH_LENGTH,
                            y=float(shot["Y"]) * _PITCH_WIDTH,
                        ),
                        body_part=shot.get("shotType", "Unknown"),
                        shot_type=shot.get("situation", "Unknown"),
                        outcome=outcome,
                        is_goal=shot["result"] == "Goal",
                        source="understat",
                    )
                )

        logger.info("get_match_shots: %d tir(s) extraits (match_id=%d)", len(shots), match_id)
        return shots

    def get_current_season_shots(
        self, team_slug: str, season: str, team_name: str
    ) -> list[ShotEvent]:
        """Tous les tirs d'une equipe sur une saison, match par match.

        Enchaine `get_team_matches` puis un `get_match_shots` par match deja
        joue (`isResult=True`), avec un delai poli entre chaque requete.
        """
        matches = self.get_team_matches(team_slug, season)
        all_shots: list[ShotEvent] = []

        for match in matches:
            if not match.get("isResult", False):
                continue
            match_id = int(match["id"])
            all_shots.extend(self.get_match_shots(match_id, team_name=team_name))
            time.sleep(self._request_delay_seconds)

        logger.info(
            "get_current_season_shots: %d tir(s) au total (team=%s, season=%s)",
            len(all_shots),
            team_slug,
            season,
        )
        return all_shots
