"""Client de scraping pour understat.com (saison en cours, non couverte par
StatsBomb Open Data).

Understat n'expose pas d'API publique documentee. Les pages `/team/...` et
`/match/...` sont des coquilles HTML quasi vides : le navigateur charge les
donnees ensuite via des appels AJAX internes (`getTeamData/{team}/{season}`
et `getMatchData/{match_id}`), qui renvoient directement du JSON. Ce module
appelle ces memes endpoints (identifies en inspectant `js/team.min.js` et
`js/match.min.js`) plutot que de parser du HTML.

Point d'attention : ces endpoints repondent 404 sans l'en-tete
`X-Requested-With: XMLHttpRequest` (verifie empiriquement) - c'est la seule
protection anti-scraping constatee, en plus du gzip standard que `requests`
gere nativement.

Usage prevu pour ce projet : recuperer les tirs de la saison en cours (non
disponibles via StatsBomb Open Data, qui s'arrete a 2022/2023) afin
d'alimenter le dashboard "live" avec de la donnee fraiche, tout en gardant
StatsBomb comme corpus d'entrainement du modele xG (features plus riches et
qualite validee).

A usage personnel/educatif (portfolio) : respecter une frequence de requetes
raisonnable (voir `request_delay_seconds`) et le robots.txt du site.
"""

from __future__ import annotations

import logging
import time
import zlib
from typing import Any

import requests

from psg_tracker.schemas import Location, ShotEvent

logger = logging.getLogger(__name__)

_BASE_URL = "https://understat.com"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

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

# Understat et StatsBomb utilisent des vocabulaires de categories differents
# pour la meme notion (ex: "OpenPlay" vs "Open Play"). Sans harmonisation,
# le one-hot encoding de `build_feature_matrix` cree des colonnes dupliquees
# par source ("situation_OpenPlay" ET "situation_Open Play"), ce qui casse
# la generalisation du modele - constate empiriquement sur les 2 premiers
# matches scrapes. On remappe donc vers le vocabulaire StatsBomb (corpus de
# reference, plus volumineux) a l'ingestion.
_SITUATION_TO_CANONICAL = {
    "OpenPlay": "Open Play",
    "FromCorner": "Corner",
    "SetPiece": "Free Kick",
    "DirectFreekick": "Free Kick",
    "Penalty": "Penalty",
}

_BODY_PART_TO_CANONICAL = {
    "RightFoot": "Right Foot",
    "LeftFoot": "Left Foot",
    "Head": "Head",
    "OtherBodyPart": "Other",
}


def _normalize_category(raw_value: str, mapping: dict[str, str], field_name: str) -> str:
    """Remappe une categorie Understat vers le vocabulaire canonique StatsBomb.

    Log un warning (sans planter) si la valeur est inconnue, afin de detecter
    silencieusement toute nouvelle categorie introduite par understat.com.
    """
    canonical = mapping.get(raw_value)
    if canonical is None:
        logger.warning(
            "Categorie Understat '%s' inconnue pour %s, conservee telle quelle",
            raw_value,
            field_name,
        )
        return raw_value
    return canonical


class UnderstatClient:
    """Appelle les endpoints JSON internes d'understat.com."""

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
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "X-Requested-With": "XMLHttpRequest",
            }
        )

    def _get_json(self, path: str) -> dict[str, Any]:
        """GET un endpoint JSON interne d'understat.com."""
        url = f"{self._base_url}/{path}"
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    def get_team_matches(self, team_slug: str, season: str) -> list[dict[str, Any]]:
        """Liste brute des matches d'une equipe pour une saison.

        Args:
            team_slug: identifiant understat de l'equipe (ex: "Paris_Saint_Germain").
            season: annee de debut de saison (ex: "2026" pour 2026/2027).
        """
        data = self._get_json(f"getTeamData/{team_slug}/{season}")
        matches: list[dict[str, Any]] = data["dates"]
        logger.info(
            "get_team_matches: %d match(es) (team=%s, season=%s)",
            len(matches),
            team_slug,
            season,
        )
        return matches

    def get_team_players(self, team_slug: str, season: str) -> list[dict[str, Any]]:
        """Liste brute des joueurs d'une equipe pour une saison, avec leur poste.

        Meme endpoint que `get_team_matches` (`getTeamData/{team}/{season}`,
        cle "players" plutot que "dates") : un appel HTTP separe (le contenu
        n'est pas mis en cache entre les deux methodes), mais reste un cout
        negligeable a l'echelle du backfill historique (~12 saisons).

        Le champ "position" est une chaine de codes espaces (ex: "F M S") :
        D=Defenseur, M=Milieu, F=Attaquant, GK=Gardien, S=est aussi entre en
        tant que remplacant (pas un poste a proprement parler). Ordonnes
        approximativement par temps de jeu decroissant a ce poste.
        """
        data = self._get_json(f"getTeamData/{team_slug}/{season}")
        players: list[dict[str, Any]] = data["players"]
        logger.info(
            "get_team_players: %d joueur(s) (team=%s, season=%s)",
            len(players),
            team_slug,
            season,
        )
        return players

    def get_match_shots(self, match_id: int, team_name: str | None = None) -> list[ShotEvent]:
        """Tous les tirs d'un match, cote domicile + exterieur.

        Args:
            match_id: id understat du match.
            team_name: si fourni, ne garde que les tirs de cette equipe.
        """
        data = self._get_json(f"getMatchData/{match_id}")
        raw: dict[str, list[dict[str, Any]]] = data["shots"]

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
                        team_id=zlib.crc32(shooter_team.encode("utf-8")),
                        minute=int(shot["minute"]),
                        second=0,
                        location=Location(
                            x=float(shot["X"]) * _PITCH_LENGTH,
                            y=float(shot["Y"]) * _PITCH_WIDTH,
                        ),
                        body_part=_normalize_category(
                            shot.get("shotType", "Unknown"), _BODY_PART_TO_CANONICAL, "shotType"
                        ),
                        shot_type=_normalize_category(
                            shot.get("situation", "Unknown"), _SITUATION_TO_CANONICAL, "situation"
                        ),
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
