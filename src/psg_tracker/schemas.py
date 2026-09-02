"""Contrats de donnees (schemas typees), partages entre ingestion et storage.

Communs aux sources StatsBomb et Understat : les coordonnees sont toujours
normalisees vers le referentiel StatsBomb (pitch 120x80, origine en bas a
gauche) au moment de l'ingestion, meme pour les tirs Understat (normalises
0-1 nativement). Ca garantit que le feature engineering (distance/angle au
but) est independant de la source.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ShotSource = Literal["statsbomb", "understat"]


class Location(BaseModel):
    """Coordonnees (x, y) sur le terrain, referentiel StatsBomb (120x80)."""

    x: float
    y: float


class ShotEvent(BaseModel):
    """Un tir, normalise quelle que soit la source d'origine."""

    event_id: str
    match_id: int
    player_id: int
    player_name: str
    team_id: int
    minute: int
    second: int
    location: Location
    body_part: str
    shot_type: str
    outcome: str
    is_goal: bool
    source: ShotSource = "statsbomb"


class MatchSummary(BaseModel):
    """Metadonnees d'un match PSG."""

    match_id: int
    match_date: str
    home_team: str
    away_team: str
    competition: str
    season: str
    source: ShotSource = "statsbomb"
