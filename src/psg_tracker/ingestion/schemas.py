"""Contrats de donnees (schemas typees) pour les events StatsBomb bruts."""

from __future__ import annotations

from pydantic import BaseModel


class Location(BaseModel):
    """Coordonnees (x, y) sur le terrain, referentiel StatsBomb (120x80)."""

    x: float
    y: float


class ShotEvent(BaseModel):
    """Un tir extrait des event data StatsBomb."""

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


class MatchSummary(BaseModel):
    """Metadonnees d'un match PSG."""

    match_id: int
    match_date: str
    home_team: str
    away_team: str
    competition: str
    season: str
