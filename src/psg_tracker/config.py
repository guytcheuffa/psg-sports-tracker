"""Configuration centralisee du projet (lue depuis .env)."""

from datetime import date
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def current_understat_season(today: date | None = None) -> str:
    """Identifiant de saison Understat en cours (annee de debut de saison).

    Understat nomme ses saisons par l'annee calendaire de leur coup d'envoi
    (ex: la saison 2026/2027 est "2026"). La saison europeenne demarre mi-
    juillet/aout : avant juillet, on considere qu'on est encore sur la saison
    qui a commence l'annee precedente.
    """
    today = today or date.today()
    season_start_year = today.year if today.month >= 7 else today.year - 1
    return str(season_start_year)


class Settings(BaseSettings):
    """Parametres globaux de l'application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    duckdb_path: Path = Path("data/processed/psg_tracker.duckdb")
    log_level: str = "INFO"

    psg_team_name: str = "Paris Saint-Germain"

    # --- StatsBomb Open Data : corpus historique pour l'entrainement du xG ---
    # source: competitions.json du depot statsbomb/open-data
    # Ligue 1 (competition_id=7) : saisons 2015/2016, 2021/2022, 2022/2023
    # Champions League (competition_id=16) : 1999/2000 -> 2018/2019
    statsbomb_ligue1_competition_id: int = 7
    statsbomb_champions_league_competition_id: int = 16

    # --- Understat : source "live" pour la saison en cours (scraping) ---
    # Understat orthographie l'equipe differemment de StatsBomb (pas de
    # tiret) : "Paris Saint Germain" vs "Paris Saint-Germain". Garder un
    # champ dedie evite un filtrage silencieusement casse.
    understat_base_url: str = "https://understat.com"
    understat_psg_slug: str = "Paris_Saint_Germain"
    understat_team_name: str = "Paris Saint Germain"


settings = Settings()
