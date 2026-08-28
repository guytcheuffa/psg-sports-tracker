"""Configuration centralisee du projet (lue depuis .env)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parametres globaux de l'application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    duckdb_path: Path = Path("data/processed/psg_tracker.duckdb")
    log_level: str = "INFO"

    # StatsBomb Open Data : Ligue 1 2015/2016 (saison du titre PSG sous Blanc)
    # source: competitions.json du depot statsbomb/open-data
    ligue1_competition_id: int = 7
    ligue1_2015_16_season_id: int = 27
    psg_team_name: str = "Paris Saint-Germain"


settings = Settings()
