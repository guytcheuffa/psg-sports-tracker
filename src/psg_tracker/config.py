"""Configuration centralisee du projet (lue depuis .env)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parametres globaux de l'application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    duckdb_path: Path = Path("data/processed/psg_tracker.duckdb")
    log_level: str = "INFO"
    psg_team_id: int = 771  # StatsBomb team id pour le PSG


settings = Settings()
