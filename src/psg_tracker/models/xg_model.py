"""Entrainement et inference du modele xG (XGBoost, classification binaire)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb


class XGModel:
    """Wrapper autour d'un classifieur XGBoost pour la probabilite de but (xG)."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model: xgb.XGBClassifier | None = None
        self._model_path = model_path

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Entraine le modele sur les features/target fournis."""
        raise NotImplementedError

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Retourne la probabilite de but (xG) pour chaque tir."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Sauvegarde le modele entraine."""
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """Charge un modele pre-entraine."""
        raise NotImplementedError
