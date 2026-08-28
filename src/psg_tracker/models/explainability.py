"""Explicabilite des predictions xG via SHAP."""

from __future__ import annotations

import pandas as pd


def compute_shap_values(model: object, features: pd.DataFrame) -> pd.DataFrame:
    """Calcule les valeurs SHAP pour un batch de tirs."""
    raise NotImplementedError
