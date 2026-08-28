"""Feature engineering pour le modele xG (partage train/inference)."""

from __future__ import annotations

import pandas as pd

GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_WIDTH = 8.0


def add_distance_to_goal(df: pd.DataFrame) -> pd.DataFrame:
    """Distance euclidienne entre le point de tir et le centre des cages."""
    raise NotImplementedError


def add_shot_angle(df: pd.DataFrame) -> pd.DataFrame:
    """Angle de tir relatif (ouverture vers les poteaux)."""
    raise NotImplementedError


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering pour l'entrainement/l'inference."""
    raise NotImplementedError
