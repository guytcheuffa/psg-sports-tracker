"""Fixtures partagees entre modules de test (donnees synthetiques xG)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psg_tracker.features.engineering import build_feature_matrix
from psg_tracker.models.xg_model import XGModel


def synthetic_shots(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Genere des tirs synthetiques ou proximite/angle correlent avec le but.

    Les tirs proches et centres (grand angle, faible distance) sont plus
    souvent des buts, pour que le modele ait un signal reel a apprendre.
    Usage exclusivement pour les tests : ce ne sont pas des donnees PSG
    reelles (voir StatsBombClient/UnderstatClient pour la vraie ingestion).
    """
    rng = np.random.default_rng(seed)
    loc_x = rng.uniform(80.0, 119.0, size=n)
    loc_y = rng.uniform(20.0, 60.0, size=n)
    distance = np.sqrt((120.0 - loc_x) ** 2 + (40.0 - loc_y) ** 2)
    goal_probability = np.clip(1.0 - distance / 45.0, 0.02, 0.9)
    is_goal = rng.uniform(size=n) < goal_probability

    return pd.DataFrame(
        {
            "player_id": rng.integers(1, 5, size=n),
            "loc_x": loc_x,
            "loc_y": loc_y,
            "body_part": rng.choice(["Right Foot", "Left Foot", "Head"], size=n),
            "shot_type": rng.choice(["Open Play", "Corner", "Penalty"], size=n),
            "is_goal": is_goal,
        }
    )


@pytest.fixture
def trained_model() -> tuple[XGModel, pd.DataFrame]:
    """Un XGModel entraine sur un jeu de tirs synthetiques + son feature matrix."""
    shots = synthetic_shots()
    features = build_feature_matrix(shots)
    model = XGModel()
    model.train(features, features["is_goal"])
    return model, features
