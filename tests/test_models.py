"""Tests unitaires du modele xG (entrainement XGBoost reel, donnees synthetiques)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psg_tracker.features.engineering import build_feature_matrix
from psg_tracker.models.xg_model import XGModel


def _synthetic_shots(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Genere des tirs synthetiques ou proximite/angle correlent avec le but.

    Les tirs proches et centres (grand angle, faible distance) sont plus
    souvent des buts, pour que le modele ait un signal reel a apprendre.
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
    shots = _synthetic_shots()
    features = build_feature_matrix(shots)
    model = XGModel()
    model.train(features, features["is_goal"])
    return model, features


def test_train_then_predict_proba_returns_valid_probabilities(
    trained_model: tuple[XGModel, pd.DataFrame],
) -> None:
    model, features = trained_model

    predictions = model.predict_proba(features)

    assert len(predictions) == len(features)
    assert predictions.between(0.0, 1.0).all()
    assert list(predictions.index) == list(features.index)


def test_model_learns_a_real_signal_not_random(
    trained_model: tuple[XGModel, pd.DataFrame],
) -> None:
    """Le xG moyen predit pour les vrais buts doit etre nettement superieur
    a celui des tirs manques, sinon le modele n'apprend rien d'utile."""
    model, features = trained_model
    predictions = model.predict_proba(features)

    avg_xg_goals = predictions[features["is_goal"]].mean()
    avg_xg_misses = predictions[~features["is_goal"]].mean()

    assert avg_xg_goals > avg_xg_misses


def test_predict_proba_before_train_raises() -> None:
    model = XGModel()
    features = build_feature_matrix(_synthetic_shots(n=5))

    with pytest.raises(RuntimeError, match="non entraine"):
        model.predict_proba(features)


def test_save_before_train_raises(tmp_path: Path) -> None:
    model = XGModel()

    with pytest.raises(RuntimeError, match="non entraine"):
        model.save(tmp_path / "model.json")


def test_save_and_load_roundtrip_produces_identical_predictions(
    trained_model: tuple[XGModel, pd.DataFrame], tmp_path: Path
) -> None:
    model, features = trained_model
    model_path = tmp_path / "xg_model.json"

    model.save(model_path)
    assert model_path.exists()
    assert model_path.with_suffix(".json.columns.json").exists()

    reloaded = XGModel()
    reloaded.load(model_path)

    original_predictions = model.predict_proba(features)
    reloaded_predictions = reloaded.predict_proba(features)

    pd.testing.assert_series_equal(original_predictions, reloaded_predictions)


def test_load_without_columns_file_raises(tmp_path: Path) -> None:
    model = XGModel()
    fake_model_path = tmp_path / "orphan.json"
    fake_model_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        model.load(fake_model_path)


def test_predict_proba_handles_missing_situation_column_at_inference(
    trained_model: tuple[XGModel, pd.DataFrame],
) -> None:
    """Un match ou une categorie de shot_type vue a l'entrainement (ex:
    'Penalty') est absente ne doit pas faire planter predict_proba : la
    colonne one-hot correspondante est simplement absente du DataFrame
    d'inference, et reindex() la complete a 0."""
    model, _ = trained_model

    inference_shots = pd.DataFrame(
        {
            "player_id": [1, 2],
            "loc_x": [110.0, 100.0],
            "loc_y": [40.0, 35.0],
            "body_part": ["Right Foot", "Head"],
            "shot_type": ["Open Play", "Open Play"],  # pas de "Corner"/"Penalty" ici
        }
    )
    inference_features = build_feature_matrix(inference_shots)

    predictions = model.predict_proba(inference_features)

    assert len(predictions) == 2
    assert predictions.between(0.0, 1.0).all()
