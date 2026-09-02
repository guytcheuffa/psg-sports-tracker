"""Tests unitaires de l'explicabilite SHAP (calcul reel, pas de mock)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from psg_tracker.models.explainability import compute_shap_values, explain_shot
from psg_tracker.models.xg_model import XGModel


def test_compute_shap_values_shape_matches_features(
    trained_model: tuple[XGModel, pd.DataFrame],
) -> None:
    model, features = trained_model

    shap_df = compute_shap_values(model, features)

    assert shap_df.shape == (len(features), len(model.feature_columns))
    assert list(shap_df.columns) == model.feature_columns
    assert list(shap_df.index) == list(features.index)


def test_shap_values_plus_base_value_reconstruct_margin_prediction(
    trained_model: tuple[XGModel, pd.DataFrame],
) -> None:
    """Verification de coherence SHAP standard : somme des contributions +
    valeur de base = prediction brute (log-odds) du modele pour chaque tir."""
    model, features = trained_model
    x = features[model.feature_columns]

    shap_df = compute_shap_values(model, features)
    margin_predictions = model.raw_model.predict(x, output_margin=True)

    import shap as shap_lib

    explainer = shap_lib.TreeExplainer(model.raw_model)
    reconstructed = shap_df.sum(axis=1).to_numpy() + explainer.expected_value

    # tolerance large car les arbres XGBoost stockent leurs valeurs de feuille
    # en float32 en interne : avec 200 estimateurs, l'erreur d'arrondi
    # cumulee sur la somme des contributions atteint facilement 1e-3.
    np.testing.assert_allclose(reconstructed, margin_predictions, atol=1e-2)


def test_explain_shot_is_sorted_by_absolute_contribution(
    trained_model: tuple[XGModel, pd.DataFrame],
) -> None:
    model, features = trained_model
    shot_index = features.index[0]

    contributions = explain_shot(model, features, shot_index)

    abs_values = contributions.abs().to_numpy()
    assert (abs_values[:-1] >= abs_values[1:]).all()
    assert set(contributions.index) == set(model.feature_columns)


def test_explain_shot_only_uses_requested_row(
    trained_model: tuple[XGModel, pd.DataFrame],
) -> None:
    model, features = trained_model
    first_index = features.index[0]
    second_index = features.index[1]

    contributions_first = explain_shot(model, features, first_index)
    contributions_second = explain_shot(model, features, second_index)

    # deux tirs differents ont (tres probablement) des contributions differentes
    assert not contributions_first.equals(contributions_second)
