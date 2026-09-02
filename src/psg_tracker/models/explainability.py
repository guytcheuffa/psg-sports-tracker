"""Explicabilite des predictions xG via SHAP."""

from __future__ import annotations

import pandas as pd
import shap

from psg_tracker.models.xg_model import XGModel


def compute_shap_values(model: XGModel, features: pd.DataFrame) -> pd.DataFrame:
    """Calcule la contribution SHAP de chaque feature pour un batch de tirs.

    Les valeurs sont dans l'espace des log-odds (sortie "margin" native du
    `TreeExplainer`), pas la probabilite finale : un signe positif pousse la
    prediction vers "plus susceptible d'etre un but", un signe negatif vers
    l'inverse. C'est le mode par defaut et le plus robuste de `TreeExplainer`
    pour les modeles d'arbres (le mode "probability" necessite une
    perturbation interventionnelle plus couteuse et un jeu de reference).

    Args:
        model: `XGModel` deja entraine ou charge.
        features: DataFrame de tirs. Reindexe automatiquement sur
            `model.feature_columns`, comme `XGModel.predict_proba`.

    Returns:
        DataFrame indexe comme `features`, une colonne par feature de
        `model.feature_columns` contenant sa contribution SHAP pour chaque
        tir.
    """
    x = features.reindex(columns=model.feature_columns, fill_value=0)
    explainer = shap.TreeExplainer(model.raw_model)
    shap_values = explainer.shap_values(x)

    return pd.DataFrame(shap_values, index=features.index, columns=model.feature_columns)


def explain_shot(model: XGModel, features: pd.DataFrame, shot_index: object) -> pd.Series:
    """Contributions SHAP d'un seul tir, triees par importance absolue decroissante.

    Pratique pour justifier une prediction individuelle dans le dashboard
    ("pourquoi ce xG pour ce tir precis ?").

    Args:
        model: `XGModel` deja entraine ou charge.
        features: DataFrame contenant (au moins) la ligne `shot_index`.
        shot_index: index (au sens pandas) du tir a expliquer.
    """
    shap_df = compute_shap_values(model, features.loc[[shot_index]])
    contributions = shap_df.loc[shot_index]
    return contributions.reindex(contributions.abs().sort_values(ascending=False).index)
