"""Chargement des donnees (DuckDB) et du modele xG pour le dashboard, avec cache Streamlit.

Separe du reste de `psg_tracker` (qui reste utilisable hors Streamlit, ex.
`scripts/train.py`) : ce module n'ajoute que la couche de cache/presentation.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import shap
import streamlit as st

from psg_tracker.features.engineering import build_feature_matrix
from psg_tracker.models.xg_model import XGModel

_SHOTS_QUERY = """
    SELECT
        s.event_id, s.match_id, s.source, s.player_id, s.player_name,
        s.minute, s.loc_x, s.loc_y, s.body_part, s.shot_type, s.outcome,
        s.is_goal,
        m.match_date, m.home_team, m.away_team, m.competition, m.season
    FROM shots s
    JOIN matches m ON s.source = m.source AND s.match_id = m.match_id
    ORDER BY m.match_date, s.minute
"""


@st.cache_data(show_spinner="Chargement des tirs depuis DuckDB...")
def load_shots_with_xg(db_path: str, model_path: str) -> pd.DataFrame:
    """Charge tous les tirs (toutes sources) et ajoute la colonne `xg_pred`.

    Args:
        db_path: chemin du fichier DuckDB (str pour rester hashable par
            `st.cache_data`, `Path` n'est pas garanti hashable selon la
            version de Streamlit).
        model_path: chemin du modele xG sauvegarde (`XGModel.save`).

    Returns:
        DataFrame issu de `_SHOTS_QUERY`, enrichi des features de
        `build_feature_matrix` et d'une colonne `xg_pred` (probabilite de
        but predite par le modele charge).
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        shots = conn.execute(_SHOTS_QUERY).fetchdf()
    finally:
        conn.close()

    if shots.empty:
        return shots

    features = build_feature_matrix(shots)

    model = XGModel()
    model.load(Path(model_path))
    features["xg_pred"] = model.predict_proba(features)

    return features


@st.cache_resource(show_spinner="Chargement du modele xG...")
def load_model(model_path: str) -> XGModel:
    """Charge (et met en cache) le modele xG entraine par `scripts/train.py`."""
    model = XGModel()
    model.load(Path(model_path))
    return model


@st.cache_resource(show_spinner="Preparation de l'explicabilite SHAP...")
def get_shap_explainer(model_path: str) -> shap.TreeExplainer:
    """Construit (une seule fois) le `TreeExplainer` SHAP pour le modele charge.

    Optimisation : `shap.TreeExplainer(...)` parcourt tout l'ensemble
    d'arbres XGBoost a la construction (non-negligeable pour 200 arbres).
    Sans ce cache, `psg_tracker.models.explainability.explain_shot` en
    reconstruit un nouveau a chaque interaction (ex. changement de tir
    selectionne), donc a chaque rerun du script Streamlit. Mis en cache une
    fois pour toute la session ici, puis reutilise directement (sans passer
    par `explain_shot`, qui reste la version "simple" pour un usage hors
    dashboard/notebook).
    """
    model = load_model(model_path)
    return shap.TreeExplainer(model.raw_model)
