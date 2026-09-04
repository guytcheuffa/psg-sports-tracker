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

# Meme joueur, nom different selon la source : StatsBomb utilise le nom
# complet a l'etat civil, Understat le nom d'usage/media (ex. "Achraf
# Hakimi Mouh" vs "Achraf Hakimi"). Sans harmonisation, un meme joueur
# apparait deux fois dans le classement et le selecteur de profil.
#
# Recense par rapprochement de chaines (difflib.SequenceMatcher, seuil
# ~0.6, puis affine par difflib.get_close_matches contre les noms de
# `player_positions` pour les joueurs 100% StatsBomb qui ne creent pas de
# doublon visible mais ratent quand meme le rattachement du poste, ex.
# "Lionel Andrés Messi Cuccittini") sur les noms distincts de
# `shots.player_name`, verifie manuellement paire par paire pour ecarter
# les faux positifs (ex. "Gonçalo Ramos" vs "Sergio Ramos" : deux joueurs
# distincts qui partagent juste un nom de famille). Cible en general le nom
# d'usage Understat (plus court), sauf quand Understat omet un accent que
# StatsBomb a correctement (ex. "Jean-Kevin Augustin" -> "Jean-Kévin
# Augustin") : dans ce cas l'orthographe correcte l'emporte. A completer si
# l'ingestion de futures saisons revele de nouveaux doublons/rattachements
# manques (rejouer les deux rapprochements difflib sur les noms distincts
# en base).
_PLAYER_NAME_ALIASES: dict[str, str] = {
    "Achraf Hakimi Mouh": "Achraf Hakimi",
    "Ángel Fabián Di María Hernández": "Ángel Di María",
    "Carlos Soler Barragán": "Carlos Soler",
    "Danilo Luís Hélio Pereira": "Danilo Pereira",
    "David Luiz Moreira Marinho": "David Luiz",
    "Edinson Roberto Cavani Gómez": "Edinson Cavani",
    "Ezequiel Iván Lavezzi": "Ezequiel Lavezzi",
    "Fabián Ruiz Peña": "Fabián",
    "Gregory van der Wiel": "Gregory Van der Wiel",
    "Hervin Scicchitano Ongenda": "Hervin Ongenda",
    "Idrissa Gana Gueye": "Idrissa Gueye",
    "Javier Matías Pastore": "Javier Pastore",
    "Jean-Kevin Augustin": "Jean-Kévin Augustin",
    "Juan Bernat Velasco": "Juan Bernat",
    "Kylian Mbappe-Lottin": "Kylian Mbappé Lottin",
    "Leandro Daniel Paredes": "Leandro Paredes",
    "Lionel Andrés Messi Cuccittini": "Lionel Messi",
    "Lucas Rodrigues Moura da Silva": "Lucas Moura",
    "Marcos Aoás Corrêa": "Marquinhos",
    "Mauro Emanuel Icardi Rivero": "Mauro Icardi",
    "Maxwell Scherrer Cabelino Andrade": "Maxwell",
    "Neymar da Silva Santos Junior": "Neymar",
    "Nordi Mukiele Mulere": "Nordi Mukiele",
    "Pablo Sarabia García": "Pablo Sarabia",
    "Renato Júnior Luz Sanches": "Renato Sanches",
    "Sergio Ramos García": "Sergio Ramos",
    "Thiago Emiliano da Silva": "Thiago Silva",
    "Vitor Machado Ferreira": "Vitinha",
    "Warren Zaire Emery": "Warren Zaïre-Emery",
    "Zlatan Ibrahimovic": "Zlatan Ibrahimović",
}


# Poste principal par joueur, derive des codes Understat (seule source a
# fournir cette info - pas StatsBomb) : D=Defenseur, M=Milieu, F=Attaquant,
# GK=Gardien. "S" (aussi entre comme remplacant) n'est pas un poste et est
# ignore.
_POSITION_LABELS: dict[str, str] = {
    "GK": "Gardien",
    "D": "Défenseur",
    "M": "Milieu",
    "F": "Attaquant",
}

_POSITIONS_QUERY = "SELECT player_name, season, position_raw FROM player_positions"


def _primary_position(position_raw: str) -> str | None:
    """Poste principal a partir d'un code Understat (ex: "F M S" -> "Attaquant").

    Le premier code reel (hors "S") est retenu : Understat ordonne les
    codes approximativement par temps de jeu decroissant a ce poste.
    """
    for token in position_raw.split():
        label = _POSITION_LABELS.get(token)
        if label is not None:
            return label
    return None


def load_player_positions(db_path: str) -> pd.DataFrame:
    """Poste dominant par joueur (mode sur toutes les saisons), nom harmonise.

    Une ligne par (joueur, saison) en base ; agregee ici en un seul poste
    "dominant" par joueur (le plus frequent parmi les postes principaux
    saison par saison) - un joueur ne change pas assez souvent de poste
    pour justifier un filtre saison-par-saison-et-poste dans le dashboard.
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        raw = conn.execute(_POSITIONS_QUERY).fetchdf()
    finally:
        conn.close()

    if raw.empty:
        return pd.DataFrame(columns=["player_name", "position"])

    raw["player_name"] = raw["player_name"].replace(_PLAYER_NAME_ALIASES)
    raw["position"] = raw["position_raw"].map(_primary_position)
    raw = raw.dropna(subset=["position"])
    if raw.empty:
        return pd.DataFrame(columns=["player_name", "position"])

    return (
        raw.groupby("player_name")["position"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )


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

    shots["player_name"] = shots["player_name"].replace(_PLAYER_NAME_ALIASES)

    positions = load_player_positions(db_path)
    shots = shots.merge(positions, on="player_name", how="left")
    shots["position"] = shots["position"].fillna("Inconnu")

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
