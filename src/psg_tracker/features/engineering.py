"""Feature engineering pour le modele xG (partage train/inference).

S'applique a un DataFrame de tirs au format canonique (colonnes issues de
`ShotEvent`/table `shots` : loc_x, loc_y, body_part, shot_type, player_id,
...), quelle que soit la source d'origine (StatsBomb ou Understat) puisque
les coordonnees sont deja normalisees vers le referentiel StatsBomb 120x80
a l'ingestion.

Note de portee (Jour 2) : le "pied preferentiel" et le "type de passe cle"
mentionnes dans le plan initial necessitent des donnees non capturees par le
schema actuel (`key_pass_id` StatsBomb, historique de tirs par joueur).
`add_preferred_foot_feature`/`compute_preferred_foot_map` ci-dessous en
fournissent une premiere version (mode du pied sur le dataset fourni). Le
mapping doit etre calcule sur le train uniquement puis applique au test pour
eviter une fuite train -> test (cf. `compute_preferred_foot_map` +
`apply_preferred_foot_feature`, utilises ainsi dans `scripts/train.py`). Le
"type de passe cle" est laisse pour une iteration ulterieure (necessite de
joindre les events de type Pass).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

# Referentiel StatsBomb (pitch 120x80 yards). Le but adverse est sur la ligne
# x=120, centre en y=40, largeur reelle ~8 yards (7.32m).
GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_WIDTH = 8.0

_FOOT_BODY_PARTS = {"Right Foot", "Left Foot"}

# Features de base produites par build_feature_matrix, utilisables telles
# quelles par un modele (numeriques/booleennes, pas d'encodage requis).
BASE_FEATURE_COLUMNS = [
    "distance_to_goal",
    "shot_angle_rad",
    "is_header",
    "is_strong_foot",
]


def add_distance_to_goal(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `distance_to_goal` : distance euclidienne au centre des cages.

    N'effectue pas de mutation sur `df` (retourne une copie).
    """
    result = df.copy()
    result["distance_to_goal"] = np.sqrt(
        (GOAL_X - result["loc_x"]) ** 2 + (GOAL_Y - result["loc_y"]) ** 2
    )
    return result


def add_shot_angle(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `shot_angle_rad` : angle (radians) sous lequel le tireur voit le but.

    Formule standard (cf. Soccermatics/FriendsOfTracking) : angle sous-tendu
    par les deux poteaux depuis le point de tir. `arctan2` est utilise plutot
    que `arctan` + correction manuelle de signe, ce qui gere nativement les
    tirs tres proches du but (angle > 90 degres) sans cas particulier.
    """
    result = df.copy()
    dx = GOAL_X - result["loc_x"]
    dy = (result["loc_y"] - GOAL_Y).abs()
    denominator = dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2
    result["shot_angle_rad"] = np.arctan2(GOAL_WIDTH * dx, denominator)
    return result


def add_is_header(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `is_header` : booleen, tir de la tete."""
    result = df.copy()
    result["is_header"] = result["body_part"] == "Head"
    return result


def compute_preferred_foot_map(df: pd.DataFrame) -> dict[int, str]:
    """Calcule le pied dominant (Right Foot/Left Foot) par `player_id`, a partir de CE DataFrame.

    Etape "fit" separee de `apply_preferred_foot_feature` ("transform") pour
    permettre un pipeline sans fuite d'information train -> test : calculer
    ce mapping sur le train uniquement, puis l'appliquer identiquement au
    train et au test (cf. `scripts/train.py`, qui splitte les tirs bruts
    *avant* le feature engineering pour cette raison precise).
    """
    foot_shots = df[df["body_part"].isin(_FOOT_BODY_PARTS)]
    preferred_foot = foot_shots.groupby("player_id")["body_part"].agg(
        lambda feet: feet.value_counts().idxmax()
    )
    return cast("dict[int, str]", preferred_foot.to_dict())


def apply_preferred_foot_feature(df: pd.DataFrame, foot_map: dict[int, str]) -> pd.DataFrame:
    """Ajoute `is_strong_foot` a partir d'un mapping `player_id -> pied dominant` deja calcule.

    Un joueur absent de `foot_map` (ex. n'apparait que dans le jeu de test)
    donne `is_strong_foot=False` par defaut (`fillna`) plutot qu'une erreur.
    """
    result = df.copy()
    result["is_strong_foot"] = (
        result["player_id"].map(foot_map) == result["body_part"]
    ).fillna(False)
    return result


def add_preferred_foot_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `is_strong_foot` : le tir a-t-il ete tire du pied dominant du joueur.

    Version "fit=transform sur le meme DataFrame" (calcule le mapping *et*
    l'applique sur `df`) : correcte pour un usage sans distinction train/test
    a proteger (modele final entraine sur 100% des donnees, inference en
    production/dashboard). Pour une evaluation train/test honnete, ne pas
    utiliser cette fonction directement : utiliser `compute_preferred_foot_map`
    sur le train puis `apply_preferred_foot_feature` sur train et test
    separement, comme le fait `scripts/train.py`.
    """
    return apply_preferred_foot_feature(df, compute_preferred_foot_map(df))


def build_feature_matrix(df: pd.DataFrame, foot_map: dict[int, str] | None = None) -> pd.DataFrame:
    """Pipeline complet de feature engineering pour l'entrainement/l'inference.

    Args:
        df: DataFrame de tirs avec au minimum les colonnes loc_x, loc_y,
            body_part, shot_type, player_id.
        foot_map: mapping `player_id -> pied dominant` deja calcule (fit sur
            un train set separe). Si `None` (defaut), le mapping est calcule
            directement sur `df` (fit=transform) - correct pour l'inference
            ou l'entrainement du modele final sur 100% des donnees, mais PAS
            pour une evaluation train/test (cf. `add_preferred_foot_feature`).

    Returns:
        DataFrame enrichi des features numeriques (distance_to_goal,
        shot_angle_rad, is_header, is_strong_foot) et de `shot_type`
        one-hot encode. Les colonnes brutes de coordonnees sont conservees.
    """
    features = add_distance_to_goal(df)
    features = add_shot_angle(features)
    features = add_is_header(features)
    features = (
        apply_preferred_foot_feature(features, foot_map)
        if foot_map is not None
        else add_preferred_foot_feature(features)
    )

    shot_type_dummies = pd.get_dummies(features["shot_type"], prefix="situation")
    features = pd.concat([features, shot_type_dummies], axis=1)

    return features


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes numeriques/booleennes exploitables telles quelles par un modele.

    Combine `BASE_FEATURE_COLUMNS` et les colonnes one-hot dynamiques
    `situation_*` (dependent des categories de `shot_type` presentes dans le
    DataFrame). Utilise a la fois a l'entrainement (pour figer la liste) et
    a l'inference (via `DataFrame.reindex`, cf. `XGModel.predict_proba`) pour
    garantir un schema de features identique entre les deux.
    """
    dynamic = [c for c in df.columns if c.startswith("situation_")]
    return [c for c in BASE_FEATURE_COLUMNS if c in df.columns] + dynamic
