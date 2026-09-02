"""Feature engineering pour le modele xG (partage train/inference).

S'applique a un DataFrame de tirs au format canonique (colonnes issues de
`ShotEvent`/table `shots` : loc_x, loc_y, body_part, shot_type, player_id,
...), quelle que soit la source d'origine (StatsBomb ou Understat) puisque
les coordonnees sont deja normalisees vers le referentiel StatsBomb 120x80
a l'ingestion.

Note de portee (Jour 2) : le "pied preferentiel" et le "type de passe cle"
mentionnes dans le plan initial necessitent des donnees non capturees par le
schema actuel (`key_pass_id` StatsBomb, historique de tirs par joueur).
`add_preferred_foot_feature` ci-dessous en fournit une premiere version
(mode du pied sur le dataset fourni) avec un avertissement explicite sur le
risque de fuite de donnees train/test si mal utilisee. Le "type de passe
cle" est laisse pour une iteration ulterieure (necessite de joindre les
events de type Pass).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Referentiel StatsBomb (pitch 120x80 yards). Le but adverse est sur la ligne
# x=120, centre en y=40, largeur reelle ~8 yards (7.32m).
GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_WIDTH = 8.0

_FOOT_BODY_PARTS = {"Right Foot", "Left Foot"}


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


def add_preferred_foot_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute `is_strong_foot` : le tir a-t-il ete tire du pied dominant du joueur.

    Le pied dominant est estime comme le pied (Right Foot/Left Foot) le plus
    frequent sur les tirs du joueur *dans le DataFrame fourni*. A appeler
    uniquement sur le jeu d'entrainement (ou en pipeline fit/transform separe)
    pour eviter toute fuite d'information de la periode de test vers
    l'entrainement.
    """
    result = df.copy()
    foot_shots = result[result["body_part"].isin(_FOOT_BODY_PARTS)]
    preferred_foot = foot_shots.groupby("player_id")["body_part"].agg(
        lambda feet: feet.value_counts().idxmax()
    )
    result["is_strong_foot"] = (
        result["player_id"].map(preferred_foot) == result["body_part"]
    ).fillna(False)
    return result


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering pour l'entrainement/l'inference.

    Args:
        df: DataFrame de tirs avec au minimum les colonnes loc_x, loc_y,
            body_part, shot_type, player_id.

    Returns:
        DataFrame enrichi des features numeriques (distance_to_goal,
        shot_angle_rad, is_header, is_strong_foot) et de `shot_type`
        one-hot encode. Les colonnes brutes de coordonnees sont conservees.
    """
    features = add_distance_to_goal(df)
    features = add_shot_angle(features)
    features = add_is_header(features)
    features = add_preferred_foot_feature(features)

    shot_type_dummies = pd.get_dummies(features["shot_type"], prefix="situation")
    features = pd.concat([features, shot_type_dummies], axis=1)

    return features
