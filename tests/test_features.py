"""Tests unitaires du feature engineering xG."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from psg_tracker.features.engineering import (
    GOAL_WIDTH,
    add_distance_to_goal,
    add_is_header,
    add_preferred_foot_feature,
    add_shot_angle,
    apply_preferred_foot_feature,
    build_feature_matrix,
    compute_preferred_foot_map,
)


def _shots_df(**overrides: object) -> pd.DataFrame:
    base = {
        "player_id": [1],
        "loc_x": [110.0],
        "loc_y": [40.0],
        "body_part": ["Right Foot"],
        "shot_type": ["Open Play"],
        "is_goal": [False],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_add_distance_to_goal_penalty_spot() -> None:
    # Le point de penalty StatsBomb est a (108, 40) -> 12 yards du but.
    df = _shots_df(loc_x=[108.0], loc_y=[40.0])

    result = add_distance_to_goal(df)

    assert result["distance_to_goal"].iloc[0] == pytest.approx(12.0)


def test_add_distance_to_goal_does_not_mutate_input() -> None:
    df = _shots_df()
    add_distance_to_goal(df)

    assert "distance_to_goal" not in df.columns


def test_add_shot_angle_centered_shot_wider_than_offcenter_shot() -> None:
    centered = add_shot_angle(_shots_df(loc_x=[110.0], loc_y=[40.0]))
    off_center = add_shot_angle(_shots_df(loc_x=[110.0], loc_y=[60.0]))

    assert centered["shot_angle_rad"].iloc[0] > off_center["shot_angle_rad"].iloc[0]


def test_add_shot_angle_closer_shot_has_wider_angle() -> None:
    close = add_shot_angle(_shots_df(loc_x=[115.0], loc_y=[40.0]))
    far = add_shot_angle(_shots_df(loc_x=[90.0], loc_y=[40.0]))

    assert close["shot_angle_rad"].iloc[0] > far["shot_angle_rad"].iloc[0]


def test_add_shot_angle_stays_within_valid_range() -> None:
    # Tir tres proche et centre : l'angle doit rester dans (0, pi), jamais negatif.
    df = _shots_df(loc_x=[119.0], loc_y=[40.0])

    result = add_shot_angle(df)

    angle = result["shot_angle_rad"].iloc[0]
    assert 0.0 < angle < math.pi


def test_add_shot_angle_known_value_on_goal_line_extended() -> None:
    # A x=120 (ligne de but), dx=0 -> angle nul par symetrie (arctan2(0, negatif) = pi,
    # mais physiquement un tir depuis la ligne de but exacte est un cas degenere ;
    # on verifie plutot une valeur non triviale a distance raisonnable et sur l'axe).
    df = _shots_df(loc_x=[112.0], loc_y=[40.0])
    result = add_shot_angle(df)

    dx = 120.0 - 112.0
    dy = 0.0
    expected = math.atan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2)
    assert result["shot_angle_rad"].iloc[0] == pytest.approx(expected)


def test_add_is_header_flags_head_shots() -> None:
    df = pd.concat(
        [_shots_df(body_part=["Head"]), _shots_df(body_part=["Right Foot"])],
        ignore_index=True,
    )

    result = add_is_header(df)

    assert result["is_header"].tolist() == [True, False]


def test_add_preferred_foot_feature_flags_majority_foot() -> None:
    df = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "loc_x": [110.0, 105.0, 100.0, 110.0],
            "loc_y": [40.0, 40.0, 40.0, 40.0],
            "body_part": ["Right Foot", "Right Foot", "Left Foot", "Left Foot"],
            "shot_type": ["Open Play"] * 4,
        }
    )

    result = add_preferred_foot_feature(df)

    # joueur 1 : pied fort = Right Foot (2/3 tirs) -> [True, True, False]
    # joueur 2 : un seul tir, forcement "pied dominant" -> True
    assert result["is_strong_foot"].tolist() == [True, True, False, True]


def test_compute_preferred_foot_map_returns_majority_foot_per_player() -> None:
    df = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "body_part": ["Right Foot", "Right Foot", "Left Foot", "Left Foot"],
        }
    )

    foot_map = compute_preferred_foot_map(df)

    assert foot_map == {1: "Right Foot", 2: "Left Foot"}


def test_apply_preferred_foot_feature_uses_given_map_not_own_data() -> None:
    # Le mapping vient d'un train set distinct : le joueur 1 y est gaucher,
    # meme si dans CE DataFrame (le "test set") il ne tire que du pied droit.
    foot_map = {1: "Left Foot"}
    df = pd.DataFrame(
        {
            "player_id": [1, 1],
            "body_part": ["Right Foot", "Left Foot"],
        }
    )

    result = apply_preferred_foot_feature(df, foot_map)

    assert result["is_strong_foot"].tolist() == [False, True]


def test_apply_preferred_foot_feature_defaults_to_false_for_unmapped_player() -> None:
    # Joueur absent du mapping (ex. n'apparait que dans le test set) : pas
    # d'erreur, is_strong_foot=False par defaut plutot qu'un NaN qui casserait
    # le modele en aval.
    df = pd.DataFrame({"player_id": [99], "body_part": ["Right Foot"]})

    result = apply_preferred_foot_feature(df, foot_map={})

    assert result["is_strong_foot"].tolist() == [False]


def test_build_feature_matrix_with_foot_map_does_not_use_own_data_for_leakage() -> None:
    # Le "test set" ci-dessous ne contient qu'un tir du pied gauche pour le
    # joueur 1 ; s'il fittait son propre mapping (comme add_preferred_foot_feature
    # seule le ferait), ce tir serait etiquete "pied fort". Avec un mapping
    # externe (fit sur un train ou ce joueur est droitier), il doit rester
    # etiquete "pied faible" : c'est exactement le scenario de fuite train/test
    # que ce paramametre `foot_map` est cense empecher (cf. scripts/train.py).
    foot_map = {1: "Right Foot"}
    df = pd.DataFrame(
        {
            "player_id": [1],
            "loc_x": [110.0],
            "loc_y": [40.0],
            "body_part": ["Left Foot"],
            "shot_type": ["Open Play"],
            "is_goal": [False],
        }
    )

    result = build_feature_matrix(df, foot_map=foot_map)

    assert result["is_strong_foot"].tolist() == [False]


def test_build_feature_matrix_returns_all_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "player_id": [1, 2],
            "loc_x": [110.0, 100.0],
            "loc_y": [40.0, 30.0],
            "body_part": ["Right Foot", "Head"],
            "shot_type": ["Open Play", "Corner"],
            "is_goal": [True, False],
        }
    )

    result = build_feature_matrix(df)

    for col in ["distance_to_goal", "shot_angle_rad", "is_header", "is_strong_foot"]:
        assert col in result.columns
    assert "situation_Open Play" in result.columns
    assert "situation_Corner" in result.columns
    # les colonnes brutes et la cible sont preservees
    assert "is_goal" in result.columns
