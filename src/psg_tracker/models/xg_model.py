"""Entrainement et inference du modele xG (XGBoost, classification binaire)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from psg_tracker.features.engineering import select_feature_columns

logger = logging.getLogger(__name__)

_DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "random_state": 42,
}


class XGModel:
    """Wrapper autour d'un classifieur XGBoost pour la probabilite de but (xG).

    La liste des colonnes de features est figee au moment de `train()` (via
    `select_feature_columns`) et persistee a cote du modele lors de `save()`.
    A l'inference, `predict_proba` reindexe le DataFrame fourni sur cette
    meme liste (colonnes manquantes -> 0, colonnes en trop ignorees) afin que
    le schema de features soit toujours identique entre entrainement et
    inference, meme si un match donne n'a pas toutes les categories de
    `shot_type` vues a l'entrainement.
    """

    def __init__(self, xgb_params: dict[str, Any] | None = None) -> None:
        self._model: xgb.XGBClassifier | None = None
        self._feature_columns: list[str] | None = None
        self._xgb_params = {**_DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    @property
    def feature_columns(self) -> list[str]:
        """Colonnes de features figees a l'entrainement (ou au chargement)."""
        if self._feature_columns is None:
            raise RuntimeError("XGModel non entraine : appeler train() ou load() d'abord.")
        return self._feature_columns

    @property
    def raw_model(self) -> xgb.XGBClassifier:
        """Le classifieur XGBoost sous-jacent (usage avance, ex: SHAP)."""
        if self._model is None:
            raise RuntimeError("XGModel non entraine : appeler train() ou load() d'abord.")
        return self._model

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Entraine le modele sur les features/target fournis.

        Args:
            features: DataFrame issu de `build_feature_matrix`.
            target: serie booleenne/0-1 alignee sur `features` (ex: `is_goal`).
        """
        self._feature_columns = select_feature_columns(features)
        x_train = features[self._feature_columns]

        model = xgb.XGBClassifier(**self._xgb_params)
        model.fit(x_train, target)
        self._model = model

        logger.info(
            "XGModel entraine: %d exemples, %d features (%s)",
            len(features),
            len(self._feature_columns),
            ", ".join(self._feature_columns),
        )

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Retourne la probabilite de but (xG) pour chaque tir.

        Leve `RuntimeError` si le modele n'a pas ete entraine ni charge.
        """
        if self._model is None or self._feature_columns is None:
            raise RuntimeError("XGModel non entraine : appeler train() ou load() d'abord.")

        x_pred = features.reindex(columns=self._feature_columns, fill_value=0)
        probabilities = self._model.predict_proba(x_pred)[:, 1]
        return pd.Series(probabilities, index=features.index, name="xg")

    def save(self, path: Path) -> None:
        """Sauvegarde le modele entraine ainsi que la liste des colonnes de features.

        Cree deux fichiers : `path` (modele XGBoost, format JSON natif) et
        `path` + `.columns.json` (liste des features, necessaire pour
        reconstruire un `predict_proba` coherent apres `load()`).
        """
        if self._model is None or self._feature_columns is None:
            raise RuntimeError("XGModel non entraine : rien a sauvegarder.")

        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

        columns_path = path.with_suffix(path.suffix + ".columns.json")
        columns_path.write_text(json.dumps(self._feature_columns), encoding="utf-8")

        logger.info("Modele sauvegarde: %s (+ %s)", path, columns_path.name)

    def load(self, path: Path) -> None:
        """Charge un modele pre-entraine et la liste des features associee."""
        columns_path = path.with_suffix(path.suffix + ".columns.json")
        if not columns_path.exists():
            raise FileNotFoundError(
                f"Fichier de colonnes introuvable: {columns_path} "
                "(le modele doit avoir ete sauvegarde via XGModel.save)"
            )

        model = xgb.XGBClassifier()
        model.load_model(str(path))
        self._model = model
        self._feature_columns = json.loads(columns_path.read_text(encoding="utf-8"))

        logger.info("Modele charge: %s (%d features)", path, len(self._feature_columns))
