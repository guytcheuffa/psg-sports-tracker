#!/usr/bin/env python3
"""Entrainement du modele xG sur les donnees reelles stockees en DuckDB.

Combine les deux sources ingerees par `scripts/ingest.py` :
    - StatsBomb Open Data : corpus historique (95 matches Ligue 1 PSG,
      saisons 2015/16, 2021/22, 2022/23), qualite de donnee elevee.
    - Understat (scraping) : saison en cours, volume encore faible en debut
      de saison. Les features utilisees (distance, angle, type de tir) ne
      dependent pas de l'identite du joueur, donc un eventuel probleme de
      matching joueur cote source ne biaise pas ces colonnes-la.

Usage:
    python scripts/train.py
    python scripts/train.py --db-path data/processed/psg_tracker.duckdb \
        --output data/models/xg_model.json --test-size 0.2

Procede en deux temps : (1) split train/test pour evaluer honnetement le
modele sur des tirs non vus, (2) reentrainement sur 100% des donnees pour le
modele final sauvegarde (usage standard en production : on ne jette pas de
signal une fois la performance validee).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from psg_tracker.config import settings
from psg_tracker.features.engineering import build_feature_matrix
from psg_tracker.models.xg_model import XGModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = _REPO_ROOT / settings.duckdb_path
_DEFAULT_OUTPUT = _REPO_ROOT / settings.model_path

_SHOTS_QUERY = """
    SELECT source, match_id, event_id, player_id, loc_x, loc_y,
           body_part, shot_type, is_goal
    FROM shots
"""


def load_shots(db_path: Path) -> pd.DataFrame:
    """Charge tous les tirs (toutes sources confondues) depuis DuckDB."""
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        df = conn.execute(_SHOTS_QUERY).fetchdf()
    finally:
        conn.close()
    logger.info(
        "load_shots: %d tir(s) charges (%s)",
        len(df),
        dict(df["source"].value_counts()) if not df.empty else {},
    )
    return df


def evaluate(model: XGModel, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Calcule les metriques standard d'un modele xG sur un jeu de test."""
    predictions = model.predict_proba(features)
    return {
        "roc_auc": float(roc_auc_score(target, predictions)),
        "log_loss": float(log_loss(target, predictions)),
        "brier_score": float(brier_score_loss(target, predictions)),
        "n_test": int(len(target)),
        "goal_rate": float(target.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=_DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    raw_shots = load_shots(args.db_path)
    if raw_shots.empty:
        raise SystemExit(f"Aucun tir trouve dans {args.db_path} (lancer scripts/ingest.py d'abord)")

    features = build_feature_matrix(raw_shots)
    target = features["is_goal"].astype(int)

    train_df, test_df, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    eval_model = XGModel()
    eval_model.train(train_df, y_train)
    metrics = evaluate(eval_model, test_df, y_test)
    logger.info("Evaluation (hold-out %.0f%%): %s", args.test_size * 100, metrics)

    final_model = XGModel()
    final_model.train(features, target)
    final_model.save(args.output)

    logger.info(
        "Entrainement termine: %d tir(s) au total (%d train / %d test), modele sauvegarde: %s",
        len(features),
        len(train_df),
        len(test_df),
        args.output,
    )


if __name__ == "__main__":
    main()
