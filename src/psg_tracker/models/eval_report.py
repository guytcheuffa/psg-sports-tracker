"""Rapport d'evaluation hold-out du modele xG (metriques + predictions brutes).

Sauvegarde par `scripts/train.py` a cote du modele (`<model_path>.eval.json`),
charge par le dashboard (onglet "Diagnostic du modele") pour tracer une
courbe ROC et une courbe de calibration a partir des VRAIES predictions
hold-out.

Point important : le modele final sauvegarde (`xg_model.json`) est
reentraine sur 100% des donnees (cf. docstring de `scripts/train.py`) - il
n'y a donc plus de "tirs de test" pour lui une fois ce fichier ecrit. Le
dashboard ne peut pas recalculer un diagnostic honnete a la volee avec ce
modele final (ce serait circulaire : evaluer un modele sur des donnees
qu'il a vues a l'entrainement). Ce rapport capture donc les predictions du
modele *intermediaire* (`eval_model` dans `scripts/train.py`, entraine sur
80% seulement) au moment de l'entrainement, pour que le dashboard affiche
un diagnostic reellement hold-out.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast


class EvalReport(TypedDict):
    """Contenu du fichier `<model_path>.eval.json`."""

    metrics: dict[str, float]
    y_true: list[int]
    y_pred: list[float]
    test_size: float
    random_state: int
    trained_at: str


def eval_report_path(model_path: Path) -> Path:
    """Chemin du rapport d'evaluation associe a un modele (convention `.eval.json`)."""
    return model_path.with_name(model_path.name + ".eval.json")


def save_eval_report(
    model_path: Path,
    metrics: dict[str, float],
    y_true: Iterable[int],
    y_pred: Iterable[float],
    test_size: float,
    random_state: int,
) -> Path:
    """Sauvegarde le rapport d'evaluation hold-out a cote du modele. Renvoie le chemin ecrit."""
    report: EvalReport = {
        "metrics": metrics,
        "y_true": [int(v) for v in y_true],
        "y_pred": [float(v) for v in y_pred],
        "test_size": test_size,
        "random_state": random_state,
        # timezone.utc plutot que l'alias datetime.UTC (3.11+, cf. pyproject) : identique
        # au runtime, mais reste executable/verifiable sur un python3.10 local si besoin.
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
    }
    path = eval_report_path(model_path)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def load_eval_report(model_path: Path) -> EvalReport | None:
    """Charge le rapport d'evaluation associe a un modele, ou None s'il n'existe pas encore."""
    path = eval_report_path(model_path)
    if not path.exists():
        return None
    return cast("EvalReport", json.loads(path.read_text(encoding="utf-8")))
