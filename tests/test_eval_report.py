"""Tests unitaires du rapport d'evaluation hold-out (save/load, cf. app/data_loader.py)."""

from __future__ import annotations

from pathlib import Path

from psg_tracker.models.eval_report import eval_report_path, load_eval_report, save_eval_report


def test_eval_report_path_appends_eval_json_suffix() -> None:
    path = eval_report_path(Path("data/models/xg_model.json"))

    assert path == Path("data/models/xg_model.json.eval.json")


def test_save_and_load_eval_report_roundtrip(tmp_path: Path) -> None:
    model_path = tmp_path / "xg_model.json"
    metrics = {
        "roc_auc": 0.772,
        "log_loss": 0.355,
        "brier_score": 0.107,
        "n_test": 3,
        "goal_rate": 0.33,
    }

    saved_path = save_eval_report(
        model_path,
        metrics,
        y_true=[0, 1, 0],
        y_pred=[0.1, 0.6, 0.2],
        test_size=0.2,
        random_state=42,
    )

    assert saved_path == model_path.with_name("xg_model.json.eval.json")
    assert saved_path.exists()

    report = load_eval_report(model_path)

    assert report is not None
    assert report["metrics"] == metrics
    assert report["y_true"] == [0, 1, 0]
    assert report["y_pred"] == [0.1, 0.6, 0.2]
    assert report["test_size"] == 0.2
    assert report["random_state"] == 42
    assert report["trained_at"]  # non vide, format ISO


def test_load_eval_report_returns_none_when_missing(tmp_path: Path) -> None:
    model_path = tmp_path / "xg_model.json"

    report = load_eval_report(model_path)

    assert report is None
