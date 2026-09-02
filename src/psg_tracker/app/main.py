"""Point d'entree de l'application Streamlit : dashboard xG du PSG.

Trois blocs : shotmap (tous les tirs filtres, positionnes sur le dernier
tiers offensif), classement buts reels vs xG cumule par joueur (sur/sous-
performance), et explicabilite SHAP d'un tir individuel selectionne.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from psg_tracker.app.data_loader import load_model, load_shots_with_xg
from psg_tracker.app.pitch import half_pitch_figure
from psg_tracker.config import settings
from psg_tracker.models.explainability import explain_shot

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DB_PATH = (
    settings.duckdb_path
    if settings.duckdb_path.is_absolute()
    else _REPO_ROOT / settings.duckdb_path
)
_MODEL_PATH = (
    settings.model_path if settings.model_path.is_absolute() else _REPO_ROOT / settings.model_path
)

_OUTCOME_COLORS = {
    True: "#22c55e",  # but : vert
    False: "#94a3b8",  # tir non converti : gris neutre
}


def _check_prerequisites() -> bool:
    """Verifie que la base et le modele existent, avec un message d'aide sinon."""
    missing = []
    if not _DB_PATH.exists():
        missing.append(
            f"- Base DuckDB introuvable ({_DB_PATH}) : "
            "lancer `python scripts/ingest.py statsbomb --discover`"
        )
    if not _MODEL_PATH.exists():
        missing.append(
            f"- Modele xG introuvable ({_MODEL_PATH}) : lancer `python scripts/train.py`"
        )
    if missing:
        st.error("Pipeline incomplet :\n\n" + "\n".join(missing))
        return False
    return True


def _apply_filters(shots: pd.DataFrame) -> pd.DataFrame:
    """Filtres sidebar : source, competition, joueur. Retourne le sous-ensemble filtre."""
    st.sidebar.header("Filtres")

    sources = sorted(shots["source"].unique())
    selected_sources = st.sidebar.multiselect("Source", sources, default=sources)

    competitions = sorted(shots["competition"].unique())
    selected_competitions = st.sidebar.multiselect(
        "Competition", competitions, default=competitions
    )

    filtered = shots[
        shots["source"].isin(selected_sources) & shots["competition"].isin(selected_competitions)
    ]

    players = sorted(filtered["player_name"].unique())
    selected_players = st.sidebar.multiselect(
        "Joueur (vide = tous)", players, default=[]
    )
    if selected_players:
        filtered = filtered[filtered["player_name"].isin(selected_players)]

    return filtered


def _render_kpis(shots: pd.DataFrame) -> None:
    """Ligne de KPI : volume, buts reels vs xG cumule (calibration visuelle du modele)."""
    total_shots = len(shots)
    total_goals = int(shots["is_goal"].sum())
    total_xg = float(shots["xg_pred"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tirs", total_shots)
    col2.metric("Buts reels", total_goals)
    col3.metric("xG cumule", f"{total_xg:.1f}")
    col4.metric(
        "Buts - xG",
        f"{total_goals - total_xg:+.1f}",
        help="Positif = plus efficace que les positions de tir ne le suggerent.",
    )


def _render_shotmap(shots: pd.DataFrame) -> None:
    """Shotmap : position des tirs sur le dernier tiers, taille = xG, couleur = but/non-but."""
    fig = half_pitch_figure()

    for is_goal, group in shots.groupby("is_goal"):
        fig.add_trace(
            go.Scatter(
                x=group["loc_x"],
                y=group["loc_y"],
                mode="markers",
                name="But" if is_goal else "Sans but",
                marker={
                    "size": (group["xg_pred"] * 40 + 6),
                    "color": _OUTCOME_COLORS[bool(is_goal)],
                    "line": {"color": "white", "width": 1},
                    "opacity": 0.85,
                },
                customdata=group[["player_name", "minute", "shot_type", "xg_pred", "source"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Minute %{customdata[1]} - %{customdata[2]}<br>"
                    "xG: %{customdata[3]:.2f} (%{customdata[4]})<extra></extra>"
                ),
            )
        )

    fig.update_layout(height=520, legend={"orientation": "h", "y": -0.05})
    st.plotly_chart(fig, width='stretch')


def _render_leaderboard(shots: pd.DataFrame, top_n: int = 10) -> None:
    """Classement buts reels vs xG cumule par joueur, trie par volume de tirs."""
    by_player = (
        shots.groupby("player_name")
        .agg(tirs=("event_id", "count"), buts=("is_goal", "sum"), xg=("xg_pred", "sum"))
        .sort_values("tirs", ascending=False)
        .head(top_n)
    )
    by_player["buts_moins_xg"] = by_player["buts"] - by_player["xg"]
    by_player = by_player.round({"xg": 2, "buts_moins_xg": 2})

    fig = go.Figure()
    fig.add_trace(go.Bar(x=by_player.index, y=by_player["buts"], name="Buts reels"))
    fig.add_trace(go.Bar(x=by_player.index, y=by_player["xg"], name="xG cumule"))
    fig.update_layout(barmode="group", height=380, xaxis_tickangle=-30)
    st.plotly_chart(fig, width='stretch')
    st.dataframe(by_player, width='stretch')


def _render_shot_explainer(shots: pd.DataFrame, model_path: Path) -> None:
    """Explicabilite SHAP d'un tir individuel choisi dans la liste filtree."""
    if shots.empty:
        st.info("Aucun tir a expliquer avec les filtres actuels.")
        return

    model = load_model(str(model_path))

    labels = (
        shots["player_name"]
        + " - min "
        + shots["minute"].astype(str)
        + " ("
        + shots["match_date"].astype(str)
        + (shots["is_goal"].map({True: ", BUT", False: ""}))
        + ")"
    )
    choice = st.selectbox(
        "Choisir un tir", options=shots.index, format_func=lambda i: labels.loc[i]
    )

    contributions = explain_shot(model, shots, choice)
    predicted_xg = float(shots.loc[choice, "xg_pred"])

    st.metric("xG predit pour ce tir", f"{predicted_xg:.2f}")

    fig = go.Figure(
        go.Bar(
            x=contributions.values,
            y=contributions.index,
            orientation="h",
            marker={"color": ["#22c55e" if v > 0 else "#ef4444" for v in contributions.values]},
        )
    )
    fig.update_layout(
        height=320,
        xaxis_title="Contribution SHAP (log-odds)",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Valeurs en espace log-odds (sortie brute de l'arbre) : positif pousse vers "
        "'plus susceptible d'etre un but', negatif vers l'inverse."
    )


def main() -> None:
    """Lance le dashboard Streamlit."""
    st.set_page_config(page_title="PSG Live Tracker", layout="wide", page_icon="⚽")
    st.title("PSG Live Sports Tracker — xG Analytics")

    if not _check_prerequisites():
        return

    shots = load_shots_with_xg(str(_DB_PATH), str(_MODEL_PATH))
    filtered = _apply_filters(shots)

    _render_kpis(filtered)

    st.subheader("Shotmap")
    _render_shotmap(filtered)

    st.subheader("Buts reels vs xG cumule (top 10 tireurs)")
    _render_leaderboard(filtered)

    st.subheader("Explicabilite d'un tir (SHAP)")
    _render_shot_explainer(filtered, _MODEL_PATH)


if __name__ == "__main__":
    main()
