"""Point d'entree de l'application Streamlit : dashboard xG du PSG.

Quatre onglets : vue d'ensemble (KPIs + shotmap), classement buts reels vs
xG cumule par joueur, profil d'un joueur (carte avec photo/monogramme en
filigrane), et explicabilite SHAP d'un tir individuel selectionne.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from psg_tracker.app import theme
from psg_tracker.app.data_loader import get_shap_explainer, load_model, load_shots_with_xg
from psg_tracker.app.pitch import half_pitch_figure
from psg_tracker.config import settings

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
    True: theme.GOLD,  # but : accent chaud, se detache de la pelouse
    False: theme.TEXT_MUTED,  # tir non converti : neutre
}


def _themed_figure(fig: go.Figure, **overrides: object) -> go.Figure:
    """Applique le template de couleurs commun, puis des overrides specifiques."""
    fig.update_layout(**theme.plotly_template())
    if overrides:
        fig.update_layout(**overrides)
    return fig


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


_SOURCE_LABELS = {"statsbomb": "StatsBomb", "understat": "Understat"}


def _apply_filters(shots: pd.DataFrame) -> pd.DataFrame:
    """Sidebar complete : identite visuelle, filtres (source, competition), infos dataset."""
    st.sidebar.markdown(theme.sidebar_brand_html(), unsafe_allow_html=True)

    sources = sorted(shots["source"].unique())
    st.sidebar.markdown(
        theme.sidebar_badges_html([_SOURCE_LABELS.get(s, s) for s in sources]),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<div class="sidebar-section-title">Filtres</div>', unsafe_allow_html=True
    )
    selected_sources = st.sidebar.multiselect("Source", sources, default=sources)

    competitions = sorted(shots["competition"].unique())
    selected_competitions = st.sidebar.multiselect(
        "Competition", competitions, default=competitions
    )

    n_matches = shots[["source", "match_date"]].drop_duplicates().shape[0]
    seasons = ", ".join(sorted(shots["season"].astype(str).unique()))
    st.sidebar.markdown(
        theme.sidebar_footer_html(len(shots), n_matches, seasons), unsafe_allow_html=True
    )

    return shots[
        shots["source"].isin(selected_sources) & shots["competition"].isin(selected_competitions)
    ]


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


def _render_shotmap(shots: pd.DataFrame, height: int = 520) -> None:
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
                    "line": {"color": theme.TEXT, "width": 1},
                    "opacity": 0.88,
                },
                customdata=group[["player_name", "minute", "shot_type", "xg_pred", "source"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Minute %{customdata[1]} - %{customdata[2]}<br>"
                    "xG: %{customdata[3]:.2f} (%{customdata[4]})<extra></extra>"
                ),
            )
        )

    _themed_figure(fig, height=height, legend={"orientation": "h", "y": -0.05})
    st.plotly_chart(fig, width="stretch")


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
    _themed_figure(fig, barmode="group", height=380, xaxis_tickangle=-30)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(by_player, width="stretch")


def _render_player_profile(shots: pd.DataFrame) -> None:
    """Carte de profil pour un joueur choisi : photo/monogramme, stats, mini-shotmap."""
    if shots.empty:
        st.info("Aucun tir avec les filtres actuels.")
        return

    players = sorted(shots["player_name"].unique())
    default_index = 0
    player = st.selectbox("Choisir un joueur", options=players, index=default_index)

    player_shots = shots[shots["player_name"] == player]
    n_matches = player_shots[["source", "match_date"]].drop_duplicates().shape[0]
    n_shots = len(player_shots)
    n_goals = int(player_shots["is_goal"].sum())
    xg_total = float(player_shots["xg_pred"].sum())

    st.markdown(
        theme.player_hero_html(
            name=player,
            subtitle=f"{n_matches} match(s) - {n_shots} tir(s) dans la selection",
            photo_data_uri=theme.player_photo_data_uri(player),
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Photo affichee uniquement si un fichier est depose localement dans "
        "`data/assets/players/` (non fourni par defaut, cf. `app/theme.py`) : "
        "sinon, monogramme genere en filigrane."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tirs", n_shots)
    col2.metric("Buts reels", n_goals)
    col3.metric("xG cumule", f"{xg_total:.2f}")
    col4.metric("Buts - xG", f"{n_goals - xg_total:+.2f}")

    st.markdown("##### Tirs de ce joueur")
    _render_shotmap(player_shots, height=420)


def _render_shot_explainer(shots: pd.DataFrame, model_path: Path) -> None:
    """Explicabilite SHAP d'un tir individuel choisi dans la liste filtree."""
    if shots.empty:
        st.info("Aucun tir a expliquer avec les filtres actuels.")
        return

    model = load_model(str(model_path))
    explainer = get_shap_explainer(str(model_path))

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

    x = shots.loc[[choice]].reindex(columns=model.feature_columns, fill_value=0)
    shap_values = explainer.shap_values(x)
    contributions = pd.Series(shap_values[0], index=model.feature_columns)
    contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    predicted_xg = float(shots.loc[choice, "xg_pred"])

    st.metric("xG predit pour ce tir", f"{predicted_xg:.2f}")

    fig = go.Figure(
        go.Bar(
            x=contributions.values,
            y=contributions.index,
            orientation="h",
            marker={
                "color": [theme.GOLD if v > 0 else theme.RED for v in contributions.values]
            },
        )
    )
    _themed_figure(
        fig,
        height=320,
        xaxis_title="Contribution SHAP (log-odds)",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Valeurs en espace log-odds (sortie brute de l'arbre) : positif pousse vers "
        "'plus susceptible d'etre un but', negatif vers l'inverse."
    )


def main() -> None:
    """Lance le dashboard Streamlit."""
    st.set_page_config(page_title="PSG Sports Tracker", layout="wide", page_icon="⚽")
    st.markdown(theme.inject_global_css(), unsafe_allow_html=True)

    if not _check_prerequisites():
        st.markdown(theme.hero_banner_html("Pipeline de donnees xG PSG"), unsafe_allow_html=True)
        return

    shots = load_shots_with_xg(str(_DB_PATH), str(_MODEL_PATH))
    st.markdown(
        theme.hero_banner_html(f"{len(shots)} tirs en base - analytics xG multi-saisons"),
        unsafe_allow_html=True,
    )

    filtered = _apply_filters(shots)
    _render_kpis(filtered)

    tab_overview, tab_leaderboard, tab_player, tab_shap = st.tabs(
        ["Vue d'ensemble", "Classement", "Profil joueur", "Explicabilite (SHAP)"]
    )

    with tab_overview:
        st.subheader("Shotmap")
        _render_shotmap(filtered)

    with tab_leaderboard:
        st.subheader("Buts reels vs xG cumule (top 10 tireurs)")
        _render_leaderboard(filtered)

    with tab_player:
        st.subheader("Profil joueur")
        _render_player_profile(filtered)

    with tab_shap:
        st.subheader("Explicabilite d'un tir")
        _render_shot_explainer(filtered, _MODEL_PATH)


if __name__ == "__main__":
    main()
