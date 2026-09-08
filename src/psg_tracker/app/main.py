"""Point d'entree de l'application Streamlit : dashboard xG du PSG.

Cinq onglets : vue d'ensemble (KPIs + shotmap), classement buts reels vs
xG cumule par joueur, profil d'un joueur (carte avec photo/monogramme en
filigrane), explicabilite SHAP d'un tir individuel selectionne, et
diagnostic du modele (courbe ROC + courbe de calibration sur le jeu de
test hold-out reel, cf. `models/eval_report.py`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from psg_tracker.app import theme
from psg_tracker.app.data_loader import (
    get_shap_explainer,
    load_eval_report_cached,
    load_model,
    load_shots_with_xg,
)
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
# Ordre football (gardien -> attaquant) plutot qu'alphabetique ; "Inconnu"
# en dernier (cf. app/data_loader.py : ~0.2% des tirs sans poste rattache,
# joueurs tres marginaux non listes par Understat cette saison-la).
_POSITION_ORDER = ["Gardien", "Défenseur", "Milieu", "Attaquant", "Inconnu"]
_GOAL_FILTER_OPTIONS = ["Tous les tirs", "Buts uniquement", "Sans but uniquement"]


_DETAILED_POSITION_DOMINANCE_THRESHOLD = 0.6


def _detailed_position_dominance(shots: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """Pour chaque poste detaille (hors "Inconnu"), le joueur le plus present et sa part.

    La couverture StatsBomb est limitee a 3 saisons sur 12 (plafond du
    dataset ouvert, pas une limite d'ingestion - cf. README) : sur cette
    fenetre reduite, certains postes fins ne refletent en realite qu'un
    seul joueur (ex. "Ailier droit" ~= Angel Di Maria a 94%). Sert a
    avertir dans la sidebar plutot qu'a laisser croire a une tendance de
    poste generalisable.
    """
    dominance: dict[str, tuple[str, float]] = {}
    known = shots[shots["position_detailed"] != "Inconnu"]
    for position, group in known.groupby("position_detailed"):
        counts = group["player_name"].value_counts()
        dominance[str(position)] = (str(counts.index[0]), float(counts.iloc[0] / counts.sum()))
    return dominance


def _apply_filters(shots: pd.DataFrame) -> pd.DataFrame:
    """Sidebar complete : identite visuelle, filtres, infos dataset.

    Filtres source/competition/saison/poste (multiselect, tout coche par
    defaut) + un filtre but/pas-but (radio, "Tous" par defaut) : combines,
    ils permettent d'isoler par ex. "tous les tirs des attaquants qui ont
    fini au but, saison 2023/2024" pour chercher des patterns dans la
    shotmap ou le classement.
    """
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

    seasons = sorted(shots["season"].astype(str).unique())
    selected_seasons = st.sidebar.multiselect("Saison", seasons, default=seasons)

    positions = [p for p in _POSITION_ORDER if p in set(shots["position"].unique())]
    selected_positions = st.sidebar.multiselect("Poste", positions, default=positions)

    detailed_present = set(shots["position_detailed"].unique())
    detailed_positions = sorted(p for p in detailed_present if p != "Inconnu")
    if "Inconnu" in detailed_present:
        detailed_positions.append("Inconnu")
    dominance = _detailed_position_dominance(shots)

    def _detailed_position_label(position: str) -> str:
        if position == "Inconnu":
            return "Inconnu (hors couverture StatsBomb)"
        top_player, share = dominance.get(position, ("", 0.0))
        if share >= _DETAILED_POSITION_DOMINANCE_THRESHOLD:
            return f"{position} ⚠️ {top_player} {share:.0%}"
        return position

    selected_detailed_positions = st.sidebar.multiselect(
        "Poste detaille (StatsBomb)",
        detailed_positions,
        default=detailed_positions,
        format_func=_detailed_position_label,
    )
    st.sidebar.markdown(
        theme.sidebar_note_html(
            icon="📊",
            title="Poste detaille : couverture partielle",
            body=(
                "Donnees StatsBomb limitees a 3 saisons sur 12 (2015/16, "
                "2021/22, 2022/23) — plafond du dataset ouvert (aucune autre "
                "saison Ligue 1 ni match PSG en Champions League n'y est "
                "publie), pas une limite d'ingestion. <strong>⚠️</strong> sur "
                "une option = poste domine a plus de 60% par un seul joueur "
                "sur ces 3 saisons : a lire comme un profil individuel, pas "
                "une tendance generale."
            ),
        ),
        unsafe_allow_html=True,
    )

    goal_filter = st.sidebar.radio("But", _GOAL_FILTER_OPTIONS, index=0)

    n_matches = shots[["source", "match_date"]].drop_duplicates().shape[0]
    seasons_label = ", ".join(seasons)
    st.sidebar.markdown(
        theme.sidebar_footer_html(len(shots), n_matches, seasons_label), unsafe_allow_html=True
    )

    filtered = shots[
        shots["source"].isin(selected_sources)
        & shots["competition"].isin(selected_competitions)
        & shots["season"].astype(str).isin(selected_seasons)
        & shots["position"].isin(selected_positions)
        & shots["position_detailed"].isin(selected_detailed_positions)
    ]
    if goal_filter == "Buts uniquement":
        filtered = filtered[filtered["is_goal"]]
    elif goal_filter == "Sans but uniquement":
        filtered = filtered[~filtered["is_goal"]]
    return filtered


def _render_kpis(shots: pd.DataFrame) -> None:
    """Ligne de KPI : volume, buts reels vs xG cumule (calibration visuelle du modele)."""
    total_shots = len(shots)
    total_goals = int(shots["is_goal"].sum())
    total_xg = float(shots["xg_pred"].sum())
    delta = total_goals - total_xg

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Tirs", f"{total_shots:,}".replace(",", " "))
    col2.metric("⚽ Buts reels", total_goals)
    col3.metric("📈 xG cumule", f"{total_xg:.1f}")
    col4.metric(
        "Efficacite (buts vs xG)",
        f"{total_goals} buts",
        delta=f"{delta:+.1f} xG",
        help=(
            "Delta positif = la selection marque plus que ce que la qualite "
            "des tirs (xG) ne le suggere - finition au-dessus de la moyenne."
        ),
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
    position = str(player_shots["position"].iloc[0])
    position_detailed = str(player_shots["position_detailed"].iloc[0])
    position_label = (
        f"{position} ({position_detailed})" if position_detailed != "Inconnu" else position
    )

    st.markdown(
        theme.player_hero_html(
            name=player,
            subtitle=(
                f"{position_label} - {n_matches} match(s) - {n_shots} tir(s) dans la selection"
            ),
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


_MIN_SHOTS_FOR_HEATMAP = 5


def _render_shot_context_heatmap(shots: pd.DataFrame, selected_index: int) -> None:
    """Heatmap de densite des tirs de la selection filtree + position du tir explique.

    Contextualise le diagramme SHAP : la carte SHAP explique *pourquoi* le
    modele estime tel xG pour ce tir precis, cette heatmap montre *ou* ce
    tir se situe par rapport aux zones ou l'equipe/le joueur filtre tire le
    plus souvent (ex. verifier si un tir bien note est aussi pris depuis
    une zone "chaude" habituelle, ou au contraire atypique).
    """
    selected = shots.loc[selected_index]
    is_goal = bool(selected["is_goal"])
    outcome_label = "⚽ BUT" if is_goal else "❌ Pas de but"
    outcome_annotation = {
        "x": selected["loc_x"],
        "y": selected["loc_y"],
        "text": f"<b>{outcome_label}</b>",
        "showarrow": True,
        "arrowhead": 2,
        "arrowcolor": theme.TEXT,
        "ax": 0,
        "ay": -35,
        "font": {"color": theme.GOLD if is_goal else theme.TEXT, "size": 13},
        "bgcolor": theme.NAVY_DARK,
        "bordercolor": theme.GOLD if is_goal else theme.TEXT_MUTED,
        "borderwidth": 1,
        "borderpad": 4,
    }

    if len(shots) < _MIN_SHOTS_FOR_HEATMAP:
        fig = half_pitch_figure()
        fig.add_trace(
            go.Scatter(
                x=[selected["loc_x"]],
                y=[selected["loc_y"]],
                mode="markers",
                marker={"size": 22, "color": theme.RED, "symbol": "star"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_annotation(outcome_annotation)
        _themed_figure(fig, height=360, showlegend=False, margin={"t": 10, "b": 0})
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Trop peu de tirs dans la selection actuelle ({len(shots)}) pour une carte "
            "de densite significative : seule la position du tir explique est affichee "
            "(resultat reel indique au-dessus du marqueur)."
        )
        return

    fig = half_pitch_figure()

    fig.add_trace(
        go.Histogram2d(
            x=shots["loc_x"],
            y=shots["loc_y"],
            xbins={"start": 60.0, "end": 122.0, "size": 5.0},
            ybins={"start": -2.0, "end": 82.0, "size": 5.0},
            colorscale=[[0.0, "rgba(20,31,77,0)"], [1.0, theme.GOLD]],
            showscale=False,
            opacity=0.65,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[selected["loc_x"]],
            y=[selected["loc_y"]],
            mode="markers",
            name="Tir explique",
            marker={
                "size": 22,
                "color": theme.RED,
                "symbol": "star",
                "line": {"color": theme.TEXT, "width": 2},
            },
            hovertemplate=(
                f"<b>{selected['player_name']}</b><br>"
                f"Minute {selected['minute']} - {selected['shot_type']}<br>"
                f"xG: {selected['xg_pred']:.2f} - {outcome_label}<extra></extra>"
            ),
        )
    )
    fig.add_annotation(outcome_annotation)

    _themed_figure(fig, height=380, showlegend=False, margin={"t": 10, "b": 0})
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Densite des tirs de la selection filtree en fond (zones les plus chaudes = "
        "plus de tirs), etoile rouge = position exacte du tir explique, avec son "
        "resultat reel (but ou non) indique au-dessus."
    )


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
    is_goal = bool(shots.loc[choice, "is_goal"])

    col_xg, col_outcome = st.columns(2)
    col_xg.metric("xG predit pour ce tir", f"{predicted_xg:.2f}")
    col_outcome.metric("Resultat reel", "⚽ But" if is_goal else "❌ Pas de but")

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

    st.markdown("##### Localisation du tir")
    _render_shot_context_heatmap(shots, choice)


def _render_model_diagnostics(model_path: Path) -> None:
    """Diagnostic du modele : metriques + courbe ROC + courbe de calibration.

    Utilise les VRAIES predictions hold-out sauvegardees par `scripts/train.py`
    (cf. `models/eval_report.py`), pas un recalcul a la volee avec le modele
    final deploye (celui-ci est reentraine sur 100% des donnees - l'evaluer
    sur les memes tirs donnerait un diagnostic circulaire, artificiellement
    optimiste).
    """
    report = load_eval_report_cached(str(model_path))
    if report is None:
        st.info(
            "Aucun rapport d'evaluation trouve a cote du modele. Relancer "
            "`python scripts/train.py` pour en generer un (genere automatiquement "
            "depuis cette version du script)."
        )
        return

    metrics = report["metrics"]
    y_true = np.array(report["y_true"])
    y_pred = np.array(report["y_pred"])
    trained_at = report["trained_at"][:10]

    st.caption(
        f"Evaluation hold-out reelle : {metrics['n_test']} tirs jamais vus par ce modele "
        f"intermediaire (split {1 - report['test_size']:.0%} entrainement / "
        f"{report['test_size']:.0%} test stratifie, random_state={report['random_state']}), "
        f"entraine le {trained_at}."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", help="1.0 = parfait, 0.5 = aleatoire.")
    col2.metric("Log loss", f"{metrics['log_loss']:.3f}", help="Plus bas = meilleur.")
    col3.metric(
        "Brier score",
        f"{metrics['brier_score']:.3f}",
        help="Plus bas = meilleur (discrimination + calibration combinees).",
    )
    col4.metric(
        "Taux de but reel",
        f"{metrics['goal_rate']:.1%}",
        help="Part de tirs termines au but dans le jeu de test (baseline naive).",
    )

    col_roc, col_calib = st.columns(2)

    with col_roc:
        st.markdown("##### Courbe ROC")
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name="Modele",
                line={"color": theme.GOLD, "width": 3},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Aleatoire",
                line={"color": theme.TEXT_MUTED, "width": 1, "dash": "dash"},
            )
        )
        _themed_figure(
            fig,
            height=340,
            xaxis_title="Taux de faux positifs",
            yaxis_title="Taux de vrais positifs",
            legend={"orientation": "h", "y": -0.22},
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(fig, width="stretch")

    with col_calib:
        st.markdown("##### Courbe de calibration")
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy="quantile")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                name="Modele",
                line={"color": theme.GOLD, "width": 3},
                marker={"size": 7},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Calibration parfaite",
                line={"color": theme.TEXT_MUTED, "width": 1, "dash": "dash"},
            )
        )
        _themed_figure(
            fig,
            height=340,
            xaxis_title="xG predit (moyenne du groupe)",
            yaxis_title="Taux de but reel (groupe)",
            legend={"orientation": "h", "y": -0.22},
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(fig, width="stretch")

    st.caption(
        "Courbe ROC : capacite du modele a distinguer tirs marques / non marques, quel que "
        "soit le seuil choisi (aire sous la courbe = ROC-AUC). Courbe de calibration : les "
        "tirs sont regroupes en 10 groupes de taille egale par xG predit croissant - le "
        "modele est bien calibre si, dans chaque groupe, le taux de but reellement observe "
        "correspond au xG moyen predit (proche de la diagonale)."
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
        theme.hero_banner_html(
            f"{len(shots)} tirs en base - analytics xG multi-saisons",
            tech_stack=["Python", "DuckDB", "XGBoost", "SHAP", "Streamlit", "Plotly"],
        ),
        unsafe_allow_html=True,
    )

    filtered = _apply_filters(shots)
    _render_kpis(filtered)

    tab_overview, tab_leaderboard, tab_player, tab_shap, tab_diagnostics = st.tabs(
        [
            "Vue d'ensemble",
            "Classement",
            "Profil joueur",
            "Explicabilite (SHAP)",
            "Diagnostic du modele",
        ]
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

    with tab_diagnostics:
        st.subheader("Diagnostic du modele")
        _render_model_diagnostics(_MODEL_PATH)

    st.markdown(theme.app_footer_html(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
