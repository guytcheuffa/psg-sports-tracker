"""Construction du fond de terrain (demi-terrain offensif) pour les shotmaps Plotly.

Referentiel StatsBomb : pitch 120x80 yards, but adverse sur la ligne x=120,
centre en y=40. Dimensions standard (surface de reparation 18 yards, surface
de but 6 yards) reprises telles quelles, independamment de la source du tir
(StatsBomb/Understat sont deja normalisees vers ce referentiel a
l'ingestion, cf. `psg_tracker.schemas`).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from psg_tracker.app.theme import GOLD, NAVY

_PITCH_LINE_COLOR = "rgba(217, 164, 65, 0.35)"  # GOLD attenue : sobre sur le fond fonce
_PITCH_FILL_COLOR = NAVY
_HALF_X_MIN = 60.0
_PITCH_X_MAX = 120.0
_PITCH_Y_MIN = 0.0
_PITCH_Y_MAX = 80.0
_GOAL_Y_CENTER = 40.0


def _rect(
    x0: float, x1: float, y0: float, y1: float, fill: str | None = None
) -> dict[str, object]:
    shape: dict[str, object] = {
        "type": "rect",
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "line": {"color": _PITCH_LINE_COLOR, "width": 1.5},
        "layer": "below",
    }
    if fill is not None:
        shape["fillcolor"] = fill
    return shape


def half_pitch_figure() -> go.Figure:
    """Figure Plotly vierge (sans traces) representant le dernier tiers offensif.

    A utiliser comme base pour une shotmap : ajouter les traces de tirs
    par-dessus (`fig.add_trace(...)`) apres appel a cette fonction.
    """
    fig = go.Figure()

    shapes: list[dict[str, object]] = [
        # Pelouse (fond plein, coherent avec le theme sombre du dashboard)
        _rect(_HALF_X_MIN, _PITCH_X_MAX, _PITCH_Y_MIN, _PITCH_Y_MAX, fill=_PITCH_FILL_COLOR),
        # Ligne mediane
        {
            "type": "line",
            "x0": _HALF_X_MIN,
            "x1": _HALF_X_MIN,
            "y0": _PITCH_Y_MIN,
            "y1": _PITCH_Y_MAX,
            "line": {"color": _PITCH_LINE_COLOR, "width": 1.5},
            "layer": "below",
        },
        # Surface de reparation (18 yards)
        _rect(102.0, _PITCH_X_MAX, 18.0, 62.0),
        # Surface de but (6 yards)
        _rect(114.0, _PITCH_X_MAX, 30.0, 50.0),
    ]

    # Cage
    shapes.append(
        {
            "type": "line",
            "x0": _PITCH_X_MAX,
            "x1": _PITCH_X_MAX,
            "y0": 36.0,
            "y1": 44.0,
            "line": {"color": GOLD, "width": 4},
            "layer": "below",
        }
    )

    fig.update_layout(shapes=shapes)

    # Point de penalty (marker plutot que shape, plus simple pour un point)
    fig.add_trace(
        go.Scatter(
            x=[108.0],
            y=[_GOAL_Y_CENTER],
            mode="markers",
            marker={"size": 4, "color": _PITCH_LINE_COLOR},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Arc de la surface de reparation (portion de cercle visible hors de la
    # surface, rayon 10 yards centre sur le point de penalty)
    theta = np.linspace(2.22, 4.06, 40)  # radians, arc oriente vers le but
    arc_x = 108.0 + 10.0 * np.cos(theta)
    arc_y = _GOAL_Y_CENTER + 10.0 * np.sin(theta)
    fig.add_trace(
        go.Scatter(
            x=arc_x,
            y=arc_y,
            mode="lines",
            line={"color": _PITCH_LINE_COLOR, "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_xaxes(
        range=[_HALF_X_MIN, _PITCH_X_MAX + 2],
        showgrid=False,
        zeroline=False,
        visible=False,
    )
    fig.update_yaxes(
        range=[_PITCH_Y_MIN - 2, _PITCH_Y_MAX + 2],
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    return fig
