"""Identite visuelle du dashboard : palette, CSS injecte, template Plotly, badge.

Regroupe tout ce qui est pure presentation (aucune logique metier) pour que
`main.py` reste lisible. Le badge est un dessin original (cercle + tour
Eiffel stylisee geometrique) inspire des couleurs du club, pas une
reproduction du blason officiel deregistre (marque protegee) : safe a
committer dans un repo public.

Portraits joueur : par defaut, aucune vraie photo (droits d'auteur presse/
agence non geres ici). `player_photo_data_uri` cherche un fichier local
optionnel sous `data/assets/players/<slug>.{jpg,png}` (dossier gitignore,
usage prive uniquement) ; a defaut, `_render_player_hero` retombe sur un
monogramme geant en filigrane genere en CSS pur.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

# --- Palette (coherente entre CSS et templates Plotly) --------------------
NAVY_DARK = "#0B1330"
NAVY = "#141F4D"
NAVY_LIGHT = "#1E2C63"
RED = "#E30613"
RED_SOFT = "#FF4D5E"
GOLD = "#D9A441"
TEXT = "#F5F6FA"
TEXT_MUTED = "#9CA3C4"
GREEN = "#22C55E"
GRAY = "#94A3B8"

_ASSETS_DIR = Path(__file__).resolve().parents[3] / "data/assets/players"


def slugify(name: str) -> str:
    """Nom de joueur -> slug de fichier (`Kylian Mbappe` -> `kylian-mbappe`)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug


def initials(name: str) -> str:
    """Initiales (1-2 lettres) utilisees pour le monogramme en filigrane."""
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def player_photo_data_uri(name: str) -> str | None:
    """Cherche une photo locale optionnelle pour ce joueur, hors du repo git.

    Convention : `data/assets/players/<slug>.jpg` ou `.png`. Ce dossier est
    gitignore (voir `.gitignore`) : sert uniquement a un usage local/prive
    (ex. demo en entretien) si l'utilisateur y depose des photos dont il a
    les droits. Rien n'est fourni par defaut.
    """
    slug = slugify(name)
    for ext, mime in ((".jpg", "image/jpeg"), (".jpeg", "image/jpeg"), (".png", "image/png")):
        path = _ASSETS_DIR / f"{slug}{ext}"
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
    return None


def psg_badge_svg(size: int = 56) -> str:
    """Badge SVG original (cercle + tour Eiffel geometrique), pas le blason officiel."""
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <circle cx="100" cy="100" r="94" fill="{NAVY_DARK}" stroke="{RED}" stroke-width="7"/>
  <circle cx="100" cy="100" r="82" fill="none" stroke="{GOLD}" stroke-width="1.5" opacity="0.5"/>
  <g fill="none" stroke="{TEXT}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round">
    <polygon points="70,128 130,128 118,100 82,100" fill="{TEXT}" opacity="0.92"/>
    <polygon points="82,100 118,100 110,74 90,74" fill="{TEXT}" opacity="0.92"/>
    <polygon points="90,74 110,74 104,50 96,50" fill="{TEXT}" opacity="0.92"/>
    <line x1="100" y1="50" x2="100" y2="36"/>
    <line x1="62" y1="128" x2="138" y2="128"/>
  </g>
  <text x="100" y="152" text-anchor="middle" font-family="Arial, sans-serif"
        font-size="26" font-weight="800" fill="{TEXT}" letter-spacing="2">PSG</text>
  <text x="100" y="168" text-anchor="middle" font-family="Arial, sans-serif"
        font-size="11" font-weight="600" fill="{GOLD}" letter-spacing="3">XG TRACKER</text>
</svg>
""".strip()


def _badge_data_uri(size: int = 56) -> str:
    encoded = base64.b64encode(psg_badge_svg(size).encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def inject_global_css() -> str:
    """Feuille de style globale (fond, cartes KPI, sidebar, onglets, header)."""
    return f"""
<style>
.stApp {{
    background:
        radial-gradient(1200px 600px at 15% -10%, {NAVY_LIGHT} 0%, transparent 60%),
        radial-gradient(1000px 500px at 110% 10%, {NAVY} 0%, transparent 55%),
        {NAVY_DARK};
}}

[data-testid="stSidebar"] {{
    background: {NAVY_DARK};
    border-right: 1px solid {NAVY_LIGHT};
}}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {{
    color: {TEXT} !important;
}}

/* Cartes KPI (st.metric) */
[data-testid="stMetric"] {{
    background: linear-gradient(160deg, {NAVY} 0%, {NAVY_DARK} 100%);
    border: 1px solid {NAVY_LIGHT};
    border-top: 3px solid {RED};
    border-radius: 12px;
    padding: 14px 16px 10px 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}}
[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}

/* Onglets */
button[data-baseweb="tab"] {{ color: {TEXT_MUTED}; font-weight: 600; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {TEXT} !important; }}
[data-baseweb="tab-highlight"] {{ background-color: {RED} !important; }}
[data-baseweb="tab-border"] {{ background-color: {NAVY_LIGHT} !important; }}

/* Bandeau d'en-tete */
.psg-header {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 6px 0 18px 0;
    border-bottom: 1px solid {NAVY_LIGHT};
    margin-bottom: 18px;
}}
.psg-header h1 {{
    margin: 0;
    font-size: 1.7rem;
    color: {TEXT};
    letter-spacing: 0.5px;
}}
.psg-header p {{
    margin: 2px 0 0 0;
    color: {TEXT_MUTED};
    font-size: 0.92rem;
}}

/* Carte "profil joueur" avec monogramme/photo en filigrane */
.player-hero {{
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    border: 1px solid {NAVY_LIGHT};
    background: linear-gradient(135deg, {NAVY} 0%, {NAVY_DARK} 100%);
    padding: 28px 26px;
    margin-bottom: 8px;
    min-height: 150px;
}}
.player-hero-bg-monogram {{
    position: absolute;
    right: -10px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 11rem;
    font-weight: 900;
    color: {TEXT};
    opacity: 0.07;
    line-height: 1;
    user-select: none;
    pointer-events: none;
}}
.player-hero-bg-photo {{
    position: absolute;
    inset: 0;
    background-size: cover;
    background-position: center 15%;
    opacity: 0.22;
    -webkit-mask-image: linear-gradient(90deg, transparent 0%, black 55%);
    mask-image: linear-gradient(90deg, transparent 0%, black 55%);
}}
.player-hero-content {{ position: relative; z-index: 1; }}
.player-hero-content h2 {{
    margin: 0 0 4px 0;
    color: {TEXT};
    font-size: 1.6rem;
}}
.player-hero-content .subtitle {{
    color: {GOLD};
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 14px;
}}
</style>
""".strip()


def header_html(subtitle: str) -> str:
    """Bandeau d'en-tete : badge + titre + sous-titre."""
    return f"""
<div class="psg-header">
    <img src="{_badge_data_uri(56)}" width="56" height="56" alt="Badge PSG xG Tracker"/>
    <div>
        <h1>PSG Live Sports Tracker</h1>
        <p>{subtitle}</p>
    </div>
</div>
""".strip()


def player_hero_html(
    name: str, subtitle: str, photo_data_uri: str | None
) -> str:
    """Carte d'en-tete du profil joueur, avec photo (si dispo) ou monogramme en filigrane."""
    if photo_data_uri:
        bg_html = (
            '<div class="player-hero-bg-photo" '
            f"style=\"background-image: url('{photo_data_uri}');\"></div>"
        )
    else:
        bg_html = f'<div class="player-hero-bg-monogram">{initials(name)}</div>'

    return f"""
<div class="player-hero">
    {bg_html}
    <div class="player-hero-content">
        <div class="subtitle">{subtitle}</div>
        <h2>{name}</h2>
    </div>
</div>
""".strip()


def plotly_template() -> dict[str, object]:
    """Layout Plotly commun a tous les graphiques du dashboard (theme sombre club)."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": TEXT, "family": "Arial, sans-serif"},
        "colorway": [RED, GOLD, TEXT_MUTED, GREEN],
        "legend": {"font": {"color": TEXT}},
        "xaxis": {"gridcolor": NAVY_LIGHT, "zerolinecolor": NAVY_LIGHT},
        "yaxis": {"gridcolor": NAVY_LIGHT, "zerolinecolor": NAVY_LIGHT},
    }
