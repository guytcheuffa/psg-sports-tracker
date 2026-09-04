"""Identite visuelle du dashboard : palette, CSS injecte, template Plotly, assets.

Regroupe tout ce qui est pure presentation (aucune logique metier) pour que
`main.py` reste lisible.

Logo/banniere club : `psg_logo_data_uri`/`psg_hero_data_uri` cherchent des
fichiers locaux optionnels sous `data/assets/branding/` (voir constantes
`_LOGO_PATH`/`_HERO_PATH`). Le blason et les visuels du club sont des
marques/images protegees : ce depot ne les fournit pas par defaut (dossier
non versionne, cf. `.gitignore`) - c'est a l'utilisateur de les y deposer
s'il en a les droits pour son usage (portfolio personnel). A defaut de
fichier, `psg_badge_svg` (dessin original, cercle + tour Eiffel stylisee)
sert de repli pour que le dashboard reste fonctionnel et publiable tel quel.

Portraits joueur : meme logique via `player_photo_data_uri`
(`data/assets/players/<slug>.{jpg,png}`) ; a defaut, un monogramme geant en
filigrane genere en CSS pur (`player_hero_html`).
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLAYERS_DIR = _REPO_ROOT / "data/assets/players"
_BRANDING_DIR = _REPO_ROOT / "data/assets/branding"
_LOGO_PATH = _BRANDING_DIR / "psg-logo.png"
_HERO_PATH = _BRANDING_DIR / "psg-hero.png"

_MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


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


def _local_image_data_uri(path: Path) -> str | None:
    """Encode un fichier image local en data URI, ou None s'il n'existe pas."""
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is None or not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def player_photo_data_uri(name: str) -> str | None:
    """Cherche une photo locale optionnelle pour ce joueur, hors du repo git.

    Convention : `data/assets/players/<slug>.jpg` ou `.png`. Ce dossier est
    gitignore (voir `.gitignore`) : sert uniquement a un usage local/prive
    (ex. demo en entretien) si l'utilisateur y depose des photos dont il a
    les droits. Rien n'est fourni par defaut.
    """
    slug = slugify(name)
    for ext in (".jpg", ".jpeg", ".png"):
        data_uri = _local_image_data_uri(_PLAYERS_DIR / f"{slug}{ext}")
        if data_uri is not None:
            return data_uri
    return None


def psg_logo_data_uri() -> str | None:
    """Logo club reel s'il a ete depose dans `data/assets/branding/psg-logo.png`."""
    return _local_image_data_uri(_LOGO_PATH)


def psg_hero_data_uri() -> str | None:
    """Photo de banniere reelle si deposee dans `data/assets/branding/psg-hero.png`."""
    return _local_image_data_uri(_HERO_PATH)


def psg_badge_svg(size: int = 56) -> str:
    """Badge SVG original (cercle + tour Eiffel geometrique) : repli si pas de vrai logo."""
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
    """Feuille de style globale (fond, cartes KPI, sidebar, onglets, banniere, header)."""
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

/* Banniere d'en-tete (photo club en fond + logo + titre) */
.psg-hero-banner {{
    position: relative;
    border-radius: 18px;
    overflow: hidden;
    min-height: 180px;
    margin-bottom: 22px;
    border: 1px solid {NAVY_LIGHT};
    display: flex;
    align-items: center;
    background: linear-gradient(135deg, {NAVY} 0%, {NAVY_DARK} 100%);
}}
.psg-hero-banner-bg {{
    position: absolute;
    inset: 0;
    background-size: cover;
    background-position: center 40%;
}}
.psg-hero-banner-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(
        100deg, {NAVY_DARK} 20%, rgba(11,19,48,0.80) 48%, rgba(11,19,48,0.30) 100%
    );
}}
.psg-hero-banner-content {{
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 22px 32px;
}}
.psg-hero-banner-content img.psg-logo {{
    width: 68px;
    height: 68px;
    border-radius: 50%;
    background: white;
    box-shadow: 0 0 0 3px {RED}, 0 6px 18px rgba(0,0,0,0.45);
    flex-shrink: 0;
}}
.psg-hero-banner-content h1 {{
    margin: 0;
    font-size: 1.9rem;
    color: {TEXT};
    letter-spacing: 0.5px;
}}
.psg-hero-banner-content p {{
    margin: 4px 0 0 0;
    color: #D7DCF2;
    font-size: 0.95rem;
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


def hero_banner_html(subtitle: str) -> str:
    """Banniere d'en-tete : logo (reel si dispo, sinon badge dessine) + titre + photo de fond."""
    logo_uri = psg_logo_data_uri() or _badge_data_uri(68)
    hero_uri = psg_hero_data_uri()

    bg_html = (
        f'<div class="psg-hero-banner-bg" style="background-image: url(\'{hero_uri}\');"></div>'
        if hero_uri
        else ""
    )

    return f"""
<div class="psg-hero-banner">
    {bg_html}
    <div class="psg-hero-banner-overlay"></div>
    <div class="psg-hero-banner-content">
        <img class="psg-logo" src="{logo_uri}" alt="Logo PSG"/>
        <div>
            <h1>PSG Live Sports Tracker</h1>
            <p>{subtitle}</p>
        </div>
    </div>
</div>
""".strip()


def player_hero_html(name: str, subtitle: str, photo_data_uri: str | None) -> str:
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
