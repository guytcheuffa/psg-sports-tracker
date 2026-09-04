"""Identite visuelle du dashboard : palette, CSS injecte, template Plotly, assets.

Regroupe tout ce qui est pure presentation (aucune logique metier) pour que
`main.py` reste lisible.

Logo/banniere club : `psg_logo_data_uri`/`psg_hero_data_uri` cherchent des
fichiers locaux optionnels sous `data/assets/branding/` (voir constantes
`_LOGO_PATH`/`_HERO_PATH`). Le blason et les visuels du club sont des
marques/images protegees ; l'utilisateur les a fournis lui-meme et choisi
de les publier dans ce depot (portfolio personnel, usage non commercial) -
ils sont donc versionnes (cf. `.gitignore`). A defaut de fichier, le repo
reste fonctionnel : `psg_badge_svg` (dessin original, cercle + tour Eiffel
stylisee) sert de repli.

Texture de fond : `psg_texture_data_uri` cherche
`data/assets/branding/bg-texture.jpg` (motif reseau discret, genere via
Canva - creation originale, pas de marque protegee) ; habille `.stApp` en
plus des logo/banniere reels. Optionnel comme le reste : sans fichier, le
degrade CSS existant suffit.

Portraits joueur : meme logique via `player_photo_data_uri`
(`data/assets/players/<slug>.{jpg,png}`), mais ce dossier reste non
versionne par defaut (photos de personnes identifiables, droit a l'image) ;
a defaut de fichier, un monogramme geant en filigrane genere en CSS pur
(`player_hero_html`).
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
_TEXTURE_PATH = _BRANDING_DIR / "bg-texture.jpg"

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


def psg_texture_data_uri() -> str | None:
    """Texture de fond (motif reseau discret, genere via Canva) si deposee sous
    `data/assets/branding/bg-texture.jpg`. Habille le fond de toute l'app ;
    reste optionnel comme les autres visuels (le degrade seul suffit en repli).
    """
    return _local_image_data_uri(_TEXTURE_PATH)


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
    texture_uri = psg_texture_data_uri()
    texture_layer = (
        f"url('{texture_uri}') center/cover no-repeat fixed,\n        " if texture_uri else ""
    )
    return f"""
<style>
.stApp {{
    background:
        radial-gradient(1200px 600px at 15% -10%, {NAVY_LIGHT} 0%, transparent 60%),
        radial-gradient(1000px 500px at 110% 10%, {NAVY} 0%, transparent 55%),
        {texture_layer}{NAVY_DARK};
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

/* Sidebar : bloc marque + badges + pied de page (comble le vide visuel) */
.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 16px;
    margin-bottom: 14px;
    border-bottom: 1px solid {NAVY_LIGHT};
}}
.sidebar-brand img {{
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: white;
    box-shadow: 0 0 0 2px {RED};
    flex-shrink: 0;
}}
.sidebar-brand h3 {{
    margin: 0;
    font-size: 0.92rem;
    color: {TEXT};
    letter-spacing: 0.5px;
}}
.sidebar-brand p {{
    margin: 2px 0 0 0;
    font-size: 0.7rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.sidebar-section-title {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {GOLD};
    font-weight: 700;
    margin: 4px 0 6px 0;
}}

.sidebar-badge-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 6px 0 16px 0;
}}
.sidebar-badge {{
    background: {NAVY};
    border: 1px solid {NAVY_LIGHT};
    color: {TEXT_MUTED};
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 999px;
}}

.sidebar-footer {{
    margin-top: 22px;
    padding-top: 14px;
    border-top: 1px solid {NAVY_LIGHT};
}}
.sidebar-footer p {{
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    line-height: 1.5;
    margin: 3px 0;
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
            <h1>PSG Sports Tracker</h1>
            <p>{subtitle}</p>
        </div>
    </div>
</div>
""".strip()


def sidebar_brand_html() -> str:
    """Bloc d'entete de la sidebar : logo + nom du projet (comble le vide visuel par defaut)."""
    logo_uri = psg_logo_data_uri() or _badge_data_uri(42)
    return f"""
<div class="sidebar-brand">
    <img src="{logo_uri}" alt="Logo PSG"/>
    <div>
        <h3>PSG XG TRACKER</h3>
        <p>Analytics tirs &amp; xG</p>
    </div>
</div>
""".strip()


def sidebar_badges_html(labels: list[str]) -> str:
    """Ligne de badges pour la sidebar (ex. sources de donnees actives)."""
    badges = "".join(f'<span class="sidebar-badge">{label}</span>' for label in labels)
    return f'<div class="sidebar-badge-row">{badges}</div>'


def sidebar_footer_html(n_shots: int, n_matches: int, seasons: str) -> str:
    """Bloc "a propos" en pied de sidebar : volumetrie du dataset + stack technique."""
    return f"""
<div class="sidebar-footer">
    <div class="sidebar-section-title">A propos</div>
    <p>{n_shots} tirs sur {n_matches} matchs ({seasons}).</p>
    <p>Modele XGBoost + explicabilite SHAP.</p>
    <p>Donnees : StatsBomb Open Data (historique) + Understat (saison en cours).</p>
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
