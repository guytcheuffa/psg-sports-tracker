# PSG Live Sports Tracker

Pipeline data engineering & data science autour des donnees evenementielles
du Paris Saint-Germain : ingestion, stockage analytique, modele de
probabilite de but (xG), et dashboard interactif.

## Stack

- **Ingestion (hybride)** :
  - [StatsBomb Open Data](https://github.com/statsbomb/open-data) — corpus historique utilise pour
    entrainer le modele xG. En pratique, seule la Ligue 1 (2015/16, 2021/22, 2022/23) contient des
    donnees PSG dans ce dataset ouvert : les 18 saisons de Champions League listees dans
    `competitions.json` (1999/2000-2018/19) ne couvrent pas les matchs du PSG (verifie
    empiriquement, pas seulement suppose). Corpus reel actuellement ingere : **95 matchs, 1432
    tirs, 235 buts**.
  - [Understat](https://understat.com) (scraping) — donnees de la saison en cours, non couvertes
    par StatsBomb Open Data, pour alimenter le dashboard "live". Understat n'expose pas d'API
    publique documentee et ne sert plus les donnees en JS inline (`var shotsData = ...`, pattern
    largement documente mais obsolete) : le client interroge directement les endpoints AJAX
    internes du site (`getTeamData/{team}/{season}`, `getMatchData/{match_id}`, identifies en
    inspectant `js/team.min.js`/`js/match.min.js`), qui repondent 404 sans l'en-tete
    `X-Requested-With: XMLHttpRequest`. Categories harmonisees vers le vocabulaire StatsBomb a
    l'ingestion (`OpenPlay` -> `Open Play`, `RightFoot` -> `Right Foot`, etc.) pour eviter des
    colonnes one-hot dupliquees par source. Coordonnees normalisees vers le referentiel StatsBomb
    (120x80). Corpus reel actuellement ingere (saison 2026/2027, tout debut de saison) : **2
    matchs, 31 tirs**.
    - *Limite de donnee connue* : sur les 2 premiers matchs de la saison 2026/2027, les scores/xG
      d'equipe renvoyes par Understat sont coherents, mais les noms de joueurs associes aux tirs
      sont visiblement errones (des noms d'une autre equipe, verifie en comparant a une saison
      complete ou les memes appels renvoient des noms corrects). C'est un probleme cote Understat
      (donnee de debut de saison pas encore stabilisee), pas un bug du client. Sans impact sur le
      modele xG : les features utilisees ne dependent pas de l'identite du joueur.
  - `scripts/ingest.py` : script d'orchestration idempotent (`INSERT OR REPLACE`), utilisable en
    mode decouverte automatique ou competition/saison par competition/saison.
  - `scripts/train.py` : entraine le modele xG sur l'integralite des tirs reels en base (les deux
    sources). Split train/test (80/20 stratifie) pour une evaluation honnete, puis reentrainement
    final sur 100% des donnees pour le modele sauvegarde. Resultat actuel (1463 tirs reels : 1432
    StatsBomb + 31 Understat) : **ROC-AUC 0.695, log loss 0.432** sur le jeu de test (293 tirs).
- **Stockage** : DuckDB, transformations SQL typees (cle composite `source + match_id`)
- **ML** : XGBoost (classification binaire xG), SHAP (explicabilite)
- **Visualisation** : Streamlit + Plotly — shotmap (demi-terrain, taille = xG, couleur = but/non-but),
  classement buts reels vs xG cumule par joueur, explicabilite SHAP d'un tir choisi. Teste via
  `streamlit.testing.v1.AppTest` (`tests/test_app.py`, base + modele synthetiques en base
  temporaire, pas de dependance aux vraies donnees en CI).
- **DevOps** : Docker (multi-copy avec README.md requis par `pyproject.toml`), GitHub Actions
  (lint/mypy/pytest sur `src`+`scripts`+`tests`), pytest + pytest-cov

## Architecture

```
src/psg_tracker/
├── ingestion/   # clients StatsBomb + Understat, schemas typees
├── storage/     # DuckDB (DDL + transformations SQL)
├── features/    # feature engineering xG (partage train/inference)
├── models/      # entrainement XGBoost + explicabilite SHAP
└── app/         # dashboard Streamlit (main.py + data_loader.py + pitch.py)
scripts/
├── ingest.py    # orchestration ingestion StatsBomb/Understat -> DuckDB
└── train.py     # entrainement xG sur les donnees reelles en base
```

## Setup

```bash
make setup      # venv + installation
make test       # tests + coverage
make lint       # ruff (src, tests, scripts)
make typecheck  # mypy strict (src, scripts)
make run        # lance le dashboard Streamlit (necessite data + modele, cf. ci-dessous)
```

Avant de lancer le dashboard : ingerer les donnees puis entrainer le modele (une fois) :

```bash
python scripts/ingest.py statsbomb --discover
python scripts/ingest.py understat
python scripts/train.py
```

### Docker

```bash
cp .env.example .env
docker compose up --build
```

Le conteneur installe le package (`pip install -e .`, deps incluses dans `pyproject.toml`) et lance
`streamlit run src/psg_tracker/app/main.py` sur le port 8501. Le dossier `data/` est monte en
volume (`docker-compose.yml`) : ingerer/entrainer en local d'abord (commandes ci-dessus) pour que
le dashboard ait des donnees a afficher, ou lancer `scripts/ingest.py`/`scripts/train.py` dans le
conteneur (`docker compose run app python scripts/ingest.py ...`). *Configuration non verifiee par
un build reel dans cet environnement (Docker indisponible ici) : verifiee par relecture uniquement
(le pipeline `pip install -e .` requiert `README.md`, present dans l'image via le Dockerfile).*

## Statut

Projet en developpement actif (vitrine technique Data Science / Data Engineering).

- [x] Jour 1 : scaffold, ingestion StatsBomb (historique) + Understat (saison en cours), DuckDB
- [x] Jour 2 : feature engineering, modele xG (XGBoost), explicabilite SHAP
- [x] Ingestion reelle StatsBomb : 95 matchs / 1432 tirs / 235 buts PSG en base (`scripts/ingest.py`)
- [x] Ingestion reelle Understat : 2 matchs / 31 tirs (saison 2026/2027, scraping via endpoints
      AJAX internes, cf. section Understat ci-dessus)
- [x] Entrainement reel du modele xG sur les 1463 tirs combines (`scripts/train.py`),
      ROC-AUC 0.695 sur le jeu de test
- [x] Jour 3 : dashboard Streamlit (shotmap, classement xG, explicabilite SHAP), teste via AppTest
- [x] CI/CD : GitHub Actions (ruff + mypy strict + pytest/coverage sur `src`+`scripts`+`tests`)
- [ ] Docker : configuration ecrite et relue, mais pas buildee dans cet environnement (pas de
      Docker disponible ici) — a valider en local avant publication
