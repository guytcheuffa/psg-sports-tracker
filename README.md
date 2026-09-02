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
    par StatsBomb Open Data, pour alimenter le dashboard "live". Coordonnees normalisees vers le
    referentiel StatsBomb (120x80) a l'ingestion pour un feature engineering unifie. Scraping non
    testable depuis certains environnements (bloque par des politiques reseau restrictives) mais
    fonctionnel en execution normale.
  - `scripts/ingest.py` : script d'orchestration idempotent (`INSERT OR REPLACE`), utilisable en
    mode decouverte automatique ou competition/saison par competition/saison.
- **Stockage** : DuckDB, transformations SQL typees (cle composite `source + match_id`)
- **ML** : XGBoost (classification binaire xG), SHAP (explicabilite)
- **Visualisation** : Streamlit + Plotly (shotmap, xG cumule, dashboard joueur)
- **DevOps** : Docker, GitHub Actions (lint/mypy/pytest), pytest

## Architecture

```
src/psg_tracker/
├── ingestion/   # client API StatsBomb + schemas typees
├── storage/     # DuckDB (DDL + transformations SQL)
├── features/    # feature engineering xG (partage train/inference)
├── models/      # entrainement XGBoost + explicabilite SHAP
└── app/         # dashboard Streamlit
```

## Setup

```bash
make setup      # venv + installation
make test       # tests + coverage
make lint       # ruff
make typecheck  # mypy
make run        # lance le dashboard Streamlit
```

## Statut

Projet en developpement actif (vitrine technique Data Science / Data Engineering).

- [x] Jour 1 : scaffold, ingestion StatsBomb (historique) + Understat (saison en cours), DuckDB
- [x] Jour 2 : feature engineering, modele xG (XGBoost), explicabilite SHAP
- [x] Ingestion reelle : 95 matchs / 1432 tirs / 235 buts PSG en base (`scripts/ingest.py`)
- [ ] Jour 3 : dashboard Streamlit, Docker, CI/CD
