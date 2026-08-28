# PSG Live Sports Tracker

Pipeline data engineering & data science autour des donnees evenementielles
du Paris Saint-Germain : ingestion, stockage analytique, modele de
probabilite de but (xG), et dashboard interactif.

## Stack

- **Ingestion** : StatsBomb Open Data (coordonnees spatiales x/y des events)
- **Stockage** : DuckDB, transformations SQL typees
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

- [x] Jour 1 : scaffold, ingestion StatsBomb, DuckDB
- [ ] Jour 2 : feature engineering, modele xG, SHAP
- [ ] Jour 3 : dashboard Streamlit, Docker, CI/CD
