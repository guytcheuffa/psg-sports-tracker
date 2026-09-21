*[English version](README.en.md)*

# PSG Sports Tracker

Pipeline data engineering & data science autour des donnees evenementielles
du Paris Saint-Germain : ingestion, stockage analytique, modele de
probabilite de but (xG), et dashboard interactif.

## Stack

- **Ingestion (hybride)** :
  - [StatsBomb Open Data](https://github.com/statsbomb/open-data) — corpus "riche" (evenements
    detailles) utilise en priorite pour entrainer le modele xG. En pratique, seule la Ligue 1
    (2015/16, 2021/22, 2022/23) contient des donnees PSG dans ce dataset ouvert, et de facon
    incomplete pour 2 des 3 saisons (26/38 et 32/38 matchs) : les 18 saisons de Champions League
    listees dans `competitions.json` (1999/2000-2018/19) ne couvrent pas les matchs du PSG
    (verifie empiriquement, pas seulement suppose). Corpus reel actuellement ingere : **95 matchs,
    1432 tirs, 235 buts**.
  - [Understat](https://understat.com) (scraping) — couvre en realite tout l'historique du club
    depuis 2014/2015 (pas seulement la saison en cours, comme suppose dans une premiere version de
    ce projet : `getTeamData/{team}/{season}` accepte n'importe quelle saison passee, pas
    seulement la courante). Sert donc a la fois a **combler les 9 saisons que StatsBomb ne
    couvre pas du tout** (2016/17-2020/21, 2023/24-2025/26) et a completer les 3 saisons qu'il
    couvre partiellement. Understat n'expose pas d'API publique documentee et ne sert plus les
    donnees en JS inline (`var shotsData = ...`, pattern largement documente mais obsolete) : le
    client interroge directement les endpoints AJAX internes du site (`getTeamData/{team}/{season}`,
    `getMatchData/{match_id}`, identifies en inspectant `js/team.min.js`/`js/match.min.js`), qui
    repondent 404 sans l'en-tete `X-Requested-With: XMLHttpRequest`. Categories harmonisees vers
    le vocabulaire StatsBomb a l'ingestion (`OpenPlay` -> `Open Play`, `RightFoot` -> `Right
    Foot`, etc.) et format de saison harmonise ("2015" -> "2015/2016") pour eviter des colonnes
    one-hot dupliquees / un affichage de saison incoherent par source. Coordonnees normalisees
    vers le referentiel StatsBomb (120x80).
    - *Deduplication inter-sources* : sur les 3 saisons ou StatsBomb et Understat se recoupent, un
      meme match reel apparaitrait deux fois (un `match_id` par source) sans filtrage - StatsBomb
      est prioritaire (plus riche), Understat ne comble que les dates que StatsBomb n'a pas. Le
      rapprochement tolere +/-1 jour : verifie empiriquement, ~6 matchs/saison ont une date de
      coup d'envoi decalee d'un jour entre les deux sources (probable artefact de fuseau horaire
      sur les matchs en soiree), une egalite stricte aurait laisse passer de vrais doublons.
    - *Homonymes inter-sources* : StatsBomb utilise le nom complet a l'etat civil, Understat le
      nom d'usage/media (ex. "Achraf Hakimi Mouh" vs "Achraf Hakimi") - sans harmonisation, un
      meme joueur apparaissait deux fois dans le classement/profil. 22 alias recenses par
      rapprochement de chaines (`difflib`) + verification manuelle, appliques au chargement dans
      `app/data_loader.py` (cf. commentaire du dictionnaire `_PLAYER_NAME_ALIASES`).
    - *Limite de donnee connue* : sur les 2 premiers matchs de la saison 2026/2027, les scores/xG
      d'equipe renvoyes par Understat sont coherents, mais les noms de joueurs associes aux tirs
      sont visiblement errones (des noms d'une autre equipe, verifie en comparant a une saison
      complete ou les memes appels renvoient des noms corrects). C'est un probleme cote Understat
      (donnee de debut de saison pas encore stabilisee), pas un bug du client. Sans impact sur le
      modele xG : les features utilisees ne dependent pas de l'identite du joueur.
    - Corpus reel actuellement ingere : **397 matchs, 4851 tirs, 721 buts** (couvre 2014/2015 a
      2026/2027 - la portion `--from-season` du backfill historique, hors les 2 derniers matchs
      "live" de la saison en cours).
  - `scripts/ingest.py` : script d'orchestration idempotent (`INSERT OR REPLACE`). StatsBomb :
    mode decouverte automatique ou competition/saison par competition/saison. Understat : une
    seule saison (`--season`) ou backfill historique complet (`--from-season 2015`).
  - `scripts/train.py` : entraine le modele xG sur l'integralite des tirs reels en base (les deux
    sources, deja deduplique a l'ingestion). Split train/test (80/20 stratifie) pour une
    evaluation honnete, puis reentrainement final sur 100% des donnees pour le modele sauvegarde.
    Resultat actuel (6283 tirs reels : 1432 StatsBomb + 4851 Understat) : **ROC-AUC 0.772, log
    loss 0.355** sur le jeu de test (1257 tirs) - en nette hausse par rapport a la version limitee
    a 4 saisons partielles (ROC-AUC 0.695), grace au volume et a la diversite de situations de tir
    supplementaires. Le split train/test se fait sur les tirs bruts *avant* le feature
    engineering : la feature `is_strong_foot` (pied dominant du joueur) est fit sur le train
    uniquement puis appliquee au test, pour eviter une fuite d'information (sinon les tirs de
    test contribuent eux-memes a definir le pied dominant utilise pour les evaluer).
- **Stockage** : DuckDB, transformations SQL typees (cle composite `source + match_id`)
- **ML** : XGBoost (classification binaire xG), SHAP (explicabilite). Features : geometrie
  (distance au but, angle de tir), technique (tete, pied dominant), contexte (type de tir en
  one-hot). *Choix deliberement ecarte* : StatsBomb expose aussi `technique`, `first_time`,
  `play_pattern` et surtout `freeze_frame` (position de tous les joueurs au moment du tir, la
  feature qui distingue generalement un xG amateur d'un xG professionnel) — non integres car
  disponibles uniquement sur les 3 saisons StatsBomb (memes limites que ci-dessus) et parce que
  `freeze_frame` demanderait un feature engineering geometrique nettement plus complexe pour un
  gain limite a une fraction du dataset. Detaille dans l'onglet "Methodologie" du dashboard.
- **Visualisation** : Streamlit + Plotly, six onglets — vue d'ensemble (shotmap demi-terrain,
  taille = xG, couleur = but/non-but, + evolution buts reels vs xG par saison), classement buts
  reels vs xG cumule par joueur (tri au choix : buts reels ou volume de tirs), profil joueur,
  explicabilite SHAP d'un tir choisi (+ heatmap de densite contextuelle), diagnostic du modele
  (courbe ROC + courbe de calibration sur le jeu de test hold-out reel, pas recalculees a partir du
  modele final reentraine sur 100% des donnees, ce qui serait circulaire), et methodologie (sources,
  pipeline, features du modele, limites connues — centralise dans le dashboard plutot que laisse
  uniquement dans ce README). Teste via `streamlit.testing.v1.AppTest` (`tests/test_app.py`, base +
  modele synthetiques en base temporaire, pas de dependance aux vraies donnees en CI).
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
python scripts/ingest.py statsbomb
python scripts/ingest.py understat --from-season 2015
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
conteneur (`docker compose run app python scripts/ingest.py ...`). *Build + lancement reels
valides localement (`docker compose up --build`) : dashboard fonctionnel sur `localhost:8501`,
theme sombre et donnees (montees via volume) correctement charges.*

## Statut

Projet en developpement actif (vitrine technique Data Science / Data Engineering).

- [x] Jour 1 : scaffold, ingestion StatsBomb (historique) + Understat (saison en cours), DuckDB
- [x] Jour 2 : feature engineering, modele xG (XGBoost), explicabilite SHAP
- [x] Ingestion reelle StatsBomb : 95 matchs / 1432 tirs / 235 buts PSG en base (`scripts/ingest.py`)
- [x] Ingestion reelle Understat : backfill historique complet 2015-2026 (scraping via endpoints
      AJAX internes, dedup inter-sources +/-1 jour, cf. section Understat ci-dessus) - 302 matchs
      / 4851 tirs / 721 buts
- [x] Entrainement reel du modele xG sur les 6283 tirs combines (`scripts/train.py`),
      ROC-AUC 0.772 sur le jeu de test (split sans fuite train/test)
- [x] Jour 3 : dashboard Streamlit (shotmap, classement xG, explicabilite SHAP), teste via AppTest
- [x] Correction d'une fuite train/test sur `is_strong_foot` (fit train uniquement / transform
      train+test) + onglet Diagnostic du modele (ROC, calibration, sur predictions hold-out reelles)
- [x] Onglet Methodologie (sources, dedup, pipeline, features, limites connues) + classement a tri
      au choix (buts reels / volume de tirs)
- [x] CI/CD : GitHub Actions (ruff + mypy strict + pytest/coverage sur `src`+`scripts`+`tests`)
- [x] Docker : build + lancement reels valides en local (`docker compose up --build`), dashboard
      fonctionnel sur `localhost:8501` (theme + donnees corrects)

## Prochaines etapes

Ameliorations identifiees mais deliberement laissees hors scope de cette premiere version, pour
rester concentre sur la profondeur du pipeline data/modele plutot que sur l'automatisation :

- **Scraping quotidien automatise (GitHub Actions)** : aujourd'hui, l'ingestion Understat est
  declenchee manuellement (`python scripts/ingest.py understat ...`) - le nom du projet reflete
  l'intention ("live" au sens "saison en cours", pas "temps reel"), pas encore un rafraichissement
  automatique. Prochaine etape : un workflow `schedule` (cron quotidien) qui installe les deps,
  relance `scripts/ingest.py understat --season <saison en cours>`, puis commit + push
  automatiquement le fichier DuckDB mis a jour (`GITHUB_TOKEN` par defaut). Question de design
  ouverte a trancher avant implementation : la pertinence d'un commit "bot" quotidien sur un
  fichier binaire qui grossit progressivement (vs. par ex. un stockage externe versionne).
- **Reentrainement periodique du modele** : decouple volontairement du scraping quotidien - un
  seul jour de nouveaux tirs ne fait pas bouger un ROC-AUC de facon statistiquement significative.
  Piste : un reentrainement (`scripts/train.py`) au rythme de la saison (ex. mensuel, ou apres
  chaque trève internationale) plutot qu'a chaque ingestion.
- **Deploiement continu du dashboard** : avec le scraping automatise ci-dessus, un deploiement sur
  une plateforme qui se redeploie a chaque push (ex. Streamlit Community Cloud) permettrait aux
  visiteurs de voir des donnees a jour sans etape manuelle (`git pull` + relance locale).

Ces trois points sont techniquement simples individuellement ; ils ont ete documentes ici comme
axes d'evolution plutot qu'implementes, pour prioriser la profondeur du pipeline existant (dedup
inter-sources, correction de fuite train/test, explicabilite SHAP, etc.) dans le temps disponible.
