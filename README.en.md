*[Version française](README.md)*

# PSG Sports Tracker

Data engineering & data science pipeline built around Paris Saint-Germain event
data: ingestion, analytical storage, an expected-goals (xG) probability model,
and an interactive dashboard.

## Stack

- **Ingestion (hybrid)**:
  - [StatsBomb Open Data](https://github.com/statsbomb/open-data) — the "rich" corpus
    (detailed events), used primarily to train the xG model. In practice, only Ligue 1
    (2015/16, 2021/22, 2022/23) contains PSG data in this open dataset, and incompletely
    for 2 of the 3 seasons (26/38 and 32/38 matches): the 18 Champions League seasons
    listed in `competitions.json` (1999/2000-2018/19) do not cover PSG matches (verified
    empirically, not just assumed). Actual corpus currently ingested: **95 matches,
    1,432 shots, 235 goals**.
  - [Understat](https://understat.com) (scraping) — actually covers the club's entire
    history since 2014/2015 (not just the current season, as assumed in an early version
    of this project: `getTeamData/{team}/{season}` accepts any past season, not only the
    current one). It therefore serves both to **fill the 9 seasons StatsBomb doesn't
    cover at all** (2016/17-2020/21, 2023/24-2025/26) and to complete the 3 seasons it
    covers partially. Understat exposes no documented public API and no longer serves
    data as inline JS (`var shotsData = ...`, a widely documented but outdated pattern):
    the client queries the site's internal AJAX endpoints directly
    (`getTeamData/{team}/{season}`, `getMatchData/{match_id}`, identified by inspecting
    `js/team.min.js`/`js/match.min.js`), which return 404 without the
    `X-Requested-With: XMLHttpRequest` header. Categories are harmonized to StatsBomb's
    vocabulary at ingestion time (`OpenPlay` -> `Open Play`, `RightFoot` -> `Right Foot`,
    etc.) and season format is harmonized ("2015" -> "2015/2016") to avoid duplicated
    one-hot columns / an inconsistent season display per source. Coordinates normalized
    to the StatsBomb reference frame (120x80).
    - *Cross-source deduplication*: on the 3 seasons where StatsBomb and Understat
      overlap, the same real match would appear twice (one `match_id` per source)
      without filtering — StatsBomb takes priority (richer), Understat only fills in the
      dates StatsBomb is missing. Matching tolerates +/-1 day: verified empirically,
      ~6 matches/season have a kickoff date offset by one day between the two sources
      (likely a timezone artifact for evening matches); a strict equality check would
      have let real duplicates through.
    - *Cross-source homonyms*: StatsBomb uses the full legal name, Understat the
      common/media name (e.g. "Achraf Hakimi Mouh" vs "Achraf Hakimi") — without
      harmonization, the same player appeared twice in the ranking/profile. 22 aliases
      identified via string matching (`difflib`) + manual verification, applied at load
      time in `app/data_loader.py` (see the comment on the `_PLAYER_NAME_ALIASES`
      dictionary).
    - *Known data limitation*: for the first 2 matches of the 2026/2027 season, the team
      scores/xG returned by Understat are consistent, but the player names attached to
      shots are visibly wrong (names from another team, verified by comparing against a
      full season where the same calls return correct names). This is an Understat-side
      issue (early-season data not yet stabilized), not a client bug. No impact on the
      xG model: the features used don't depend on player identity.
    - Actual corpus currently ingested: **397 matches, 4,851 shots, 721 goals** (covers
      2014/2015 to 2026/2027 — the `--from-season` portion of the historical backfill,
      excluding the last 2 "live" matches of the current season).
  - `scripts/ingest.py`: idempotent orchestration script (`INSERT OR REPLACE`).
    StatsBomb: automatic discovery mode or competition/season by competition/season.
    Understat: a single season (`--season`) or full historical backfill
    (`--from-season 2015`).
  - `scripts/train.py`: trains the xG model on the full set of real shots in the
    database (both sources, already deduplicated at ingestion). Train/test split
    (80/20 stratified) for an honest evaluation, then final retraining on 100% of the
    data for the saved model. Current result (6,283 real shots: 1,432 StatsBomb +
    4,851 Understat): **ROC-AUC 0.772, log loss 0.355** on the test set (1,257 shots)
    — a clear improvement over the version limited to 4 partial seasons
    (ROC-AUC 0.695), thanks to the additional volume and diversity of shot situations.
    The train/test split happens on raw shots *before* feature engineering: the
    `is_strong_foot` feature (player's dominant foot) is fit on train only, then applied
    to test, to avoid information leakage (otherwise the test shots would themselves
    help define the dominant foot used to evaluate them).
- **Storage**: DuckDB, typed SQL transformations (composite key `source + match_id`)
- **ML**: XGBoost (binary xG classification), SHAP (explainability). Features:
  geometry (distance to goal, shot angle), technique (header, dominant foot), context
  (one-hot shot type). *Deliberately excluded*: StatsBomb also exposes `technique`,
  `first_time`, `play_pattern`, and above all `freeze_frame` (the position of every
  player at the moment of the shot — the feature that typically separates an amateur
  xG model from a professional one) — not included because it's only available for the
  3 StatsBomb seasons (same limitations as above) and because `freeze_frame` would
  require significantly more complex geometric feature engineering for a gain limited
  to a fraction of the dataset. Detailed in the dashboard's "Methodology" tab.
- **Visualization**: Streamlit + Plotly, six tabs — overview (half-pitch shotmap,
  size = xG, color = goal/no goal, + real goals vs xG trend by season), real goals vs
  cumulative xG ranking by player (sortable by real goals or shot volume), player
  profile, SHAP explainability for a selected shot (+ contextual density heatmap),
  model diagnostics (ROC curve + calibration curve on the real hold-out test set, not
  recomputed from the final model retrained on 100% of the data, which would be
  circular), and methodology (sources, pipeline, model features, known limitations —
  centralized in the dashboard rather than left only in this README). Tested via
  `streamlit.testing.v1.AppTest` (`tests/test_app.py`, synthetic data + model in a
  temporary database, no dependency on real data in CI).
- **DevOps**: Docker (multi-copy, `README.md` required by `pyproject.toml`),
  GitHub Actions (lint/mypy/pytest on `src`+`scripts`+`tests`), pytest + pytest-cov

## Architecture

```
src/psg_tracker/
├── ingestion/   # StatsBomb + Understat clients, typed schemas
├── storage/     # DuckDB (DDL + SQL transformations)
├── features/    # xG feature engineering (shared train/inference)
├── models/      # XGBoost training + SHAP explainability
└── app/         # Streamlit dashboard (main.py + data_loader.py + pitch.py)
scripts/
├── ingest.py    # StatsBomb/Understat ingestion orchestration -> DuckDB
└── train.py     # xG training on the real data in the database
```

## Setup

```bash
make setup      # venv + install
make test       # tests + coverage
make lint       # ruff (src, tests, scripts)
make typecheck  # strict mypy (src, scripts)
make run        # launches the Streamlit dashboard (requires data + model, see below)
```

Before launching the dashboard: ingest the data, then train the model (once):

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

The container installs the package (`pip install -e .`, deps included in
`pyproject.toml`) and launches `streamlit run src/psg_tracker/app/main.py` on port
8501. The `data/` folder is mounted as a volume (`docker-compose.yml`): ingest/train
locally first (commands above) so the dashboard has data to display, or run
`scripts/ingest.py`/`scripts/train.py` inside the container
(`docker compose run app python scripts/ingest.py ...`). *Real build + run validated
locally (`docker compose up --build`): dashboard working on `localhost:8501`, dark
theme and data (mounted via volume) loading correctly.*

## Status

Project in active development (Data Science / Data Engineering technical showcase).

- [x] Day 1: scaffold, StatsBomb ingestion (historical) + Understat (current season),
      DuckDB
- [x] Day 2: feature engineering, xG model (XGBoost), SHAP explainability
- [x] Real StatsBomb ingestion: 95 matches / 1,432 shots / 235 PSG goals in the
      database (`scripts/ingest.py`)
- [x] Real Understat ingestion: full historical backfill 2015-2026 (scraping via
      internal AJAX endpoints, cross-source dedup +/-1 day, see Understat section
      above) — 302 matches / 4,851 shots / 721 goals
- [x] Real xG model training on the 6,283 combined shots (`scripts/train.py`),
      ROC-AUC 0.772 on the test set (leak-free split)
- [x] Day 3: Streamlit dashboard (shotmap, xG ranking, SHAP explainability), tested
      via AppTest
- [x] Fixed a train/test leak on `is_strong_foot` (fit on train only / transform
      train+test) + Model Diagnostics tab (ROC, calibration, on real hold-out
      predictions)
- [x] Methodology tab (sources, dedup, pipeline, features, known limitations) +
      ranking sortable by choice (real goals / shot volume)
- [x] CI/CD: GitHub Actions (ruff + strict mypy + pytest/coverage on
      `src`+`scripts`+`tests`)
- [x] Docker: real build + run validated locally (`docker compose up --build`),
      dashboard working on `localhost:8501` (theme + data correct)

## Next steps

Improvements identified but deliberately left out of scope for this first version, to
stay focused on the depth of the data/model pipeline rather than on automation:

- **Automated daily scraping (GitHub Actions)**: today, Understat ingestion is
  triggered manually (`python scripts/ingest.py understat ...`) — the project's name
  reflects the intent ("live" in the sense of "current season", not "real-time" yet),
  not yet an automatic refresh. Next step: a `schedule` workflow (daily cron) that
  installs deps, reruns `scripts/ingest.py understat --season <current season>`, then
  automatically commits + pushes the updated DuckDB file (default `GITHUB_TOKEN`).
  Open design question to settle before implementation: whether a daily "bot" commit
  on a binary file that keeps growing makes sense (vs., say, external versioned
  storage).
- **Periodic model retraining**: deliberately decoupled from daily scraping — a single
  day of new shots won't move a ROC-AUC in a statistically significant way. Direction:
  retraining (`scripts/train.py`) on a season-level cadence (e.g. monthly, or after
  each international break) rather than on every ingestion.
- **Continuous dashboard deployment**: with the automated scraping above, deploying to
  a platform that redeploys on every push (e.g. Streamlit Community Cloud) would let
  visitors see up-to-date data without a manual step (`git pull` + local relaunch).

These three points are each simple on their own from a technical standpoint; they are
documented here as directions for future work rather than implemented, in order to
prioritize the depth of the existing pipeline (cross-source dedup, train/test leak
fix, SHAP explainability, etc.) within the available time.
