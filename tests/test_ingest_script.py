"""Tests unitaires de la logique d'orchestration de `scripts/ingest.py`.

Couvre les fonctions pures (dedup inter-sources, formatage de saison) et le
garde-fou sur une liste de saisons vide - jusqu'ici non testees (seul le
`StatsBombClient`/`UnderstatClient` avait des tests, pas le script qui les
orchestre). Importe via `pythonpath = ["src", "scripts"]` (cf. pyproject.toml) :
`scripts/ingest.py` est un point d'entree CLI, pas un module du package
`psg_tracker` installe.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ingest import (
    _date_matches_within_one_day,
    _understat_season_to_range,
    ingest_understat_seasons,
)


def test_date_matches_within_one_day_exact_match() -> None:
    assert _date_matches_within_one_day("2022-08-14", {"2022-08-14"})


def test_date_matches_within_one_day_tolerates_one_day_offset() -> None:
    # Ecart d'un jour constate empiriquement entre StatsBomb et Understat
    # (cf. docstring) : doit etre traite comme le meme match.
    assert _date_matches_within_one_day("2022-08-14", {"2022-08-13"})
    assert _date_matches_within_one_day("2022-08-14", {"2022-08-15"})


def test_date_matches_within_one_day_rejects_larger_gap() -> None:
    assert not _date_matches_within_one_day("2022-08-14", {"2022-08-16"})


def test_date_matches_within_one_day_empty_reference_set() -> None:
    assert not _date_matches_within_one_day("2022-08-14", set())


def test_date_matches_within_one_day_falls_back_on_unparseable_date() -> None:
    # Chaine de date invalide (ex. champ manquant/vide en amont) : pas de
    # crash, comparaison stricte en repli plutot qu'une exception propagee.
    assert not _date_matches_within_one_day("", {"2022-08-14"})
    assert _date_matches_within_one_day("not-a-date", {"not-a-date"})


def test_understat_season_to_range_formats_start_and_end_year() -> None:
    assert _understat_season_to_range("2015") == "2015/2016"
    assert _understat_season_to_range("2026") == "2026/2027"


def test_ingest_understat_seasons_with_empty_list_does_not_crash() -> None:
    # Bug corrige : `seasons[0]`/`seasons[-1]` (log final) levaient IndexError
    # sur une liste vide, ex. `--from-season` poste apres la saison en cours.
    manager = MagicMock()

    ingest_understat_seasons(manager, seasons=[])

    manager.get_match_dates.assert_not_called()
    manager.insert_matches.assert_not_called()
