"""Tests unitaires du client StatsBomb (reponses API mockees, aucun appel reseau)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from psg_tracker.ingestion.statsbomb_client import StatsBombClient

MATCHES_FIXTURE: list[dict[str, Any]] = [
    {
        "match_id": 1,
        "match_date": "2016-05-14",
        "home_team": {"home_team_name": "Paris Saint-Germain"},
        "away_team": {"away_team_name": "Nantes"},
        "competition": {"competition_name": "Ligue 1"},
        "season": {"season_name": "2015/2016"},
    },
    {
        "match_id": 2,
        "match_date": "2016-04-09",
        "home_team": {"home_team_name": "Lyon"},
        "away_team": {"away_team_name": "Marseille"},
        "competition": {"competition_name": "Ligue 1"},
        "season": {"season_name": "2015/2016"},
    },
]

EVENTS_FIXTURE: list[dict[str, Any]] = [
    {
        "id": "evt-1",
        "type": {"name": "Pass"},
        "team": {"id": 771, "name": "Paris Saint-Germain"},
    },
    {
        "id": "evt-2",
        "type": {"name": "Shot"},
        "team": {"id": 771, "name": "Paris Saint-Germain"},
        "player": {"id": 10, "name": "Zlatan Ibrahimovic"},
        "minute": 23,
        "second": 5,
        "location": [110.5, 38.0],
        "shot": {
            "body_part": {"name": "Right Foot"},
            "type": {"name": "Open Play"},
            "outcome": {"name": "Goal"},
        },
    },
    {
        "id": "evt-3",
        "type": {"name": "Shot"},
        "team": {"id": 999, "name": "Nantes"},
        "player": {"id": 20, "name": "Adversaire"},
        "minute": 40,
        "second": 12,
        "location": [100.0, 30.0],
        "shot": {
            "body_part": {"name": "Left Foot"},
            "type": {"name": "Open Play"},
            "outcome": {"name": "Off T"},
        },
    },
]


def _mock_response(payload: object) -> MagicMock:
    """Construit une fausse `requests.Response` renvoyant `payload` en JSON."""
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def client() -> StatsBombClient:
    return StatsBombClient()


def test_get_competitions_returns_parsed_json(client: StatsBombClient) -> None:
    payload = [{"competition_id": 7, "season_id": 27, "competition_name": "Ligue 1"}]
    with patch.object(client._session, "get", return_value=_mock_response(payload)) as mock_get:
        result = client.get_competitions()

    assert result == payload
    mock_get.assert_called_once()
    assert mock_get.call_args.args[0].endswith("competitions.json")


def test_get_matches_filters_by_team_name(client: StatsBombClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_response(MATCHES_FIXTURE)):
        matches = client.get_matches(
            competition_id=7, season_id=27, team_name="Paris Saint-Germain"
        )

    assert len(matches) == 1
    assert matches[0].match_id == 1
    assert matches[0].home_team == "Paris Saint-Germain"


def test_get_matches_without_filter_returns_all(client: StatsBombClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_response(MATCHES_FIXTURE)):
        matches = client.get_matches(competition_id=7, season_id=27)

    assert len(matches) == 2


def test_get_shot_events_ignores_non_shot_events(client: StatsBombClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_response(EVENTS_FIXTURE)):
        shots = client.get_shot_events(match_id=1)

    assert len(shots) == 2
    assert {s.event_id for s in shots} == {"evt-2", "evt-3"}


def test_get_shot_events_filters_by_team_and_flags_goal(client: StatsBombClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_response(EVENTS_FIXTURE)):
        shots = client.get_shot_events(match_id=1, team_name="Paris Saint-Germain")

    assert len(shots) == 1
    shot = shots[0]
    assert shot.player_name == "Zlatan Ibrahimovic"
    assert shot.is_goal is True
    assert shot.location.x == 110.5
    assert shot.location.y == 38.0


def test_get_shot_events_non_goal_is_flagged_false(client: StatsBombClient) -> None:
    with patch.object(client._session, "get", return_value=_mock_response(EVENTS_FIXTURE)):
        shots = client.get_shot_events(match_id=1, team_name="Nantes")

    assert len(shots) == 1
    assert shots[0].is_goal is False
    assert shots[0].outcome == "Off T"


def test_get_json_propagates_http_error(client: StatsBombClient) -> None:
    response = MagicMock()
    response.raise_for_status.side_effect = requests_http_error()
    with patch.object(client._session, "get", return_value=response):
        with pytest.raises(Exception):  # noqa: B017 - on verifie juste la propagation
            client.get_competitions()


def requests_http_error() -> Exception:
    import requests

    return requests.HTTPError("404 Not Found")
