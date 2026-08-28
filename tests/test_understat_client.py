"""Tests unitaires du client Understat (pages HTML mockees, aucun appel reseau)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from psg_tracker.ingestion.understat_client import UnderstatClient

DATES_DATA: list[dict[str, Any]] = [
    {
        "id": "101",
        "isResult": True,
        "h": {"title": "Paris Saint Germain"},
        "a": {"title": "Marseille"},
    },
    {
        "id": "102",
        "isResult": False,
        "h": {"title": "Paris Saint Germain"},
        "a": {"title": "Lyon"},
    },
]

SHOTS_DATA: dict[str, list[dict[str, Any]]] = {
    "h": [
        {
            "id": "5001",
            "minute": "23",
            "result": "Goal",
            "X": "0.91",
            "Y": "0.52",
            "xG": "0.34",
            "player": "Kylian Mbappe",
            "player_id": "2371",
            "h_a": "h",
            "h_team": "Paris Saint Germain",
            "a_team": "Marseille",
            "situation": "OpenPlay",
            "shotType": "RightFoot",
        }
    ],
    "a": [
        {
            "id": "5002",
            "minute": "60",
            "result": "MissedShots",
            "X": "0.85",
            "Y": "0.40",
            "xG": "0.10",
            "player": "Adversaire",
            "player_id": "9999",
            "h_a": "a",
            "h_team": "Paris Saint Germain",
            "a_team": "Marseille",
            "situation": "FromCorner",
            "shotType": "Head",
        }
    ],
}


def _js_escaped_var(var_name: str, payload: object) -> str:
    """Reproduit le format d'echappement JS utilise par understat.com."""
    raw_json = json.dumps(payload)
    # understat echappe les guillemets simples et non-ASCII a la maniere JS
    escaped = raw_json.replace("\\", "\\\\").replace("'", "\\'")
    return f"var {var_name} = JSON.parse('{escaped}');"


def _html_page(var_name: str, payload: object) -> str:
    return f"<html><body><script>{_js_escaped_var(var_name, payload)}</script></body></html>"


def _mock_response(html: str) -> MagicMock:
    response = MagicMock()
    response.text = html
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def client() -> UnderstatClient:
    return UnderstatClient(request_delay_seconds=0.0)


def test_extract_json_var_decodes_standard_json_escaping(client: UnderstatClient) -> None:
    html = _html_page("datesData", DATES_DATA)
    result = client._extract_json_var(html, "datesData")
    assert result == DATES_DATA


def test_extract_json_var_missing_variable_raises(client: UnderstatClient) -> None:
    html = "<html><body><script>var other = 1;</script></body></html>"
    with pytest.raises(ValueError, match="datesData"):
        client._extract_json_var(html, "datesData")


def test_get_team_matches_parses_dates_data(client: UnderstatClient) -> None:
    html = _html_page("datesData", DATES_DATA)
    with patch.object(client._session, "get", return_value=_mock_response(html)):
        matches = client.get_team_matches("Paris_Saint_Germain", "2026")

    assert len(matches) == 2
    assert matches[0]["id"] == "101"


def test_get_match_shots_parses_both_sides_and_scales_coordinates(client: UnderstatClient) -> None:
    html = _html_page("shotsData", SHOTS_DATA)
    with patch.object(client._session, "get", return_value=_mock_response(html)):
        shots = client.get_match_shots(match_id=101)

    assert len(shots) == 2
    goal = next(s for s in shots if s.is_goal)
    assert goal.player_name == "Kylian Mbappe"
    assert goal.location.x == pytest.approx(0.91 * 120.0)
    assert goal.location.y == pytest.approx(0.52 * 80.0)
    assert goal.source == "understat"
    assert goal.event_id == "understat-5001"


def test_get_match_shots_filters_by_team_name(client: UnderstatClient) -> None:
    html = _html_page("shotsData", SHOTS_DATA)
    with patch.object(client._session, "get", return_value=_mock_response(html)):
        shots = client.get_match_shots(match_id=101, team_name="Marseille")

    assert len(shots) == 1
    assert shots[0].outcome == "Off T"
    assert shots[0].is_goal is False


def test_get_match_shots_maps_result_to_outcome(client: UnderstatClient) -> None:
    html = _html_page("shotsData", SHOTS_DATA)
    with patch.object(client._session, "get", return_value=_mock_response(html)):
        shots = client.get_match_shots(match_id=101)

    outcomes = {s.event_id: s.outcome for s in shots}
    assert outcomes["understat-5001"] == "Goal"
    assert outcomes["understat-5002"] == "Off T"


def test_get_current_season_shots_skips_unplayed_matches(client: UnderstatClient) -> None:
    dates_html = _html_page("datesData", DATES_DATA)
    shots_html = _html_page("shotsData", SHOTS_DATA)

    responses = [_mock_response(dates_html), _mock_response(shots_html)]
    with patch.object(client._session, "get", side_effect=responses) as mock_get:
        shots = client.get_current_season_shots(
            "Paris_Saint_Germain", "2026", "Paris Saint Germain"
        )

    # 1 seul match "isResult=True" (match id 102 est ignore) -> seul le tir PSG
    # est garde, celui de Marseille est filtre par team_name.
    assert mock_get.call_count == 2
    assert len(shots) == 1
    assert shots[0].player_name == "Kylian Mbappe"
    assert mock_get.call_args_list[1].args[0].endswith("match/101")
