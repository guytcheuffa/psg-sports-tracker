"""Tests unitaires du client Understat (reponses JSON mockees, aucun appel reseau)."""

from __future__ import annotations

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


def _mock_response(payload: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def client() -> UnderstatClient:
    return UnderstatClient(request_delay_seconds=0.0)


def test_get_team_matches_parses_dates(client: UnderstatClient) -> None:
    payload = {"dates": DATES_DATA, "players": [], "statistics": {}}
    with patch.object(client._session, "get", return_value=_mock_response(payload)) as mock_get:
        matches = client.get_team_matches("Paris_Saint_Germain", "2026")

    assert len(matches) == 2
    assert matches[0]["id"] == "101"
    called_url = mock_get.call_args.args[0]
    assert called_url.endswith("getTeamData/Paris_Saint_Germain/2026")


def test_get_match_shots_parses_both_sides_and_scales_coordinates(client: UnderstatClient) -> None:
    payload = {"shots": SHOTS_DATA, "rosters": {}, "tmpl": {}}
    with patch.object(client._session, "get", return_value=_mock_response(payload)):
        shots = client.get_match_shots(match_id=101)

    assert len(shots) == 2
    goal = next(s for s in shots if s.is_goal)
    assert goal.player_name == "Kylian Mbappe"
    assert goal.location.x == pytest.approx(0.91 * 120.0)
    assert goal.location.y == pytest.approx(0.52 * 80.0)
    assert goal.source == "understat"
    assert goal.event_id == "understat-5001"
    # Vocabulaire Understat brut ("OpenPlay", "RightFoot") remappe vers le
    # vocabulaire canonique StatsBomb ("Open Play", "Right Foot") pour eviter
    # les colonnes one-hot dupliquees entre sources.
    assert goal.shot_type == "Open Play"
    assert goal.body_part == "Right Foot"


def test_get_match_shots_filters_by_team_name(client: UnderstatClient) -> None:
    payload = {"shots": SHOTS_DATA, "rosters": {}, "tmpl": {}}
    with patch.object(client._session, "get", return_value=_mock_response(payload)):
        shots = client.get_match_shots(match_id=101, team_name="Marseille")

    assert len(shots) == 1
    assert shots[0].outcome == "Off T"
    assert shots[0].is_goal is False


def test_get_match_shots_maps_result_to_outcome(client: UnderstatClient) -> None:
    payload = {"shots": SHOTS_DATA, "rosters": {}, "tmpl": {}}
    with patch.object(client._session, "get", return_value=_mock_response(payload)):
        shots = client.get_match_shots(match_id=101)

    outcomes = {s.event_id: s.outcome for s in shots}
    assert outcomes["understat-5001"] == "Goal"
    assert outcomes["understat-5002"] == "Off T"

    by_id = {s.event_id: s for s in shots}
    assert by_id["understat-5002"].shot_type == "Corner"  # "FromCorner" normalise
    assert by_id["understat-5002"].body_part == "Head"


def test_get_match_shots_falls_back_and_warns_on_unknown_category(
    client: UnderstatClient, caplog: pytest.LogCaptureFixture
) -> None:
    unknown_shot = {
        "h": [
            {
                "id": "5003",
                "minute": "10",
                "result": "MissedShots",
                "X": "0.80",
                "Y": "0.45",
                "player": "Joueur Test",
                "player_id": "1",
                "h_a": "h",
                "h_team": "Paris Saint Germain",
                "a_team": "Marseille",
                "situation": "SomeNewCategory",
                "shotType": "SomeNewBodyPart",
            }
        ],
        "a": [],
    }
    payload = {"shots": unknown_shot, "rosters": {}, "tmpl": {}}
    with (
        patch.object(client._session, "get", return_value=_mock_response(payload)),
        caplog.at_level("WARNING"),
    ):
        shots = client.get_match_shots(match_id=101)

    assert shots[0].shot_type == "SomeNewCategory"
    assert shots[0].body_part == "SomeNewBodyPart"
    assert "inconnue" in caplog.text


def test_get_current_season_shots_skips_unplayed_matches(client: UnderstatClient) -> None:
    dates_payload = {"dates": DATES_DATA, "players": [], "statistics": {}}
    shots_payload = {"shots": SHOTS_DATA, "rosters": {}, "tmpl": {}}

    responses = [_mock_response(dates_payload), _mock_response(shots_payload)]
    with patch.object(client._session, "get", side_effect=responses) as mock_get:
        shots = client.get_current_season_shots(
            "Paris_Saint_Germain", "2026", "Paris Saint Germain"
        )

    # 1 seul match "isResult=True" (match id 102 est ignore) -> seul le tir PSG
    # est garde, celui de Marseille est filtre par team_name.
    assert mock_get.call_count == 2
    assert len(shots) == 1
    assert shots[0].player_name == "Kylian Mbappe"
    assert mock_get.call_args_list[1].args[0].endswith("getMatchData/101")


def test_session_sends_xhr_header_required_by_understat() -> None:
    # Verifie empiriquement le 2026-09-02 : ces endpoints renvoient 404 sans
    # cet en-tete (seule protection anti-scraping constatee).
    client = UnderstatClient()
    assert client._session.headers["X-Requested-With"] == "XMLHttpRequest"
