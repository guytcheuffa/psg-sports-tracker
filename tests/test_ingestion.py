"""Tests unitaires du client StatsBomb (reponses API mockees)."""

from psg_tracker.ingestion.statsbomb_client import StatsBombClient


def test_client_instantiation() -> None:
    client = StatsBombClient()
    assert client is not None
