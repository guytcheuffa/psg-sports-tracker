"""Point d'entree de l'application Streamlit : dashboard PSG xG."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Lance le dashboard (shotmap, xG cumule, stats joueur)."""
    st.set_page_config(page_title="PSG Live Tracker", layout="wide")
    st.title("PSG Live Sports Tracker")
    st.info("Pipeline en cours de construction - Jour 1: ingestion & storage.")


if __name__ == "__main__":
    main()
