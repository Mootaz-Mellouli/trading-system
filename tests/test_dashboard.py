"""Headless smoke test: the dashboard must load, compute and render
all tabs without raising, using the synthetic data source."""
from streamlit.testing.v1 import AppTest


def test_dashboard_runs_headless():
    at = AppTest.from_file("dashboard/app.py", default_timeout=120)
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    for expected in ("Candles", "FVGs detected", "Signals", "Trades"):
        assert expected in labels
    # "Analyse ICT" tab: bias cards (one per timeframe) + session clock
    assert any(label.startswith("Biais") for label in labels)
    assert "Session active" in labels
