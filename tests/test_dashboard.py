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
    # "Analyse ICT" tab: bias cards, AMD badge, score panel, session clock
    assert any(label.startswith("Biais") for label in labels)
    assert "Phase AMD" in labels
    assert "Haussier" in labels and "Baissier" in labels   # confluence scores
    assert "Session active" in labels
