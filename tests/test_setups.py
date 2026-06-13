"""Trade setup builder: exact entry/SL/TP levels, the OTE->OB->FVG fallback,
both directions, the Signal bridge, knobs, and 'suggestion not promise'."""
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from core.detectors.ote import OTEZone
from core.analysis.setups import TradeSetup, build_setup

TS = pd.Timestamp("2026-01-05 12:00", tz="UTC")


def _analysis(nearest_zones=()):
    return NS(symbol="X", generated_at=TS, nearest_zones=list(nearest_zones))


def _scores(bull, bear):
    return NS(bullish=NS(strength=bull), bearish=NS(strength=bear))


def _bull_ote():
    return OTEZone(direction="bullish", created_at=TS, confirmed_at=TS,
                   leg_low=1.0000, leg_high=1.0100,
                   top=1.0038, bottom=1.0021, sweet=1.00295)


def test_bullish_ote_setup_exact_levels():
    s = build_setup(_analysis(), _scores(72.0, 20.0), [_bull_ote()])
    assert s.direction == "bullish" and s.basis == "OTE"
    assert s.entry_zone_top == pytest.approx(1.0038)
    assert s.entry_zone_bottom == pytest.approx(1.0021)
    assert s.entry == pytest.approx(1.00295)     # OTE sweet spot
    assert s.stop_loss == pytest.approx(1.0000)   # leg origin
    assert s.risk == pytest.approx(0.00295)
    assert s.tp1 == pytest.approx(1.0059)         # entry + 1R
    assert s.tp2 == pytest.approx(1.00885)        # entry + 2R
    assert s.tp3 == pytest.approx(1.0118)         # entry + 3R
    assert "Suggestion only" in s.reason


def test_no_setup_when_lean_is_weak():
    assert build_setup(_analysis(), _scores(52.0, 48.0), [_bull_ote()]) is None  # margin
    assert build_setup(_analysis(), _scores(40.0, 5.0), [_bull_ote()]) is None   # < min
    # strong lean but no entry zone on that side -> still None
    assert build_setup(_analysis(), _scores(80.0, 5.0), []) is None


def test_orderblock_fallback_when_no_ote():
    a = _analysis([NS(source="orderblock", kind="bullish",
                      top=1.0050, bottom=1.0030)])
    s = build_setup(a, _scores(70.0, 10.0), ote_zones=[])
    assert s.basis == "order block"
    assert s.entry == pytest.approx(1.0040)       # zone mid
    assert s.stop_loss == pytest.approx(1.0030)   # zone bottom
    assert s.tp1 == pytest.approx(1.0050)         # entry + 1R (risk 0.0010)


def test_bearish_setup_is_mirrored():
    a = _analysis([NS(source="orderblock", kind="bearish",
                      top=1.0070, bottom=1.0050)])
    s = build_setup(a, _scores(15.0, 75.0), ote_zones=[])
    assert s.direction == "bearish"
    assert s.entry == pytest.approx(1.0060)
    assert s.stop_loss == pytest.approx(1.0070)   # zone top
    assert s.tp1 == pytest.approx(1.0050)         # entry - 1R
    assert s.tp3 == pytest.approx(1.0030)
    assert s.to_signal().direction == "short"


def test_rr_targets_are_a_knob():
    s = build_setup(_analysis(), _scores(72.0, 20.0), [_bull_ote()],
                    rr_targets=(2.0, 4.0, 6.0))
    assert s.tp1 == pytest.approx(1.00295 + 2 * 0.00295)
    assert s.tp3 == pytest.approx(1.00295 + 6 * 0.00295)


def test_to_signal_bridge_and_determinism():
    args = (_analysis(), _scores(72.0, 20.0), [_bull_ote()])
    a, b = build_setup(*args), build_setup(*args)
    key = lambda s: (s.direction, s.basis, s.entry, s.stop_loss,
                     s.tp1, s.tp2, s.tp3)
    assert key(a) == key(b)                       # deterministic (ids aside)
    sig = a.to_signal()
    assert sig.direction == "long"
    assert sig.take_profit == pytest.approx(1.0059)
    assert "ict_setup" in sig.detections and "OTE" in sig.detections
    assert sig.risk_reward > 0


def test_setup_has_no_promise_fields():
    s = build_setup(_analysis(), _scores(72.0, 20.0), [_bull_ote()])
    for forbidden in ("win_rate", "probability", "confidence", "guaranteed",
                      "expected_profit", "edge"):
        assert not hasattr(s, forbidden)
    assert isinstance(s, TradeSetup)
