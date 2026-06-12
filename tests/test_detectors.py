"""Unit tests with HAND-CRAFTED candles where the correct answer is known.

Run from the project root:  pytest tests/ -v
"""
import pandas as pd
import pytest

from core.data.provider import CachedProvider, SyntheticProvider, COLUMNS
from core.detectors.fvg import detect_fvgs
from core.detectors.swings import detect_swings, detect_structure_events
from core.strategies.base import FVGRetestStrategy


def make_df(rows: list[tuple]) -> pd.DataFrame:
    """rows = [(open, high, low, close), ...] on a 15m index."""
    idx = pd.date_range("2026-01-05 08:00", periods=len(rows), freq="15min",
                        tz="UTC", name="time")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000.0
    return df


# ---------------------------------------------------------------- FVG ------

def fvg_fixture() -> pd.DataFrame:
    quiet = [(1.0000, 1.0003, 0.9999, 1.0002)] * 16          # rows 0..15
    rows = quiet + [
        (1.0002, 1.0032, 1.0001, 1.0030),  # 16: displacement up
        (1.0030, 1.0040, 1.0010, 1.0035),  # 17: gap -> low 1.0010 > high[15] 1.0003
        (1.0035, 1.0036, 1.0020, 1.0024),  # 18
        (1.0022, 1.0024, 1.0014, 1.0018),  # 19
        (1.0025, 1.0026, 1.0008, 1.0012),  # 20: re-enters zone (mitigation)
        (1.0012, 1.0016, 1.0000, 1.0002),  # 21: traverses zone (filled)
    ]
    return make_df(rows)


def test_fvg_detected_with_exact_zone():
    df = fvg_fixture()
    fvgs = detect_fvgs(df, min_displacement_atr=0.8)
    bullish = [f for f in fvgs if f.kind == "bullish"]
    assert len(bullish) == 1
    f = bullish[0]
    assert f.created_at == df.index[17]
    assert f.bottom == pytest.approx(1.0003)   # high of candle 15
    assert f.top == pytest.approx(1.0010)      # low of candle 17
    assert f.displacement_atr >= 0.8


def test_fvg_fill_tracking():
    df = fvg_fixture()
    f = [x for x in detect_fvgs(df, 0.8) if x.kind == "bullish"][0]
    assert f.mitigated_at == df.index[20]      # first touch of the zone
    assert f.filled_at == df.index[21]         # full traversal


def test_fvg_displacement_filter():
    df = fvg_fixture()
    assert len(detect_fvgs(df, min_displacement_atr=100.0)) == 0


# ------------------------------------------------------------- swings ------

def zigzag_fixture() -> pd.DataFrame:
    closes = [1.0000, 1.0010, 1.0020, 1.0030, 1.0040,   # rise -> peak @4
              1.0030, 1.0020, 1.0010, 1.0000,           # fall -> trough @8
              1.0015, 1.0030, 1.0045, 1.0060,           # rise -> HH peak @12
              1.0045, 1.0030, 1.0020, 1.0010,           # fall -> HL trough @16
              1.0020, 1.0030, 1.0040, 1.0050,           # rise -> LH peak @20
              1.0040, 1.0030, 1.0028, 1.0026]
    rows = [(c, c + 0.0002, c - 0.0002, c) for c in closes]
    return make_df(rows)


def test_swing_points_and_labels():
    df = zigzag_fixture()
    swings = detect_swings(df, lookback=2)
    got = [(df.index.get_loc(s.time), s.kind, s.label) for s in swings]
    assert got == [(4, "high", ""), (8, "low", ""), (12, "high", "HH"),
                   (16, "low", "HL"), (20, "high", "LH")]


def test_swing_confirmation_lag():
    """A swing must only be knowable `lookback` bars after it happens."""
    df = zigzag_fixture()
    for s in detect_swings(df, lookback=2):
        assert s.confirmed_at == df.index[df.index.get_loc(s.time) + 2]


def test_structure_events_no_lookahead():
    df = zigzag_fixture()
    swings = detect_swings(df, lookback=2)
    events = detect_structure_events(df, swings)
    # Only one break: close[11]=1.0045 crosses peak@4 (1.0042),
    # which is only usable from its confirmation at bar 6.
    assert len(events) == 1
    e = events[0]
    assert (e.kind, e.direction) == ("BOS", "bullish")
    assert e.time == df.index[11]
    assert e.price == pytest.approx(1.0042)


# ------------------------------------------------- strategy + data ---------

def test_strategy_is_deterministic():
    df = SyntheticProvider(seed=7).fetch("TEST", "15m", 600)
    strat = FVGRetestStrategy()
    a = strat.generate("TEST", df)
    b = strat.generate("TEST", df)
    key = lambda sigs: [(s.time, s.direction, s.entry, s.stop_loss) for s in sigs]
    assert key(a) == key(b)
    assert len(a) > 0
    for s in a:
        assert s.direction in ("long", "short")
        assert s.risk_reward > 0


def test_synthetic_provider_contract():
    df = SyntheticProvider().fetch("TEST", "15m", 300)
    assert list(df.columns) == COLUMNS
    assert len(df) == 300
    assert df.index.tz is not None
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_cached_provider_roundtrip(tmp_path):
    p = CachedProvider(SyntheticProvider(), cache_dir=tmp_path)
    df1 = p.fetch("TEST", "15m", 100)
    assert (tmp_path / "TEST_15m.parquet").exists()
    df2 = p.fetch("TEST", "15m", 100)
    pd.testing.assert_frame_equal(df1, df2, check_freq=False)
