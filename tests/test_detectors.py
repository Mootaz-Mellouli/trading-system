"""Unit tests with HAND-CRAFTED candles where the correct answer is known.

Run from the project root:  pytest tests/ -v
"""
import pandas as pd
import pytest

from core.data.provider import CachedProvider, SyntheticProvider, COLUMNS
from core.detectors.fvg import detect_fvgs
from core.detectors.liquidity import detect_liquidity
from core.detectors.orderblocks import detect_orderblocks
from core.detectors.ote import detect_ote
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


# ------------------------------------------------------- order blocks ------

def ob_bullish_fixture() -> pd.DataFrame:
    quiet = [(1.0000, 1.0003, 0.9999, 1.0002)] * 16          # rows 0..15
    rows = quiet + [
        (1.0002, 1.0006, 0.9996, 0.9998),  # 16: bearish OB candle, zone [0.9996, 1.0006]
        (0.9998, 1.0032, 0.9997, 1.0030),  # 17: bullish displacement, closes above 1.0006
        (1.0030, 1.0038, 1.0010, 1.0034),  # 18: stays above the zone (fresh)
        (1.0034, 1.0035, 1.0004, 1.0012),  # 19: wick into zone -> tested
        (1.0012, 1.0018, 1.0008, 1.0009),  # 20: holds above the zone
        (1.0009, 1.0016, 0.9988, 0.9990),  # 21: closes below 0.9996 -> invalidated
    ]
    return make_df(rows)


def ob_bearish_fixture() -> pd.DataFrame:
    quiet = [(1.0000, 1.0003, 0.9999, 1.0002)] * 16          # rows 0..15
    rows = quiet + [
        (1.0002, 1.0008, 0.9998, 1.0006),  # 16: bullish OB candle, zone [0.9998, 1.0008]
        (1.0006, 1.0007, 0.9972, 0.9975),  # 17: bearish displacement, closes below 0.9998
        (0.9975, 0.9990, 0.9966, 0.9970),  # 18: stays below the zone (fresh)
        (0.9970, 1.0000, 0.9968, 0.9988),  # 19: wick into zone -> tested
        (0.9990, 0.9995, 0.9985, 0.9992),  # 20: holds below the zone
        (0.9992, 1.0015, 0.9991, 1.0012),  # 21: closes above 1.0008 -> invalidated
    ]
    return make_df(rows)


def test_bullish_orderblock_exact_zone():
    df = ob_bullish_fixture()
    obs = detect_orderblocks(df, min_displacement_atr=1.0)
    assert len(obs) == 1
    ob = obs[0]
    assert ob.kind == "bullish"
    assert ob.created_at == df.index[16]      # the OB candle itself
    assert ob.confirmed_at == df.index[17]    # the displacement candle
    assert ob.top == pytest.approx(1.0006)    # high of candle 16
    assert ob.bottom == pytest.approx(0.9996)  # low of candle 16
    assert ob.displacement_atr >= 1.0


def test_bearish_orderblock_exact_zone():
    df = ob_bearish_fixture()
    obs = detect_orderblocks(df, min_displacement_atr=1.0)
    assert len(obs) == 1
    ob = obs[0]
    assert ob.kind == "bearish"
    assert ob.created_at == df.index[16]
    assert ob.confirmed_at == df.index[17]
    assert ob.top == pytest.approx(1.0008)
    assert ob.bottom == pytest.approx(0.9998)


def test_orderblock_state_lifecycle():
    for fixture in (ob_bullish_fixture, ob_bearish_fixture):
        df = fixture()
        ob = detect_orderblocks(df, min_displacement_atr=1.0)[0]
        assert ob.tested_at == df.index[19]       # first wick into the zone
        assert ob.invalidated_at == df.index[21]  # first close beyond the far side
        assert ob.state == "invalidated"
        # state at intermediate points in time, via truncated data
        assert detect_orderblocks(df.iloc[:19], 1.0)[0].state == "fresh"
        assert detect_orderblocks(df.iloc[:20], 1.0)[0].state == "tested"


def test_orderblock_no_lookahead():
    """The OB must only exist once the displacement candle (row 17) has
    closed, and never before — whatever prefix of the data we feed in."""
    df = ob_bullish_fixture()
    for k in range(len(df) + 1):
        obs = detect_orderblocks(df.iloc[:k], min_displacement_atr=1.0)
        if k <= 17:  # prefix ends before the displacement candle exists
            assert obs == []
        else:
            assert len(obs) == 1
            assert obs[0].confirmed_at == df.index[17]


def test_orderblock_displacement_filter():
    df = ob_bullish_fixture()
    assert detect_orderblocks(df, min_displacement_atr=100.0) == []


# ---------------------------------------------------------- liquidity ------

def liquidity_fixture() -> pd.DataFrame:
    """Two near-equal swing highs (one pool of equal highs at 1.0040), one
    swing low (sellside pool at 0.9990), then a wick that sweeps the highs."""
    quiet = [(1.0000, 1.0003, 0.9999, 1.0002)] * 16          # rows 0..15
    rows = quiet + [
        (1.0002, 1.0012, 1.0000, 1.0010),  # 16
        (1.0010, 1.0040, 1.0008, 1.0030),  # 17: swing high #1 @ 1.0040 (confirms @19)
        (1.0030, 1.0032, 1.0012, 1.0014),  # 18
        (1.0014, 1.0016, 0.9990, 0.9992),  # 19: swing low @ 0.9990 (confirms @21)
        (0.9992, 1.0010, 0.9992, 1.0008),  # 20
        (1.0008, 1.0024, 1.0006, 1.0022),  # 21
        (1.0022, 1.0039, 1.0020, 1.0030),  # 22: swing high #2 @ 1.0039 (confirms @24,
                                           #     equal high: 0.0001 from #1)
        (1.0030, 1.0032, 1.0014, 1.0016),  # 23
        (1.0016, 1.0018, 1.0002, 1.0004),  # 24
        (1.0004, 1.0044, 1.0002, 1.0030),  # 25: pierces 1.0040 by 0.0004, closes
                                           #     back below -> buyside sweep
    ]
    return make_df(rows)


def test_liquidity_pools_exact():
    df = liquidity_fixture()
    pools, _ = detect_liquidity(df, detect_swings(df, lookback=2))
    buy = [p for p in pools if p.side == "buyside"]
    sell = [p for p in pools if p.side == "sellside"]
    assert len(buy) == 1 and len(sell) == 1
    b = buy[0]
    assert b.price == pytest.approx(1.0040)    # extreme of the two equal highs
    assert b.created_at == df.index[17]        # first member swing
    assert b.confirmed_at == df.index[24]      # 'equal highs' knowable when #2 confirms
    assert b.swing_times == [df.index[17], df.index[22]]
    assert b.is_equal
    s = sell[0]
    assert s.price == pytest.approx(0.9990)
    assert s.created_at == df.index[19]
    assert s.confirmed_at == df.index[21]
    assert not s.is_equal
    assert s.status == "untaken"               # 0.9990 never traded through


def test_liquidity_sweep_exact():
    df = liquidity_fixture()
    pools, sweeps = detect_liquidity(df, detect_swings(df, lookback=2))
    assert len(sweeps) == 1
    sw = sweeps[0]
    assert sw.time == df.index[25]
    assert sw.level == pytest.approx(1.0040)
    assert sw.side == "buyside"
    assert 0 < sw.pierce_atr <= 0.5
    b = [p for p in pools if p.side == "buyside"][0]
    assert b.taken_at == df.index[25]
    assert b.swept_at == df.index[25]
    assert b.status == "swept"


def test_liquidity_take_without_sweep():
    # close ABOVE the level -> real break, taken but NOT swept
    df = liquidity_fixture()
    df.iloc[25] = [1.0004, 1.0050, 1.0002, 1.0046, 1000.0]
    pools, sweeps = detect_liquidity(df, detect_swings(df, lookback=2))
    assert sweeps == []
    b = [p for p in pools if p.side == "buyside"][0]
    assert b.taken_at == df.index[25]
    assert b.swept_at is None
    assert b.status == "taken"
    # wick deeper than max_pierce_atr -> also just a take, not a hunt
    df2 = liquidity_fixture()
    pools2, sweeps2 = detect_liquidity(df2, detect_swings(df2, lookback=2),
                                       max_pierce_atr=0.01)
    assert sweeps2 == []
    assert [p for p in pools2 if p.side == "buyside"][0].status == "taken"


def test_liquidity_equal_grouping_tolerance():
    # tighter tolerance -> the 0.0001 gap no longer groups -> two pools
    df = liquidity_fixture()
    pools, sweeps = detect_liquidity(df, detect_swings(df, lookback=2),
                                     tolerance_atr=0.05)
    buy = [p for p in pools if p.side == "buyside"]
    assert len(buy) == 2
    assert not any(p.is_equal for p in buy)
    assert len(sweeps) == 2                    # bar 25 wick sweeps both levels


def test_liquidity_no_lookahead():
    """Pools/sweeps must only exist once knowable: pool #1 when swing #1
    confirms (bar 19), 'equal highs' when swing #2 confirms (bar 24),
    the sweep only once bar 25 has closed."""
    df = liquidity_fixture()
    for k in range(len(df) + 1):
        part = df.iloc[:k]
        pools, sweeps = detect_liquidity(part, detect_swings(part, lookback=2))
        buy = [p for p in pools if p.side == "buyside"]
        sell = [p for p in pools if p.side == "sellside"]
        assert len(buy) == (1 if k >= 20 else 0)
        if buy:
            assert buy[0].is_equal == (k >= 25)
        assert len(sell) == (1 if k >= 22 else 0)
        assert len(sweeps) == (1 if k >= 26 else 0)


def test_liquidity_min_pool_age():
    df = liquidity_fixture()
    swings = detect_swings(df, lookback=2)
    # age gate beyond the data -> no swing's liquidity ever becomes usable
    pools, sweeps = detect_liquidity(df, swings, min_pool_age_bars=9)
    assert pools == [] and sweeps == []
    # age 6: swing #1 usable from bar 23, swing #2 (bar 22+6=28) never -> no
    # equal-highs grouping, single-member pool still swept at bar 25
    pools6, sweeps6 = detect_liquidity(df, swings, min_pool_age_bars=6)
    buy6 = [p for p in pools6 if p.side == "buyside"]
    assert len(buy6) == 1
    assert not buy6[0].is_equal
    assert len(sweeps6) == 1


# --------------------------------------------------------------- OTE -------

def ote_fixture() -> pd.DataFrame:
    """Quiet warm-up (for ATR), a clean bullish impulse leg low@16=1.0000 ->
    high@20=1.0100, then a retrace into the 62-79% band [1.0021, 1.0038]."""
    quiet = [(1.0050, 1.0053, 1.0047, 1.0050)] * 14          # rows 0..13
    rows = quiet + [
        (1.0050, 1.0050, 1.0030, 1.0035),  # 14
        (1.0035, 1.0035, 1.0010, 1.0015),  # 15
        (1.0015, 1.0016, 1.0000, 1.0005),  # 16: swing low @ 1.0000 (conf @18)
        (1.0005, 1.0030, 1.0004, 1.0028),  # 17
        (1.0028, 1.0060, 1.0026, 1.0058),  # 18
        (1.0058, 1.0090, 1.0056, 1.0088),  # 19
        (1.0088, 1.0100, 1.0086, 1.0095),  # 20: swing high @ 1.0100 (conf @22)
        (1.0095, 1.0096, 1.0070, 1.0072),  # 21
        (1.0072, 1.0074, 1.0050, 1.0052),  # 22
        (1.0052, 1.0053, 1.0030, 1.0033),  # 23: retrace into OTE band -> tested
        (1.0033, 1.0045, 1.0031, 1.0042),  # 24
        (1.0042, 1.0060, 1.0040, 1.0058),  # 25
    ]
    return make_df(rows)


def test_ote_zone_exact_levels():
    df = ote_fixture()
    zones = detect_ote(df, detect_swings(df, lookback=2))
    bull = [z for z in zones if z.direction == "bullish"]
    assert len(bull) == 1
    z = bull[0]
    assert z.created_at == df.index[20]       # leg-end swing high
    assert z.confirmed_at == df.index[22]      # knowable only when it confirms
    assert z.leg_low == pytest.approx(1.0000)
    assert z.leg_high == pytest.approx(1.0100)
    assert z.top == pytest.approx(1.0038)      # 62% retracement
    assert z.bottom == pytest.approx(1.0021)   # 79% retracement
    assert z.sweet == pytest.approx(1.00295)   # 70.5%
    assert z.entered_at == df.index[23]        # first trade into the band
    assert z.invalidated_at is None
    assert z.state == "tested"


def test_ote_leg_filter():
    df = ote_fixture()
    assert detect_ote(df, detect_swings(df, lookback=2),
                      min_leg_atr=100.0) == []


def test_ote_no_lookahead():
    """The bullish OTE may only exist once the leg-end swing high (bar 20)
    has confirmed, at bar 22 -> present from a 23-bar prefix onward."""
    df = ote_fixture()
    for k in range(len(df) + 1):
        part = df.iloc[:k]
        bull = [z for z in detect_ote(part, detect_swings(part, lookback=2))
                if z.direction == "bullish"]
        assert len(bull) == (1 if k >= 23 else 0)
        if bull:
            assert bull[0].confirmed_at == df.index[22]


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
