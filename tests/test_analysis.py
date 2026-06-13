"""Tests for core/analysis/range_session.py — fixed timestamps everywhere,
time is always passed in explicitly (determinism rule: no wall-clock reads).
"""
from datetime import time

import pandas as pd
import pytest

from core.analysis.amd import UNDEFINED, amd_from_session, detect_amd
from core.analysis.range_session import (
    OFF_HOURS, dealing_range, session_bounds, session_levels, session_now,
    session_spans,
)
from core.detectors.swings import detect_swings
from tests.test_detectors import make_df, zigzag_fixture


# ------------------------------------------------------- dealing range -----
# zigzag_fixture swings (lookback=2): high@4 = 1.0042, low@8 = 0.9998,
# high@12 = 1.0062, low@16 = 1.0008, high@20 = 1.0052; last close = 1.0026.

def test_dealing_range_exact():
    df = zigzag_fixture()
    dr = dealing_range(df, detect_swings(df, lookback=2), n_swings=1)
    assert dr.high == pytest.approx(1.0052)    # latest swing high (@20)
    assert dr.low == pytest.approx(1.0008)     # latest swing low (@16)
    assert dr.high_time == df.index[20]
    assert dr.low_time == df.index[16]
    assert dr.equilibrium == pytest.approx(1.0030)
    assert dr.price == pytest.approx(1.0026)   # last close
    assert dr.position_pct == pytest.approx(100 * 0.0018 / 0.0044)  # 40.9%
    assert dr.zone == "discount"


def test_dealing_range_n_swings_extremity():
    # with n_swings=2 the wider extremes (high@12, low@8) define the range
    df = zigzag_fixture()
    dr = dealing_range(df, detect_swings(df, lookback=2), n_swings=2)
    assert dr.high == pytest.approx(1.0062)
    assert dr.low == pytest.approx(0.9998)
    assert dr.high_time == df.index[12]
    assert dr.low_time == df.index[8]


def test_dealing_range_zones():
    # push the last close near the range high -> premium
    df = zigzag_fixture()
    df.iloc[-1, df.columns.get_loc("close")] = 1.0050
    df.iloc[-1, df.columns.get_loc("high")] = 1.0050
    dr = dealing_range(df, detect_swings(df, lookback=2), n_swings=1)
    assert dr.position_pct == pytest.approx(100 * 0.0042 / 0.0044)  # 95.5%
    assert dr.zone == "premium"
    # the equilibrium band is a knob: widen it and 40.9% becomes equilibrium
    df2 = zigzag_fixture()
    dr2 = dealing_range(df2, detect_swings(df2, lookback=2), n_swings=1,
                        eq_tolerance_pct=10.0)
    assert dr2.zone == "equilibrium"


def test_dealing_range_needs_both_sides():
    df = zigzag_fixture()
    assert dealing_range(df, []) is None
    # rows 0..9: high@4 confirmed (bar 6) but low@8 confirms only at bar 10
    part = df.iloc[:10]
    assert dealing_range(part, detect_swings(part, lookback=2)) is None


def test_dealing_range_no_lookahead():
    """The range may only widen when the new extreme swing CONFIRMS:
    swing high@12 (1.0062) confirms at bar 14, so prefixes up to 14 bars
    must still use high@4 (1.0042)."""
    df = zigzag_fixture()
    for k in range(11, len(df) + 1):
        part = df.iloc[:k]
        dr = dealing_range(part, detect_swings(part, lookback=2), n_swings=1)
        assert dr is not None
        assert dr.high == pytest.approx(1.0042 if k <= 14 else
                                        1.0062 if k <= 22 else 1.0052)


# ----------------------------------------------------------- AMD phase -----

def amd_fixture() -> pd.DataFrame:
    """Accumulation range [1.0000, 1.0050]; acc_end after bar 2. Then an
    inside bar, a sellside grab (bar 4), an inside bar, then a bullish
    expansion close (bar 6)."""
    return make_df([
        (1.0025, 1.0048, 1.0002, 1.0030),  # 0  inside (accumulation)
        (1.0030, 1.0049, 1.0010, 1.0040),  # 1  inside
        (1.0040, 1.0050, 1.0020, 1.0045),  # 2  inside  <- acc_end here
        (1.0045, 1.0049, 1.0030, 1.0035),  # 3  inside
        (1.0035, 1.0047, 0.9985, 1.0010),  # 4  sellside grab (wick down, close back in)
        (1.0010, 1.0035, 1.0008, 1.0030),  # 5  inside
        (1.0030, 1.0075, 1.0028, 1.0070),  # 6  bullish expansion close > 1.0050
    ])


def test_amd_accumulation_then_manipulation_then_distribution():
    df = amd_fixture()
    acc_end = df.index[2]
    # only the first inside bar after acc_end -> still accumulation
    acc = detect_amd(df.iloc[:4], 1.0000, 1.0050, acc_end)
    assert (acc.phase, acc.direction) == ("accumulation", "neutral")
    assert acc.manipulated_at is None and acc.distributed_at is None
    # through the grab but before the expansion -> manipulation (bullish bias)
    man = detect_amd(df.iloc[:6], 1.0000, 1.0050, acc_end)
    assert (man.phase, man.direction) == ("manipulation", "bullish")
    assert man.manipulation_side == "sellside"
    assert man.manipulated_at == df.index[4]
    assert man.distributed_at is None
    # full data -> distribution up
    dist = detect_amd(df, 1.0000, 1.0050, acc_end)
    assert (dist.phase, dist.direction) == ("distribution", "bullish")
    assert dist.manipulated_at == df.index[4]
    assert dist.distributed_at == df.index[6]


def test_amd_undefined_when_no_bars_after_accumulation():
    df = amd_fixture()
    res = detect_amd(df.iloc[:3], 1.0000, 1.0050, df.index[2])
    assert res.phase == UNDEFINED
    bad = detect_amd(df, 1.0050, 1.0000, df.index[2])  # inverted range
    assert bad.phase == UNDEFINED


def test_amd_from_session():
    df = session_df()  # Asian range 0.9950..1.0050 on 2026-01-05, ends 07:00
    res = amd_from_session(df, "asia", max_pierce_atr=None)
    assert res.acc_low == pytest.approx(0.9950)
    assert res.acc_high == pytest.approx(1.0050)
    assert res.acc_end == pd.Timestamp("2026-01-05 07:00", tz="UTC")
    assert res.phase == "manipulation"          # sellside grab at 09:00, no expansion
    assert res.manipulation_side == "sellside"
    assert res.direction == "bullish"
    assert res.manipulated_at == pd.Timestamp("2026-01-05 09:00", tz="UTC")
    # a day with no Asian-window bars -> None
    part = df[df.index >= pd.Timestamp("2026-01-05 08:00", tz="UTC")]
    assert amd_from_session(part, "asia") is None


# ---------------------------------------------------- sessions / killzones -

def test_session_now_fixed_timestamps():
    at = lambda hhmm: pd.Timestamp(f"2026-06-10 {hhmm}", tz="UTC")
    assert session_now(at("00:00")) == "asia"
    assert session_now(at("03:00")) == "asia"
    assert session_now(at("06:59")) == "asia"
    assert session_now(at("07:00")) == "london_kz"   # start inclusive
    assert session_now(at("09:59")) == "london_kz"
    assert session_now(at("10:30")) == OFF_HOURS     # gap before NY
    assert session_now(at("12:00")) == "ny_kz"
    assert session_now(at("14:59")) == "ny_kz"
    assert session_now(at("15:00")) == "london_close"
    assert session_now(at("16:59")) == "london_close"
    assert session_now(at("17:00")) == OFF_HOURS
    assert session_now(at("23:00")) == OFF_HOURS


def test_session_now_is_utc_based():
    # 04:00 Paris in June = 02:00 UTC -> still Asia
    assert session_now(pd.Timestamp("2026-06-10 04:00", tz="Europe/Paris")) == "asia"
    # naive timestamps are refused: time must be passed in explicitly
    with pytest.raises(ValueError):
        session_now(pd.Timestamp("2026-06-10 03:00"))


def test_session_boundaries_are_configurable():
    custom = {"weird": (time(22, 0), time(2, 0))}  # wraps midnight
    at = lambda hhmm: pd.Timestamp(f"2026-06-10 {hhmm}", tz="UTC")
    assert session_now(at("23:00"), sessions=custom) == "weird"
    assert session_now(at("01:00"), sessions=custom) == "weird"
    assert session_now(at("12:00"), sessions=custom) == OFF_HOURS


def test_session_bounds_display_timezone():
    start, end = session_bounds("london_kz", "2026-06-10", tz="Europe/Paris")
    assert start == pd.Timestamp("2026-06-10 09:00", tz="Europe/Paris")  # CEST
    assert end == pd.Timestamp("2026-06-10 12:00", tz="Europe/Paris")
    assert str(start.tz) == "Europe/Paris"


def session_df() -> pd.DataFrame:
    """Two days of hourly bars; extremes planted inside and around the
    CURRENT day's Asia window to prove the masking is exact."""
    idx = pd.date_range("2026-01-04 00:00", "2026-01-05 13:00", freq="1h",
                        tz="UTC", name="time")
    df = pd.DataFrame({"open": 1.0000, "high": 1.0010, "low": 0.9990,
                       "close": 1.0000, "volume": 1000.0}, index=idx)
    ts = lambda s: pd.Timestamp(s, tz="UTC")
    df.loc[ts("2026-01-04 02:00"), "high"] = 1.0100  # PREVIOUS day's asia
    df.loc[ts("2026-01-04 04:00"), "low"] = 0.9900   # -> ignored
    df.loc[ts("2026-01-05 03:00"), "high"] = 1.0050  # current asia high
    df.loc[ts("2026-01-05 05:00"), "low"] = 0.9950   # current asia low
    df.loc[ts("2026-01-05 07:00"), "high"] = 1.0070  # 07:00 is OUTSIDE [00,07)
    df.loc[ts("2026-01-05 09:00"), "low"] = 0.9930   # london -> ignored
    return df


def test_session_levels_current_day_asia():
    levels = session_levels(session_df(), "asia")
    assert levels.high == pytest.approx(1.0050)
    assert levels.low == pytest.approx(0.9950)
    assert levels.start == pd.Timestamp("2026-01-05 00:00", tz="UTC")
    assert levels.end == pd.Timestamp("2026-01-05 07:00", tz="UTC")
    assert levels.session == "asia"


def test_session_spans_cover_index():
    df = session_df()  # spans 2026-01-04 00:00 .. 2026-01-05 13:00 UTC
    spans = session_spans(df.index)
    asia = [(s, e) for s, e, name in spans if name == "asia"]
    assert (pd.Timestamp("2026-01-04 00:00", tz="UTC"),
            pd.Timestamp("2026-01-04 07:00", tz="UTC")) in asia
    assert (pd.Timestamp("2026-01-05 00:00", tz="UTC"),
            pd.Timestamp("2026-01-05 07:00", tz="UTC")) in asia
    for start, end, _ in spans:  # every span overlaps the data
        assert end > df.index[0] and start <= df.index[-1]


def test_session_levels_none_when_no_bars():
    df = session_df()
    part = df[df.index >= pd.Timestamp("2026-01-05 08:00", tz="UTC")]
    assert session_levels(part, "asia") is None
