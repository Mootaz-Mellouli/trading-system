"""Backtester tests on hand-crafted candles with FORCED outcomes."""
import pandas as pd
import pytest

from core.backtest.engine import Backtester, BTConfig
from core.strategies.base import Signal
from tests.test_detectors import make_df

CFG = BTConfig(initial_equity=10_000, risk_pct=1.0, spread=0.0, slippage=0.0,
               entry_expiry_bars=10)


def long_signal(df, bar: int) -> Signal:
    return Signal(symbol="T", time=df.index[bar], direction="long",
                  entry=1.0010, stop_loss=1.0000, take_profit=1.0030,
                  reason="test", detections=[])


def test_take_profit_hit_exact_pnl():
    rows = [(1.0020, 1.0022, 1.0018, 1.0020)] * 6 + [
        (1.0015, 1.0016, 1.0009, 1.0012),  # 6: dips to entry -> fill @1.0010
        (1.0012, 1.0020, 1.0011, 1.0019),  # 7
        (1.0019, 1.0035, 1.0018, 1.0032),  # 8: hits TP 1.0030
    ]
    df = make_df(rows)
    res = Backtester().run(df, [long_signal(df, 5)], CFG)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.outcome == "tp"
    assert t.entry_time == df.index[6] and t.exit_time == df.index[8]
    # risk 1% of 10k = 100 EUR over 10 pips -> units 100000; 20 pips gain = +200
    assert t.pnl == pytest.approx(200.0)
    assert t.r_multiple == pytest.approx(2.0)
    assert res.equity_curve.iloc[-1] == pytest.approx(10_200.0)


def test_stop_loss_hit_loses_exactly_risk():
    rows = [(1.0020, 1.0022, 1.0018, 1.0020)] * 6 + [
        (1.0015, 1.0016, 1.0009, 1.0012),  # 6: fill @1.0010
        (1.0012, 1.0013, 0.9999, 1.0001),  # 7: breaks SL 1.0000
    ]
    df = make_df(rows)
    res = Backtester().run(df, [long_signal(df, 5)], CFG)
    t = res.trades[0]
    assert t.outcome == "sl"
    assert t.pnl == pytest.approx(-100.0)   # exactly the 1% risked
    assert t.r_multiple == pytest.approx(-1.0)


def test_ambiguous_bar_counts_as_loss():
    """Candle touches BOTH SL and TP -> pessimistic engine books the loss."""
    rows = [(1.0020, 1.0022, 1.0018, 1.0020)] * 6 + [
        (1.0015, 1.0016, 1.0009, 1.0012),  # 6: fill
        (1.0012, 1.0040, 0.9995, 1.0025),  # 7: huge bar through SL AND TP
    ]
    df = make_df(rows)
    res = Backtester().run(df, [long_signal(df, 5)], CFG)
    assert res.trades[0].outcome == "sl"


def test_unfilled_order_expires():
    rows = [(1.0050, 1.0052, 1.0048, 1.0050)] * 20  # never dips to 1.0010
    df = make_df(rows)
    res = Backtester().run(df, [long_signal(df, 5)], CFG)
    assert len(res.trades) == 0
    assert res.expired == 1


def test_one_position_at_a_time():
    rows = [(1.0020, 1.0022, 1.0018, 1.0020)] * 6 + [
        (1.0015, 1.0016, 1.0009, 1.0012),  # 6: fill sig A
        (1.0012, 1.0020, 1.0011, 1.0019),  # 7: sig B arrives while A open
        (1.0019, 1.0035, 1.0018, 1.0032),  # 8: A exits at TP
    ]
    df = make_df(rows)
    sigs = [long_signal(df, 5), long_signal(df, 6)]
    sigs[1].time = df.index[7]
    res = Backtester().run(df, sigs, CFG)
    assert len(res.trades) == 1
    assert res.skipped_busy == 1


def test_costs_reduce_pnl():
    rows = [(1.0020, 1.0022, 1.0018, 1.0020)] * 6 + [
        (1.0015, 1.0016, 1.0009, 1.0012),
        (1.0012, 1.0020, 1.0011, 1.0019),
        (1.0019, 1.0035, 1.0018, 1.0032),
    ]
    df = make_df(rows)
    costly = BTConfig(initial_equity=10_000, risk_pct=1.0,
                      spread=0.0002, slippage=0.0, entry_expiry_bars=10)
    res = Backtester().run(df, [long_signal(df, 5)], costly)
    t = res.trades[0]
    assert t.entry_price == pytest.approx(1.0012)  # entry worsened by spread
    assert t.r_multiple < 2.0                       # edge shrank
