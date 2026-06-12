"""Market structure: swing points, HH/HL/LH/LL labels, BOS and CHoCH.

Definitions used (the knobs to negotiate with the trader):
  - Swing high at i: high[i] is the strict maximum of the window i±lookback.
    Swing low: symmetric. `lookback` controls structure sensitivity —
    bigger = fewer, more significant swings.
  - Labels: each new swing high is HH if above the previous swing high,
    else LH. Each swing low is HL if above the previous swing low, else LL.
  - BOS (break of structure): a candle CLOSES beyond the last swing point
    in the direction of the current trend (continuation).
  - CHoCH (change of character): a candle CLOSES beyond the last swing
    point AGAINST the current trend (potential reversal). The first break
    sets the trend.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SwingPoint:
    time: pd.Timestamp          # when the swing candle occurred
    confirmed_at: pd.Timestamp  # when it became KNOWABLE (lookback bars later)
    price: float
    kind: str    # 'high' | 'low'
    label: str   # 'HH' | 'LH' | 'HL' | 'LL' | ''


@dataclass
class StructureEvent:
    time: pd.Timestamp
    price: float          # the swing level that was broken
    kind: str             # 'BOS' | 'CHoCH'
    direction: str        # 'bullish' | 'bearish'


def detect_swings(df: pd.DataFrame, lookback: int = 5) -> list[SwingPoint]:
    highs, lows = df["high"], df["low"]
    swings: list[SwingPoint] = []
    last_high: float | None = None
    last_low: float | None = None
    n = len(df)
    for i in range(lookback, n - lookback):
        win_h = highs.iloc[i - lookback: i + lookback + 1]
        win_l = lows.iloc[i - lookback: i + lookback + 1]
        if highs.iloc[i] == win_h.max() and (win_h == highs.iloc[i]).sum() == 1:
            label = "" if last_high is None else ("HH" if highs.iloc[i] > last_high else "LH")
            swings.append(SwingPoint(df.index[i], df.index[i + lookback],
                                     highs.iloc[i], "high", label))
            last_high = highs.iloc[i]
        if lows.iloc[i] == win_l.min() and (win_l == lows.iloc[i]).sum() == 1:
            label = "" if last_low is None else ("HL" if lows.iloc[i] > last_low else "LL")
            swings.append(SwingPoint(df.index[i], df.index[i + lookback],
                                     lows.iloc[i], "low", label))
            last_low = lows.iloc[i]
    return swings


def detect_structure_events(df: pd.DataFrame, swings: list[SwingPoint]) -> list[StructureEvent]:
    """Walk forward through closes; emit BOS/CHoCH when a close breaks the
    most recent confirmed swing high/low. Trend starts undefined; the first
    break defines it."""
    events: list[StructureEvent] = []
    trend: str | None = None  # 'bullish' | 'bearish'
    swing_iter = iter(swings)
    pending = next(swing_iter, None)
    active_high: SwingPoint | None = None
    active_low: SwingPoint | None = None

    for t, close in df["close"].items():
        # activate swings only once they are CONFIRMED (no lookahead bias)
        while pending is not None and pending.confirmed_at <= t:
            if pending.kind == "high":
                active_high = pending
            else:
                active_low = pending
            pending = next(swing_iter, None)

        if active_high and close > active_high.price:
            kind = "BOS" if trend == "bullish" else ("CHoCH" if trend == "bearish" else "BOS")
            events.append(StructureEvent(t, active_high.price, kind, "bullish"))
            trend = "bullish"
            active_high = None  # consumed
        elif active_low and close < active_low.price:
            kind = "BOS" if trend == "bearish" else ("CHoCH" if trend == "bullish" else "BOS")
            events.append(StructureEvent(t, active_low.price, kind, "bearish"))
            trend = "bearish"
            active_low = None
    return events
