"""Optimal Trade Entry (OTE) zones.

OTE is the 62%-79% retracement band of a confirmed impulse leg (70.5% is
the classic sweet spot). A leg is a move between two consecutive opposite
swing points:
  - bullish leg  = swing low -> swing high (expect continuation UP after a
    retracement down into the band),
  - bearish leg  = swing high -> swing low (expect continuation DOWN after a
    retracement up into the band).

No lookahead: a leg is only knowable once its LATER swing is confirmed, so
confirmed_at = that swing's confirmed_at. State is tracked only on candles
after confirmed_at:
  - 'fresh' until price first trades into the band -> 'tested',
  - 'invalidated' once a candle CLOSES beyond the leg origin (a full
    retracement: below leg_low for a bullish leg, above leg_high for a
    bearish one).

Tunable rules (the knobs to negotiate with the trader):
  - fib_low / fib_high: the retracement band (default 0.62 / 0.79).
  - sweet: the sweet-spot level inside the band (default 0.705).
  - min_leg_atr: the leg's range must be >= this multiple of ATR at the
    leg-end bar (filters out noise legs). 0 disables the filter.
  - atr_period: ATR length for that filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.detectors.fvg import atr
from core.detectors.swings import SwingPoint


@dataclass
class OTEZone:
    direction: str              # 'bullish' | 'bearish' (expected continuation)
    created_at: pd.Timestamp    # the leg-end swing time
    confirmed_at: pd.Timestamp  # leg-end swing confirmed_at (when KNOWABLE)
    leg_low: float
    leg_high: float
    top: float                  # upper bound of the 62-79% band
    bottom: float               # lower bound of the 62-79% band
    sweet: float                # the sweet-spot level (70.5%)
    entered_at: pd.Timestamp | None = None
    invalidated_at: pd.Timestamp | None = None
    meta: dict = field(default_factory=dict)

    @property
    def state(self) -> str:
        if self.invalidated_at is not None:
            return "invalidated"
        if self.entered_at is not None:
            return "tested"
        return "fresh"


def detect_ote(df: pd.DataFrame, swings: list[SwingPoint],
               fib_low: float = 0.62, fib_high: float = 0.79,
               sweet: float = 0.705, min_leg_atr: float = 1.0,
               atr_period: int = 14) -> list[OTEZone]:
    """One OTE zone per qualifying impulse leg, state tracked forward."""
    a = atr(df, atr_period)
    out: list[OTEZone] = []
    for prev, cur in zip(swings, swings[1:]):
        if prev.kind == cur.kind:
            continue  # need alternating swings to form a leg
        leg_low = min(prev.price, cur.price)
        leg_high = max(prev.price, cur.price)
        rng = leg_high - leg_low
        if rng <= 0:
            continue
        i = df.index.get_loc(cur.time)
        av = a.iloc[i]
        if np.isnan(av) or av == 0 or rng / av < min_leg_atr:
            continue
        direction = "bullish" if cur.kind == "high" else "bearish"
        if direction == "bullish":  # retrace down from the high
            lvl_lo = leg_high - fib_low * rng
            lvl_hi = leg_high - fib_high * rng
            sweet_lvl = leg_high - sweet * rng
        else:                       # retrace up from the low
            lvl_lo = leg_low + fib_low * rng
            lvl_hi = leg_low + fib_high * rng
            sweet_lvl = leg_low + sweet * rng
        out.append(OTEZone(
            direction, cur.time, cur.confirmed_at, leg_low, leg_high,
            top=max(lvl_lo, lvl_hi), bottom=min(lvl_lo, lvl_hi),
            sweet=sweet_lvl))
    _track_states(df, out)
    return out


def _track_states(df: pd.DataFrame, zones: list[OTEZone]) -> None:
    for z in zones:
        after = df[df.index > z.confirmed_at]
        if after.empty:
            continue
        entered = (after["low"] <= z.top) & (after["high"] >= z.bottom)
        if z.direction == "bullish":
            broken = after["close"] < z.leg_low
        else:
            broken = after["close"] > z.leg_high
        if entered.any():
            z.entered_at = after.index[entered.argmax()]
        if broken.any():
            z.invalidated_at = after.index[broken.argmax()]


def ote_to_frame(zones: list[OTEZone]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"direction": z.direction, "created_at": z.created_at,
          "confirmed_at": z.confirmed_at, "top": z.top, "bottom": z.bottom,
          "sweet": z.sweet, "leg_low": z.leg_low, "leg_high": z.leg_high,
          "state": z.state, "entered_at": z.entered_at,
          "invalidated_at": z.invalidated_at}
         for z in zones]
    )
