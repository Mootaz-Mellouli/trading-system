"""AMD / Power-of-3 phase: Accumulation -> Manipulation -> Distribution.

The classic ICT daily/session profile:
  - Accumulation: price consolidates inside a reference range (e.g. the
    Asian session range).
  - Manipulation: a liquidity grab — price pierces ONE side of the range by
    a wick then closes back inside (the "Judas swing").
  - Distribution: the real expansion, in the OPPOSITE direction of the grab
    (a close beyond the far side of the range). Grabbing sellside (lows)
    precedes a bullish expansion; grabbing buyside (highs) precedes bearish.

This classifies the phase reached as of the last candle, working only from
bars AFTER the accumulation window closed (`acc_end`). Determinism rule:
the reference range and `as_of` come from the data passed in — never the
wall clock.

Tunable rules (the knobs to negotiate with the trader):
  - max_pierce_atr: a grab's wick beyond the range may be at most this many
    ATR (deeper = a genuine breakout, not a manipulation). None disables.
  - expansion_atr: distribution requires a close at least this many ATR
    beyond the far side (filters marginal closes). 0 = any close beyond.
  - atr_period: ATR length for both tolerances.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.analysis.range_session import DEFAULT_SESSIONS, session_levels
from core.detectors.fvg import atr

UNDEFINED = "undefined"


@dataclass
class AMDPhase:
    phase: str                  # 'accumulation'|'manipulation'|'distribution'|'undefined'
    direction: str              # 'bullish'|'bearish'|'neutral' (expected/true move)
    acc_low: float
    acc_high: float
    acc_end: pd.Timestamp
    as_of: pd.Timestamp
    manipulation_side: str | None = None  # 'buyside'|'sellside' grabbed
    manipulated_at: pd.Timestamp | None = None
    distributed_at: pd.Timestamp | None = None
    meta: dict = field(default_factory=dict)


def detect_amd(df: pd.DataFrame, acc_low: float, acc_high: float,
               acc_end: pd.Timestamp, max_pierce_atr: float | None = 1.0,
               expansion_atr: float = 0.0, atr_period: int = 14) -> AMDPhase:
    """Classify the AMD phase from candles after the accumulation window."""
    as_of = df.index[-1]
    base = AMDPhase("accumulation", "neutral", acc_low, acc_high, acc_end, as_of)
    if acc_high <= acc_low:
        base.phase = UNDEFINED
        return base
    a = atr(df, atr_period)
    after = df[df.index > acc_end]
    if after.empty:
        base.phase = UNDEFINED
        return base

    side: str | None = None
    manip_at: pd.Timestamp | None = None
    for t, row in after.iterrows():
        av = a.loc[t]
        pierce_ok = (max_pierce_atr is None or np.isnan(av) or av == 0)
        # a grab pierces one side by a wick and CLOSES back inside the range
        if row["high"] > acc_high and row["close"] <= acc_high:
            depth = row["high"] - acc_high
            if pierce_ok or depth <= max_pierce_atr * av:
                side, manip_at = "buyside", t
                break
        if row["low"] < acc_low and row["close"] >= acc_low:
            depth = acc_low - row["low"]
            if pierce_ok or depth <= max_pierce_atr * av:
                side, manip_at = "sellside", t
                break

    # distribution: a close beyond the far side, after the grab if one exists
    dist_from = after[after.index >= manip_at] if manip_at is not None else after
    dist_at: pd.Timestamp | None = None
    dist_dir = "neutral"
    for t, row in dist_from.iterrows():
        av = a.loc[t]
        margin = 0.0 if (np.isnan(av) or av == 0) else expansion_atr * av
        broke_up = row["close"] > acc_high + margin
        broke_dn = row["close"] < acc_low - margin
        # the true move runs opposite the grab; with no grab, take either break
        if side == "sellside" and broke_up:
            dist_at, dist_dir = t, "bullish"; break
        if side == "buyside" and broke_dn:
            dist_at, dist_dir = t, "bearish"; break
        if side is None and (broke_up or broke_dn):
            dist_at, dist_dir = t, ("bullish" if broke_up else "bearish"); break

    if dist_at is not None:
        base.phase, base.direction = "distribution", dist_dir
    elif manip_at is not None:
        base.phase = "manipulation"
        base.direction = "bullish" if side == "sellside" else "bearish"
    base.manipulation_side, base.manipulated_at = side, manip_at
    base.distributed_at = dist_at
    return base


def amd_from_session(df: pd.DataFrame, session: str = "asia",
                     sessions: dict = DEFAULT_SESSIONS,
                     max_pierce_atr: float | None = 1.0,
                     expansion_atr: float = 0.0,
                     atr_period: int = 14) -> AMDPhase | None:
    """AMD phase using a session's range as the accumulation window
    (Asian range by default). None when that session has no bars yet."""
    levels = session_levels(df, session, sessions)
    if levels is None:
        return None
    return detect_amd(df, levels.low, levels.high, levels.end,
                      max_pierce_atr, expansion_atr, atr_period)
