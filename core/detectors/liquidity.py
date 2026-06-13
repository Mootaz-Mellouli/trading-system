"""Liquidity pools (buyside / sellside) and liquidity sweeps.

Definitions:
  - Every confirmed swing high is resting BUYSIDE liquidity (buy stops sit
    above it); every confirmed swing low is SELLSIDE liquidity. A pool is
    only usable from the swing's confirmed_at bar (no lookahead), plus an
    optional extra min_pool_age_bars delay measured from the swing bar.
  - Equal highs/lows: when a newly usable swing's price is within
    tolerance_atr * ATR (ATR at the activation bar) of a still-untaken
    same-side pool, it merges into that pool. The pool's level becomes the
    extreme of its members and the pool counts as 'equal highs/lows' — a
    stronger draw on liquidity.
  - Taken: a pool is taken the first time price trades STRICTLY beyond its
    level (an exact touch is not beyond).
  - Sweep: the taking candle pierces the level by at most
    max_pierce_atr * ATR (a wick hunt) and CLOSES back on the original
    side of the level. A sweep also takes the pool.

Tunable rules (the knobs to negotiate with the trader):
  - tolerance_atr: max distance between swing prices, in ATR multiples,
    to group them as equal highs/lows.
  - max_pierce_atr: max wick beyond the level, in ATR multiples, for a
    take to count as a sweep (deeper = a real breakout, not a hunt).
  - min_pool_age_bars: extra bars after the swing bar before its liquidity
    counts (0 = usable as soon as the swing is confirmed).
  - atr_period: ATR length used for both tolerances.
If ATR is not yet defined at a bar (warm-up), no merging happens there and
a take cannot qualify as a sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.detectors.fvg import atr
from core.detectors.swings import SwingPoint


@dataclass
class LiquidityPool:
    side: str                   # 'buyside' | 'sellside'
    price: float                # extreme of member swing prices (the level)
    created_at: pd.Timestamp    # time of the first member swing
    confirmed_at: pd.Timestamp  # when the pool, in its current form, became usable
    swing_times: list = field(default_factory=list)  # member swing bars (2+ = equal highs/lows)
    taken_at: pd.Timestamp | None = None   # first trade strictly beyond the level
    swept_at: pd.Timestamp | None = None   # taken by a wick that closed back inside
    meta: dict = field(default_factory=dict)

    @property
    def is_equal(self) -> bool:
        return len(self.swing_times) >= 2

    @property
    def status(self) -> str:
        if self.swept_at is not None:
            return "swept"
        if self.taken_at is not None:
            return "taken"
        return "untaken"


@dataclass
class LiquiditySweep:
    time: pd.Timestamp
    level: float
    side: str          # 'buyside' | 'sellside' (the pool that was swept)
    pierce_atr: float  # wick beyond the level, in ATR multiples


def detect_liquidity(df: pd.DataFrame, swings: list[SwingPoint],
                     tolerance_atr: float = 0.1, max_pierce_atr: float = 0.5,
                     min_pool_age_bars: int = 0,
                     atr_period: int = 14) -> tuple[list[LiquidityPool], list[LiquiditySweep]]:
    """Walk forward bar by bar; pools enter play only once usable, takes and
    sweeps are evaluated on each bar against the pools active at that bar."""
    a = atr(df, atr_period)
    highs, lows = df["high"].values, df["low"].values
    closes = df["close"].values
    times = df.index
    n = len(df)

    # bar index from which each swing's liquidity is usable
    activations: list[tuple[int, SwingPoint]] = []
    for s in swings:
        act = max(times.get_loc(s.confirmed_at),
                  times.get_loc(s.time) + min_pool_age_bars)
        if act < n:
            activations.append((act, s))
    activations.sort(key=lambda x: x[0])

    pools: list[LiquidityPool] = []
    active: list[LiquidityPool] = []
    sweeps: list[LiquiditySweep] = []
    k = 0

    for i in range(n):
        # 1. activate newly usable swings; merge equal highs/lows
        while k < len(activations) and activations[k][0] <= i:
            s = activations[k][1]
            k += 1
            side = "buyside" if s.kind == "high" else "sellside"
            tol = None if np.isnan(a.iloc[i]) else tolerance_atr * a.iloc[i]
            target = None
            if tol is not None:
                for p in active:
                    if p.side == side and abs(p.price - s.price) <= tol:
                        target = p
                        break
            if target is not None:
                target.swing_times.append(s.time)
                target.price = (max(target.price, s.price) if side == "buyside"
                                else min(target.price, s.price))
                target.confirmed_at = times[i]
            else:
                p = LiquidityPool(side, s.price, s.time, times[i],
                                  swing_times=[s.time])
                pools.append(p)
                active.append(p)

        # 2. take / sweep checks against this bar
        remaining: list[LiquidityPool] = []
        for p in active:
            beyond = highs[i] > p.price if p.side == "buyside" else lows[i] < p.price
            if not beyond:
                remaining.append(p)
                continue
            p.taken_at = times[i]
            pierce = (highs[i] - p.price if p.side == "buyside"
                      else p.price - lows[i])
            closed_back = (closes[i] < p.price if p.side == "buyside"
                           else closes[i] > p.price)
            if (closed_back and not np.isnan(a.iloc[i]) and a.iloc[i] > 0
                    and pierce <= max_pierce_atr * a.iloc[i]):
                p.swept_at = times[i]
                sweeps.append(LiquiditySweep(times[i], p.price, p.side,
                                             round(pierce / a.iloc[i], 2)))
        active = remaining

    return pools, sweeps


def pools_to_frame(pools: list[LiquidityPool]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"side": p.side, "price": p.price, "created_at": p.created_at,
          "confirmed_at": p.confirmed_at, "n_swings": len(p.swing_times),
          "is_equal": p.is_equal, "status": p.status,
          "taken_at": p.taken_at, "swept_at": p.swept_at}
         for p in pools]
    )


def sweeps_to_frame(sweeps: list[LiquiditySweep]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"time": s.time, "level": s.level, "side": s.side,
          "pierce_atr": s.pierce_atr}
         for s in sweeps]
    )
