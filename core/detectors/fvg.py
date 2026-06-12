"""Fair Value Gap (FVG) detection.

Definition (3-candle pattern, candles i-1, i, i+1):
  - Bullish FVG: low[i+1] > high[i-1]. The gap zone is [high[i-1], low[i+1]].
  - Bearish FVG: high[i+1] < low[i-1]. The gap zone is [high[i+1], low[i-1]].
The middle candle i is the 'displacement' candle.

Tunable rules (the knobs to negotiate with the trader):
  - min_displacement_atr: middle candle body must be >= this multiple of ATR
    (filters out micro-gaps from noise). 0 disables the filter.
  - Fill tracking: an FVG is 'mitigated' the first time price trades back
    into the zone, and 'filled' when price fully traverses it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


@dataclass
class FVG:
    kind: str                 # 'bullish' | 'bearish'
    created_at: pd.Timestamp  # time of the candle that completes the pattern (i+1)
    top: float
    bottom: float
    displacement_atr: float   # middle candle body size, in ATR multiples
    mitigated_at: pd.Timestamp | None = None
    filled_at: pd.Timestamp | None = None
    meta: dict = field(default_factory=dict)

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2


def detect_fvgs(df: pd.DataFrame, min_displacement_atr: float = 0.8,
                atr_period: int = 14) -> list[FVG]:
    """Scan the whole dataframe; returns FVGs with fill status tracked forward."""
    a = atr(df, atr_period)
    highs, lows = df["high"].values, df["low"].values
    opens, closes = df["open"].values, df["close"].values
    times = df.index
    out: list[FVG] = []

    for i in range(1, len(df) - 1):
        if np.isnan(a.iloc[i]) or a.iloc[i] == 0:
            continue
        body_atr = abs(closes[i] - opens[i]) / a.iloc[i]
        if body_atr < min_displacement_atr:
            continue
        if lows[i + 1] > highs[i - 1]:  # bullish gap
            out.append(FVG("bullish", times[i + 1],
                           top=lows[i + 1], bottom=highs[i - 1],
                           displacement_atr=round(body_atr, 2)))
        elif highs[i + 1] < lows[i - 1]:  # bearish gap
            out.append(FVG("bearish", times[i + 1],
                           top=lows[i - 1], bottom=highs[i + 1],
                           displacement_atr=round(body_atr, 2)))

    _track_fills(df, out)
    return out


def _track_fills(df: pd.DataFrame, fvgs: list[FVG]) -> None:
    for f in fvgs:
        after = df[df.index > f.created_at]
        if after.empty:
            continue
        if f.kind == "bullish":
            entered = after["low"] <= f.top
            crossed = after["low"] <= f.bottom
        else:
            entered = after["high"] >= f.bottom
            crossed = after["high"] >= f.top
        if entered.any():
            f.mitigated_at = after.index[entered.argmax()]
        if crossed.any():
            f.filled_at = after.index[crossed.argmax()]


def fvgs_to_frame(fvgs: list[FVG]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"kind": f.kind, "created_at": f.created_at, "top": f.top,
          "bottom": f.bottom, "displacement_atr": f.displacement_atr,
          "mitigated_at": f.mitigated_at, "filled_at": f.filled_at}
         for f in fvgs]
    )
