"""Order block (OB) detection.

Definition (2-candle pattern, candles j = i-1 and i):
  - Bullish OB: candle j is bearish (close < open) and candle i is a bullish
    displacement — its body is >= min_displacement_atr * ATR AND it closes
    above high[j]. The zone is [low[j], high[j]].
  - Bearish OB: symmetric (bullish candle j, bearish displacement closing
    below low[j]).
Breaks are measured on CLOSES, consistent with BOS/CHoCH in swings.py.

An OB is only knowable once the displacement candle has closed, so
confirmed_at = the displacement candle's time. State tracking starts on the
candle AFTER confirmed_at (no lookahead).

State lifecycle, tracked forward:
  - 'fresh' until price first re-enters the zone,
  - 'tested' from the first re-entry (wick into the zone counts),
  - 'invalidated' once a candle CLOSES beyond the far side of the zone
    (below the bottom for bullish, above the top for bearish).

Tunable rules (the knobs to negotiate with the trader):
  - min_displacement_atr: displacement candle body must be >= this multiple
    of ATR. 0 disables the filter.
  - atr_period: ATR length used for the displacement filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.detectors.fvg import atr


@dataclass
class OrderBlock:
    kind: str                   # 'bullish' | 'bearish'
    created_at: pd.Timestamp    # time of the OB candle itself
    confirmed_at: pd.Timestamp  # time of the displacement candle (when KNOWABLE)
    top: float                  # OB candle high
    bottom: float               # OB candle low
    displacement_atr: float     # displacement candle body size, in ATR multiples
    tested_at: pd.Timestamp | None = None       # first re-entry into the zone
    invalidated_at: pd.Timestamp | None = None  # first close beyond the far side
    meta: dict = field(default_factory=dict)

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def state(self) -> str:
        if self.invalidated_at is not None:
            return "invalidated"
        if self.tested_at is not None:
            return "tested"
        return "fresh"


def detect_orderblocks(df: pd.DataFrame, min_displacement_atr: float = 1.0,
                       atr_period: int = 14) -> list[OrderBlock]:
    """Scan the whole dataframe; returns OBs with state tracked forward."""
    a = atr(df, atr_period)
    opens, closes = df["open"].values, df["close"].values
    highs, lows = df["high"].values, df["low"].values
    times = df.index
    out: list[OrderBlock] = []

    for i in range(1, len(df)):
        if np.isnan(a.iloc[i]) or a.iloc[i] == 0:
            continue
        body_atr = abs(closes[i] - opens[i]) / a.iloc[i]
        if body_atr < min_displacement_atr:
            continue
        j = i - 1
        if closes[i] > opens[i] and closes[j] < opens[j] and closes[i] > highs[j]:
            out.append(OrderBlock("bullish", times[j], times[i],
                                  top=highs[j], bottom=lows[j],
                                  displacement_atr=round(body_atr, 2)))
        elif closes[i] < opens[i] and closes[j] > opens[j] and closes[i] < lows[j]:
            out.append(OrderBlock("bearish", times[j], times[i],
                                  top=highs[j], bottom=lows[j],
                                  displacement_atr=round(body_atr, 2)))

    _track_states(df, out)
    return out


def _track_states(df: pd.DataFrame, obs: list[OrderBlock]) -> None:
    for ob in obs:
        after = df[df.index > ob.confirmed_at]
        if after.empty:
            continue
        if ob.kind == "bullish":
            entered = after["low"] <= ob.top
            broken = after["close"] < ob.bottom
        else:
            entered = after["high"] >= ob.bottom
            broken = after["close"] > ob.top
        if entered.any():
            ob.tested_at = after.index[entered.argmax()]
        if broken.any():
            ob.invalidated_at = after.index[broken.argmax()]


def orderblocks_to_frame(obs: list[OrderBlock]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"kind": ob.kind, "created_at": ob.created_at,
          "confirmed_at": ob.confirmed_at, "top": ob.top, "bottom": ob.bottom,
          "displacement_atr": ob.displacement_atr, "state": ob.state,
          "tested_at": ob.tested_at, "invalidated_at": ob.invalidated_at}
         for ob in obs]
    )
