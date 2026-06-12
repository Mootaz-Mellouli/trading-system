"""Strategy layer: combines detector outputs into Signals.

A Signal is plain data — the same object is consumed by the backtester,
the Telegram bot, and the dashboard. Strategies must be deterministic:
same candles in, same signals out.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import uuid4

import pandas as pd

from core.detectors.fvg import detect_fvgs
from core.detectors.swings import detect_swings, detect_structure_events


@dataclass
class Signal:
    symbol: str
    time: pd.Timestamp
    direction: str            # 'long' | 'short'
    entry: float
    stop_loss: float
    take_profit: float
    reason: str
    detections: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        return round(reward / risk, 2) if risk else 0.0


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, symbol: str, df: pd.DataFrame) -> list[Signal]:
        ...


class FVGRetestStrategy(BaseStrategy):
    """Starter example (NOT the client's final rules — a scaffold to iterate on):

    Long setup:
      1. A bullish CHoCH occurs (structure flips up), and
      2. a bullish FVG was created within `recent` bars before/around it, and
      3. price later retests (re-enters) that FVG zone.
    Entry at the FVG top, stop below the FVG bottom, TP at fixed R multiple.
    Short is symmetric. Every threshold is a constructor argument so the
    trader's numbers can be plugged in without code changes.
    """

    name = "fvg_retest"

    def __init__(self, min_displacement_atr: float = 0.8, swing_lookback: int = 5,
                 recent: int = 30, rr: float = 2.0, stop_buffer: float = 0.0001):
        self.min_displacement_atr = min_displacement_atr
        self.swing_lookback = swing_lookback
        self.recent = recent
        self.rr = rr
        self.stop_buffer = stop_buffer

    def generate(self, symbol: str, df: pd.DataFrame) -> list[Signal]:
        fvgs = detect_fvgs(df, self.min_displacement_atr)
        swings = detect_swings(df, self.swing_lookback)
        events = detect_structure_events(df, swings)
        chochs = [e for e in events if e.kind == "CHoCH"]
        signals: list[Signal] = []
        step = df.index[1] - df.index[0]

        for f in fvgs:
            if f.mitigated_at is None:
                continue  # never retested -> no entry
            window_start = f.created_at - self.recent * step
            window_end = f.created_at + self.recent * step
            aligned = [c for c in chochs
                       if window_start <= c.time <= window_end
                       and ((c.direction == "bullish" and f.kind == "bullish")
                            or (c.direction == "bearish" and f.kind == "bearish"))]
            if not aligned:
                continue
            # The signal exists only once ALL conditions are known:
            # the retest AND the confirming CHoCH (whichever came last).
            t_signal = max([f.mitigated_at] + [c.time for c in aligned
                                               if c.time <= f.mitigated_at] or [f.mitigated_at])
            late_chochs = [c.time for c in aligned if c.time > f.mitigated_at]
            if late_chochs:
                t_signal = min(late_chochs)  # earliest moment everything aligned
            if f.kind == "bullish":
                entry, sl = f.top, f.bottom - self.stop_buffer
                tp = entry + self.rr * (entry - sl)
                direction = "long"
            else:
                entry, sl = f.bottom, f.top + self.stop_buffer
                tp = entry - self.rr * (sl - entry)
                direction = "short"
            signals.append(Signal(
                symbol=symbol, time=t_signal, direction=direction,
                entry=round(entry, 5), stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                reason=(f"{f.kind} CHoCH + retest of {f.kind} FVG "
                        f"[{f.bottom:.5f}–{f.top:.5f}] "
                        f"(displacement {f.displacement_atr}x ATR)"),
                detections=["choch", "fvg", "fvg_retest"],
            ))
        return signals
