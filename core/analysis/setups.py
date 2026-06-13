"""Trade setup builder — a deterministic, display-only suggestion.

build_setup() turns an ICTAnalysis + its scores into at most one TradeSetup:
a direction, an entry zone (top/bottom), a stop, and three staged targets
(TP1/2/3). It is a SUGGESTION a human confirms (rule 6, rewritten) — not a
command, not a promise (rule 7). It reuses the plain-data rule (rule 5):
`to_signal()` yields a `Signal` so the setup flows through the existing
pessimistic backtester and the future bot unchanged.

How it is built (all explicit, all tunable):
  - Direction = the side with the higher confluence strength, but only if it
    clears `min_strength` AND beats the other side by `margin`. Otherwise no
    setup (None) — we do not force a trade out of a flat read.
  - Entry zone = the active OTE band if present, else the nearest fresh order
    block, else the nearest open FVG, on the chosen side.
  - Stop = just beyond the zone (OTE: beyond the leg origin), with a buffer.
  - TP1/2/3 = entry ± rr_i * risk, where risk = |entry - stop|.

No randomness, no wall-clock, no LLM (rule 2). Same inputs -> same setup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pandas as pd

from core.strategies.base import Signal


@dataclass
class TradeSetup:
    symbol: str
    time: pd.Timestamp
    direction: str            # 'bullish' | 'bearish'
    basis: str                # 'OTE' | 'order block' | 'FVG'
    entry_zone_top: float
    entry_zone_bottom: float
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    bullish_strength: float
    bearish_strength: float
    reason: str
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)

    def to_signal(self) -> Signal:
        """Plain-data bridge: TP1 is the Signal take-profit (rule 5)."""
        return Signal(
            symbol=self.symbol, time=self.time,
            direction="long" if self.direction == "bullish" else "short",
            entry=round(self.entry, 5), stop_loss=round(self.stop_loss, 5),
            take_profit=round(self.tp1, 5), reason=self.reason,
            detections=["ict_setup", self.basis], id=self.id)


def _pick_direction(bull: float, bear: float, min_strength: float,
                    margin: float) -> str | None:
    hi, lo, direction = ((bull, bear, "bullish") if bull >= bear
                         else (bear, bull, "bearish"))
    if hi >= min_strength and (hi - lo) >= margin:
        return direction
    return None


def _entry_zone(direction: str, ote_zones, nearest_zones):
    """(basis, top, bottom, entry, sl_anchor) for the chosen side, or None."""
    kind = direction  # zones are tagged 'bullish'/'bearish'
    ote = next((z for z in reversed(ote_zones or [])
                if z.direction == kind and z.state != "invalidated"), None)
    if ote is not None:
        anchor = ote.leg_low if direction == "bullish" else ote.leg_high
        return "OTE", ote.top, ote.bottom, ote.sweet, anchor
    for source, label in (("orderblock", "order block"), ("fvg", "FVG")):
        z = next((z for z in nearest_zones
                  if z.source == source and z.kind == kind), None)
        if z is not None:
            mid = (z.top + z.bottom) / 2
            anchor = z.bottom if direction == "bullish" else z.top
            return label, z.top, z.bottom, mid, anchor
    return None


def build_setup(analysis, scores, ote_zones=None, *,
                min_strength: float = 50.0, margin: float = 10.0,
                sl_buffer: float = 0.0,
                rr_targets: tuple[float, float, float] = (1.0, 2.0, 3.0)
                ) -> TradeSetup | None:
    """At most one setup for the current read, or None when the lean is weak
    or no entry zone exists on the chosen side."""
    bull, bear = scores.bullish.strength, scores.bearish.strength
    direction = _pick_direction(bull, bear, min_strength, margin)
    if direction is None:
        return None
    picked = _entry_zone(direction, ote_zones, analysis.nearest_zones)
    if picked is None:
        return None
    basis, top, bottom, entry, anchor = picked

    if direction == "bullish":
        stop = anchor - sl_buffer
        risk = entry - stop
        if risk <= 0:
            return None
        tp1, tp2, tp3 = (entry + r * risk for r in rr_targets)
    else:
        stop = anchor + sl_buffer
        risk = stop - entry
        if risk <= 0:
            return None
        tp1, tp2, tp3 = (entry - r * risk for r in rr_targets)

    strength = bull if direction == "bullish" else bear
    reason = (f"{direction} setup — {basis} entry "
              f"[{bottom:.5f}–{top:.5f}], confluence strength {strength:.0f} "
              f"(bull {bull:.0f} / bear {bear:.0f}). Suggestion only.")
    return TradeSetup(
        symbol=analysis.symbol, time=analysis.generated_at,
        direction=direction, basis=basis,
        entry_zone_top=top, entry_zone_bottom=bottom, entry=entry,
        stop_loss=stop, tp1=tp1, tp2=tp2, tp3=tp3,
        bullish_strength=bull, bearish_strength=bear, reason=reason)
