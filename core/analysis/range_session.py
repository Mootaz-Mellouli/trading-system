"""Dealing range (premium / discount / equilibrium) and session utilities.

Dealing range:
  The range between the most recent significant swing high and swing low.
  Significance is parameterized: the most recent `n_swings` CONFIRMED swing
  highs (resp. lows) are considered and the range takes their extremes —
  n_swings=1 means the latest swing pair, larger values favour price
  extremity over recency. Only swings with confirmed_at <= the last bar are
  used (no lookahead). Position is measured at the last close:
    premium  > 50% of the range, discount < 50%,
    equilibrium when within eq_tolerance_pct of the midpoint (45-55% by
    default) — the equilibrium band takes precedence.

Sessions / kill zones:
  All boundaries are UTC times-of-day, configurable via the `sessions`
  mapping (defaults below). Time is ALWAYS passed in explicitly — never
  read from the wall clock (determinism rule). A bar belongs to a session
  when its open timestamp falls in [start, end). Sessions may wrap
  midnight (start > end). Timezones other than UTC are for DISPLAY only
  (see session_bounds).

Tunable rules (the knobs to negotiate with the trader):
  - n_swings: how many recent confirmed swings define the range extremes.
  - eq_tolerance_pct: half-width of the equilibrium band, in % of range.
  - sessions: name -> (start, end) UTC times-of-day.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from core.detectors.swings import SwingPoint

DEFAULT_SESSIONS: dict[str, tuple[time, time]] = {
    "asia": (time(0, 0), time(7, 0)),
    "london_kz": (time(7, 0), time(10, 0)),
    "ny_kz": (time(12, 0), time(15, 0)),
    "london_close": (time(15, 0), time(17, 0)),
}
OFF_HOURS = "off_hours"


# ------------------------------------------------------- dealing range -----

@dataclass
class DealingRange:
    high: float
    low: float
    high_time: pd.Timestamp   # swing bar that set the range high
    low_time: pd.Timestamp    # swing bar that set the range low
    equilibrium: float        # 50% of the range
    price: float              # reference price (last close)
    position_pct: float       # 0 = range low, 100 = range high
    zone: str                 # 'premium' | 'discount' | 'equilibrium'


def dealing_range(df: pd.DataFrame, swings: list[SwingPoint],
                  n_swings: int = 5,
                  eq_tolerance_pct: float = 5.0) -> DealingRange | None:
    """Range between the extremes of the last `n_swings` confirmed swing
    highs and lows; None when no confirmed high or low exists yet."""
    last = df.index[-1]
    highs = [s for s in swings if s.kind == "high" and s.confirmed_at <= last]
    lows = [s for s in swings if s.kind == "low" and s.confirmed_at <= last]
    if not highs or not lows:
        return None
    top = max(highs[-n_swings:], key=lambda s: s.price)
    bottom = min(lows[-n_swings:], key=lambda s: s.price)
    width = top.price - bottom.price
    if width <= 0:
        return None
    price = float(df["close"].iloc[-1])
    position = (price - bottom.price) / width * 100
    if abs(position - 50.0) <= eq_tolerance_pct:
        zone = "equilibrium"
    elif position > 50.0:
        zone = "premium"
    else:
        zone = "discount"
    return DealingRange(top.price, bottom.price, top.time, bottom.time,
                        equilibrium=(top.price + bottom.price) / 2,
                        price=price, position_pct=position, zone=zone)


# ---------------------------------------------------- sessions / killzones -

@dataclass
class SessionLevels:
    session: str
    start: pd.Timestamp  # UTC
    end: pd.Timestamp    # UTC
    high: float
    low: float


def _as_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(
            "timestamp must be tz-aware; pass time in explicitly in UTC "
            "(never read the wall clock)")
    return ts.tz_convert("UTC")


def _td(t: time) -> pd.Timedelta:
    return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)


def session_now(ts, sessions: dict[str, tuple[time, time]] = DEFAULT_SESSIONS) -> str:
    """Session name whose [start, end) UTC window contains the given
    (explicitly passed, tz-aware) timestamp; OFF_HOURS when none does."""
    t = _as_utc(ts).time()
    for name, (start, end) in sessions.items():
        if start <= end:
            if start <= t < end:
                return name
        elif t >= start or t < end:  # session wrapping midnight
            return name
    return OFF_HOURS


def session_bounds(session: str, on_date,
                   sessions: dict[str, tuple[time, time]] = DEFAULT_SESSIONS,
                   tz: str = "UTC") -> tuple[pd.Timestamp, pd.Timestamp]:
    """Concrete start/end of a session on the given UTC date. `tz` converts
    the result for DISPLAY only — the session is defined in UTC."""
    start_t, end_t = sessions[session]
    d = pd.Timestamp(on_date)
    d = (d.tz_localize("UTC") if d.tzinfo is None else d.tz_convert("UTC")).normalize()
    start = d + _td(start_t)
    end = d + _td(end_t)
    if end <= start:  # session wrapping midnight
        end += pd.Timedelta(days=1)
    return start.tz_convert(tz), end.tz_convert(tz)


def session_spans(index: pd.DatetimeIndex,
                  sessions: dict[str, tuple[time, time]] = DEFAULT_SESSIONS
                  ) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Concrete (start, end, name) session windows overlapping the given
    UTC index span — display helper for chart shading."""
    if len(index) == 0:
        return []
    first, last = index[0], index[-1]
    days = pd.date_range(first.normalize() - pd.Timedelta(days=1),
                         last.normalize(), freq="1D")
    spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for day in days:
        for name in sessions:
            start, end = session_bounds(name, day, sessions)
            if end <= first or start > last:
                continue
            spans.append((start, end, name))
    return spans


def session_levels(df: pd.DataFrame, session: str = "asia",
                   sessions: dict[str, tuple[time, time]] = DEFAULT_SESSIONS) -> SessionLevels | None:
    """High/low of the given session on the LAST bar's UTC day (for a
    wrapping session: the occurrence ending that day). None when the data
    has no bars in that window."""
    start_t, end_t = sessions[session]
    day = df.index[-1].normalize()
    start = day + _td(start_t)
    end = day + _td(end_t)
    if end <= start:  # wraps midnight: starts the previous day
        start -= pd.Timedelta(days=1)
    window = df[(df.index >= start) & (df.index < end)]
    if window.empty:
        return None
    return SessionLevels(session, start, end,
                         high=float(window["high"].max()),
                         low=float(window["low"].min()))
