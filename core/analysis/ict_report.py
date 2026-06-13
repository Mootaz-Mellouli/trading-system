"""ICT analysis aggregator: one structured, serializable snapshot.

build_ict_analysis() runs every detector across the bias timeframes and the
entry timeframe and assembles a single ICTAnalysis object for the dashboard
(and any future head). It is pure data assembly: deterministic, no
wall-clock reads (the snapshot is timestamped with the last CANDLE time),
no LLM calls.

The confluence section is a CHECKLIST, not a verdict (non-negotiable rule:
the decision is human). Each item is computed by a simple explicit rule,
with NO weighting, NO aggregate score, NO trade recommendation:
  - structure per timeframe -> the direction of the last BOS/CHoCH
    (neutral when no structure event exists yet);
  - dealing-range position  -> discount reads bullish (favorable to longs),
    premium reads bearish, equilibrium reads neutral;
  - nearest open FVG / fresh order block -> reads as the zone's own kind;
  - liquidity draw -> the nearest untaken pool: buyside (above) reads
    bullish (price tends to be drawn to resting liquidity), sellside reads
    bearish;
  - last sweep -> a buyside sweep (stops above taken, close back inside)
    reads bearish; a sellside sweep reads bullish;
  - active session -> always neutral (time is context, not direction).
Never add a score, verdict or recommendation field to these objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from core.analysis.amd import AMDPhase, amd_from_session
from core.analysis.range_session import (
    DEFAULT_SESSIONS, DealingRange, dealing_range, session_now,
)
from core.analysis.scoring import ICTScores, build_scores
from core.analysis.setups import TradeSetup, build_setup
from core.data.provider import BaseProvider
from core.detectors.fvg import atr, detect_fvgs
from core.detectors.liquidity import LiquiditySweep, detect_liquidity
from core.detectors.orderblocks import detect_orderblocks
from core.detectors.ote import OTEZone, detect_ote
from core.detectors.swings import detect_swings, detect_structure_events
from core.strategies.base import BaseStrategy, FVGRetestStrategy, Signal

_ZONE_READING = {"discount": "bullish", "premium": "bearish",
                 "equilibrium": "neutral"}
_POOL_READING = {"buyside": "bullish", "sellside": "bearish"}
_SWEEP_READING = {"buyside": "bearish", "sellside": "bullish"}


@dataclass
class StructureView:
    timeframe: str
    trend: str                # 'bullish' | 'bearish' | 'neutral'
    last_event_kind: str | None        # 'BOS' | 'CHoCH'
    last_event_direction: str | None
    last_event_price: float | None
    last_event_time: pd.Timestamp | None
    range_zone: str | None             # 'premium' | 'discount' | 'equilibrium'
    range_position_pct: float | None


@dataclass
class NearbyZone:
    source: str               # 'fvg' | 'orderblock'
    kind: str                 # 'bullish' | 'bearish'
    top: float
    bottom: float
    state: str                # fvg: 'open' | 'mitigated'; ob: 'fresh'
    distance_atr: float       # nearest edge to current price; 0 = price inside
    created_at: pd.Timestamp


@dataclass
class PoolView:
    side: str                 # 'buyside' | 'sellside'
    price: float
    is_equal: bool            # equal highs/lows (stronger pool)
    distance_atr: float


@dataclass
class ConfluenceItem:
    element: str
    reading: str              # 'bullish' | 'bearish' | 'neutral' — nothing else
    key_level: float | None
    detail: str


@dataclass
class ICTAnalysis:
    """One ICT snapshot. Carries the confluence checklist AND (per rule 6,
    rewritten) deterministic scores + a display-only setup suggestion. It
    still has NO single verdict/recommendation field — both score directions
    are presented side by side and the setup is explicitly a suggestion."""
    symbol: str
    generated_at: pd.Timestamp     # last entry-tf candle time (data, not wall clock)
    entry_timeframe: str
    bias_timeframes: list[str]
    price: float                   # last close on the entry timeframe
    bias: list[StructureView]
    entry_structure: StructureView
    nearest_zones: list[NearbyZone]        # open FVGs + fresh OBs, by distance
    liquidity_pools: list[PoolView]        # untaken only, by distance
    recent_sweeps: list[LiquiditySweep]
    dealing_range: DealingRange | None
    active_session: str
    signals: list[Signal] = field(default_factory=list)
    confluence: list[ConfluenceItem] = field(default_factory=list)
    ote_zones: list[OTEZone] = field(default_factory=list)
    amd: AMDPhase | None = None
    scores: ICTScores | None = None
    setup: TradeSetup | None = None        # display-only suggestion, may be None

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _structure_view(df: pd.DataFrame, timeframe: str, swing_lookback: int,
                    n_swings: int, eq_tolerance_pct: float):
    swings = detect_swings(df, swing_lookback)
    events = detect_structure_events(df, swings)
    dr = dealing_range(df, swings, n_swings, eq_tolerance_pct)
    last = events[-1] if events else None
    view = StructureView(
        timeframe=timeframe,
        trend=last.direction if last else "neutral",
        last_event_kind=last.kind if last else None,
        last_event_direction=last.direction if last else None,
        last_event_price=last.price if last else None,
        last_event_time=last.time if last else None,
        range_zone=dr.zone if dr else None,
        range_position_pct=dr.position_pct if dr else None,
    )
    return view, swings, dr


def _edge_distance_atr(price: float, top: float, bottom: float,
                       atr_value: float) -> float:
    if bottom <= price <= top:
        return 0.0
    gap = bottom - price if price < bottom else price - top
    return round(gap / atr_value, 2)


def build_ict_analysis(
    provider: BaseProvider, symbol: str, entry_tf: str,
    bias_tfs: tuple[str, ...] = ("1d", "1h"),
    bars: int = 400,
    swing_lookback: int = 5,
    min_displacement_atr: float = 0.8,      # FVG displacement filter
    ob_min_displacement_atr: float = 1.0,   # order block displacement filter
    atr_period: int = 14,
    tolerance_atr: float = 0.1,             # equal highs/lows grouping
    max_pierce_atr: float = 0.5,            # sweep wick limit
    min_pool_age_bars: int = 0,
    n_swings: int = 5,                      # dealing range significance
    eq_tolerance_pct: float = 5.0,          # equilibrium band half-width
    sessions: dict = DEFAULT_SESSIONS,
    n_recent_sweeps: int = 5,
    strategies: list[BaseStrategy] | None = None,
    ote_fib_low: float = 0.62,              # OTE band knobs
    ote_fib_high: float = 0.79,
    ote_sweet: float = 0.705,
    ote_min_leg_atr: float = 1.0,
    amd_session: str = "asia",              # accumulation reference window
    score_weights: dict | None = None,      # confluence weights (knobs)
    score_proximity_atr: float = 3.0,
    setup_min_strength: float = 50.0,       # setup gating knobs
    setup_margin: float = 10.0,
    setup_sl_buffer: float = 0.0,
    setup_rr_targets: tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> ICTAnalysis:
    """Fetch candles per timeframe, run all detectors, return one snapshot."""
    df = provider.fetch(symbol, entry_tf, bars)
    price = float(df["close"].iloc[-1])
    now = df.index[-1]

    # --- bias timeframes: structure + range position only -------------------
    bias_views = [
        _structure_view(provider.fetch(symbol, tf, bars), tf,
                        swing_lookback, n_swings, eq_tolerance_pct)[0]
        for tf in bias_tfs
    ]

    # --- entry timeframe: everything ----------------------------------------
    entry_view, swings, dr = _structure_view(df, entry_tf, swing_lookback,
                                             n_swings, eq_tolerance_pct)
    fvgs = detect_fvgs(df, min_displacement_atr, atr_period)
    obs = detect_orderblocks(df, ob_min_displacement_atr, atr_period)
    pools, sweeps = detect_liquidity(df, swings, tolerance_atr,
                                     max_pierce_atr, min_pool_age_bars,
                                     atr_period)
    atr_last = float(atr(df, atr_period).iloc[-1])
    have_atr = not np.isnan(atr_last) and atr_last > 0

    zones: list[NearbyZone] = []
    pool_views: list[PoolView] = []
    if have_atr:  # without a defined ATR, distances are meaningless
        for f in fvgs:
            if f.filled_at is not None:
                continue
            zones.append(NearbyZone(
                "fvg", f.kind, f.top, f.bottom,
                "open" if f.mitigated_at is None else "mitigated",
                _edge_distance_atr(price, f.top, f.bottom, atr_last),
                f.created_at))
        for ob in obs:
            if ob.state != "fresh":
                continue
            zones.append(NearbyZone(
                "orderblock", ob.kind, ob.top, ob.bottom, "fresh",
                _edge_distance_atr(price, ob.top, ob.bottom, atr_last),
                ob.created_at))
        zones.sort(key=lambda z: z.distance_atr)
        pool_views = sorted(
            (PoolView(p.side, p.price, p.is_equal,
                      round(abs(price - p.price) / atr_last, 2))
             for p in pools if p.status == "untaken"),
            key=lambda p: p.distance_atr)

    ote_zones = detect_ote(df, swings, ote_fib_low, ote_fib_high, ote_sweet,
                           ote_min_leg_atr, atr_period)
    amd = amd_from_session(df, amd_session, sessions, max_pierce_atr,
                           atr_period=atr_period)

    strategies = strategies if strategies is not None else [FVGRetestStrategy()]
    signals = [s for strat in strategies for s in strat.generate(symbol, df)]

    analysis = ICTAnalysis(
        symbol=symbol, generated_at=now, entry_timeframe=entry_tf,
        bias_timeframes=list(bias_tfs), price=price,
        bias=bias_views, entry_structure=entry_view,
        nearest_zones=zones, liquidity_pools=pool_views,
        recent_sweeps=sweeps[-n_recent_sweeps:],
        dealing_range=dr,
        active_session=session_now(now, sessions),
        signals=signals,
        ote_zones=ote_zones, amd=amd,
    )
    analysis.confluence = _build_confluence(analysis)
    analysis.scores = build_scores(analysis, ote_zones, amd,
                                   weights=score_weights,
                                   proximity_atr=score_proximity_atr)
    analysis.setup = build_setup(analysis, analysis.scores, ote_zones,
                                 min_strength=setup_min_strength,
                                 margin=setup_margin, sl_buffer=setup_sl_buffer,
                                 rr_targets=setup_rr_targets)
    return analysis


def _build_confluence(a: ICTAnalysis) -> list[ConfluenceItem]:
    """One item per element, by the explicit rules in the module docstring.
    A checklist for a human — never aggregate these into a score."""
    items: list[ConfluenceItem] = []

    for sv in [*a.bias, a.entry_structure]:
        detail = (f"{sv.last_event_kind} {sv.last_event_direction} @ "
                  f"{sv.last_event_price}" if sv.last_event_kind
                  else "no structure event yet")
        items.append(ConfluenceItem(f"structure_{sv.timeframe}", sv.trend,
                                    sv.last_event_price, detail))

    if a.dealing_range is not None:
        dr = a.dealing_range
        items.append(ConfluenceItem(
            "dealing_range_position", _ZONE_READING[dr.zone], dr.equilibrium,
            f"price at {dr.position_pct:.1f}% of range -> {dr.zone}"))
    else:
        items.append(ConfluenceItem("dealing_range_position", "neutral",
                                    None, "no confirmed range yet"))

    for source, element in (("fvg", "nearest_fvg"),
                            ("orderblock", "nearest_orderblock")):
        zone = next((z for z in a.nearest_zones if z.source == source), None)
        if zone is None:
            items.append(ConfluenceItem(element, "neutral", None,
                                        f"no {zone_name(source)}"))
        else:
            items.append(ConfluenceItem(
                element, zone.kind, (zone.top + zone.bottom) / 2,
                f"{zone.kind} {zone_name(source)} "
                f"[{zone.bottom:.5f}-{zone.top:.5f}] "
                f"{zone.distance_atr} ATR away ({zone.state})"))

    pool = a.liquidity_pools[0] if a.liquidity_pools else None
    if pool is None:
        items.append(ConfluenceItem("liquidity_draw", "neutral", None,
                                    "no untaken pool"))
    else:
        items.append(ConfluenceItem(
            "liquidity_draw", _POOL_READING[pool.side], pool.price,
            f"untaken {pool.side} liquidity"
            f"{' (equal highs/lows)' if pool.is_equal else ''} "
            f"{pool.distance_atr} ATR away"))

    sweep = a.recent_sweeps[-1] if a.recent_sweeps else None
    if sweep is None:
        items.append(ConfluenceItem("last_sweep", "neutral", None,
                                    "no recent sweep"))
    else:
        items.append(ConfluenceItem(
            "last_sweep", _SWEEP_READING[sweep.side], sweep.level,
            f"{sweep.side} sweep @ {sweep.level} "
            f"(pierce {sweep.pierce_atr} ATR)"))

    items.append(ConfluenceItem("session", "neutral", None,
                                f"active session: {a.active_session}"))
    return items


def zone_name(source: str) -> str:
    return "open FVG" if source == "fvg" else "fresh order block"
