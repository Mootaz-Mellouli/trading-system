"""ICT confluence scoring — deterministic, tunable, traceable.

Turns detector outputs into per-element scores and an aggregate
**confluence strength** per direction (bullish / bearish), each on a 0-100
scale. This is NOT a win-probability or a profit forecast (rule 7): it is
a weighted reading of how many ICT elements lean a given way, and every
point traces back to the element that produced it (rule 6, rewritten).

Each directional element scores BOTH sides 0-100 (e.g. a nearby bullish
order block scores bullish, 0 bearish). The aggregate per direction is the
weighted average of the directional elements; weights are knobs (rule 4).
The session element is timing FAVORABILITY (0-100, directionless) — shown
on its own, never blended into a direction.

No randomness, no wall-clock, no LLM: same detector outputs in -> identical
scores out (rule 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Directional elements that feed the aggregate, with default weights (knobs).
DEFAULT_WEIGHTS: dict[str, float] = {
    "structure": 0.25, "liquidity": 0.20, "orderblock": 0.15,
    "fvg": 0.10, "ote": 0.15, "amd": 0.15,
}
# Session timing favorability (directionless), 0-100 (knob).
SESSION_FAVORABILITY: dict[str, float] = {
    "london_kz": 100.0, "ny_kz": 100.0, "asia": 55.0,
    "london_close": 45.0, "off_hours": 15.0,
}


@dataclass
class ElementScore:
    element: str
    bullish: float            # 0-100
    bearish: float            # 0-100
    detail: str = ""


@dataclass
class DirectionScore:
    direction: str            # 'bullish' | 'bearish'
    strength: float           # 0-100 weighted confluence strength
    components: dict = field(default_factory=dict)  # element -> its directional score


@dataclass
class ICTScores:
    """Per-element + aggregate scores. No verdict/recommendation field — both
    directions are presented side by side; the human decides."""
    elements: list[ElementScore]
    bullish: DirectionScore
    bearish: DirectionScore
    session_favorability: float
    weights: dict


def _clamp(x: float) -> float:
    return round(max(0.0, min(100.0, x)), 1)


def _proximity(distance_atr: float, scale: float) -> float:
    """1.0 when inside (distance 0), fading linearly to 0 at `scale` ATR."""
    if scale <= 0:
        return 1.0 if distance_atr <= 0 else 0.0
    return max(0.0, 1.0 - distance_atr / scale)


def _structure_score(trends: list[str]) -> ElementScore:
    n = len(trends) or 1
    bull = 100.0 * sum(t == "bullish" for t in trends) / n
    bear = 100.0 * sum(t == "bearish" for t in trends) / n
    return ElementScore("structure", _clamp(bull), _clamp(bear),
                        f"{sum(t=='bullish' for t in trends)}↑/"
                        f"{sum(t=='bearish' for t in trends)}↓ of {n} TF")


def _zone_score(element: str, kind: str | None, distance_atr: float | None,
                scale: float, base: float = 100.0) -> ElementScore:
    if kind is None or distance_atr is None:
        return ElementScore(element, 0.0, 0.0, "none")
    s = base * _proximity(distance_atr, scale)
    detail = f"{kind} {distance_atr} ATR away"
    return (ElementScore(element, _clamp(s), 0.0, detail) if kind == "bullish"
            else ElementScore(element, 0.0, _clamp(s), detail))


def _liquidity_score(pool_side: str | None, pool_dist: float | None,
                     sweep_side: str | None, scale: float,
                     pool_base: float = 60.0, sweep_base: float = 40.0) -> ElementScore:
    bull = bear = 0.0
    bits = []
    if pool_side is not None and pool_dist is not None:
        s = pool_base * _proximity(pool_dist, scale)
        # resting liquidity is a draw: buyside (above) pulls price up
        if pool_side == "buyside":
            bull += s
        else:
            bear += s
        bits.append(f"{pool_side} pool {pool_dist} ATR")
    if sweep_side is not None:
        # a sweep takes liquidity then reverses: buyside sweep -> bearish
        if sweep_side == "buyside":
            bear += sweep_base
        else:
            bull += sweep_base
        bits.append(f"{sweep_side} sweep")
    return ElementScore("liquidity", _clamp(bull), _clamp(bear),
                        " + ".join(bits) or "none")


def _ote_score(direction: str | None, state: str | None,
               inside_base: float = 100.0, fresh_base: float = 50.0) -> ElementScore:
    if direction is None or state in (None, "invalidated"):
        return ElementScore("ote", 0.0, 0.0, "none")
    s = inside_base if state == "tested" else fresh_base
    detail = f"{direction} OTE ({state})"
    return (ElementScore("ote", _clamp(s), 0.0, detail) if direction == "bullish"
            else ElementScore("ote", 0.0, _clamp(s), detail))


def _amd_score(phase: str | None, direction: str | None,
               dist_base: float = 100.0, manip_base: float = 60.0) -> ElementScore:
    if phase == "distribution" and direction in ("bullish", "bearish"):
        s = dist_base
    elif phase == "manipulation" and direction in ("bullish", "bearish"):
        s = manip_base
    else:
        return ElementScore("amd", 0.0, 0.0, phase or "none")
    detail = f"{phase} -> {direction}"
    return (ElementScore("amd", _clamp(s), 0.0, detail) if direction == "bullish"
            else ElementScore("amd", 0.0, _clamp(s), detail))


def _aggregate(direction: str, elements: dict[str, ElementScore],
               weights: dict) -> DirectionScore:
    total_w = sum(weights.get(e, 0.0) for e in elements) or 1.0
    comp = {e: (es.bullish if direction == "bullish" else es.bearish)
            for e, es in elements.items()}
    strength = sum(weights.get(e, 0.0) * comp[e] for e in elements) / total_w
    return DirectionScore(direction, _clamp(strength), comp)


def build_scores(analysis, ote_zones=None, amd=None, *,
                 weights: dict | None = None, proximity_atr: float = 3.0,
                 session_favorability: dict | None = None) -> ICTScores:
    """Score an ICTAnalysis (duck-typed) plus OTE zones and an AMD phase.

    `analysis` must expose: bias (list of views with .trend), entry_structure
    (.trend), nearest_zones (.source/.kind/.distance_atr), liquidity_pools
    (.side/.distance_atr), recent_sweeps (.side), active_session.
    """
    weights = weights or DEFAULT_WEIGHTS
    fav_map = session_favorability or SESSION_FAVORABILITY

    trends = [sv.trend for sv in analysis.bias] + [analysis.entry_structure.trend]
    structure = _structure_score(trends)

    fvg = next((z for z in analysis.nearest_zones if z.source == "fvg"), None)
    ob = next((z for z in analysis.nearest_zones if z.source == "orderblock"), None)
    fvg_es = _zone_score("fvg", fvg.kind if fvg else None,
                         fvg.distance_atr if fvg else None, proximity_atr)
    ob_es = _zone_score("orderblock", ob.kind if ob else None,
                        ob.distance_atr if ob else None, proximity_atr)

    pool = analysis.liquidity_pools[0] if analysis.liquidity_pools else None
    sweep = analysis.recent_sweeps[-1] if analysis.recent_sweeps else None
    liquidity = _liquidity_score(pool.side if pool else None,
                                 pool.distance_atr if pool else None,
                                 sweep.side if sweep else None, proximity_atr)

    active_ote = next((z for z in reversed(ote_zones or [])
                       if z.state != "invalidated"), None)
    ote_es = _ote_score(active_ote.direction if active_ote else None,
                        active_ote.state if active_ote else None)

    amd_es = _amd_score(amd.phase if amd else None,
                        amd.direction if amd else None)

    fav = _clamp(fav_map.get(analysis.active_session, 30.0))
    session = ElementScore("session", fav, fav,
                           f"{analysis.active_session} favorability {fav}")

    directional = {"structure": structure, "liquidity": liquidity,
                   "orderblock": ob_es, "fvg": fvg_es, "ote": ote_es,
                   "amd": amd_es}
    return ICTScores(
        elements=list(directional.values()) + [session],
        bullish=_aggregate("bullish", directional, weights),
        bearish=_aggregate("bearish", directional, weights),
        session_favorability=fav,
        weights=dict(weights),
    )
