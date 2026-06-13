"""Scoring layer: exact per-element numbers, plus determinism, bounds,
tunable weights, and the guarantee that no verdict field exists."""
from types import SimpleNamespace as NS

import pytest

from core.analysis.scoring import (
    DEFAULT_WEIGHTS, ICTScores, _amd_score, _liquidity_score, _ote_score,
    _proximity, _structure_score, _zone_score, build_scores,
)


# ---- pure element scorers (exact) ----------------------------------------

def test_proximity_fades_linearly():
    assert _proximity(0.0, 3.0) == 1.0
    assert _proximity(1.5, 3.0) == 0.5
    assert _proximity(3.0, 3.0) == 0.0
    assert _proximity(5.0, 3.0) == 0.0   # never negative


def test_structure_score_fraction_of_timeframes():
    es = _structure_score(["bullish", "bullish", "neutral"])
    assert es.bullish == pytest.approx(66.7)
    assert es.bearish == 0.0


def test_zone_score_scales_with_distance():
    assert _zone_score("fvg", "bullish", 0.0, 3.0).bullish == 100.0
    assert _zone_score("fvg", "bullish", 1.5, 3.0).bullish == 50.0
    bear = _zone_score("orderblock", "bearish", 0.0, 3.0)
    assert (bear.bullish, bear.bearish) == (0.0, 100.0)
    none = _zone_score("fvg", None, None, 3.0)
    assert (none.bullish, none.bearish) == (0.0, 0.0)


def test_liquidity_score_pool_draw_plus_sweep_reversal():
    es = _liquidity_score("buyside", 0.0, "sellside", 3.0)
    # buyside pool draws up (60) AND sellside sweep reverses up (40) -> 100
    assert es.bullish == 100.0 and es.bearish == 0.0
    es2 = _liquidity_score("sellside", 0.0, "buyside", 3.0)
    assert es2.bearish == 100.0 and es2.bullish == 0.0


def test_ote_and_amd_scores():
    assert _ote_score("bullish", "tested").bullish == 100.0
    assert _ote_score("bullish", "fresh").bullish == 50.0
    assert _ote_score("bearish", "invalidated").bearish == 0.0
    assert _amd_score("distribution", "bullish").bullish == 100.0
    assert _amd_score("manipulation", "bearish").bearish == 60.0
    assert _amd_score("accumulation", "neutral").bullish == 0.0


# ---- orchestrator --------------------------------------------------------

def _all_bullish_analysis():
    return NS(
        bias=[NS(trend="bullish")],
        entry_structure=NS(trend="bullish"),
        nearest_zones=[NS(source="fvg", kind="bullish", distance_atr=0.0),
                       NS(source="orderblock", kind="bullish", distance_atr=0.0)],
        liquidity_pools=[NS(side="buyside", distance_atr=0.0)],
        recent_sweeps=[NS(side="sellside")],
        active_session="ny_kz",
    )


def test_build_scores_all_aligned():
    a = _all_bullish_analysis()
    scores = build_scores(a, ote_zones=[NS(direction="bullish", state="tested")],
                          amd=NS(phase="distribution", direction="bullish"))
    assert scores.bullish.strength == 100.0
    assert scores.bearish.strength == 0.0
    assert scores.session_favorability == 100.0   # ny_kz kill zone
    # every element traces into the bullish components
    assert set(scores.bullish.components) == {
        "structure", "liquidity", "orderblock", "fvg", "ote", "amd"}


def test_build_scores_bounded_and_deterministic():
    a = _all_bullish_analysis()
    s1 = build_scores(a)              # no OTE/AMD supplied -> those score 0
    s2 = build_scores(a)
    assert s1 == s2                   # determinism
    for es in s1.elements:
        assert 0.0 <= es.bullish <= 100.0 and 0.0 <= es.bearish <= 100.0
    assert 0.0 <= s1.bullish.strength <= 100.0
    assert 0.0 <= s1.bearish.strength <= 100.0


def test_weights_are_a_knob():
    # structure bullish, AMD bearish, everything else neutral
    a = NS(bias=[], entry_structure=NS(trend="bullish"),
           nearest_zones=[], liquidity_pools=[], recent_sweeps=[],
           active_session="off_hours")
    amd = NS(phase="distribution", direction="bearish")
    heavy_structure = build_scores(a, amd=amd,
                                   weights={"structure": 1.0, "amd": 0.0})
    heavy_amd = build_scores(a, amd=amd,
                             weights={"structure": 0.0, "amd": 1.0})
    assert heavy_structure.bullish.strength == 100.0
    assert heavy_structure.bearish.strength == 0.0
    assert heavy_amd.bearish.strength == 100.0
    assert heavy_amd.bullish.strength == 0.0


def test_no_verdict_field():
    scores = build_scores(_all_bullish_analysis())
    for forbidden in ("verdict", "recommendation", "winner", "signal",
                      "decision", "action", "probability", "win_rate"):
        assert not hasattr(scores, forbidden)
    assert isinstance(scores, ICTScores)
