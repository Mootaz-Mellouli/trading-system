"""Tests for build_ict_analysis() on synthetic data: object shape, allowed
readings, determinism, serializability — and the absence of any verdict."""
import json

import pytest

from core.analysis.ict_report import build_ict_analysis
from core.analysis.range_session import DEFAULT_SESSIONS, OFF_HOURS
from core.data.provider import BaseProvider, SyntheticProvider

ALLOWED_READINGS = {"bullish", "bearish", "neutral"}


class FrozenProvider(BaseProvider):
    """Fetch synthetic candles once and replay them: SyntheticProvider
    stamps its index with the current time, so two raw fetches are not
    guaranteed identical across a clock tick."""

    def __init__(self, seed: int = 7):
        self._inner = SyntheticProvider(seed=seed)
        self._cache: dict = {}

    def fetch(self, symbol, interval, bars):
        key = (symbol, interval, bars)
        if key not in self._cache:
            self._cache[key] = self._inner.fetch(symbol, interval, bars)
        return self._cache[key]


@pytest.fixture(scope="module")
def analysis():
    return build_ict_analysis(FrozenProvider(), "TEST", "15m",
                              bias_tfs=("1h",))


def test_ict_analysis_shape(analysis):
    a = analysis
    assert a.symbol == "TEST"
    assert a.entry_timeframe == "15m"
    assert a.bias_timeframes == ["1h"]
    assert [b.timeframe for b in a.bias] == ["1h"]
    assert a.entry_structure.timeframe == "15m"
    assert a.price > 0
    assert a.generated_at.tzinfo is not None
    assert a.active_session in set(DEFAULT_SESSIONS) | {OFF_HOURS}
    assert a.dealing_range is None or a.dealing_range.zone in (
        "premium", "discount", "equilibrium")
    # every confluence element appears exactly once
    elements = [c.element for c in a.confluence]
    assert len(elements) == len(set(elements))
    assert {"structure_1h", "structure_15m", "dealing_range_position",
            "nearest_fvg", "nearest_orderblock", "liquidity_draw",
            "last_sweep", "session"} == set(elements)


def test_ict_analysis_readings_are_only_the_three_allowed(analysis):
    for sv in [*analysis.bias, analysis.entry_structure]:
        assert sv.trend in ALLOWED_READINGS
    for c in analysis.confluence:
        assert c.reading in ALLOWED_READINGS
        assert isinstance(c.element, str) and c.element
        assert isinstance(c.detail, str) and c.detail
        assert c.key_level is None or isinstance(c.key_level, float)


def test_ict_analysis_zones_and_pools_sorted_by_distance(analysis):
    a = analysis
    dists = [z.distance_atr for z in a.nearest_zones]
    assert dists == sorted(dists)
    for z in a.nearest_zones:
        assert z.source in ("fvg", "orderblock")
        assert z.kind in ("bullish", "bearish")
        assert z.top >= z.bottom
        assert z.distance_atr >= 0
        assert z.state == "fresh" if z.source == "orderblock" \
            else z.state in ("open", "mitigated")
    pool_dists = [p.distance_atr for p in a.liquidity_pools]
    assert pool_dists == sorted(pool_dists)
    for p in a.liquidity_pools:
        assert p.side in ("buyside", "sellside")
        assert isinstance(p.is_equal, bool)


def test_ict_analysis_is_deterministic():
    provider = FrozenProvider()
    a = build_ict_analysis(provider, "TEST", "15m", bias_tfs=("1h",))
    b = build_ict_analysis(provider, "TEST", "15m", bias_tfs=("1h",))
    assert a.confluence == b.confluence
    assert a.nearest_zones == b.nearest_zones
    assert a.liquidity_pools == b.liquidity_pools
    assert a.recent_sweeps == b.recent_sweeps
    assert a.entry_structure == b.entry_structure
    assert a.dealing_range == b.dealing_range
    # Signal ids are random by design; compare their content instead
    key = lambda sigs: [(s.time, s.direction, s.entry, s.stop_loss) for s in sigs]
    assert key(a.signals) == key(b.signals)


def test_ict_analysis_is_serializable(analysis):
    payload = json.dumps(analysis.to_dict())
    parsed = json.loads(payload)
    assert parsed["symbol"] == "TEST"
    assert isinstance(parsed["confluence"], list)
    assert isinstance(parsed["generated_at"], str)


def test_ict_analysis_scores_and_setup_shape(analysis):
    a = analysis
    # scores exist now (rule 6 rewritten) and expose BOTH directions
    assert a.scores is not None
    assert 0.0 <= a.scores.bullish.strength <= 100.0
    assert 0.0 <= a.scores.bearish.strength <= 100.0
    assert 0.0 <= a.scores.session_favorability <= 100.0
    for es in a.scores.elements:
        assert 0.0 <= es.bullish <= 100.0 and 0.0 <= es.bearish <= 100.0
    # detectors attached
    for z in a.ote_zones:
        assert z.direction in ("bullish", "bearish")
    assert a.amd is None or a.amd.phase in (
        "accumulation", "manipulation", "distribution", "undefined")
    # setup is optional; when present it is a complete, directional suggestion
    if a.setup is not None:
        assert a.setup.direction in ("bullish", "bearish")
        assert a.setup.tp1 != a.setup.tp2 != a.setup.tp3
        assert a.setup.to_signal().direction in ("long", "short")


def test_ict_analysis_has_no_collapsed_verdict(analysis):
    """Rule 6 (rewritten): scores and a display-only setup ARE allowed, but
    nothing may collapse them into a single buy/sell verdict, and scores are
    never framed as probability. Both directions stay visible."""
    for forbidden in ("verdict", "recommendation", "decision", "action",
                      "winner", "probability", "win_rate"):
        assert not hasattr(analysis, forbidden)
        assert not hasattr(analysis.scores, forbidden)
    # both sides are always present (no single direction field)
    assert analysis.scores.bullish is not None
    assert analysis.scores.bearish is not None
    # the checklist stays a checklist: no per-item score/weight
    assert not hasattr(analysis.confluence[0], "score")
    assert not hasattr(analysis.confluence[0], "weight")
