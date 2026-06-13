"""Analysis dashboard (read-only — no order buttons, by design).

Run from the project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import pandas as pd
import streamlit as st

from core.analysis.ict_report import build_ict_analysis
from core.analysis.range_session import DEFAULT_SESSIONS, session_bounds, session_spans
from core.backtest.engine import Backtester, BTConfig, trades_to_frame
from core.data.provider import (
    BinanceProvider, CachedProvider, CoinbaseProvider, KrakenProvider,
    SyntheticProvider, YFinanceProvider)
from core.detectors.fvg import detect_fvgs
from core.detectors.liquidity import detect_liquidity
from core.detectors.orderblocks import detect_orderblocks
from core.detectors.swings import detect_swings, detect_structure_events
from core.strategies.base import FVGRetestStrategy
from viz.chart import build_chart
from viz.report import build_report

st.set_page_config(page_title="Trading system", layout="wide")
st.title("Trading system — analysis dashboard")

# Curated symbols per source. The first entry is the default selection.
# Binance pairs (crypto + PAXG gold) for the serious crypto feed; Yahoo
# tickers (forex needs '=X', crypto BASE-QUOTE) for the validation source.
SYMBOLS_BINANCE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "BTCEUR", "ETHEUR",
    "PAXGUSDT",            # PAX Gold: 1 token = 1 troy oz of gold
]
# Coinbase (US-based -> works on Streamlit Cloud). Products use BASE-QUOTE.
SYMBOLS_COINBASE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "LINK-USD",
    "BTC-EUR", "ETH-EUR",
]
# Kraken (US-accessible). Pairs use Kraken naming (BTC = XBT); incl. PAXG gold.
SYMBOLS_KRAKEN = [
    "XBTUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD",
    "DOGEUSD", "AVAXUSD", "LINKUSD",
    "XBTEUR", "ETHEUR",
    "PAXGUSD",            # PAX Gold: 1 token = 1 troy oz of gold
]
SYMBOLS_YAHOO = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "BTC-USD", "ETH-USD", "BTC-EUR", "ETH-EUR",
    "GC=F", "SI=F", "CL=F",            # or / argent / pétrole (futures)
    "^GSPC", "^NDX", "^DJI",            # S&P 500 / Nasdaq 100 / Dow Jones
]

# French display labels (the client is francophone). Core stays in English.
TREND_FR = {"bullish": "Haussier", "bearish": "Baissier", "neutral": "Neutre"}
READING_FR = TREND_FR
ARROW = {"bullish": "↗", "bearish": "↘", "neutral": "→"}
ZONE_FR = {"premium": "Premium", "discount": "Discount",
           "equilibrium": "Équilibre", None: "—"}
ETAT_FR = {"open": "Ouvert", "mitigated": "Mitigé", "fresh": "Frais"}
SESSION_FR = {"asia": "Asie", "london_kz": "KZ Londres", "ny_kz": "KZ New York",
              "london_close": "Clôture Londres", "off_hours": "Hors session"}
AMD_FR = {"accumulation": "Accumulation", "manipulation": "Manipulation",
          "distribution": "Distribution", "undefined": "Indéfini"}


def fr_element(name: str) -> str:
    if name.startswith("structure_"):
        return f"Structure {name.removeprefix('structure_')}"
    return {"dealing_range_position": "Position dans le range",
            "nearest_fvg": "FVG le plus proche",
            "nearest_orderblock": "Order block le plus proche",
            "liquidity_draw": "Liquidité (aimant)",
            "last_sweep": "Dernier sweep",
            "session": "Session"}.get(name, name)


def _reading_color(value: str) -> str:
    return {"Haussier": "color: #1D9E75; font-weight: 600",
            "Baissier": "color: #D85A30; font-weight: 600"}.get(
        value, "color: #888780")

# ----------------------------------------------------------- sidebar -------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Synthetic (offline)", "Live (Coinbase)",
                                 "Live (Kraken)", "Live (Binance)",
                                 "Live (Yahoo Finance)"],
                      help="Coinbase/Kraken marchent sur Streamlit Cloud ; "
                           "Binance est bloqué depuis les serveurs US.")
    _symbols_by_source = {"Coinbase": SYMBOLS_COINBASE, "Kraken": SYMBOLS_KRAKEN,
                          "Binance": SYMBOLS_BINANCE}
    symbol_options = next((lst for key, lst in _symbols_by_source.items()
                           if key in source), SYMBOLS_YAHOO)
    symbol = st.selectbox("Symbol", symbol_options, index=0,
                          help="Crypto + or (PAXG) via Binance/Kraken · "
                               "forex/indices via Yahoo. Liste selon la source.")
    interval = st.selectbox("Timeframe", ["5m", "15m", "1h", "1d"], index=1)
    bars = st.slider("Candles", 200, 1500, 600, step=100)

    st.header("Strategy knobs")
    min_disp = st.slider("Min displacement (× ATR)", 0.0, 3.0, 0.8, 0.1,
                         help="How strong the impulse candle must be for an FVG to count")
    lookback = st.slider("Swing lookback", 2, 15, 5,
                         help="Higher = fewer, more significant swings")
    rr = st.slider("Risk:reward target", 1.0, 5.0, 2.0, 0.5)

    st.header("Analyse ICT")
    bias_tfs = st.multiselect("TF de biais", ["5m", "15m", "1h", "1d"],
                              default=["1d", "1h"],
                              help="Timeframes hauts pour le biais directionnel")
    ob_min_disp = st.slider("OB : déplacement min (× ATR)", 0.0, 3.0, 1.0, 0.1)
    tol_atr = st.slider("Highs/lows égaux : tolérance (× ATR)",
                        0.0, 0.5, 0.1, 0.05)
    max_pierce = st.slider("Sweep : mèche max (× ATR)", 0.1, 2.0, 0.5, 0.1)
    eq_band = st.slider("Bande équilibre (± % du range)", 0.0, 15.0, 5.0, 1.0)

    st.header("Score & setups")
    with st.expander("Poids du score (confluence)"):
        w_structure = st.slider("Structure", 0.0, 1.0, 0.25, 0.05)
        w_liquidity = st.slider("Liquidité", 0.0, 1.0, 0.20, 0.05)
        w_orderblock = st.slider("Order block", 0.0, 1.0, 0.15, 0.05)
        w_fvg = st.slider("FVG", 0.0, 1.0, 0.10, 0.05)
        w_ote = st.slider("OTE", 0.0, 1.0, 0.15, 0.05)
        w_amd = st.slider("AMD", 0.0, 1.0, 0.15, 0.05)
    setup_min_strength = st.slider("Setup : force min", 0.0, 100.0, 50.0, 5.0,
                                   help="Force de confluence minimale pour proposer un setup")
    setup_margin = st.slider("Setup : marge directionnelle", 0.0, 50.0, 10.0, 5.0,
                             help="Écart minimal entre les deux directions")

    st.header("Backtest")
    equity0 = st.number_input("Starting equity", 1_000, 1_000_000, 10_000, step=1_000)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25)
    spread_pips = st.number_input("Spread (price units)", 0.0, 0.01, 0.00012,
                                  step=0.00001, format="%.5f")


def make_provider(source: str):
    if source.startswith("Synthetic"):
        return SyntheticProvider(seed=7)
    if "Coinbase" in source:
        return CachedProvider(CoinbaseProvider())
    if "Kraken" in source:
        return CachedProvider(KrakenProvider())
    if "Binance" in source:
        return CachedProvider(BinanceProvider())
    return CachedProvider(YFinanceProvider())


@st.cache_data(ttl=300, show_spinner="Loading candles…")
def load_data(source: str, symbol: str, interval: str, bars: int) -> pd.DataFrame:
    return make_provider(source).fetch(symbol, interval, bars)


@st.cache_data(ttl=300, show_spinner="Analyse ICT…")
def load_analysis(source: str, symbol: str, interval: str, bars: int,
                  bias_tfs: tuple, min_disp: float, lookback: int, rr: float,
                  ob_min_disp: float, tol_atr: float, max_pierce: float,
                  eq_band: float, weights: tuple, setup_min_strength: float,
                  setup_margin: float):
    score_weights = dict(zip(
        ("structure", "liquidity", "orderblock", "fvg", "ote", "amd"), weights))
    return build_ict_analysis(
        make_provider(source), symbol, interval, bias_tfs=bias_tfs, bars=bars,
        swing_lookback=lookback, min_displacement_atr=min_disp,
        ob_min_displacement_atr=ob_min_disp, tolerance_atr=tol_atr,
        max_pierce_atr=max_pierce, eq_tolerance_pct=eq_band,
        score_weights=score_weights, setup_min_strength=setup_min_strength,
        setup_margin=setup_margin,
        strategies=[FVGRetestStrategy(min_displacement_atr=min_disp,
                                      swing_lookback=lookback, rr=rr)])


try:
    df = load_data(source, symbol, interval, bars)
except Exception as exc:  # bad symbol, no network, Yahoo limits...
    st.error(f"Could not load data: {exc}")
    st.info("Check the symbol format (forex needs '=X', e.g. EURUSD=X) "
            "or switch to the synthetic source.")
    st.stop()

# ------------------------------------------------------- computation -------
strategy = FVGRetestStrategy(min_displacement_atr=min_disp,
                             swing_lookback=lookback, rr=rr)
fvgs = detect_fvgs(df, min_disp)
swings = detect_swings(df, lookback)
events = detect_structure_events(df, swings)
signals = strategy.generate(symbol, df)
orderblocks = detect_orderblocks(df, ob_min_disp)
pools, sweeps = detect_liquidity(df, swings, tolerance_atr=tol_atr,
                                 max_pierce_atr=max_pierce)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Candles", len(df))
c2.metric("FVGs detected", len(fvgs))
c3.metric("Structure events", len(events))
c4.metric("Signals", len(signals))

tab_ict, tab_chart, tab_bt, tab_signals = st.tabs(
    ["🧭 Analyse ICT", "📈 Chart", "🧪 Backtest", "📋 Signals"])

# The ICT tab is DISPLAY-ONLY (rule 6, rewritten): it may show deterministic
# scores and a trade-setup suggestion, but NEVER an execute/sizing button and
# NEVER a single collapsed buy/sell verdict. Both score directions stay
# visible; the setup is explicitly a suggestion. The decision is human.
with tab_ict:
    analysis = None
    try:
        analysis = load_analysis(source, symbol, interval, bars,
                                 tuple(bias_tfs), min_disp, lookback, rr,
                                 ob_min_disp, tol_atr, max_pierce, eq_band,
                                 (w_structure, w_liquidity, w_orderblock,
                                  w_fvg, w_ote, w_amd),
                                 setup_min_strength, setup_margin)
    except Exception as exc:
        st.error(f"Analyse ICT impossible : {exc}")

    if analysis is not None:
        # ------------------------------------------------ bias panel -------
        views = [*analysis.bias, analysis.entry_structure]
        amd = analysis.amd
        cards = st.columns(len(views) + 1)
        for col, sv in zip(cards, views):
            pos = (f" · {sv.range_position_pct:.0f} %"
                   if sv.range_position_pct is not None else "")
            col.metric(f"Biais {sv.timeframe}",
                       f"{ARROW[sv.trend]} {TREND_FR[sv.trend]}",
                       delta=f"{ZONE_FR.get(sv.range_zone, '—')}{pos}",
                       delta_color="off")
        amd_dir = (f"{ARROW[amd.direction]} {TREND_FR[amd.direction]}"
                   if amd and amd.direction != "neutral" else "—")
        cards[-1].metric("Phase AMD",
                         AMD_FR.get(amd.phase, "—") if amd else "—",
                         delta=amd_dir, delta_color="off")

        col_chart, col_panels = st.columns([3, 2])

        with col_chart:
            spans = [(s0, s1, SESSION_FR.get(name, name))
                     for s0, s1, name in session_spans(df.index)]
            st.plotly_chart(
                build_chart(df, fvgs, swings, events, signals,
                            orderblocks=orderblocks, pools=pools,
                            sweeps=sweeps, session_spans=spans,
                            ote=analysis.ote_zones,
                            title=f"{symbol} {interval} — vue ICT"),
                width='stretch', key="chart_ict")
            st.caption("Zones ombrées = FVG · rectangles bordés = order blocks "
                       "· lignes pointillées = liquidité (épaisses = highs/lows "
                       "égaux) · X = sweeps · rectangles violets = OTE · "
                       "bandes verticales = sessions.")

        with col_panels:
            sc = analysis.scores
            if sc is not None:
                st.subheader("Force de confluence")
                gb, gs = st.columns(2)
                gb.metric("Haussier", f"{sc.bullish.strength:.0f}/100")
                gs.metric("Baissier", f"{sc.bearish.strength:.0f}/100")
                st.progress(sc.bullish.strength / 100,
                            text=f"Haussier {sc.bullish.strength:.0f}")
                st.progress(sc.bearish.strength / 100,
                            text=f"Baissier {sc.bearish.strength:.0f}")
                st.dataframe(pd.DataFrame(
                    [{"Élément": fr_element(e.element), "Haussier": e.bullish,
                      "Baissier": e.bearish, "Détail": e.detail}
                     for e in sc.elements]),
                    width='stretch', hide_index=True)
                st.caption("Force de confluence (0–100), pondérée et traçable — "
                           "PAS une probabilité de gain. Les deux sens sont "
                           "montrés ; la décision reste humaine.")

            st.subheader("Setup (suggestion)")
            su = analysis.setup
            if su is not None:
                st.markdown(
                    f"**{ARROW[su.direction]} {TREND_FR[su.direction]}** "
                    f"· base : {su.basis}")
                st.dataframe(pd.DataFrame([
                    {"Niveau": "Zone d'entrée (haut)", "Prix": su.entry_zone_top},
                    {"Niveau": "Zone d'entrée (bas)", "Prix": su.entry_zone_bottom},
                    {"Niveau": "Entrée", "Prix": su.entry},
                    {"Niveau": "Stop loss", "Prix": su.stop_loss},
                    {"Niveau": "TP1", "Prix": su.tp1},
                    {"Niveau": "TP2", "Prix": su.tp2},
                    {"Niveau": "TP3", "Prix": su.tp3},
                ]).style.format({"Prix": "{:.5f}"}),
                    width='stretch', hide_index=True)
                st.caption("Suggestion déterministe à confirmer par un humain — "
                           "aucun ordre n'est passé depuis ce tableau de bord.")
            else:
                st.info("Aucun setup : confluence trop faible ou pas de zone "
                        "d'entrée nette.")

            st.subheader("Confluences")
            conf_df = pd.DataFrame(
                [{"Élément": fr_element(c.element),
                  "Lecture": READING_FR[c.reading],
                  "Niveau clé": c.key_level, "Détail": c.detail}
                 for c in analysis.confluence])
            st.dataframe(
                conf_df.style.map(_reading_color, subset=["Lecture"])
                .format({"Niveau clé": "{:.5f}"}, na_rep="—"),
                width='stretch', hide_index=True)
            st.caption("Liste de contrôle — aucun score, aucune "
                       "recommandation : la décision reste humaine.")

            st.subheader("Zones les plus proches")
            if analysis.nearest_zones:
                st.dataframe(pd.DataFrame(
                    [{"Type": "FVG" if z.source == "fvg" else "Order block",
                      "Sens": TREND_FR[z.kind], "Bas": z.bottom, "Haut": z.top,
                      "Distance (ATR)": z.distance_atr,
                      "État": ETAT_FR.get(z.state, z.state)}
                     for z in analysis.nearest_zones]),
                    width='stretch', hide_index=True)
            else:
                st.info("Aucune zone ouverte (FVG ou order block).")

            st.subheader("Carte de liquidité")
            if analysis.liquidity_pools:
                st.dataframe(pd.DataFrame(
                    [{"Côté": "Buyside" if p.side == "buyside" else "Sellside",
                      "Niveau": p.price, "Égaux": "✓" if p.is_equal else "",
                      "Distance (ATR)": p.distance_atr}
                     for p in analysis.liquidity_pools]),
                    width='stretch', hide_index=True)
            else:
                st.info("Aucun pool de liquidité non pris.")
            if analysis.recent_sweeps:
                st.caption("Derniers sweeps : " + " · ".join(
                    f"{s.side} @ {s.level:.5f} ({s.time:%d/%m %H:%M})"
                    for s in analysis.recent_sweeps))

            st.subheader("Dealing range")
            dr = analysis.dealing_range
            if dr is not None:
                st.progress(
                    min(max(dr.position_pct, 0.0), 100.0) / 100,
                    text=f"{dr.position_pct:.1f} % du range — {ZONE_FR[dr.zone]}")
                st.caption(f"Bas {dr.low:.5f} · Équilibre {dr.equilibrium:.5f} "
                           f"· Haut {dr.high:.5f}")
            else:
                st.info("Range non défini (pas encore de swings confirmés).")

            st.subheader("Horloge des sessions")
            st.metric("Session active",
                      SESSION_FR.get(analysis.active_session,
                                     analysis.active_session))
            sched = []
            for name in DEFAULT_SESSIONS:
                u0, u1 = session_bounds(name, analysis.generated_at)
                p0, p1 = session_bounds(name, analysis.generated_at,
                                        tz="Europe/Paris")
                sched.append({"Session": SESSION_FR.get(name, name),
                              "UTC": f"{u0:%H:%M}–{u1:%H:%M}",
                              "Paris": f"{p0:%H:%M}–{p1:%H:%M}"})
            st.dataframe(pd.DataFrame(sched), width='stretch', hide_index=True)
            st.caption(f"Heure de référence : dernière bougie "
                       f"({analysis.generated_at:%d/%m/%Y %H:%M} UTC) — "
                       f"jamais l'horloge système.")

with tab_chart:
    st.plotly_chart(
        build_chart(df, fvgs, swings, events, signals,
                    title=f"{symbol} {interval}"),
        width='stretch')
    st.caption("Shaded zones = FVGs (green bullish / red bearish, ending where "
               "filled) · HH/HL/LH/LL = swing labels · arrows = BOS/CHoCH · "
               "triangles = strategy signals (hover for entry/SL/TP).")

with tab_bt:
    cfg = BTConfig(initial_equity=float(equity0), risk_pct=risk_pct,
                   spread=spread_pips, slippage=0.00003)
    result = Backtester().run(df, signals, cfg)
    m = result.metrics
    if m.get("trades", 0) == 0:
        st.warning("No trades were filled with the current settings.")
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Trades", m["trades"])
        k2.metric("Win rate", f"{m['win_rate_pct']}%")
        k3.metric("Profit factor", m["profit_factor"])
        k4.metric("Net return", f"{m['net_return_pct']}%")
        k5.metric("Max drawdown", f"{m['max_drawdown_pct']}%")
        st.plotly_chart(build_report(result, "Backtest"), width='stretch')
        with st.expander("All trades"):
            st.dataframe(trades_to_frame(result.trades), width='stretch')
        st.caption("Pessimistic engine: limit fills only after signal time, full "
                   "spread+slippage on entry, ambiguous candles count as losses, "
                   "one position at a time.")
    st.warning("These results describe the PAST. They are a filter against bad "
               "strategies, never a promise about the future.", icon="⚠️")

with tab_signals:
    if signals:
        st.dataframe(pd.DataFrame(
            [{"time": s.time, "direction": s.direction, "entry": s.entry,
              "stop_loss": s.stop_loss, "take_profit": s.take_profit,
              "RR": s.risk_reward, "reason": s.reason, "id": s.id}
             for s in signals]).sort_values("time", ascending=False),
            width='stretch')
    else:
        st.info("No signals with the current settings.")
