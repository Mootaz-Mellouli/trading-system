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
    BinanceProvider, CachedProvider, SyntheticProvider, YFinanceProvider)
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
    source = st.radio("Source", ["Synthetic (offline)", "Live (Binance)",
                                 "Live (Yahoo Finance)"])
    symbol_options = SYMBOLS_BINANCE if "Binance" in source else SYMBOLS_YAHOO
    symbol = st.selectbox("Symbol", symbol_options, index=0,
                          help="Crypto + or (PAXG) via Binance · "
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

    st.header("Backtest")
    equity0 = st.number_input("Starting equity", 1_000, 1_000_000, 10_000, step=1_000)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25)
    spread_pips = st.number_input("Spread (price units)", 0.0, 0.01, 0.00012,
                                  step=0.00001, format="%.5f")


def make_provider(source: str):
    if source.startswith("Synthetic"):
        return SyntheticProvider(seed=7)
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
                  eq_band: float):
    return build_ict_analysis(
        make_provider(source), symbol, interval, bias_tfs=bias_tfs, bars=bars,
        swing_lookback=lookback, min_displacement_atr=min_disp,
        ob_min_displacement_atr=ob_min_disp, tolerance_atr=tol_atr,
        max_pierce_atr=max_pierce, eq_tolerance_pct=eq_band,
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

# The ICT tab is a READ-ONLY CHECKLIST surface (non-negotiable rule 6):
# never display a buy/sell verdict, a confluence score, or any aggregated
# recommendation here. The decision is human.
with tab_ict:
    analysis = None
    try:
        analysis = load_analysis(source, symbol, interval, bars,
                                 tuple(bias_tfs), min_disp, lookback, rr,
                                 ob_min_disp, tol_atr, max_pierce, eq_band)
    except Exception as exc:
        st.error(f"Analyse ICT impossible : {exc}")

    if analysis is not None:
        # ------------------------------------------------ bias panel -------
        views = [*analysis.bias, analysis.entry_structure]
        for col, sv in zip(st.columns(len(views)), views):
            pos = (f" · {sv.range_position_pct:.0f} %"
                   if sv.range_position_pct is not None else "")
            col.metric(f"Biais {sv.timeframe}",
                       f"{ARROW[sv.trend]} {TREND_FR[sv.trend]}",
                       delta=f"{ZONE_FR.get(sv.range_zone, '—')}{pos}",
                       delta_color="off")

        col_chart, col_panels = st.columns([3, 2])

        with col_chart:
            spans = [(s0, s1, SESSION_FR.get(name, name))
                     for s0, s1, name in session_spans(df.index)]
            st.plotly_chart(
                build_chart(df, fvgs, swings, events, signals,
                            orderblocks=orderblocks, pools=pools,
                            sweeps=sweeps, session_spans=spans,
                            title=f"{symbol} {interval} — vue ICT"),
                width='stretch', key="chart_ict")
            st.caption("Zones ombrées = FVG · rectangles bordés = order blocks "
                       "· lignes pointillées = liquidité (épaisses = highs/lows "
                       "égaux) · X = sweeps · bandes verticales = sessions.")

        with col_panels:
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
