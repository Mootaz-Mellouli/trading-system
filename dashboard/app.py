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

from core.backtest.engine import Backtester, BTConfig, trades_to_frame
from core.data.provider import CachedProvider, SyntheticProvider, YFinanceProvider
from core.detectors.fvg import detect_fvgs
from core.detectors.swings import detect_swings, detect_structure_events
from core.strategies.base import FVGRetestStrategy
from viz.chart import build_chart
from viz.report import build_report

st.set_page_config(page_title="Trading system", layout="wide")
st.title("Trading system — analysis dashboard")

# ----------------------------------------------------------- sidebar -------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Synthetic (offline)", "Live (Yahoo Finance)"])
    symbol = st.text_input("Symbol", "EURUSD=X",
                           help="Forex: EURUSD=X · Crypto: BTC-EUR, BTC-USD")
    interval = st.selectbox("Timeframe", ["5m", "15m", "1h", "1d"], index=1)
    bars = st.slider("Candles", 200, 1500, 600, step=100)

    st.header("Strategy knobs")
    min_disp = st.slider("Min displacement (× ATR)", 0.0, 3.0, 0.8, 0.1,
                         help="How strong the impulse candle must be for an FVG to count")
    lookback = st.slider("Swing lookback", 2, 15, 5,
                         help="Higher = fewer, more significant swings")
    rr = st.slider("Risk:reward target", 1.0, 5.0, 2.0, 0.5)

    st.header("Backtest")
    equity0 = st.number_input("Starting equity", 1_000, 1_000_000, 10_000, step=1_000)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25)
    spread_pips = st.number_input("Spread (price units)", 0.0, 0.01, 0.00012,
                                  step=0.00001, format="%.5f")


@st.cache_data(ttl=300, show_spinner="Loading candles…")
def load_data(source: str, symbol: str, interval: str, bars: int) -> pd.DataFrame:
    if source.startswith("Synthetic"):
        return SyntheticProvider(seed=7).fetch(symbol, interval, bars)
    return CachedProvider(YFinanceProvider()).fetch(symbol, interval, bars)


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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Candles", len(df))
c2.metric("FVGs detected", len(fvgs))
c3.metric("Structure events", len(events))
c4.metric("Signals", len(signals))

tab_chart, tab_bt, tab_signals = st.tabs(["📈 Chart", "🧪 Backtest", "📋 Signals"])

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
