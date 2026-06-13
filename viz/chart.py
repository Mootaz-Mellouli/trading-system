"""Interactive chart: the 'does my code see what the trader sees?' tool.

Renders candles + detected FVGs (shaded zones), order blocks (bordered
rectangles), liquidity pools (dashed level lines, thicker for equal
highs/lows) and sweeps (X markers), session windows (shaded vertical
bands), swing labels (HH/HL/LH/LL), BOS/CHoCH markers, and strategy
signals. Output is a standalone HTML file the trader
can open, zoom and pan in any browser.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

GREEN = "rgba(29,158,117,0.18)"
GREEN_LINE = "rgba(29,158,117,0.6)"
RED = "rgba(216,90,48,0.18)"
RED_LINE = "rgba(216,90,48,0.6)"
OB_GREEN = "rgba(29,158,117,0.07)"
OB_GREEN_LINE = "rgba(29,158,117,0.9)"
OB_RED = "rgba(216,90,48,0.07)"
OB_RED_LINE = "rgba(216,90,48,0.9)"
LIQ_BUY_LINE = "rgba(29,158,117,0.85)"
LIQ_SELL_LINE = "rgba(216,90,48,0.85)"


OTE_LINE = "rgba(124,93,191,0.9)"
OTE_FILL = "rgba(124,93,191,0.10)"


def build_chart(df: pd.DataFrame, fvgs=None, swings=None, events=None,
                signals=None, orderblocks=None, pools=None, sweeps=None,
                session_spans=None, ote=None, title: str = "Chart") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="price",
        increasing_line_color="#1D9E75", decreasing_line_color="#D85A30",
    ))

    end_time = df.index[-1]

    for s0, s1, label in (session_spans or []):  # (start, end, name) windows
        fig.add_vrect(x0=s0, x1=s1, fillcolor="rgba(98,114,164,0.06)",
                      line_width=0, layer="below",
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=9, color="#9b9a93"))
    for f in (fvgs or []):
        until = f.filled_at or end_time
        fill = GREEN if f.kind == "bullish" else RED
        line = GREEN_LINE if f.kind == "bullish" else RED_LINE
        fig.add_shape(type="rect", x0=f.created_at, x1=until,
                      y0=f.bottom, y1=f.top, fillcolor=fill,
                      line=dict(color=line, width=1), layer="below")

    for ob in (orderblocks or []):
        until = ob.invalidated_at or end_time
        fill = OB_GREEN if ob.kind == "bullish" else OB_RED
        line = OB_GREEN_LINE if ob.kind == "bullish" else OB_RED_LINE
        fig.add_shape(type="rect", x0=ob.created_at, x1=until,
                      y0=ob.bottom, y1=ob.top, fillcolor=fill,
                      line=dict(color=line, width=2,
                                dash="solid" if ob.state == "fresh" else "dot"),
                      layer="below")

    for p in (pools or []):  # only untaken pools are still resting liquidity
        if p.taken_at is not None:
            continue
        color = LIQ_BUY_LINE if p.side == "buyside" else LIQ_SELL_LINE
        fig.add_shape(type="line", x0=p.created_at, x1=end_time,
                      y0=p.price, y1=p.price,
                      line=dict(color=color, dash="dash",
                                width=2.5 if p.is_equal else 1.2))

    for z in (ote or []):  # optimal trade entry bands (62-79% retracement)
        if z.invalidated_at is not None:
            continue
        fig.add_shape(type="rect", x0=z.created_at, x1=end_time,
                      y0=z.bottom, y1=z.top, fillcolor=OTE_FILL,
                      line=dict(color=OTE_LINE, width=1, dash="dash"),
                      layer="below")
        fig.add_annotation(x=z.created_at, y=z.top, text="OTE", showarrow=False,
                           xanchor="left", yshift=7,
                           font=dict(size=9, color="#7C5DBF"))

    for sw in (sweeps or []):
        color = "#1D9E75" if sw.side == "buyside" else "#D85A30"
        fig.add_trace(go.Scatter(
            x=[sw.time], y=[sw.level], mode="markers",
            marker=dict(symbol="x", size=11, color=color),
            name=f"{sw.side} sweep",
            hovertext=f"{sw.side} liquidity sweep @ {sw.level} "
                      f"(pierce {sw.pierce_atr} ATR)",
            hoverinfo="text", showlegend=False,
        ))

    for s in (swings or []):
        if not s.label:
            continue
        fig.add_annotation(
            x=s.time, y=s.price, text=s.label, showarrow=False,
            yshift=14 if s.kind == "high" else -14,
            font=dict(size=10, color="#888780"),
        )

    for e in (events or []):
        color = "#1D9E75" if e.direction == "bullish" else "#D85A30"
        fig.add_annotation(
            x=e.time, y=e.price, text=e.kind, showarrow=True,
            arrowhead=2, arrowsize=0.8, arrowcolor=color, ay=-28 if e.direction == "bullish" else 28,
            font=dict(size=10, color=color),
        )

    for sig in (signals or []):
        color = "#1D9E75" if sig.direction == "long" else "#D85A30"
        symbol = "triangle-up" if sig.direction == "long" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[sig.time], y=[sig.entry], mode="markers",
            marker=dict(symbol=symbol, size=13, color=color,
                        line=dict(width=1, color="white")),
            name=f"{sig.direction} signal",
            hovertext=f"{sig.reason}<br>entry {sig.entry} | SL {sig.stop_loss} "
                      f"| TP {sig.take_profit} | RR {sig.risk_reward}",
            hoverinfo="text", showlegend=False,
        ))

    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_rangeslider_visible=False, height=620,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h"),
    )
    return fig


def save_chart(fig: go.Figure, path: str) -> None:
    fig.write_html(path, include_plotlyjs="cdn")
