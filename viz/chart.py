"""Interactive chart: the 'does my code see what the trader sees?' tool.

Renders candles + detected FVGs (shaded zones), swing labels (HH/HL/LH/LL),
BOS/CHoCH markers, and strategy signals. Output is a standalone HTML file
the trader can open, zoom and pan in any browser.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

GREEN = "rgba(29,158,117,0.18)"
GREEN_LINE = "rgba(29,158,117,0.6)"
RED = "rgba(216,90,48,0.18)"
RED_LINE = "rgba(216,90,48,0.6)"


def build_chart(df: pd.DataFrame, fvgs=None, swings=None, events=None,
                signals=None, title: str = "Chart") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="price",
        increasing_line_color="#1D9E75", decreasing_line_color="#D85A30",
    ))

    end_time = df.index[-1]
    for f in (fvgs or []):
        until = f.filled_at or end_time
        fill = GREEN if f.kind == "bullish" else RED
        line = GREEN_LINE if f.kind == "bullish" else RED_LINE
        fig.add_shape(type="rect", x0=f.created_at, x1=until,
                      y0=f.bottom, y1=f.top, fillcolor=fill,
                      line=dict(color=line, width=1), layer="below")

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
