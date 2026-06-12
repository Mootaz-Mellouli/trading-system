"""Backtest report figure: equity curve, drawdown, per-trade R."""
from __future__ import annotations

from plotly.subplots import make_subplots
import plotly.graph_objects as go


def build_report(result, title: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.06,
                        subplot_titles=("Equity curve", "Drawdown %", "Per-trade R multiple"))
    eq = result.equity_curve
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines",
                             line=dict(color="#534AB7", width=2), name="equity"),
                  row=1, col=1)
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines",
                             fill="tozeroy", line=dict(color="#D85A30", width=1),
                             name="drawdown"), row=2, col=1)
    if result.trades:
        xs = [t.exit_time for t in result.trades]
        rs = [t.r_multiple for t in result.trades]
        colors = ["#1D9E75" if r > 0 else "#D85A30" for r in rs]
        fig.add_trace(go.Bar(x=xs, y=rs, marker_color=colors, name="R"),
                      row=3, col=1)
    fig.update_layout(title=title, template="plotly_white", height=760,
                      showlegend=False, margin=dict(l=50, r=20, t=70, b=30))
    return fig
