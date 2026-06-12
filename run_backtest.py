"""Milestone 2: backtest the strategy and produce an honest report.

Usage:
  python run_backtest.py                       # synthetic data (offline)
  python run_backtest.py --live EURUSD=X 15m   # real data
"""
from __future__ import annotations

import sys

from viz.report import build_report

from core.backtest.engine import Backtester, BTConfig, trades_to_frame
from core.data.provider import CachedProvider, SyntheticProvider, YFinanceProvider
from core.strategies.base import FVGRetestStrategy


def main() -> None:
    live = "--live" in sys.argv
    if live:
        i = sys.argv.index("--live")
        symbol = sys.argv[i + 1] if len(sys.argv) > i + 1 else "EURUSD=X"
        interval = sys.argv[i + 2] if len(sys.argv) > i + 2 else "15m"
        provider = CachedProvider(YFinanceProvider())
        spread = 0.00012  # ~1.2 pip round EURUSD retail spread; tune per market
    else:
        symbol, interval = "EURUSD (synthetic)", "15m"
        provider = SyntheticProvider(seed=7)
        spread = 0.00012

    df = provider.fetch(symbol, interval, bars=600)
    signals = FVGRetestStrategy().generate(symbol, df)
    cfg = BTConfig(initial_equity=10_000, risk_pct=1.0,
                   spread=spread, slippage=0.00003)
    result = Backtester().run(df, signals, cfg)

    print(f"Backtest: {symbol} {interval} | {len(df)} candles | "
          f"{len(signals)} signals\n")
    print("WITH realistic costs (spread + slippage):")
    for k, v in result.metrics.items():
        print(f"  {k:18s} {v}")

    # Same run with zero costs: shows how much costs eat the edge
    free = Backtester().run(df, signals, BTConfig(initial_equity=10_000, risk_pct=1.0))
    print("\nWITH zero costs (fantasy mode, for comparison):")
    for k in ("trades", "win_rate_pct", "profit_factor", "net_return_pct"):
        print(f"  {k:18s} {free.metrics.get(k)}")

    fig = build_report(result, f"{symbol} {interval} — FVGRetestStrategy backtest")
    fig.write_html("backtest_report.html", include_plotlyjs="cdn")
    if result.trades:
        trades_to_frame(result.trades).to_csv("trades.csv", index=False)
    print("\nWrote backtest_report.html and trades.csv")


if __name__ == "__main__":
    main()
