"""Milestone 1 demo: data -> detectors -> strategy -> chart.

Run with synthetic data (works offline):   python run_demo.py
Run with real Yahoo Finance data:          python run_demo.py --live EURUSD=X 15m
"""
from __future__ import annotations

import sys

from core.data.provider import CachedProvider, SyntheticProvider, YFinanceProvider
from core.detectors.fvg import detect_fvgs, fvgs_to_frame
from core.detectors.swings import detect_swings, detect_structure_events
from core.strategies.base import FVGRetestStrategy
from viz.chart import build_chart, save_chart


def main() -> None:
    live = "--live" in sys.argv
    if live:
        i = sys.argv.index("--live")
        symbol = sys.argv[i + 1] if len(sys.argv) > i + 1 else "EURUSD=X"
        interval = sys.argv[i + 2] if len(sys.argv) > i + 2 else "15m"
        provider = CachedProvider(YFinanceProvider())
    else:
        symbol, interval = "EURUSD (synthetic)", "15m"
        provider = SyntheticProvider(seed=7)

    df = provider.fetch(symbol, interval, bars=600)
    print(f"Loaded {len(df)} candles  {df.index[0]} -> {df.index[-1]}")

    fvgs = detect_fvgs(df, min_displacement_atr=0.8)
    swings = detect_swings(df, lookback=5)
    events = detect_structure_events(df, swings)
    signals = FVGRetestStrategy().generate(symbol, df)

    open_fvgs = [f for f in fvgs if f.filled_at is None]
    print(f"FVGs: {len(fvgs)} detected ({len(open_fvgs)} still open)")
    print(f"Swing points: {len(swings)} | structure events: "
          f"{sum(e.kind == 'BOS' for e in events)} BOS, "
          f"{sum(e.kind == 'CHoCH' for e in events)} CHoCH")
    print(f"Signals from FVGRetestStrategy: {len(signals)}")
    for s in signals[-5:]:
        print(f"  [{s.time}] {s.direction.upper():5s} @ {s.entry}  "
              f"SL {s.stop_loss}  TP {s.take_profit}  RR {s.risk_reward}  "
              f"id={s.id}")

    fig = build_chart(df, fvgs, swings, events, signals,
                      title=f"{symbol} {interval} — FVG + structure detection")
    save_chart(fig, "chart.html")
    fvgs_to_frame(fvgs).to_csv("fvgs_detected.csv", index=False)
    print("Wrote chart.html and fvgs_detected.csv")


if __name__ == "__main__":
    main()
