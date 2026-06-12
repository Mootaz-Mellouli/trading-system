# Testing & running guide

## 1. Setup (once)

Requires Python 3.10+.

```bash
cd trading-system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest
```

## 2. Three ways to run

**A. Offline demo (synthetic data, works anywhere, no internet):**
```bash
python run_demo.py
```
Outputs `chart.html` (open in a browser) and `fvgs_detected.csv`.

**B. Live data (real market data from Yahoo Finance):**
```bash
python run_demo.py --live EURUSD=X 15m
python run_demo.py --live GBPUSD=X 5m
python run_demo.py --live BTC-USD 1h
```
Forex symbols need the `=X` suffix; crypto uses `BTC-USD` style. Intraday
intervals are limited to ~60 days of history on Yahoo — fine for visual
validation, not enough for serious backtesting (paid feed comes later).
Data is cached in `data_cache/` (parquet) so re-runs are fast.

**C. Automated tests:**
```bash
python -m pytest tests/ -v
```

## 3. What the automated tests prove

| Test | Guarantees |
|---|---|
| `test_fvg_detected_with_exact_zone` | FVG found on a hand-built 3-candle gap, with exact top/bottom prices |
| `test_fvg_fill_tracking` | Mitigation = first touch of zone; filled = full traversal, exact candle |
| `test_fvg_displacement_filter` | The ATR threshold actually filters weak gaps |
| `test_swing_points_and_labels` | A known zigzag yields exactly HH / HL / LH labels on the right candles |
| `test_swing_confirmation_lag` | A swing is only usable `lookback` bars after it occurs |
| `test_structure_events_no_lookahead` | BOS fires on the right close, using only information available at that time |
| `test_strategy_is_deterministic` | Same candles in → identical signals out, every run |
| `test_synthetic_provider_contract` | Data contract: columns, UTC index, high ≥ body ≥ low |
| `test_cached_provider_roundtrip` | Parquet cache returns identical data |

The no-lookahead tests are the most important ones in the repo: a backtest
that accidentally peeks at the future will look profitable and lose money
live. Every new detector must ship with an equivalent test.

## 4. Manual validation scenario (the client session)

Goal: verify the code "sees" what the trader sees, BEFORE writing strategy
rules. Allow ~1 hour, screen-sharing.

1. Run live on his exact market and timeframe, e.g.
   `python run_demo.py --live EURUSD=X 15m`, open `chart.html`.
2. Put TradingView (his marked-up chart) side by side with `chart.html`
   on the same symbol/timeframe/date range.
3. For each FVG zone on our chart, ask: "Would you have marked this one?
   If not, why?" Record every disagreement — each one is a missing rule.
4. Same for swing labels (HH/HL/LH/LL) and BOS/CHoCH arrows: "Is this a
   break of structure for you? Do you use the close or the wick?"
5. Find 2–3 zones HE marked that our chart missed. Ask what makes them
   valid — usually reveals a threshold (zone too small, move too weak).
6. Tune the knobs live and re-run until the detections roughly match
   his eye:

| Knob | Where | Meaning | If client says... |
|---|---|---|---|
| `min_displacement_atr` | `detect_fvgs(...)` | How strong the impulse candle must be | "too many tiny gaps" → raise it |
| `lookback` | `detect_swings(...)` | Structure sensitivity | "structure too noisy" → raise it |
| `recent`, `rr`, `stop_buffer` | `FVGRetestStrategy(...)` | Confluence window, target, stop padding | per his risk rules |

7. Screenshot agreed examples AND rejected examples. Write the agreed
   numeric thresholds into a spec note he confirms (email is fine).

**Acceptance criteria for milestone 1:** on one full week of his market,
the client agrees with ≥ 80% of detections, and every disagreement is
explained by a written rule we can implement next.

## 5. Known limitations (deliberate, for now)

- The example `FVGRetestStrategy` is a scaffold, NOT his rules — no kill
  zones, no higher-timeframe bias, no liquidity sweeps yet. Expect it to
  be noisy.
- No backtester yet: signal markers on the chart say nothing about
  profitability. That's milestone 2, and no real money moves before it.
- Yahoo data has no real bid/ask spread; cost modeling comes with the
  backtester.

## 6. Troubleshooting

- `No data returned for ...` → wrong symbol format (forex needs `=X`),
  or interval/period limit hit; try `1h` or `1d` first.
- Stale data → delete the `data_cache/` folder.
- `ModuleNotFoundError: core` → run commands from the project root
  (the folder containing `run_demo.py`).
- Empty chart in browser → the HTML loads plotly.js from a CDN, so it
  needs internet the first time it's opened.
