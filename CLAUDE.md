# CLAUDE.md

Semi-automated ICT/SMC trading system for a client. Two heads on one shared
core: (1) a signal bot where a human confirms trades, (2) a read-only
analysis dashboard. The core does all the thinking; heads only consume it.

## Commands

```bash
python run_demo.py                        # detectors + chart, synthetic data
python run_demo.py --live EURUSD=X 15m    # real Yahoo Finance data
python run_backtest.py [--live SYM TF]    # backtest -> backtest_report.html
streamlit run dashboard/app.py            # analysis dashboard (port 8501)
python -m pytest tests/ -v                # full test suite (keep at 100%)
```

Yahoo symbols: forex needs `=X` (EURUSD=X), crypto is `BTC-USD` / `BTC-EUR`.
Intraday data is capped at ~60 days. Candle cache: `data_cache/` (parquet).

## Architecture

```
core/data/provider.py      BaseProvider + YFinance/Synthetic/Cached.
                           Contract: DataFrame, UTC DatetimeIndex,
                           columns open/high/low/close/volume. Everything
                           downstream depends ONLY on this contract.
core/detectors/fvg.py      Fair value gaps + ATR + fill tracking.
core/detectors/swings.py   Swing points (with confirmed_at), HH/HL/LH/LL,
                           BOS/CHoCH on closes.
core/strategies/base.py    Signal dataclass + BaseStrategy + FVGRetestStrategy
                           (a SCAFFOLD — not the client's real rules yet).
core/backtest/engine.py    Pessimistic backtester (see rules below).
viz/chart.py, viz/report.py  Plotly figures shared by CLI and dashboard.
dashboard/app.py           Streamlit, READ-ONLY by design.
tests/                     Hand-crafted candles with known correct answers.
```

## Non-negotiable rules

1. **No lookahead, ever.** A pattern may only be used from the moment it was
   knowable: swings activate at `confirmed_at` (= swing bar + lookback), FVG
   signals are dated when ALL conditions aligned, backtest fills start on the
   candle AFTER signal time. Every new detector MUST ship with a
   no-lookahead test (see `test_structure_events_no_lookahead`). Three
   lookahead bugs were already caught in this codebase; assume the next
   change introduces one until a test proves otherwise.
2. **Determinism.** Same candles in -> identical signals out. No randomness,
   no wall-clock reads, no LLM calls anywhere in the signal path. LLMs are
   allowed only AFTER the core decided (e.g. phrasing alert text).
3. **Backtester stays pessimistic.** Candle touches both SL and TP -> count
   the LOSS. SL checked before TP on the fill bar. Full spread+slippage on
   entry. Unfilled limit orders expire. One position at a time. Never
   "improve" results by relaxing these.
4. **Every threshold is a knob.** No magic numbers inside logic — thresholds
   are constructor/function parameters (`min_displacement_atr`, `lookback`,
   `rr`...), surfaced in the dashboard sidebar. The client tunes them; we
   don't hardcode his rules.
5. **Signals are plain data** (`Signal` dataclass with unique `id`). The
   backtester, future Telegram bot, and dashboard all consume the same
   object. No head contains trading logic.
6. **Dashboard is read-only.** Never add order/execution buttons to it.
7. Money expectations live in the docs, not the code: this system executes
   the client's rules with discipline; it does not promise profit. Don't add
   copy that implies otherwise.

## Conventions

- Python 3.10+, type hints, dataclasses for domain objects.
- All timestamps UTC, tz-aware.
- Prices: floats rounded for display only, never rounded mid-computation.
- Tests use `make_df()` in `tests/test_detectors.py` to build hand-crafted
  candles; assert EXACT zones/prices/timestamps, not just counts.
- Dashboard smoke test uses `streamlit.testing.v1.AppTest` — keep it green.
- Plotly HTML exports use `include_plotlyjs="cdn"` to stay small.

## Roadmap (next milestones)

1. Telegram signal bot: scheduler -> run strategies -> alert with inline
   ✅ Execute / ❌ Skip buttons; signal expiry; idempotent by signal id;
   alert-only mode until execution exists. Log every signal/decision to DB.
2. Client's real strategy rules as a new `BaseStrategy` subclass (kill
   zones, HTF bias, liquidity sweeps) once his spec arrives — do NOT extend
   FVGRetestStrategy.
3. Broker/exchange execution module (last; paper/sandbox first).

## Known limitations (intentional for now)

- Yahoo data: no real bid/ask, ~60d intraday history; fine for validation,
  switch provider for serious backtesting.
- FVGRetestStrategy is noisy by design — confluence filters come with the
  client's spec.
- No persistence layer yet beyond the parquet candle cache.
