"""Backtest engine: replays Signals over historical candles.

Design principles (all deliberately PESSIMISTIC — a backtest should
under-promise):
  - A signal becomes a LIMIT order at signal.entry, working from the first
    candle AFTER signal.time. Never filled on information from the past.
  - The order expires unfilled after `entry_expiry_bars`.
  - If a candle touches both SL and TP, the trade counts as a LOSS
    (intrabar order of events is unknowable from OHLC).
  - On the fill candle itself, SL is checked before TP.
  - Costs: full spread + slippage charged on entry price.
  - One position at a time (a human confirms trades one by one).
  - Position size: fixed % of CURRENT equity risked per trade
    (units = risk_amount / stop_distance), compounding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.strategies.base import Signal


@dataclass
class BTConfig:
    initial_equity: float = 10_000.0
    risk_pct: float = 1.0          # % of equity risked per trade
    spread: float = 0.0            # full spread, in price units (e.g. 0.00012 EURUSD)
    slippage: float = 0.0          # extra adverse fill, in price units
    entry_expiry_bars: int = 20    # cancel unfilled limit order after N bars


@dataclass
class Trade:
    signal_id: str
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float            # actual fill incl. costs
    exit_time: pd.Timestamp
    exit_price: float
    outcome: str                  # 'tp' | 'sl' | 'end_of_data'
    units: float
    pnl: float
    r_multiple: float


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series       # equity after each closed trade
    skipped_busy: int             # signals ignored because a position was open
    expired: int                  # limit orders never filled
    config: BTConfig

    @property
    def metrics(self) -> dict:
        t = self.trades
        if not t:
            return {"trades": 0}
        pnls = np.array([x.pnl for x in t])
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        gross_win, gross_loss = wins.sum(), -losses.sum()
        eq = self.equity_curve
        peak = eq.cummax()
        max_dd = ((eq - peak) / peak).min() * 100
        return {
            "trades": len(t),
            "win_rate_pct": round(100 * len(wins) / len(t), 1),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
            "expectancy_R": round(float(np.mean([x.r_multiple for x in t])), 3),
            "net_return_pct": round(100 * (eq.iloc[-1] / self.config.initial_equity - 1), 2),
            "max_drawdown_pct": round(float(max_dd), 2),
            "skipped_busy": self.skipped_busy,
            "expired_orders": self.expired,
        }


class Backtester:
    def run(self, df: pd.DataFrame, signals: list[Signal],
            config: BTConfig | None = None) -> BacktestResult:
        cfg = config or BTConfig()
        equity = cfg.initial_equity
        trades: list[Trade] = []
        eq_times, eq_values = [df.index[0]], [equity]
        busy_until: pd.Timestamp | None = None
        skipped = expired = 0

        highs, lows = df["high"].values, df["low"].values
        closes = df["close"].values
        times = df.index

        for sig in sorted(signals, key=lambda s: s.time):
            if busy_until is not None and sig.time <= busy_until:
                skipped += 1
                continue
            start = times.searchsorted(sig.time, side="right")
            if start >= len(df):
                continue
            trade = self._simulate(sig, start, highs, lows, closes, times,
                                   equity, cfg)
            if trade is None:
                expired += 1
                continue
            trades.append(trade)
            equity += trade.pnl
            busy_until = trade.exit_time
            eq_times.append(trade.exit_time)
            eq_values.append(equity)

        curve = pd.Series(eq_values, index=pd.DatetimeIndex(eq_times), name="equity")
        return BacktestResult(trades, curve, skipped, expired, cfg)

    def _simulate(self, sig: Signal, start: int, highs, lows, closes, times,
                  equity: float, cfg: BTConfig) -> Trade | None:
        is_long = sig.direction == "long"
        cost = cfg.spread + cfg.slippage

        # ---- entry: limit order ----
        fill_idx = None
        last = min(start + cfg.entry_expiry_bars, len(times))
        for j in range(start, last):
            touched = lows[j] <= sig.entry if is_long else highs[j] >= sig.entry
            if touched:
                fill_idx = j
                break
        if fill_idx is None:
            return None
        entry_exec = sig.entry + cost if is_long else sig.entry - cost
        stop_dist = abs(entry_exec - sig.stop_loss)
        if stop_dist <= 0:
            return None
        units = (equity * cfg.risk_pct / 100) / stop_dist

        # ---- exit: walk forward, SL checked before TP (pessimistic) ----
        for j in range(fill_idx, len(times)):
            hit_sl = lows[j] <= sig.stop_loss if is_long else highs[j] >= sig.stop_loss
            hit_tp = highs[j] >= sig.take_profit if is_long else lows[j] <= sig.take_profit
            if hit_sl:
                return self._close(sig, times[fill_idx], entry_exec,
                                   times[j], sig.stop_loss, "sl", units, is_long, stop_dist)
            if hit_tp:
                return self._close(sig, times[fill_idx], entry_exec,
                                   times[j], sig.take_profit, "tp", units, is_long, stop_dist)
        return self._close(sig, times[fill_idx], entry_exec,
                           times[-1], closes[-1], "end_of_data", units, is_long, stop_dist)

    @staticmethod
    def _close(sig, t_in, p_in, t_out, p_out, outcome, units, is_long, stop_dist) -> Trade:
        pnl = (p_out - p_in) * units if is_long else (p_in - p_out) * units
        r = (p_out - p_in) / stop_dist if is_long else (p_in - p_out) / stop_dist
        return Trade(sig.id, sig.symbol, sig.direction, t_in, round(p_in, 6),
                     t_out, round(p_out, 6), outcome, round(units, 2),
                     round(pnl, 2), round(r, 3))


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in trades])
