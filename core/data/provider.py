"""Data layer: fetches OHLCV candles from a provider, with local caching.

All providers return a pandas DataFrame with:
  - DatetimeIndex (UTC)
  - columns: open, high, low, close, volume
This contract is what the rest of the system depends on. Swap providers
freely (yfinance, broker API, exchange API) without touching detectors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


class BaseProvider(ABC):
    @abstractmethod
    def fetch(self, symbol: str, interval: str, bars: int) -> pd.DataFrame:
        """Return the most recent `bars` candles for symbol at interval."""


class YFinanceProvider(BaseProvider):
    """Free data via Yahoo Finance. Good for prototyping.

    Notes:
      - Forex symbols use the '=X' suffix, e.g. 'EURUSD=X'.
      - Intraday intervals ('15m', '5m', '1h') are limited to ~60 days
        of history. For serious backtesting, switch to a paid feed
        (Polygon, Dukascopy dumps) or the broker's own history.
    """

    def fetch(self, symbol: str, interval: str, bars: int) -> pd.DataFrame:
        import yfinance as yf  # imported lazily: not needed for demo mode

        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}
        raw = yf.download(
            symbol,
            interval=interval,
            period=period_map.get(interval, "60d"),
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            raise RuntimeError(f"No data returned for {symbol} {interval}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[COLUMNS]
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "time"
        return df.tail(bars)


class SyntheticProvider(BaseProvider):
    """Deterministic synthetic forex-like data for demos and unit tests.

    Generates a random walk with periodic impulsive 'displacement' moves so
    that FVGs and structure breaks actually appear on the chart.
    """

    def __init__(self, seed: int = 7):
        self.seed = seed

    def fetch(self, symbol: str, interval: str, bars: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        price = 1.0850
        opens, highs, lows, closes = [], [], [], []
        drift = 0.0
        for i in range(bars):
            # Every ~70 bars, start a 6-candle impulsive move (up or down)
            if i % 70 == 10:
                drift = rng.choice([-1, 1]) * rng.uniform(0.00045, 0.0009)
            elif i % 70 == 16:
                drift = 0.0
            o = price
            body = drift + rng.normal(0, 0.00022)
            c = o + body
            wick = abs(rng.normal(0, 0.00012))
            h = max(o, c) + wick * (0.3 if drift else 1.0)
            l = min(o, c) - wick * (0.3 if drift else 1.0)
            opens.append(o); highs.append(h); lows.append(l); closes.append(c)
            price = c
        step = pd.Timedelta(interval.replace("m", "min").replace("d", "D"))
        end = pd.Timestamp.now(tz="UTC").floor(step)
        idx = pd.date_range(end=end, periods=bars, freq=step, name="time")
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": rng.integers(800, 5000, bars).astype(float)},
            index=idx,
        )
        return df


class CachedProvider(BaseProvider):
    """Wraps any provider with a local parquet cache (one file per symbol/interval)."""

    def __init__(self, inner: BaseProvider, cache_dir: str | Path = "data_cache"):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> Path:
        safe = symbol.replace("/", "_").replace("=", "_")
        return self.cache_dir / f"{safe}_{interval}.parquet"

    def fetch(self, symbol: str, interval: str, bars: int) -> pd.DataFrame:
        df = self.inner.fetch(symbol, interval, bars)
        path = self._path(symbol, interval)
        if path.exists():  # merge with what we already have
            old = pd.read_parquet(path)
            df = pd.concat([old, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(path)
        return df.tail(bars)
