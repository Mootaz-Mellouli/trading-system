"""Provider tests. The Binance provider is exercised with the network
stubbed out (the _get seam) so the suite stays offline and deterministic."""
import pandas as pd
import pytest

from core.data.provider import COLUMNS, BinanceProvider


def _kline(open_ms: int, o, h, l, c, v):
    """A Binance kline row (only the first 6 fields matter to us)."""
    return [open_ms, str(o), str(h), str(l), str(c), str(v),
            open_ms + 1, "0", 1, "0", "0", "0"]


# 1-minute apart, ascending by open time (as Binance returns them).
KLINES = [
    _kline(1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 10.0),
    _kline(1_700_000_060_000, 100.5, 102.0, 100.0, 101.5, 12.0),
    _kline(1_700_000_120_000, 101.5, 103.0, 101.0, 102.5, 11.0),
    _kline(1_700_000_180_000, 102.5, 104.0, 102.0, 103.5, 13.0),
    _kline(1_700_000_240_000, 103.5, 105.0, 103.0, 104.5, 14.0),
]


class FakeBinance(BinanceProvider):
    """Serves canned klines from a master list, honouring endTime/limit
    the way the real API does, and records every request for assertions."""

    def __init__(self, klines, max_limit=1000):
        super().__init__()
        self.MAX_LIMIT = max_limit
        self._klines = klines
        self.calls: list[dict] = []

    def _get(self, params):
        self.calls.append(dict(params))
        end = params.get("endTime")
        pool = [k for k in self._klines if end is None or k[0] <= end]
        return pool[-params["limit"]:]  # ascending, the most recent `limit`


def test_binance_provider_contract():
    df = FakeBinance(KLINES).fetch("BTCUSDT", "15m", 5)
    assert list(df.columns) == COLUMNS
    assert len(df) == 5
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert df.index.name == "time"
    assert df.index.is_monotonic_increasing
    assert (df.dtypes == "float64").all()
    # exact values from the first/last canned candle
    assert df["open"].iloc[0] == pytest.approx(100.0)
    assert df["close"].iloc[-1] == pytest.approx(104.5)
    assert df.index[0] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")


def test_binance_provider_tail_and_validates_interval():
    df = FakeBinance(KLINES).fetch("ethusdt", "1h", 2)  # also lower-case symbol
    assert len(df) == 2
    assert df["close"].tolist() == pytest.approx([103.5, 104.5])
    with pytest.raises(ValueError):
        FakeBinance(KLINES).fetch("BTCUSDT", "7m", 2)  # not a Binance interval


def test_binance_provider_paginates_backwards():
    fake = FakeBinance(KLINES, max_limit=2)
    df = fake.fetch("BTCUSDT", "5m", 5)
    assert len(fake.calls) == 3                 # 2 + 2 + 1
    assert fake.calls[0].get("endTime") is None  # newest batch first
    assert all(c["endTime"] is not None for c in fake.calls[1:])
    assert df.index.is_monotonic_increasing      # reassembled in order
    assert df["close"].tolist() == pytest.approx(
        [100.5, 101.5, 102.5, 103.5, 104.5])


def test_binance_provider_raises_when_empty():
    with pytest.raises(RuntimeError):
        FakeBinance([]).fetch("BTCUSDT", "15m", 5)
