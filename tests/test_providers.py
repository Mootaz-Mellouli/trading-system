"""Provider tests. The Binance provider is exercised with the network
stubbed out (the _get seam) so the suite stays offline and deterministic."""
import pandas as pd
import pytest

from core.data.provider import (
    COLUMNS, BinanceProvider, CoinbaseProvider, KrakenProvider)


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


# ----------------------------------------------------------- Coinbase ------
# Coinbase rows are [time, low, high, open, close, volume], newest-first.
CB_CANDLES = [  # ascending master list, 1h apart
    [1_700_000_000, 99.0, 101.0, 100.0, 100.5, 10.0],
    [1_700_003_600, 100.0, 102.0, 100.5, 101.5, 12.0],
    [1_700_007_200, 101.0, 103.0, 101.5, 102.5, 11.0],
    [1_700_010_800, 102.0, 104.0, 102.5, 103.5, 13.0],
    [1_700_014_400, 103.0, 105.0, 103.5, 104.5, 14.0],
]


class FakeCoinbase(CoinbaseProvider):
    def __init__(self, candles, max_candles=300):
        super().__init__()
        self.MAX_CANDLES = max_candles
        self._candles = candles
        self.calls: list = []

    def _get(self, product, params):
        self.calls.append((product, dict(params)))
        if "end" in params:
            end = int(pd.Timestamp(params["end"]).timestamp())
            pool = [c for c in self._candles if c[0] <= end]
        else:
            pool = list(self._candles)
        batch = pool[-self.MAX_CANDLES:]
        return list(reversed(batch))            # Coinbase returns newest-first


def test_coinbase_provider_contract_and_column_order():
    df = FakeCoinbase(CB_CANDLES).fetch("BTC-USD", "1h", 5)
    assert list(df.columns) == COLUMNS
    assert len(df) == 5
    assert str(df.index.tz) == "UTC" and df.index.name == "time"
    assert df.index.is_monotonic_increasing
    # the [t, low, high, open, close, vol] order is mapped correctly
    assert df["open"].iloc[0] == pytest.approx(100.0)
    assert df["high"].iloc[0] == pytest.approx(101.0)
    assert df["low"].iloc[0] == pytest.approx(99.0)
    assert df["close"].iloc[-1] == pytest.approx(104.5)


def test_coinbase_provider_paginates_backwards():
    fake = FakeCoinbase(CB_CANDLES, max_candles=2)
    df = fake.fetch("BTC-USD", "1h", 5)
    assert len(fake.calls) == 3                 # 2 + 2 + 1
    assert "end" not in fake.calls[0][1]        # newest window first
    assert all("end" in c[1] for c in fake.calls[1:])
    assert df.index.is_monotonic_increasing
    assert df["close"].tolist() == pytest.approx(
        [100.5, 101.5, 102.5, 103.5, 104.5])


def test_coinbase_validates_interval():
    with pytest.raises(ValueError):
        FakeCoinbase(CB_CANDLES).fetch("BTC-USD", "7m", 5)


# ------------------------------------------------------------- Kraken ------
# Kraken rows: [time, open, high, low, close, vwap, volume, count] (strings).
KRAKEN_ROWS = [
    [1_700_000_000, "100.0", "101.0", "99.0", "100.5", "100.2", "10.0", 5],
    [1_700_003_600, "100.5", "102.0", "100.0", "101.5", "101.0", "12.0", 7],
    [1_700_007_200, "101.5", "103.0", "101.0", "102.5", "102.1", "11.0", 6],
]


class FakeKraken(KrakenProvider):
    def __init__(self, payload):
        super().__init__()
        self._payload = payload
        self.calls: list = []

    def _get(self, params):
        self.calls.append(dict(params))
        return self._payload


def test_kraken_provider_contract_and_key_extraction():
    payload = {"error": [], "result": {"XXBTZUSD": KRAKEN_ROWS, "last": 1_700_007_200}}
    df = FakeKraken(payload).fetch("XBTUSD", "1h", 3)
    assert list(df.columns) == COLUMNS
    assert len(df) == 3
    assert str(df.index.tz) == "UTC" and df.index.is_monotonic_increasing
    assert (df.dtypes == "float64").all()       # string fields coerced
    assert df["open"].iloc[0] == pytest.approx(100.0)
    assert df["close"].iloc[-1] == pytest.approx(102.5)
    assert df["high"].iloc[1] == pytest.approx(102.0)


def test_kraken_provider_tail_and_errors():
    payload = {"error": [], "result": {"XXBTZUSD": KRAKEN_ROWS, "last": 0}}
    df = FakeKraken(payload).fetch("XBTUSD", "1h", 2)
    assert df["close"].tolist() == pytest.approx([101.5, 102.5])
    with pytest.raises(ValueError):
        FakeKraken(payload).fetch("XBTUSD", "2h", 2)   # not a Kraken interval
    err = {"error": ["EQuery:Unknown asset pair"], "result": {}}
    with pytest.raises(RuntimeError):
        FakeKraken(err).fetch("NOPE", "1h", 2)
