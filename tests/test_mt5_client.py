from __future__ import annotations

import importlib
import sys
import types

import pandas as pd
import pytest


def make_fake_mt5(initialize_ok: bool = True, rates=None, symbol_select_ok: bool = True):
    if rates is None:
        rates = [
            {
                "time": 1704067200,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "tick_volume": 10,
                "spread": 5,
                "real_volume": 10,
            }
        ]

    fake = types.SimpleNamespace()
    fake.TIMEFRAME_M1 = 1
    fake.TIMEFRAME_M2 = 2
    fake.TIMEFRAME_M3 = 3
    fake.TIMEFRAME_M4 = 4
    fake.TIMEFRAME_M5 = 5
    fake.TIMEFRAME_M6 = 6
    fake.TIMEFRAME_M10 = 10
    fake.TIMEFRAME_M12 = 12
    fake.TIMEFRAME_M15 = 15
    fake.TIMEFRAME_M20 = 20
    fake.TIMEFRAME_M30 = 30
    fake.TIMEFRAME_H1 = 16385
    fake.TIMEFRAME_H2 = 16386
    fake.TIMEFRAME_H3 = 16387
    fake.TIMEFRAME_H4 = 16388
    fake.TIMEFRAME_H6 = 16390
    fake.TIMEFRAME_H8 = 16392
    fake.TIMEFRAME_H12 = 16396
    fake.TIMEFRAME_D1 = 16408
    fake.TIMEFRAME_W1 = 32769
    fake.TIMEFRAME_MN1 = 49153

    fake.initialize = lambda: initialize_ok
    fake.last_error = lambda: (500, "init error")
    fake.copy_rates_range = lambda symbol, tf, start, end: rates
    fake.shutdown = lambda: None
    fake.symbol_select = lambda symbol, selected: symbol_select_ok
    fake.symbol_info_tick = lambda symbol: types.SimpleNamespace(time=1704067200)
    return fake


def import_mt5_client_with_fake(fake_mt5):
    sys.modules.pop("src.data.mt5_client", None)
    sys.modules["MetaTrader5"] = fake_mt5
    module = importlib.import_module("src.data.mt5_client")
    return module


def test_resolve_supported_timeframe():
    module = import_mt5_client_with_fake(make_fake_mt5())
    value, label = module._resolve_timeframe("h1")
    assert value == 16385
    assert label == "H1"


def test_fetch_mt5_bars_returns_normalized_df():
    module = import_mt5_client_with_fake(make_fake_mt5())
    req = module.BarRequest(
        symbol="XAUUSD",
        timeframe="H1",
        start=pd.Timestamp("2024-01-01T00:00:00Z"),
        end=pd.Timestamp("2024-01-02T00:00:00Z"),
    )
    bars = module.fetch_mt5_bars(req)
    assert list(["time", "open", "high", "low", "close"]) == list(bars.columns[:5])
    assert bars["time"].dtype == "datetime64[ns, UTC]"


def test_connector_raises_if_mt5_init_fails():
    module = import_mt5_client_with_fake(make_fake_mt5(initialize_ok=False))
    with pytest.raises(RuntimeError, match="MetaTrader 5"):
        module.MT5Connector()


def test_obtener_ohlcv_raises_if_symbol_select_fails():
    module = import_mt5_client_with_fake(make_fake_mt5(symbol_select_ok=False))
    connector = module.MT5Connector()
    with pytest.raises(RuntimeError, match="symbol_select failed"):
        connector.obtener_ohlcv(
            symbol="XAUUSD",
            timeframe="H1",
            fecha_inicio=pd.Timestamp("2024-01-01T00:00:00Z").to_pydatetime(),
            fecha_fin=pd.Timestamp("2024-01-02T00:00:00Z").to_pydatetime(),
        )


def test_obtener_ohlcv_empty_rates_reports_last_error():
    module = import_mt5_client_with_fake(make_fake_mt5(rates=[]))
    connector = module.MT5Connector()
    with pytest.raises(RuntimeError, match="last_error"):
        connector.obtener_ohlcv(
            symbol="XAUUSD",
            timeframe="H1",
            fecha_inicio=pd.Timestamp("2024-01-01T00:00:00Z").to_pydatetime(),
            fecha_fin=pd.Timestamp("2024-01-02T00:00:00Z").to_pydatetime(),
        )

def test_mt5_epoch_is_converted_with_utc_basis():
    module = import_mt5_client_with_fake(make_fake_mt5())
    ts = module._mt5_epoch_to_utc(1704067200)
    assert ts.isoformat() == "2024-01-01T00:00:00+00:00"
