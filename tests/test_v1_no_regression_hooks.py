from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pandas as pd

from src.strategy import build_strategy_hooks
from src.strategy.tf_dc_atr import params_from_dict


def _fake_mt5_module():
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
    fake.initialize = lambda: True
    fake.last_error = lambda: (0, "ok")
    fake.copy_rates_range = lambda *args, **kwargs: []
    fake.shutdown = lambda: None
    fake.symbol_select = lambda *args, **kwargs: True
    fake.symbol_info_tick = lambda *args, **kwargs: types.SimpleNamespace(time=1704067200)
    return fake


def _engine_module():
    sys.modules.setdefault("MetaTrader5", _fake_mt5_module())
    sys.modules.pop("src.backtest.engine", None)
    return importlib.import_module("src.backtest.engine")


def _bars(n: int = 300) -> pd.DataFrame:
    t0 = pd.Timestamp("2023-01-01T00:00:00Z")
    rows = []
    px = 1800.0
    for i in range(n):
        px += 0.3
        rows.append(
            {
                "time": t0 + pd.Timedelta(hours=i),
                "open": px,
                "high": px + 0.6,
                "low": px - 0.6,
                "close": px + 0.1,
                "spread": 0.05,
            }
        )
    return pd.DataFrame(rows)


def _config() -> dict:
    return {
        "strategy_name": "tf_dc_atr",
        "strategy": {
            "ema_period": 10,
            "slope_lookback": 2,
            "slope_atr_multiplier": 0.25,
            "atr_period": 5,
            "donchian_period": 10,
            "atr_min_median_period": 20,
            "atr_min_multiplier": 0.70,
        },
        "costs": {
            "default_spread": 0.05,
            "slippage": 0.01,
            "commission_per_side": 0.0,
            "use_bid_ask_ohlc": False,
        },
        "backtest": {
            "cooldown_bars": 5,
            "time_stop_bars": 3,
            "entry_hour_start_utc": 7,
            "entry_hour_end_utc": 20,
        },
    }


def test_v1_summary_metrics_unchanged_with_default_and_explicit_hooks(tmp_path: Path):
    engine = _engine_module()
    config = _config()
    params = params_from_dict(config)
    bars = _bars()

    run_a = tmp_path / "a"
    run_b = tmp_path / "b"

    trades_a, equity_a, summary_a = engine.run_backtest(bars, params, config, "XAUUSD", run_a)
    hooks = build_strategy_hooks(config)
    trades_b, equity_b, summary_b = engine.run_backtest(
        bars,
        params,
        config,
        "XAUUSD",
        run_b,
        strategy_hooks=hooks,
    )

    assert len(trades_a) == len(trades_b)
    assert float(summary_a["net_r"]) == float(summary_b["net_r"])
    assert float(equity_a["equity_r"].min()) == float(equity_b["equity_r"].min())
    assert summary_a["trades"] == summary_b["trades"]
    assert summary_a["net_r"] == summary_b["net_r"]
