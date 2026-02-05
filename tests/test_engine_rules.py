from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.costs import CostModel, entry_fill_price, exit_fill_price_market
from src.backtest.engine import run_backtest, update_trailing_stop
from src.strategy.tf_dc_atr import StrategyParams


def synthetic_trending_bars(n: int = 260) -> pd.DataFrame:
    t0 = pd.Timestamp("2023-01-01T00:00:00Z")
    rows = []
    price = 100.0
    for i in range(n):
        price += 0.05
        rows.append(
            {
                "time": t0 + pd.Timedelta(hours=i),
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price + 0.05,
                "spread": 0.05,
            }
        )
    return pd.DataFrame(rows)


def test_trailing_monotonic():
    assert update_trailing_stop("long", 100.0, 99.0) == 100.0
    assert update_trailing_stop("long", 100.0, 101.0) == 101.0
    assert update_trailing_stop("short", 100.0, 101.0) == 100.0
    assert update_trailing_stop("short", 100.0, 99.0) == 99.0


def test_cooldown_and_timestop(tmp_path: Path):
    bars = synthetic_trending_bars()
    params = StrategyParams(
        ema_period=10,
        slope_lookback=2,
        atr_period=5,
        donchian_period=10,
        atr_min_median_period=20,
    )
    config = {
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
    trades, _equity, _summary = run_backtest(bars, params, config, "XAUUSD", tmp_path)
    if len(trades) >= 2:
        gaps = pd.to_datetime(trades["entry_time"]).diff().dropna() / pd.Timedelta(hours=1)
        assert (gaps >= 5).all()
    assert set(trades["reason"]).issubset({"time_stop", "stop_intrabar"})


def test_costs_consistency_functions():
    model = CostModel(default_spread=0.2, slippage=0.1, commission_per_side=0.01)
    long_entry = entry_fill_price("long", 100.0, 0.2, model)
    long_exit = exit_fill_price_market("long", 100.0, 0.2, model)
    short_entry = entry_fill_price("short", 100.0, 0.2, model)
    short_exit = exit_fill_price_market("short", 100.0, 0.2, model)

    assert long_entry == 100.2
    assert long_exit == 99.8
    assert short_entry == 99.8
    assert short_exit == 100.2
