from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.backtest.engine import run_backtest
from src.strategy.tf_dc_atr import StrategyParams


def synthetic_bars(n: int = 260) -> pd.DataFrame:
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


def base_params() -> StrategyParams:
    return StrategyParams(
        ema_period=10,
        slope_lookback=2,
        atr_period=5,
        donchian_period=10,
        atr_min_median_period=20,
    )


def base_config() -> dict:
    return {
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


def test_summary_and_resolved_config_include_data_contract_metadata(tmp_path: Path):
    trades, equity, summary = run_backtest(
        synthetic_bars(),
        base_params(),
        base_config(),
        "XAUUSD",
        tmp_path,
    )
    assert isinstance(trades, pd.DataFrame)
    assert isinstance(equity, pd.DataFrame)

    assert summary["time_basis"] == "UTC_EPOCH_FROM_MT5"
    assert summary["price_basis"] == "MT5_RATES_OHLC"
    assert summary["spread_mode"] == "column_or_default"
    assert summary["slippage_mode"] == "fixed_absolute_per_fill"
    assert summary["spread_units"] == "price"
    assert summary["slippage_units"] == "price"
    assert summary["config_hash"]
    assert summary["sample_first_time_utc"].endswith("+00:00")
    assert summary["sample_last_time_utc"].endswith("+00:00")

    resolved = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["data_contract"]["time_basis"] == "UTC_EPOCH_FROM_MT5"
    assert resolved["data_contract"]["price_basis"] == "MT5_RATES_OHLC"
    assert resolved["config_hash"] == summary["config_hash"]


def test_abort_if_bid_ask_mode_enabled_and_columns_missing(tmp_path: Path):
    config = base_config()
    config["costs"]["use_bid_ask_ohlc"] = True
    with pytest.raises(ValueError, match="use_bid_ask_ohlc=true"):
        run_backtest(synthetic_bars(), base_params(), config, "XAUUSD", tmp_path)
