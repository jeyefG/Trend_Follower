from __future__ import annotations

import pandas as pd

from src.backtest.costs import CostModel, stop_fill_price
from src.backtest.engine import _in_entry_hours
from src.strategy.tf_dc_atr import StrategyParams, compute_features


def synthetic_bars(n: int = 320) -> pd.DataFrame:
    t0 = pd.Timestamp("2023-01-01T00:00:00Z")
    times = [t0 + pd.Timedelta(hours=i) for i in range(n)]
    close = pd.Series([1800 + 0.2 * i for i in range(n)])
    out = pd.DataFrame(
        {
            "time": times,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "spread": 0.2,
        }
    )
    return out


def test_no_lookahead_donchian_shift():
    bars = synthetic_bars()
    features = compute_features(bars, StrategyParams())
    idx = 250
    assert features.loc[idx, "donchian_high55"] == bars["high"].iloc[idx - 54 : idx + 1].max()
    assert features.loc[idx, "long_breakout"] == (
        features.loc[idx, "close"] > features.loc[idx - 1, "donchian_high55"]
    )


def test_stop_gap_and_intrabar_deterministic():
    model = CostModel(default_spread=0.2, slippage=0.1, commission_per_side=0.01)
    assert stop_fill_price("long", stop_price=100.0, bar_open=99.0, model=model) == 98.9
    assert stop_fill_price("long", stop_price=100.0, bar_open=101.0, model=model) == 99.9
    assert stop_fill_price("short", stop_price=100.0, bar_open=101.0, model=model) == 101.1
    assert stop_fill_price("short", stop_price=100.0, bar_open=99.0, model=model) == 100.1


def test_entry_hours_only_restrict_entries():
    ts_ok = pd.Timestamp("2023-01-01T10:00:00Z")
    ts_bad = pd.Timestamp("2023-01-01T23:00:00Z")
    assert _in_entry_hours(ts_ok, 7, 20)
    assert not _in_entry_hours(ts_bad, 7, 20)
