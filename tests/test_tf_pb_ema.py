from __future__ import annotations

import pandas as pd

from src.strategy.tf_pb_ema import TFPBEMAHooks, TFPBEMAParams, compute_features


def _base_bars(n: int = 260) -> pd.DataFrame:
    t0 = pd.Timestamp("2024-01-01T00:00:00Z")
    rows = []
    price = 2000.0
    for i in range(n):
        price += 0.4
        rows.append(
            {
                "time": t0 + pd.Timedelta(hours=i),
                "open": price,
                "high": price + 1.2,
                "low": price - 1.2,
                "close": price + 0.2,
                "spread": 0.05,
            }
        )
    return pd.DataFrame(rows)


def test_pullback_signal_long_triggers_on_synthetic_data():
    bars = _base_bars()
    idx = 230
    f0 = compute_features(bars, TFPBEMAParams())
    ema20 = float(f0.loc[idx, "ema20"])
    ema50 = float(f0.loc[idx, "ema50"])

    # Fuerza patrón de pullback long exacto: low <= ema20, low > ema50, close > ema20
    bars.loc[idx, "low"] = (ema20 + ema50) / 2
    bars.loc[idx, "close"] = ema20 + 0.2

    features = compute_features(bars, TFPBEMAParams())
    assert bool(features.loc[idx, "trend_long_allowed"])
    assert bool(features.loc[idx, "signal_long_close_t"])


def test_trailing_is_monotonic_for_long_and_short():
    hooks = TFPBEMAHooks(params=TFPBEMAParams())
    bars = hooks.build_signal_frame(_base_bars())

    long_trailing = []
    short_trailing = []
    long_current = float("-inf")
    short_current = float("inf")
    for idx in range(220, 240):
        long_cand = hooks.update_trailing(bars, idx, {"side": "long", "bars_in_trade": 1, "mfe_r": 0.0})
        short_cand = hooks.update_trailing(bars, idx, {"side": "short", "bars_in_trade": 1, "mfe_r": 0.0})
        long_current = max(long_current, float(long_cand))
        short_current = min(short_current, float(short_cand))
        long_trailing.append(long_current)
        short_trailing.append(short_current)

    assert all(a <= b for a, b in zip(long_trailing, long_trailing[1:]))
    assert all(a >= b for a, b in zip(short_trailing, short_trailing[1:]))


def test_time_stop_triggers_at_48_bars():
    hooks = TFPBEMAHooks(params=TFPBEMAParams(time_stop_bars=48))
    bars = _base_bars()
    assert hooks.check_time_stop(bars, 200, {"side": "long", "bars_in_trade": 47, "mfe_r": 2.0}) is None
    assert hooks.check_time_stop(bars, 201, {"side": "long", "bars_in_trade": 48, "mfe_r": 0.0}) is not None
