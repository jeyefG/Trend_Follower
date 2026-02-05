from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class StrategyParams:
    ema_period: int = 200
    slope_lookback: int = 24
    slope_atr_multiplier: float = 0.25
    atr_period: int = 14
    donchian_period: int = 55
    atr_min_median_period: int = 200
    atr_min_multiplier: float = 0.70


def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")


def compute_features(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Compute leakage-safe features for TF-DC-ATR v1."""
    _validate_ohlc(df)
    out = df.copy()

    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    tr_components = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift(1)).abs(),
            (out["low"] - out["close"].shift(1)).abs(),
        ],
        axis=1,
    )
    out["atr14"] = tr_components.max(axis=1).ewm(alpha=1 / params.atr_period, adjust=False).mean()

    out["ema200"] = out["close"].ewm(span=params.ema_period, adjust=False).mean()
    out["slope24"] = out["ema200"] - out["ema200"].shift(params.slope_lookback)

    out["donchian_high55"] = out["high"].rolling(params.donchian_period, min_periods=params.donchian_period).max()
    out["donchian_low55"] = out["low"].rolling(params.donchian_period, min_periods=params.donchian_period).min()

    out["median_atr200"] = out["atr14"].rolling(
        params.atr_min_median_period,
        min_periods=params.atr_min_median_period,
    ).median()

    out["trend_long_allowed"] = (out["close"] > out["ema200"]) & (
        out["slope24"] > params.slope_atr_multiplier * out["atr14"]
    )
    out["trend_short_allowed"] = (out["close"] < out["ema200"]) & (
        out["slope24"] < -params.slope_atr_multiplier * out["atr14"]
    )

    # Donchian breakout must use t-1 channel to avoid leakage.
    out["long_breakout"] = out["close"] > out["donchian_high55"].shift(1)
    out["short_breakout"] = out["close"] < out["donchian_low55"].shift(1)

    out["signal_long_close_t"] = out["trend_long_allowed"] & out["long_breakout"]
    out["signal_short_close_t"] = out["trend_short_allowed"] & out["short_breakout"]

    return out


def build_signal_frame(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    features = compute_features(df, params)
    columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "atr14",
        "ema200",
        "slope24",
        "donchian_high55",
        "donchian_low55",
        "median_atr200",
        "signal_long_close_t",
        "signal_short_close_t",
    ]
    if "spread" in features.columns:
        columns.append("spread")
    return features[columns]


def params_from_dict(config: Dict) -> StrategyParams:
    p = config.get("strategy", {})
    return StrategyParams(
        ema_period=p.get("ema_period", 200),
        slope_lookback=p.get("slope_lookback", 24),
        slope_atr_multiplier=p.get("slope_atr_multiplier", 0.25),
        atr_period=p.get("atr_period", 14),
        donchian_period=p.get("donchian_period", 55),
        atr_min_median_period=p.get("atr_min_median_period", 200),
        atr_min_multiplier=p.get("atr_min_multiplier", 0.70),
    )
