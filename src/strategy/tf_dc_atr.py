from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.strategy import EntryIntent, ExitIntent


@dataclass(frozen=True)
class StrategyParams:
    ema_period: int = 200
    slope_lookback: int = 24
    slope_atr_multiplier: float = 0.25
    atr_period: int = 14
    donchian_period: int = 55
    atr_min_median_period: int = 200
    atr_min_multiplier: float = 0.70


@dataclass(frozen=True)
class TFDCATRHooks:
    params: StrategyParams
    time_stop_bars: int = 120
    chandelier_lookback: int = 22
    chandelier_atr_multiplier: float = 3.0
    initial_stop_atr_multiplier: float = 2.8
    strategy_name: str = "tf_dc_atr"

    def build_signal_frame(self, bars: pd.DataFrame) -> pd.DataFrame:
        return build_signal_frame(bars, self.params)

    def compute_entry(self, bars: pd.DataFrame, idx: int, state: dict[str, Any]) -> EntryIntent | None:
        bar = bars.iloc[idx]
        atr = float(bar["atr14"])
        if bool(bar.get("signal_long_close_t", False)):
            return EntryIntent(
                direction="long",
                exec_idx=idx + 1,
                stop_price=0.0,
                metadata={"initial_r": self.initial_stop_atr_multiplier * atr},
            )
        if bool(bar.get("signal_short_close_t", False)):
            return EntryIntent(
                direction="short",
                exec_idx=idx + 1,
                stop_price=0.0,
                metadata={"initial_r": self.initial_stop_atr_multiplier * atr},
            )
        return None

    def update_trailing(self, bars: pd.DataFrame, idx: int, position_state: dict[str, Any]) -> float | None:
        window = bars.iloc[max(0, idx - self.chandelier_lookback + 1) : idx + 1]
        atr = float(bars.iloc[idx]["atr14"])
        if position_state["side"] == "long":
            return float(window["high"].max() - self.chandelier_atr_multiplier * atr)
        return float(window["low"].min() + self.chandelier_atr_multiplier * atr)

    def check_time_stop(self, bars: pd.DataFrame, idx: int, position_state: dict[str, Any]) -> ExitIntent | None:
        if position_state["bars_in_trade"] >= self.time_stop_bars and position_state["mfe_r"] < 1.0:
            return ExitIntent(reason="time_stop", exec_idx=idx + 1)
        return None


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


def hooks_from_dict(config: Dict) -> TFDCATRHooks:
    bt = config.get("backtest", {})
    return TFDCATRHooks(params=params_from_dict(config), time_stop_bars=int(bt.get("time_stop_bars", 120)))
