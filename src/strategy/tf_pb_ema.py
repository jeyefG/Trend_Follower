from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.strategy import EntryIntent, ExitIntent


@dataclass(frozen=True)
class TFPBEMAParams:
    ema_fast_period: int = 20
    ema_trend_period: int = 50
    ema_slow_period: int = 200
    atr_period: int = 14
    slope_lookback: int = 24
    atr_min_median_period: int = 200
    atr_min_multiplier: float = 0.8
    initial_stop_atr_multiplier: float = 1.2
    chandelier_lookback: int = 10
    chandelier_atr_multiplier: float = 2.0
    time_stop_bars: int = 48


@dataclass(frozen=True)
class TFPBEMAHooks:
    params: TFPBEMAParams
    strategy_name: str = "tf_pb_ema"

    def build_signal_frame(self, bars: pd.DataFrame) -> pd.DataFrame:
        features = compute_features(bars, self.params)
        columns = [
            "time",
            "open",
            "high",
            "low",
            "close",
            "atr14",
            "median_atr200",
            "signal_long_close_t",
            "signal_short_close_t",
        ]
        if "spread" in features.columns:
            columns.append("spread")
        return features[columns]

    def compute_entry(self, bars: pd.DataFrame, idx: int, state: dict[str, Any]) -> EntryIntent | None:
        bar = bars.iloc[idx]
        entry_open = float(bars.iloc[idx + 1]["open"])
        atr = float(bar["atr14"])
        if bool(bar.get("signal_long_close_t", False)):
            stop = min(float(bar["low"]), entry_open - self.params.initial_stop_atr_multiplier * atr)
            return EntryIntent(direction="long", exec_idx=idx + 1, stop_price=stop)
        if bool(bar.get("signal_short_close_t", False)):
            stop = max(float(bar["high"]), entry_open + self.params.initial_stop_atr_multiplier * atr)
            return EntryIntent(direction="short", exec_idx=idx + 1, stop_price=stop)
        return None

    def update_trailing(self, bars: pd.DataFrame, idx: int, position_state: dict[str, Any]) -> float | None:
        window = bars.iloc[max(0, idx - self.params.chandelier_lookback + 1) : idx + 1]
        atr = float(bars.iloc[idx]["atr14"])
        if position_state["side"] == "long":
            return float(window["high"].max() - self.params.chandelier_atr_multiplier * atr)
        return float(window["low"].min() + self.params.chandelier_atr_multiplier * atr)

    def check_time_stop(self, bars: pd.DataFrame, idx: int, position_state: dict[str, Any]) -> ExitIntent | None:
        if position_state["bars_in_trade"] >= self.params.time_stop_bars:
            return ExitIntent(reason="time_stop", exec_idx=idx + 1)
        return None


def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")


def compute_features(df: pd.DataFrame, params: TFPBEMAParams) -> pd.DataFrame:
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

    out["ema20"] = out["close"].ewm(span=params.ema_fast_period, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=params.ema_trend_period, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=params.ema_slow_period, adjust=False).mean()
    out["slope24"] = out["ema200"] - out["ema200"].shift(params.slope_lookback)

    out["median_atr200"] = out["atr14"].rolling(
        params.atr_min_median_period,
        min_periods=params.atr_min_median_period,
    ).median()

    out["trend_long_allowed"] = (out["ema50"] > out["ema200"]) & (out["slope24"] > 0)
    out["trend_short_allowed"] = (out["ema50"] < out["ema200"]) & (out["slope24"] < 0)

    long_pullback = (out["low"] <= out["ema20"]) & (out["low"] > out["ema50"]) & (out["close"] > out["ema20"])
    short_pullback = (out["high"] >= out["ema20"]) & (out["high"] < out["ema50"]) & (out["close"] < out["ema20"])

    out["signal_long_close_t"] = out["trend_long_allowed"] & long_pullback
    out["signal_short_close_t"] = out["trend_short_allowed"] & short_pullback
    return out


def params_from_dict(config: Dict) -> TFPBEMAParams:
    p = config.get("strategy", {})
    return TFPBEMAParams(
        ema_fast_period=p.get("ema_fast_period", 20),
        ema_trend_period=p.get("ema_trend_period", 50),
        ema_slow_period=p.get("ema_slow_period", 200),
        atr_period=p.get("atr_period", 14),
        slope_lookback=p.get("slope_lookback", 24),
        atr_min_median_period=p.get("atr_min_median_period", 200),
        atr_min_multiplier=p.get("atr_min_multiplier", 0.8),
        initial_stop_atr_multiplier=p.get("initial_stop_atr_multiplier", 1.2),
        chandelier_lookback=p.get("chandelier_lookback", 10),
        chandelier_atr_multiplier=p.get("chandelier_atr_multiplier", 2.0),
        time_stop_bars=p.get("time_stop_bars", 48),
    )


def hooks_from_dict(config: Dict) -> TFPBEMAHooks:
    return TFPBEMAHooks(params=params_from_dict(config))
