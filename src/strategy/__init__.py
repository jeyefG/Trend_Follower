from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class EntryIntent:
    direction: str
    exec_idx: int
    stop_price: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitIntent:
    reason: str
    exec_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)


class StrategyHooks(Protocol):
    strategy_name: str

    def build_signal_frame(self, bars: pd.DataFrame) -> pd.DataFrame:
        ...

    def compute_entry(self, bars: pd.DataFrame, idx: int, state: dict[str, Any]) -> EntryIntent | None:
        ...

    def update_trailing(self, bars: pd.DataFrame, idx: int, position_state: dict[str, Any]) -> float | None:
        ...

    def check_time_stop(self, bars: pd.DataFrame, idx: int, position_state: dict[str, Any]) -> ExitIntent | None:
        ...


def build_strategy_hooks(config: dict):
    strategy_name = config.get("strategy_name", "tf_dc_atr")
    if strategy_name == "tf_dc_atr":
        from src.strategy.tf_dc_atr import hooks_from_dict

        return hooks_from_dict(config)
    if strategy_name == "tf_pb_ema":
        from src.strategy.tf_pb_ema import hooks_from_dict

        return hooks_from_dict(config)
    raise ValueError(f"Unknown strategy_name: {strategy_name}")
