from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class BarRequest:
    symbol: str
    timeframe: str
    start: pd.Timestamp
    end: pd.Timestamp


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    if out["time"].isna().any():
        raise ValueError("Invalid timestamps")
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return out


def load_csv(path: str) -> pd.DataFrame:
    return normalize_bars(pd.read_csv(path))
