from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_windows(index: Iterable[pd.Timestamp], train_bars: int, test_bars: int, step_bars: int) -> List[WalkForwardWindow]:
    idx = list(index)
    windows: List[WalkForwardWindow] = []
    start = 0
    while start + train_bars + test_bars <= len(idx):
        windows.append(
            WalkForwardWindow(
                train_start=idx[start],
                train_end=idx[start + train_bars - 1],
                test_start=idx[start + train_bars],
                test_end=idx[start + train_bars + test_bars - 1],
            )
        )
        start += step_bars
    return windows
