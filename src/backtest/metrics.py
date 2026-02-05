from __future__ import annotations

from typing import Dict

import pandas as pd


def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> Dict:
    if trades.empty:
        return {
            "expectancy_net_r": 0.0,
            "win_rate": 0.0,
            "payoff": 0.0,
            "trades_per_week": 0.0,
            "max_dd_r": 0.0,
            "go": False,
        }

    wins = trades[trades["net_r"] > 0]
    losses = trades[trades["net_r"] <= 0]
    expectancy = float(trades["net_r"].mean())
    win_rate = float(len(wins) / len(trades))
    avg_win = float(wins["net_r"].mean()) if len(wins) else 0.0
    avg_loss = abs(float(losses["net_r"].mean())) if len(losses) else 0.0
    payoff = avg_win / avg_loss if avg_loss else 0.0

    days = max((equity["time"].max() - equity["time"].min()).days, 1)
    trades_per_week = float(len(trades) / (days / 7))

    eq = equity["equity_r"].astype(float)
    dd = eq.cummax() - eq
    max_dd_r = float(dd.max())

    go = (
        expectancy >= 0.10
        and max_dd_r <= 20.0
        and 0.5 <= trades_per_week <= 8.0
    )

    return {
        "expectancy_net_r": expectancy,
        "win_rate": win_rate,
        "payoff": payoff,
        "trades_per_week": trades_per_week,
        "max_dd_r": max_dd_r,
        "go": bool(go),
    }
