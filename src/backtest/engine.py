from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from src.backtest.costs import (
    CostModel,
    entry_fill_price,
    exit_fill_price_market,
    resolve_spread,
    stop_fill_price,
)
from src.strategy.tf_dc_atr import StrategyParams, build_signal_frame


@dataclass
class Position:
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    initial_r: float
    stop_price: float
    trailing_stop: float
    bars_in_trade: int
    mfe_r: float


def _setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"backtest_{log_file}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _in_entry_hours(ts: pd.Timestamp, start_hour: int, end_hour: int) -> bool:
    return start_hour <= ts.hour <= end_hour


def _chandelier_stop(side: str, window: pd.DataFrame, atr: float, atr_mult: float) -> float:
    if side == "long":
        return float(window["high"].max() - atr_mult * atr)
    return float(window["low"].min() + atr_mult * atr)




def update_trailing_stop(side: str, current_stop: float, chandelier_stop: float) -> float:
    if side == "long":
        return max(current_stop, chandelier_stop)
    return min(current_stop, chandelier_stop)
def _stop_triggered(pos: Position, bar: pd.Series) -> bool:
    if pos.side == "long":
        return bar["low"] <= pos.stop_price
    return bar["high"] >= pos.stop_price


def run_backtest(
    bars: pd.DataFrame,
    strategy_params: StrategyParams,
    config: Dict,
    symbol: str,
    run_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(run_dir / "run.log")

    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    signal_df = build_signal_frame(bars, strategy_params)
    model = CostModel(**config["costs"])
    bt = config["backtest"]

    cooldown_bars = int(bt["cooldown_bars"])
    time_stop_bars = int(bt["time_stop_bars"])
    entry_start = int(bt["entry_hour_start_utc"])
    entry_end = int(bt["entry_hour_end_utc"])
    atr_min_mult = float(strategy_params.atr_min_multiplier)

    trades: List[Dict] = []
    equity_rows: List[Dict] = []

    pos: Optional[Position] = None
    pending_entry: Optional[Dict] = None
    pending_exit: Optional[str] = None
    cooldown_remaining = 0
    equity = 0.0

    for i in range(len(signal_df) - 1):
        bar = signal_df.iloc[i]
        next_bar = signal_df.iloc[i + 1]

        # apply pending market exit at next open
        if pending_exit and pos is not None:
            spread = resolve_spread(next_bar, model)
            px = exit_fill_price_market(pos.side, float(next_bar["open"]), spread, model)
            r_mult = (px - pos.entry_price) / pos.initial_r if pos.side == "long" else (pos.entry_price - px) / pos.initial_r
            net_r = r_mult - (2 * model.commission_per_side / pos.initial_r)
            equity += net_r
            trades.append(
                {
                    "symbol": symbol,
                    "entry_time": pos.entry_time,
                    "exit_time": next_bar["time"],
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "exit_price": px,
                    "reason": pending_exit,
                    "gross_r": r_mult,
                    "net_r": net_r,
                    "mfe_r": pos.mfe_r,
                }
            )
            logger.info("EXIT market reason=%s side=%s px=%.5f net_r=%.4f", pending_exit, pos.side, px, net_r)
            pos = None
            pending_exit = None
            cooldown_remaining = cooldown_bars

        # apply pending entry at next open
        if pending_entry and pos is None and cooldown_remaining == 0:
            spread = resolve_spread(next_bar, model)
            epx = entry_fill_price(pending_entry["side"], float(next_bar["open"]), spread, model)
            atr = float(bar["atr14"])
            initial_r = 2.8 * atr
            stop = epx - initial_r if pending_entry["side"] == "long" else epx + initial_r
            pos = Position(
                side=pending_entry["side"],
                entry_time=next_bar["time"],
                entry_price=epx,
                initial_r=initial_r,
                stop_price=stop,
                trailing_stop=stop,
                bars_in_trade=0,
                mfe_r=0.0,
            )
            logger.info("ENTRY side=%s px=%.5f stop=%.5f", pending_entry["side"], epx, stop)
            pending_entry = None

        # active position management for current bar (intrabar stop)
        if pos is not None:
            pos.bars_in_trade += 1
            if pos.side == "long":
                mfe = (float(bar["high"]) - pos.entry_price) / pos.initial_r
            else:
                mfe = (pos.entry_price - float(bar["low"])) / pos.initial_r
            pos.mfe_r = max(pos.mfe_r, mfe)

            if _stop_triggered(pos, bar):
                stop_px = stop_fill_price(pos.side, pos.stop_price, float(bar["open"]), model)
                r_mult = (stop_px - pos.entry_price) / pos.initial_r if pos.side == "long" else (pos.entry_price - stop_px) / pos.initial_r
                net_r = r_mult - (2 * model.commission_per_side / pos.initial_r)
                equity += net_r
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_time": pos.entry_time,
                        "exit_time": bar["time"],
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "exit_price": stop_px,
                        "reason": "stop_intrabar",
                        "gross_r": r_mult,
                        "net_r": net_r,
                        "mfe_r": pos.mfe_r,
                    }
                )
                logger.info("EXIT stop side=%s px=%.5f net_r=%.4f", pos.side, stop_px, net_r)
                pos = None
                cooldown_remaining = cooldown_bars
            else:
                win = signal_df.iloc[max(0, i - 21) : i + 1]
                chand = _chandelier_stop(pos.side, win, float(bar["atr14"]), atr_mult=3.0)
                pos.trailing_stop = update_trailing_stop(pos.side, pos.trailing_stop, chand)
                pos.stop_price = update_trailing_stop(pos.side, pos.stop_price, pos.trailing_stop)

                if pos.bars_in_trade >= time_stop_bars and pos.mfe_r < 1.0:
                    pending_exit = "time_stop"

        # signal generation at close[t], execute open[t+1]
        if pos is None and pending_entry is None and cooldown_remaining == 0:
            spread_t = resolve_spread(bar, model)
            spread_ok = spread_t <= 0.12 * float(bar["atr14"])
            atr_ok = float(bar["atr14"]) >= atr_min_mult * float(bar["median_atr200"])
            hour_ok = _in_entry_hours(bar["time"], entry_start, entry_end)
            if spread_ok and atr_ok and hour_ok:
                if bool(bar["signal_long_close_t"]):
                    pending_entry = {"side": "long"}
                    logger.info("SIGNAL long at %s", bar["time"])
                elif bool(bar["signal_short_close_t"]):
                    pending_entry = {"side": "short"}
                    logger.info("SIGNAL short at %s", bar["time"])
            else:
                logger.info(
                    "BLOCKED filters spread_ok=%s atr_ok=%s hour_ok=%s time=%s",
                    spread_ok,
                    atr_ok,
                    hour_ok,
                    bar["time"],
                )

        if cooldown_remaining > 0 and pos is None:
            cooldown_remaining -= 1

        equity_rows.append({"time": bar["time"], "equity_r": equity})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    summary = {
        "symbol": symbol,
        "trades": int(len(trades_df)),
        "net_r": float(trades_df["net_r"].sum()) if len(trades_df) else 0.0,
    }

    trades_df.to_csv(run_dir / "trades.csv", index=False)
    equity_df.to_csv(run_dir / "equity.csv", index=False)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return trades_df, equity_df, summary
