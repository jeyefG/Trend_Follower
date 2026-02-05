from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    default_spread: float
    slippage: float
    commission_per_side: float
    use_bid_ask_ohlc: bool = False


def resolve_spread(row, model: CostModel) -> float:
    spread = row.get("spread", None)
    if spread is None:
        return model.default_spread
    return float(spread)


def entry_fill_price(side: str, open_price: float, spread: float, model: CostModel) -> float:
    half_spread = spread / 2
    if side == "long":
        return open_price + half_spread + model.slippage
    return open_price - half_spread - model.slippage


def exit_fill_price_market(side: str, open_price: float, spread: float, model: CostModel) -> float:
    half_spread = spread / 2
    if side == "long":
        return open_price - half_spread - model.slippage
    return open_price + half_spread + model.slippage


def stop_fill_price(side: str, stop_price: float, bar_open: float, model: CostModel) -> float:
    """Deterministic stop+gap model with worst-case slippage."""
    if side == "long":
        if bar_open <= stop_price:  # gap through stop
            return bar_open - model.slippage
        return stop_price - model.slippage

    if bar_open >= stop_price:  # gap through stop
        return bar_open + model.slippage
    return stop_price + model.slippage


def total_commission(model: CostModel) -> float:
    return 2.0 * model.commission_per_side
