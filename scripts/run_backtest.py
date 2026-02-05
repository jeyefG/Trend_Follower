from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics
from src.data.mt5_client import load_csv
from src.strategy.tf_dc_atr import params_from_dict


def load_config(base_path: Path, symbol: str, symbol_config: Path | None) -> dict:
    with base_path.open("r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    if symbol_config and symbol_config.exists():
        with symbol_config.open("r", encoding="utf-8") as fh:
            override = yaml.safe_load(fh)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value

    base["symbol"] = symbol
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--config", default="configs/strategy_v1.yaml")
    parser.add_argument("--symbol-config", default=None)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config), args.symbol, Path(args.symbol_config) if args.symbol_config else None)
    bars = load_csv(args.data)
    params = params_from_dict(config)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts") / args.symbol / run_id
    trades, equity, summary = run_backtest(bars, params, config, args.symbol, out_dir)
    summary.update(compute_metrics(trades, equity))

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        import json

        json.dump(summary, fh, indent=2, default=str)

    print(out_dir)


if __name__ == "__main__":
    main()
