from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

import sys

# --- ensure project root is on PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics
from src.data.mt5_client import BarRequest, fetch_mt5_bars, load_csv
from src.strategy.tf_dc_atr import params_from_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def parse_utc(ts: str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "strategy_v1.yaml"))
    parser.add_argument("--symbol-config", default=str(PROJECT_ROOT / "configs" / "symbols" / "XAUUSD.yaml"))
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--start", default=None, help="Inicio UTC: 2023-01-01 or 2023-01-01T00:00:00Z")
    parser.add_argument("--end", default=None, help="Fin UTC: 2024-01-01 or 2024-01-01T00:00:00Z")
    parser.add_argument("--data", default=None, help="CSV opcional como fallback")
    args = parser.parse_args()

    config = load_config(Path(args.config), args.symbol, Path(args.symbol_config) if args.symbol_config else None)
    if args.data:
        bars = load_csv(args.data)
    else:
        if not args.start or not args.end:
            parser.error("Debe indicar --start y --end para descargar desde MT5 cuando no usa --data.")
        request = BarRequest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=parse_utc(args.start),
            end=parse_utc(args.end),
        )
        bars = fetch_mt5_bars(request)
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
