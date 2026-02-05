from __future__ import annotations

import argparse

import pandas as pd

from src.data.mt5_client import BarRequest, fetch_mt5_bars
from src.walkforward.protocol import build_windows


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
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--start", required=True, help="Inicio UTC")
    parser.add_argument("--end", required=True, help="Fin UTC")
    parser.add_argument("--train-bars", type=int, default=24 * 180)
    parser.add_argument("--test-bars", type=int, default=24 * 30)
    parser.add_argument("--step-bars", type=int, default=24 * 30)
    args = parser.parse_args()

    request = BarRequest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=parse_utc(args.start),
        end=parse_utc(args.end),
    )
    bars = fetch_mt5_bars(request)
    windows = build_windows(bars["time"], args.train_bars, args.test_bars, args.step_bars)
    print(f"windows={len(windows)}")


if __name__ == "__main__":
    main()
