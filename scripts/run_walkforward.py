from __future__ import annotations

import argparse

from src.data.mt5_client import load_csv
from src.walkforward.protocol import build_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-bars", type=int, default=24 * 180)
    parser.add_argument("--test-bars", type=int, default=24 * 30)
    parser.add_argument("--step-bars", type=int, default=24 * 30)
    args = parser.parse_args()

    bars = load_csv(args.data)
    windows = build_windows(bars["time"], args.train_bars, args.test_bars, args.step_bars)
    print(f"windows={len(windows)}")


if __name__ == "__main__":
    main()
