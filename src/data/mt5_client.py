from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

_TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

TIME_BASIS = "UTC_EPOCH_FROM_MT5"


@dataclass(frozen=True)
class BarRequest:
    symbol: str
    timeframe: str
    start: pd.Timestamp
    end: pd.Timestamp


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe).upper()


def _resolve_timeframe(timeframe: str | int) -> tuple[int, str]:
    if isinstance(timeframe, int):
        return timeframe, str(timeframe)
    key = _normalize_timeframe(timeframe)
    if key not in _TIMEFRAME_MAP:
        raise ValueError(f"Unsupported MT5 timeframe: {timeframe}")
    return _TIMEFRAME_MAP[key], key


def _mt5_epoch_to_utc(ts: int | float) -> datetime:
    return datetime.utcfromtimestamp(int(ts)).replace(tzinfo=timezone.utc)


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


@dataclass
class MT5Connector:
    """Simple MT5 connector for OHLCV retrieval."""

    def __post_init__(self) -> None:
        if not mt5.initialize():
            raise RuntimeError(f"No se pudo conectar a MetaTrader 5: {mt5.last_error()}")

    def shutdown(self) -> None:
        mt5.shutdown()

    def obtener_ohlcv(
        self,
        symbol: str,
        timeframe: str | int,
        fecha_inicio: datetime,
        fecha_fin: datetime,
    ) -> pd.DataFrame:
        timeframe_value, timeframe_label = _resolve_timeframe(timeframe)
        rates = mt5.copy_rates_range(symbol, timeframe_value, fecha_inicio, fecha_fin)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No se pudieron obtener datos {timeframe_label} desde MT5.")
        df = pd.DataFrame(rates)
        df["time"] = df["time"].apply(_mt5_epoch_to_utc)
        return normalize_bars(df)

    def obtener_h1(self, symbol: str, fecha_inicio: datetime, fecha_fin: datetime) -> pd.DataFrame:
        return self.obtener_ohlcv(symbol, "H1", fecha_inicio, fecha_fin)

    def obtener_m5(self, symbol: str, fecha_inicio: datetime, fecha_fin: datetime) -> pd.DataFrame:
        return self.obtener_ohlcv(symbol, "M5", fecha_inicio, fecha_fin)

    def server_now(self, symbol: str) -> pd.Timestamp:
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError("No se pudo obtener tick para inferir hora del servidor MT5.")
        return pd.Timestamp(_mt5_epoch_to_utc(int(tick.time)))


def fetch_mt5_bars(request: BarRequest) -> pd.DataFrame:
    connector = MT5Connector()
    try:
        return connector.obtener_ohlcv(
            symbol=request.symbol,
            timeframe=request.timeframe,
            fecha_inicio=request.start.to_pydatetime(),
            fecha_fin=request.end.to_pydatetime(),
        )
    finally:
        connector.shutdown()


def load_csv(path: str) -> pd.DataFrame:
    return normalize_bars(pd.read_csv(path))


__all__ = ["BarRequest", "MT5Connector", "TIME_BASIS", "fetch_mt5_bars", "load_csv", "normalize_bars"]
