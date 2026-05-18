"""価格データ取得モジュール (yfinance利用)"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Optional

import pandas as pd
import yfinance as yf


_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}
_TTL_SEC = 60


def fetch_ohlc(symbol: str, interval: str = "1h", period: str = "60d") -> pd.DataFrame:
    """OHLCV をDataFrameで返す。60秒TTLのプロセス内キャッシュ付き。"""
    key = (symbol, interval, period)
    now = time.time()
    if key in _CACHE:
        ts, df = _CACHE[key]
        if now - ts < _TTL_SEC:
            return df.copy()

    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"価格データ取得失敗: {symbol} {interval} {period}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    _CACHE[key] = (now, df.copy())
    return df


def latest_price(symbol: str) -> Optional[float]:
    try:
        df = fetch_ohlc(symbol, "5m", "1d")
        return float(df["Close"].iloc[-1])
    except Exception:
        return None
