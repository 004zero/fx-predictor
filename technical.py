"""テクニカル指標計算モジュール"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(s, fast) - ema(s, slow)
    sig_line = ema(macd_line, signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(s, n)
    std = s.rolling(n).std()
    return mid + k * std, mid, mid - k * std


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def ichimoku(df: pd.DataFrame):
    high9 = df["High"].rolling(9).max()
    low9 = df["Low"].rolling(9).min()
    tenkan = (high9 + low9) / 2

    high26 = df["High"].rolling(26).max()
    low26 = df["Low"].rolling(26).min()
    kijun = (high26 + low26) / 2

    span_a = ((tenkan + kijun) / 2).shift(26)
    high52 = df["High"].rolling(52).max()
    low52 = df["Low"].rolling(52).min()
    span_b = ((high52 + low52) / 2).shift(26)
    chikou = df["Close"].shift(-26)
    return tenkan, kijun, span_a, span_b, chikou


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["SMA20"] = sma(c, 20)
    out["SMA50"] = sma(c, 50)
    out["SMA200"] = sma(c, 200)
    out["EMA12"] = ema(c, 12)
    out["EMA26"] = ema(c, 26)
    out["RSI14"] = rsi(c, 14)
    out["MACD"], out["MACDsig"], out["MACDhist"] = macd(c)
    out["BBu"], out["BBm"], out["BBl"] = bollinger(c)
    out["ATR14"] = atr(out, 14)
    out["StK"], out["StD"] = stochastic(out)
    t, k, sa, sb, ch = ichimoku(out)
    out["Tenkan"], out["Kijun"], out["SpanA"], out["SpanB"], out["Chikou"] = t, k, sa, sb, ch
    return out


def signal_summary(df: pd.DataFrame) -> dict:
    """直近バーから簡易シグナル集計。スコア -100(売り) ~ +100(買い) """
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else row
    score = 0
    reasons: list[str] = []

    if row["Close"] > row["SMA20"]:
        score += 10; reasons.append("Close > SMA20")
    else:
        score -= 10; reasons.append("Close < SMA20")

    if row["SMA20"] > row["SMA50"]:
        score += 10; reasons.append("SMA20 > SMA50 (短期上向き)")
    else:
        score -= 10; reasons.append("SMA20 < SMA50 (短期下向き)")

    if not np.isnan(row["SMA200"]):
        if row["Close"] > row["SMA200"]:
            score += 15; reasons.append("Close > SMA200 (長期上昇トレンド)")
        else:
            score -= 15; reasons.append("Close < SMA200 (長期下降トレンド)")

    if row["RSI14"] < 30:
        score += 15; reasons.append(f"RSI={row['RSI14']:.1f} 売られすぎ")
    elif row["RSI14"] > 70:
        score -= 15; reasons.append(f"RSI={row['RSI14']:.1f} 買われすぎ")

    if row["MACD"] > row["MACDsig"] and prev["MACD"] <= prev["MACDsig"]:
        score += 20; reasons.append("MACDゴールデンクロス")
    elif row["MACD"] < row["MACDsig"] and prev["MACD"] >= prev["MACDsig"]:
        score -= 20; reasons.append("MACDデッドクロス")
    elif row["MACD"] > row["MACDsig"]:
        score += 5
    else:
        score -= 5

    if row["Close"] < row["BBl"]:
        score += 10; reasons.append("BB下抜け (反発期待)")
    elif row["Close"] > row["BBu"]:
        score -= 10; reasons.append("BB上抜け (反落リスク)")

    if row["StK"] < 20 and row["StK"] > row["StD"]:
        score += 10; reasons.append("ストキャス底値反転")
    elif row["StK"] > 80 and row["StK"] < row["StD"]:
        score -= 10; reasons.append("ストキャス天井反落")

    if not np.isnan(row["SpanA"]) and not np.isnan(row["SpanB"]):
        cloud_top = max(row["SpanA"], row["SpanB"])
        cloud_bot = min(row["SpanA"], row["SpanB"])
        if row["Close"] > cloud_top:
            score += 10; reasons.append("一目雲上抜け")
        elif row["Close"] < cloud_bot:
            score -= 10; reasons.append("一目雲下抜け")

    score = max(-100, min(100, score))
    if score >= 40:
        verdict = "強い買い"
    elif score >= 15:
        verdict = "買い"
    elif score <= -40:
        verdict = "強い売り"
    elif score <= -15:
        verdict = "売り"
    else:
        verdict = "中立"
    return {"score": score, "verdict": verdict, "reasons": reasons}
