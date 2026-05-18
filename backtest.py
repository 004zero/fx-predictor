"""シンプルなウォークフォワード・バックテスト"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from predictor import _build_features, FEATURES


@dataclass
class BacktestResult:
    trades: int
    win_rate: float
    avg_pips: float
    total_pips: float
    max_dd_pips: float
    equity_curve: pd.Series


def run_backtest(
    df: pd.DataFrame,
    pair: str,
    horizon: int = 4,
    train_window: int = 1000,
    step: int = 50,
    threshold: float = 0.6,
) -> BacktestResult:
    # pip単位を銘柄から推定
    if pair == "USDJPY":
        pip = 0.01
    elif pair == "EURUSD":
        pip = 0.0001
    elif pair == "GOLD":
        pip = 0.1
    elif pair == "BITCOIN":
        pip = 1.0
    else:
        pip = 0.0001

    x = _build_features(df).dropna()
    if len(x) < train_window + step * 2:
        raise ValueError("バックテストに十分なデータがありません")

    y = (x["Close"].shift(-horizon) > x["Close"]).astype(int)
    feat = x[FEATURES]
    valid = feat.notna().all(axis=1) & y.notna()
    feat = feat[valid]
    y = y[valid]
    closes = x["Close"][valid]

    trades = []
    equity = [0.0]
    i = train_window
    while i + horizon < len(feat):
        Xtr = feat.iloc[i - train_window : i]
        ytr = y.iloc[i - train_window : i]
        model = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
        )
        model.fit(Xtr, ytr)
        Xte = feat.iloc[i : i + step]
        proba = model.predict_proba(Xte)[:, 1]
        for j, p in enumerate(proba):
            idx = i + j
            if idx + horizon >= len(closes):
                break
            entry = closes.iloc[idx]
            exit_ = closes.iloc[idx + horizon]
            if p >= threshold:
                pips = (exit_ - entry) / pip
                trades.append(pips)
                equity.append(equity[-1] + pips)
            elif p <= 1 - threshold:
                pips = (entry - exit_) / pip
                trades.append(pips)
                equity.append(equity[-1] + pips)
        i += step

    if not trades:
        return BacktestResult(0, 0.0, 0.0, 0.0, 0.0, pd.Series([0.0]))

    arr = np.array(trades)
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = (eq - peak).min()
    return BacktestResult(
        trades=len(arr),
        win_rate=float((arr > 0).mean()),
        avg_pips=float(arr.mean()),
        total_pips=float(arr.sum()),
        max_dd_pips=float(dd),
        equity_curve=eq,
    )
