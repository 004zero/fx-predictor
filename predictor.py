"""機械学習による短期方向予想モジュール
RandomForestで「Nバー先のClose上昇 vs 下落」を分類学習。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from technical import add_all_indicators


FEATURES = [
    "ret1", "ret5", "ret10",
    "RSI14", "MACD", "MACDsig", "MACDhist",
    "SMA20_dist", "SMA50_dist", "SMA200_dist",
    "BB_pos", "ATR14", "StK", "StD",
    "vol_ratio",
]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = add_all_indicators(df).copy()
    x["ret1"] = x["Close"].pct_change(1)
    x["ret5"] = x["Close"].pct_change(5)
    x["ret10"] = x["Close"].pct_change(10)
    x["SMA20_dist"] = (x["Close"] - x["SMA20"]) / x["Close"]
    x["SMA50_dist"] = (x["Close"] - x["SMA50"]) / x["Close"]
    x["SMA200_dist"] = (x["Close"] - x["SMA200"]) / x["Close"]
    bb_range = (x["BBu"] - x["BBl"]).replace(0, np.nan)
    x["BB_pos"] = (x["Close"] - x["BBl"]) / bb_range
    vol_mean = x["Volume"].rolling(20).mean().replace(0, np.nan)
    x["vol_ratio"] = x["Volume"] / vol_mean
    return x


@dataclass
class Prediction:
    direction: str   # "上昇" / "下落"
    proba_up: float  # 0~1
    confidence: float  # 0~1 (|0.5-proba|*2)
    accuracy: float  # ホールドアウト精度
    n_train: int
    n_test: int


def train_and_predict(df: pd.DataFrame, horizon: int = 4) -> Prediction:
    """horizon バー先の Close が上がっているかを分類学習し、最後尾を予測"""
    if len(df) < 250:
        raise ValueError("学習に十分なデータがありません(250本以上必要)")

    x = _build_features(df).dropna()
    if len(x) < 200:
        raise ValueError("有効サンプルが不足しています")

    y = (x["Close"].shift(-horizon) > x["Close"]).astype(int)
    feat = x[FEATURES]
    valid = feat.notna().all(axis=1) & y.notna()
    feat = feat[valid].iloc[:-horizon]
    y = y[valid].iloc[:-horizon]

    X_train, X_test, y_train, y_test = train_test_split(
        feat, y, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    acc = float(model.score(X_test, y_test))

    last = _build_features(df).iloc[[-1]][FEATURES]
    if last.isna().any().any():
        last = last.fillna(method="ffill").fillna(0)
    proba_up = float(model.predict_proba(last)[0, 1])
    direction = "上昇" if proba_up >= 0.5 else "下落"
    confidence = abs(proba_up - 0.5) * 2
    return Prediction(
        direction=direction,
        proba_up=proba_up,
        confidence=confidence,
        accuracy=acc,
        n_train=len(X_train),
        n_test=len(X_test),
    )


def feature_importance(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
    x = _build_features(df).dropna()
    y = (x["Close"].shift(-horizon) > x["Close"]).astype(int)
    feat = x[FEATURES].iloc[:-horizon]
    y = y.iloc[:-horizon]
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(feat, y)
    return pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
