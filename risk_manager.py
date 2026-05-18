"""資金管理・ロット計算・SL/TP提案 (FX/Gold/BTC対応)"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradePlan:
    units: float          # ロット (FX: 万通貨 / Gold: oz / BTC: BTC)
    units_label: str      # 単位の表示文字列
    sl_price: float
    tp_price: float
    sl_points: float      # pip/point単位の距離
    tp_points: float
    risk_jpy: float
    rr: float


def pip_size_from_cfg(pair_cfg: dict) -> float:
    return float(pair_cfg.get("pip", 0.0001))


def units_label_for(category: str) -> str:
    return {"fx": "万通貨", "metal": "oz", "crypto": "BTC"}.get(category, "単位")


def calc_plan(
    pair_cfg: dict,
    entry: float,
    direction: str,
    atr: float,
    balance_jpy: float,
    risk_pct: float,
    rr: float = 2.0,
    sl_atr_mult: float = 1.5,
) -> TradePlan:
    pip = pip_size_from_cfg(pair_cfg)
    point_jpy = float(pair_cfg.get("point_jpy", 100))  # 1pip/point動いた時の円損益（1単位あたり）
    category = pair_cfg.get("category", "fx")

    sl_dist = max(atr * sl_atr_mult, pip * 10)
    sl_points = sl_dist / pip
    tp_points = sl_points * rr

    if direction.startswith("買"):
        sl = entry - sl_dist
        tp = entry + sl_dist * rr
    else:
        sl = entry + sl_dist
        tp = entry - sl_dist * rr

    risk_jpy = balance_jpy * risk_pct / 100.0
    # 1単位 × sl_points × point_jpy = リスク
    raw_units = risk_jpy / max(sl_points * point_jpy, 1.0)

    if category == "fx":
        units = max(0.1, round(raw_units, 1))   # 万通貨単位
    elif category == "metal":
        units = max(0.1, round(raw_units, 2))   # oz
    else:  # crypto
        units = max(0.001, round(raw_units, 4)) # BTC

    decimals = 5 if pip < 0.01 else (3 if pip < 1 else 1)
    return TradePlan(
        units=units,
        units_label=units_label_for(category),
        sl_price=round(sl, decimals),
        tp_price=round(tp, decimals),
        sl_points=round(sl_points, 1),
        tp_points=round(tp_points, 1),
        risk_jpy=round(risk_jpy, 0),
        rr=rr,
    )
