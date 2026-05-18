"""資金管理・ロット計算・SL/TP提案"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradePlan:
    lot_10k: float        # 何万通貨
    sl_price: float
    tp_price: float
    sl_pips: float
    tp_pips: float
    risk_jpy: float
    rr: float             # リスクリワード


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def calc_plan(
    pair: str,
    entry: float,
    direction: str,   # "買い" / "売り"
    atr: float,
    balance_jpy: float,
    risk_pct: float,
    rr: float = 2.0,
    pip_value_jpy_per_10k: float = 100.0,
    sl_atr_mult: float = 1.5,
) -> TradePlan:
    pip = pip_size(pair)
    sl_dist = max(atr * sl_atr_mult, pip * 10)   # 最低10pips
    sl_pips = sl_dist / pip
    tp_pips = sl_pips * rr

    if direction.startswith("買"):
        sl = entry - sl_dist
        tp = entry + sl_dist * rr
    else:
        sl = entry + sl_dist
        tp = entry - sl_dist * rr

    # リスク金額からロット計算 (JPY建てペア前提のpip_value)
    risk_jpy = balance_jpy * risk_pct / 100.0
    lot_10k = risk_jpy / (sl_pips * pip_value_jpy_per_10k)
    lot_10k = max(0.1, round(lot_10k, 1))

    return TradePlan(
        lot_10k=lot_10k,
        sl_price=round(sl, 5),
        tp_price=round(tp, 5),
        sl_pips=round(sl_pips, 1),
        tp_pips=round(tp_pips, 1),
        risk_jpy=round(risk_jpy, 0),
        rr=rr,
    )
