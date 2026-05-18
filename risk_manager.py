"""資金管理・ロット計算・SL/TP提案 (業者ロット定義対応)"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradePlan:
    lots: float           # 推奨ロット数 (業者ロット単位)
    units: float          # 実通貨数量 (10000通貨 等)
    units_label: str      # 単位ラベル
    sl_price: float
    tp_price: float
    sl_points: float
    tp_points: float
    risk_jpy: float
    rr: float


def pip_size_from_cfg(pair_cfg: dict) -> float:
    return float(pair_cfg.get("pip", 0.0001))


def units_label_for(category: str) -> str:
    return {"fx": "通貨", "metal": "oz", "crypto": "BTC"}.get(category, "単位")


def units_per_lot(pair_cfg: dict, broker_preset: dict) -> float:
    """この銘柄カテゴリで 1ロット が何通貨/oz/BTC に相当するか"""
    cat = pair_cfg.get("category", "fx")
    if cat == "fx":
        return float(broker_preset.get("fx_units_per_lot", 10_000))
    if cat == "metal":
        return float(broker_preset.get("gold_units_per_lot", 10))
    return float(broker_preset.get("btc_units_per_lot", 0.01))


def calc_plan(
    pair_cfg: dict,
    entry: float,
    direction: str,
    atr: float,
    balance_jpy: float,
    risk_pct: float,
    rr: float = 2.0,
    sl_atr_mult: float = 1.5,
    broker_preset: dict | None = None,
) -> TradePlan:
    pip = pip_size_from_cfg(pair_cfg)
    point_jpy = float(pair_cfg.get("point_jpy", 100))   # 1単位×1pipの円損益
    category = pair_cfg.get("category", "fx")

    if broker_preset is None:
        broker_preset = {
            "fx_units_per_lot": 10_000,
            "gold_units_per_lot": 10,
            "btc_units_per_lot": 0.01,
            "min_lot": 0.1,
            "lot_step": 0.1,
        }
    upl = units_per_lot(pair_cfg, broker_preset)
    min_lot = float(broker_preset.get("min_lot", 0.1))
    lot_step = float(broker_preset.get("lot_step", 0.1))

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
    raw_units = risk_jpy / max(sl_points * point_jpy, 1.0)

    # 通貨数量をロット数に変換
    raw_lots = raw_units / upl

    # 業者の最小ロット・刻みに丸める
    lots = max(min_lot, round(raw_lots / lot_step) * lot_step)
    # 浮動小数点の桁数を整える
    lots = round(lots, 4 if lot_step < 0.1 else (2 if lot_step < 1 else 1))

    units = lots * upl

    decimals = 5 if pip < 0.01 else (3 if pip < 1 else 1)
    return TradePlan(
        lots=lots,
        units=units,
        units_label=units_label_for(category),
        sl_price=round(sl, decimals),
        tp_price=round(tp, decimals),
        sl_points=round(sl_points, 1),
        tp_points=round(tp_points, 1),
        risk_jpy=round(risk_jpy, 0),
        rr=rr,
    )
