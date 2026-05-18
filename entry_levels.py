"""リアルタイム エントリーレベル算出
「今この価格帯で買い/売り」を判定するために、
- 直近のサポート/レジスタンス (ピボット & スイング高安)
- ATR ベースの押し目/戻り目ゾーン
- ボリンジャー & EMA 乖離
- 統合シグナル方向
を組み合わせて買いゾーン/売りゾーンを生成する。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd

from technical import add_all_indicators, atr as atr_func


@dataclass
class Zone:
    kind: Literal["buy", "sell"]
    low: float          # ゾーン下端
    high: float         # ゾーン上端
    target: float       # 推奨利確 (TP)
    stop: float         # 推奨損切 (SL)
    rr: float
    label: str          # 表示用ラベル ("押し目買い", "戻り売り" など)
    confidence: int     # 0~100
    reason: str


@dataclass
class LiveSnapshot:
    price: float
    bias: Literal["買い", "売り", "中立"]
    zones: list[Zone]
    note: str           # 今の状況の一言コメント
    support: list[float]
    resistance: list[float]
    in_zone: Optional[Zone]    # 現在価格が含まれるゾーン
    in_zone_action: str        # "今すぐ買い" "今すぐ売り" "様子見" のいずれか


def _pivot_levels(df: pd.DataFrame) -> dict:
    """前日のH/L/Cからクラシック・ピボット算出"""
    daily = df.resample("1D").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    if len(daily) < 2:
        last = df.iloc[-1]
        return {"P": float(last["Close"]), "R1": np.nan, "R2": np.nan, "S1": np.nan, "S2": np.nan}
    prev = daily.iloc[-2]
    p = (prev["High"] + prev["Low"] + prev["Close"]) / 3
    r1 = 2 * p - prev["Low"]
    s1 = 2 * p - prev["High"]
    r2 = p + (prev["High"] - prev["Low"])
    s2 = p - (prev["High"] - prev["Low"])
    return {
        "P": float(p), "R1": float(r1), "R2": float(r2),
        "S1": float(s1), "S2": float(s2),
    }


def _swing_levels(df: pd.DataFrame, window: int = 5, lookback: int = 60) -> tuple[list[float], list[float]]:
    """直近のスイング高安を抽出してサポート/レジスタンスを返す"""
    recent = df.tail(lookback)
    highs, lows = [], []
    h, l = recent["High"].values, recent["Low"].values
    for i in range(window, len(recent) - window):
        if h[i] == max(h[i - window:i + window + 1]):
            highs.append(float(h[i]))
        if l[i] == min(l[i - window:i + window + 1]):
            lows.append(float(l[i]))
    return sorted(set([round(x, 5) for x in highs]), reverse=True)[:4], \
           sorted(set([round(x, 5) for x in lows]))[:4]


def compute_live_snapshot(
    df: pd.DataFrame,
    pair_cfg: dict,
    integrated_score: float,    # -100~+100
    atr_mult_zone: float = 0.5,
    atr_mult_target: float = 2.0,
    atr_mult_stop: float = 1.5,
) -> LiveSnapshot:
    pip = float(pair_cfg.get("pip", 0.0001))
    decimals = 5 if pip < 0.01 else (3 if pip < 1 else 1)
    x = add_all_indicators(df).dropna()
    if len(x) < 30:
        last = float(df["Close"].iloc[-1])
        return LiveSnapshot(
            price=last, bias="中立", zones=[], note="データ不足",
            support=[], resistance=[], in_zone=None, in_zone_action="様子見",
        )

    last = x.iloc[-1]
    price = float(last["Close"])
    atr = float(atr_func(x).iloc[-1])
    pivots = _pivot_levels(df)
    sw_res, sw_sup = _swing_levels(df)

    # 統合方向
    if integrated_score >= 15:
        bias = "買い"
    elif integrated_score <= -15:
        bias = "売り"
    else:
        bias = "中立"

    # サポート/レジスタンスを統合
    resistance = sorted(set(
        [pivots["R1"], pivots["R2"]] + sw_res + [float(last["BBu"])]
    ))
    support = sorted(set(
        [pivots["S1"], pivots["S2"]] + sw_sup + [float(last["BBl"])]
    ), reverse=True)
    resistance = [r for r in resistance if r > price and not np.isnan(r)][:3]
    support = [s for s in support if s < price and not np.isnan(s)][:3]

    zones: list[Zone] = []

    # === 買いゾーン候補 ===
    # 1) 押し目買い: 現在より下のサポート ± 0.5*ATR
    for i, sup in enumerate(support[:2]):
        z_low = sup - atr * atr_mult_zone
        z_high = sup + atr * atr_mult_zone
        target = price + atr * atr_mult_target if i == 0 else (resistance[0] if resistance else price + atr * atr_mult_target * 1.5)
        stop = sup - atr * atr_mult_stop
        rr = (target - z_high) / max(z_high - stop, 1e-9)
        if rr < 0.8:
            continue
        conf = 70 if bias == "買い" else (50 if bias == "中立" else 30)
        conf -= i * 10
        zones.append(Zone(
            kind="buy",
            low=round(z_low, decimals), high=round(z_high, decimals),
            target=round(target, decimals), stop=round(stop, decimals),
            rr=round(rr, 2),
            label=f"押し目買い#{i+1} (S{i+1}付近)",
            confidence=max(0, min(100, conf)),
            reason=f"サポート{sup:.{decimals}f}付近への押し目。ATR={atr:.{decimals}f}",
        ))

    # === 売りゾーン候補 ===
    for i, res in enumerate(resistance[:2]):
        z_low = res - atr * atr_mult_zone
        z_high = res + atr * atr_mult_zone
        target = price - atr * atr_mult_target if i == 0 else (support[0] if support else price - atr * atr_mult_target * 1.5)
        stop = res + atr * atr_mult_stop
        rr = (z_low - target) / max(stop - z_low, 1e-9)
        if rr < 0.8:
            continue
        conf = 70 if bias == "売り" else (50 if bias == "中立" else 30)
        conf -= i * 10
        zones.append(Zone(
            kind="sell",
            low=round(z_low, decimals), high=round(z_high, decimals),
            target=round(target, decimals), stop=round(stop, decimals),
            rr=round(rr, 2),
            label=f"戻り売り#{i+1} (R{i+1}付近)",
            confidence=max(0, min(100, conf)),
            reason=f"レジスタンス{res:.{decimals}f}付近への戻り。ATR={atr:.{decimals}f}",
        ))

    # === 現在価格がどのゾーンに入っているか ===
    in_zone: Optional[Zone] = None
    for z in zones:
        if z.low <= price <= z.high:
            in_zone = z
            break

    in_zone_action = "様子見"
    if in_zone:
        if in_zone.kind == "buy" and bias != "売り":
            in_zone_action = "今すぐ買い"
        elif in_zone.kind == "sell" and bias != "買い":
            in_zone_action = "今すぐ売り"
        else:
            in_zone_action = f"{in_zone.kind}ゾーン内 (方向と逆なので注意)"

    # コメント生成
    if bias == "買い" and in_zone_action == "今すぐ買い":
        note = "🟢 押し目買いゾーン到達。エントリー検討タイミング。"
    elif bias == "売り" and in_zone_action == "今すぐ売り":
        note = "🔴 戻り売りゾーン到達。エントリー検討タイミング。"
    elif bias == "買い":
        next_buy = next((z for z in zones if z.kind == "buy"), None)
        if next_buy:
            d = next_buy.high - price
            note = f"買い目線。次の押し目買いゾーン {next_buy.low:.{decimals}f}〜{next_buy.high:.{decimals}f} まで {d:+.{decimals}f}"
        else:
            note = "買い目線だが、押し目ゾーンが遠い。"
    elif bias == "売り":
        next_sell = next((z for z in zones if z.kind == "sell"), None)
        if next_sell:
            d = price - next_sell.low
            note = f"売り目線。次の戻り売りゾーン {next_sell.low:.{decimals}f}〜{next_sell.high:.{decimals}f} まで {d:+.{decimals}f}"
        else:
            note = "売り目線だが、戻りゾーンが遠い。"
    else:
        note = "中立。レンジ内、明確なエントリーは見送り推奨。"

    # confidence で並び替え
    zones.sort(key=lambda z: -z.confidence)

    return LiveSnapshot(
        price=price,
        bias=bias,
        zones=zones[:4],
        note=note,
        support=[round(s, decimals) for s in support],
        resistance=[round(r, decimals) for r in resistance],
        in_zone=in_zone,
        in_zone_action=in_zone_action,
    )
