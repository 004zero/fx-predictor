"""テクニカル + ML + ファンダメンタル統合シグナルエンジン"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntegratedSignal:
    final_score: float        # -100 ~ +100
    verdict: str              # 強い買い/買い/中立/売り/強い売り
    tech_score: float
    ml_score: float
    fund_score: float
    risk_today: int
    reason: list[str]


def integrate(tech: dict, ml_proba_up: float, fund: dict, weights=(0.4, 0.4, 0.2)) -> IntegratedSignal:
    """
    tech: technical.signal_summary 戻り値
    ml_proba_up: 0~1 (predictor の proba_up)
    fund: fundamental.fundamental_bias 戻り値
    """
    w_t, w_m, w_f = weights
    tech_score = float(tech.get("score", 0))
    ml_score = (ml_proba_up - 0.5) * 200   # -100~+100
    fund_score = float(fund.get("bias", 0))

    final = w_t * tech_score + w_m * ml_score + w_f * fund_score

    risk_today = int(fund.get("risk_today", 0))
    if risk_today >= 30:
        # 当日に重要指標がある場合は中立寄りに絞る
        final *= 0.6

    if final >= 40:
        verdict = "強い買い"
    elif final >= 15:
        verdict = "買い"
    elif final <= -40:
        verdict = "強い売り"
    elif final <= -15:
        verdict = "売り"
    else:
        verdict = "中立"

    reasons = [
        f"テクニカル: {tech.get('verdict','?')} (score={tech_score:+.0f})",
        f"ML予想: {'上昇' if ml_proba_up>=0.5 else '下落'} (proba_up={ml_proba_up:.2f})",
        f"ファンダ: bias={fund_score:+.0f} / 当日リスク={risk_today}",
    ] + list(tech.get("reasons", []))[:5]

    return IntegratedSignal(
        final_score=round(final, 1),
        verdict=verdict,
        tech_score=tech_score,
        ml_score=round(ml_score, 1),
        fund_score=fund_score,
        risk_today=risk_today,
        reason=reasons,
    )
