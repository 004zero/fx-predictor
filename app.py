"""FX予想ダッシュボード - Streamlit UI (スマホ最適化版)

起動:
    streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from data_fetcher import fetch_ohlc
from technical import add_all_indicators, signal_summary
from technical import atr as atr_func
from fundamental import FundamentalProvider
from predictor import train_and_predict, feature_importance
from signal_engine import integrate
from risk_manager import calc_plan, pip_size
from backtest import run_backtest


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


@st.cache_resource
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def fundamental_provider(cfg):
    f = cfg["fundamental"]
    return FundamentalProvider(
        weekly_url=f["weekly_schedule_url"],
        blog_url=f["blog_top_url"],
        analysis_url=f["analysis_url"],
        cache_minutes=f.get("cache_minutes", 30),
    )


MOBILE_CSS = """
<style>
/* スマホ最適化: 横スクロール抑制・余白圧縮・タップしやすいサイズ */
@media (max-width: 768px) {
    .block-container {padding: 0.6rem 0.6rem 4rem 0.6rem !important; max-width: 100% !important;}
    h1 {font-size: 1.4rem !important;}
    h2 {font-size: 1.15rem !important;}
    h3 {font-size: 1.0rem !important;}
    [data-testid="stMetricLabel"] {font-size: 0.75rem !important;}
    [data-testid="stMetricValue"] {font-size: 1.1rem !important;}
    [data-testid="stMetricDelta"] {font-size: 0.75rem !important;}
    .stTabs [data-baseweb="tab-list"] {flex-wrap: wrap; gap: 4px;}
    .stTabs [data-baseweb="tab"] {font-size: 0.8rem; padding: 6px 10px;}
    button[kind="primary"], .stButton button {min-height: 44px;}
    div[data-testid="stHorizontalBlock"] {gap: 0.3rem !important;}
}
/* PCでもコンパクト */
.block-container {padding-top: 1rem;}
[data-testid="stSidebarNav"] {display:none;}
/* タッチ操作向けスクロール */
[data-testid="stDataFrame"] {-webkit-overflow-scrolling: touch;}
/* シグナルカードの強調 */
.signal-banner {
    padding: 12px 16px; border-radius: 12px; margin: 8px 0;
    font-size: 1.1rem; font-weight: 700; text-align: center;
}
.signal-buy {background: linear-gradient(135deg,#1b5e20,#26a69a); color:#fff;}
.signal-sell {background: linear-gradient(135deg,#b71c1c,#ef5350); color:#fff;}
.signal-neutral {background: linear-gradient(135deg,#37474f,#78909c); color:#fff;}
</style>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0e1117">
"""


def chart(df: pd.DataFrame, title: str, compact: bool = False) -> go.Figure:
    d = add_all_indicators(df).tail(200 if compact else 300)
    height = 480 if compact else 700
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=(title, "MACD", "RSI"),
    )
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="価格", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)
    for col, color in [("SMA20", "#42a5f5"), ("SMA50", "#ffa726"), ("SMA200", "#ab47bc")]:
        if col in d:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, line=dict(width=1, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BBu"], name="BB+", line=dict(width=1, dash="dot", color="gray"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BBl"], name="BB-", line=dict(width=1, dash="dot", color="gray"),
                              fill="tonexty", fillcolor="rgba(128,128,128,0.08)", showlegend=False), row=1, col=1)

    fig.add_trace(go.Bar(x=d.index, y=d["MACDhist"], name="Hist", showlegend=False,
                          marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in d["MACDhist"].fillna(0)]), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD", line=dict(color="#42a5f5", width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACDsig"], name="Signal", line=dict(color="#ffa726", width=1), showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["RSI14"], name="RSI", line=dict(color="#ab47bc", width=1), showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=3, col=1)

    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        margin=dict(l=4, r=4, t=30, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def signal_banner(verdict: str, score: float):
    cls = "signal-neutral"
    icon = "⚪"
    if verdict in ("買い", "強い買い"):
        cls = "signal-buy"; icon = "🟢"
    elif verdict in ("売り", "強い売り"):
        cls = "signal-sell"; icon = "🔴"
    st.markdown(
        f'<div class="signal-banner {cls}">{icon} {verdict}　(スコア {score:+.0f})</div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="FX予想",
        layout="centered",   # スマホ向けにcentered
        page_icon="📈",
        initial_sidebar_state="collapsed",
    )
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)

    cfg = load_config()
    fp = fundamental_provider(cfg)

    st.title("📈 FX予想")

    # ---- サイドバー ----
    with st.sidebar:
        st.header("⚙️ 設定")
        pair = st.selectbox("通貨ペア", list(cfg["pairs"].keys()), index=0)
        interval = st.selectbox(
            "時間足",
            list(cfg["intervals"].keys()),
            index=2,
            format_func=lambda k: cfg["intervals"][k]["label"],
        )
        period = cfg["intervals"][interval]["period"]
        horizon = st.slider("予想バー数 (Nバー先)", 1, 24, 4)

        st.divider()
        st.subheader("💰 資金管理")
        bal = st.number_input("口座残高 (円)", value=int(cfg["risk"]["account_balance_jpy"]), step=10000)
        risk_pct = st.slider("1トレード許容リスク %", 0.1, 5.0, float(cfg["risk"]["risk_per_trade_pct"]), 0.1)
        rr = st.slider("リスクリワード比", 1.0, 5.0, 2.0, 0.1)

        st.divider()
        do_bt = st.checkbox("バックテスト実行", value=False)
        bt_threshold = st.slider("BT エントリー閾値", 0.5, 0.9, 0.6, 0.01)

        st.divider()
        if st.button("🔄 キャッシュクリア", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    symbol = cfg["pairs"][pair]

    # ---- クイック選択 (モバイル向け、本文上部) ----
    qc1, qc2 = st.columns(2)
    with qc1:
        pair = st.selectbox(
            "ペア", list(cfg["pairs"].keys()),
            index=list(cfg["pairs"].keys()).index(pair), key="qp_pair", label_visibility="collapsed",
        )
    with qc2:
        interval = st.selectbox(
            "足", list(cfg["intervals"].keys()),
            index=list(cfg["intervals"].keys()).index(interval),
            format_func=lambda k: cfg["intervals"][k]["label"],
            key="qp_int", label_visibility="collapsed",
        )
    symbol = cfg["pairs"][pair]
    period = cfg["intervals"][interval]["period"]

    # ---- データ取得 ----
    with st.spinner(f"{pair} データ取得中..."):
        try:
            df = fetch_ohlc(symbol, interval=interval, period=period)
        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            return

    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    pip = pip_size(pair)
    chg_pips = (last_close - prev_close) / pip

    # ---- 現値カード (2列) ----
    c1, c2 = st.columns(2)
    c1.metric(f"{pair}", f"{last_close:.3f}", f"{chg_pips:+.1f} pips")
    c2.metric("更新", df.index[-1].strftime("%m/%d %H:%M"), cfg["intervals"][interval]["label"])

    # ---- 予想計算 ----
    tech = signal_summary(add_all_indicators(df).dropna())
    try:
        pred = train_and_predict(df, horizon=horizon)
        ml_ok = True
    except Exception as e:
        st.warning(f"ML予想失敗: {e}")
        ml_ok = False
        pred = None

    with st.spinner("ファンダメンタル取得中..."):
        fund = fp.fundamental_bias(pair)

    sig = integrate(tech, pred.proba_up if ml_ok else 0.5, fund)

    # ---- メインシグナル (大きなバナー) ----
    signal_banner(sig.verdict, sig.final_score)

    # サブメトリクス2x2
    m1, m2 = st.columns(2)
    m1.metric("テクニカル", tech["verdict"], f"{tech['score']:+.0f}")
    if ml_ok:
        m2.metric("ML上昇確率", f"{pred.proba_up*100:.1f}%", f"精度 {pred.accuracy*100:.1f}%")
    else:
        m2.metric("ML上昇確率", "N/A")
    m3, m4 = st.columns(2)
    m3.metric("ファンダ", f"{sig.fund_score:+.0f}", f"当日リスク {sig.risk_today}")

    atr_val = float(atr_func(df).iloc[-1])
    direction = "買い" if sig.final_score >= 0 else "売り"
    plan = calc_plan(
        pair=pair, entry=last_close, direction=direction, atr=atr_val,
        balance_jpy=bal, risk_pct=risk_pct, rr=rr,
        pip_value_jpy_per_10k=cfg["risk"]["pip_value_jpy_per_10k"],
    )
    m4.metric("推奨ロット", f"{plan.lot_10k}万", f"SL{plan.sl_pips}/TP{plan.tp_pips}")

    if sig.risk_today >= 30:
        st.warning("⚠️ 本日は重要指標・要人発言あり。指標前後はポジション縮小推奨。")

    with st.expander("📋 判断根拠を見る"):
        for r in sig.reason:
            st.markdown(f"- {r}")

    # ---- タブ ----
    tab_chart, tab_plan, tab_fund, tab_ml, tab_bt = st.tabs(
        ["📊チャート", "💼プラン", "📰ファンダ", "🤖ML", "🧪BT"]
    )

    with tab_chart:
        st.plotly_chart(
            chart(df, f"{pair} {cfg['intervals'][interval]['label']}", compact=True),
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False},
        )

    with tab_plan:
        st.subheader(f"💼 {direction}プラン")
        p1, p2 = st.columns(2)
        p1.metric("エントリー", f"{last_close:.3f}")
        p2.metric("RR", f"1 : {plan.rr}")
        p3, p4 = st.columns(2)
        p3.metric("SL", f"{plan.sl_price:.3f}", f"-{plan.sl_pips}p")
        p4.metric("TP", f"{plan.tp_price:.3f}", f"+{plan.tp_pips}p")
        st.info(
            f"ロット: **{plan.lot_10k}万通貨**\n\n"
            f"最大損失見込み: **{plan.risk_jpy:,.0f}円**\n\n"
            f"(残高{bal:,}円 × リスク{risk_pct}%)"
        )
        st.caption("※ ATR×1.5基準のSL幅。pip価値はJPYクロス前提(100円/pip/万通貨)")

    with tab_fund:
        st.subheader("📰 今週の経済指標・要人発言")
        events = fp.fetch_weekly_events()
        today = datetime.now().strftime("%Y-%m-%d")
        if not events:
            st.info("羊飼いブログから取得できませんでした。")
        else:
            edf = pd.DataFrame([e.__dict__ for e in events])
            edf = edf.sort_values(["date", "time"])
            today_events = edf[edf["date"] == today]
            if len(today_events):
                st.warning(f"📌 本日のイベント {len(today_events)}件")
                st.dataframe(today_events[["time", "currency", "name", "importance"]],
                              use_container_width=True, hide_index=True)
            with st.expander("今週分すべて見る"):
                st.dataframe(edf[["date", "weekday", "time", "currency", "name", "importance"]],
                              use_container_width=True, hide_index=True)

        st.subheader("📝 最新の相場分析")
        for h in fp.fetch_analysis_headlines(6):
            st.markdown(f"- [{h['title']}]({h['url']})")
        st.caption("出典: 羊飼いのFXブログ https://kissfx.com/")

    with tab_ml:
        if ml_ok and pred:
            mlc1, mlc2 = st.columns(2)
            mlc1.metric("予測方向", pred.direction)
            mlc2.metric("信頼度", f"{pred.confidence*100:.0f}%")
            mlc3, mlc4 = st.columns(2)
            mlc3.metric("上昇確率", f"{pred.proba_up*100:.2f}%")
            mlc4.metric("テスト精度", f"{pred.accuracy*100:.1f}%")
            st.caption(f"学習{pred.n_train}本 / テスト{pred.n_test}本")
            st.subheader("特徴量重要度")
            try:
                imp = feature_importance(df, horizon=horizon)
                st.bar_chart(imp, height=240)
            except Exception as e:
                st.info(f"計算失敗: {e}")
        else:
            st.info("MLモデル利用不可。データ本数を増やしてください。")

    with tab_bt:
        if do_bt:
            with st.spinner("バックテスト中..."):
                try:
                    res = run_backtest(
                        df, pair, horizon=horizon,
                        train_window=min(800, max(300, len(df) // 2)),
                        step=50, threshold=bt_threshold,
                    )
                    b1, b2 = st.columns(2)
                    b1.metric("勝率", f"{res.win_rate*100:.1f}%")
                    b2.metric("トレード数", res.trades)
                    b3, b4 = st.columns(2)
                    b3.metric("平均pips", f"{res.avg_pips:+.1f}")
                    b4.metric("合計pips", f"{res.total_pips:+.1f}")
                    st.metric("最大ドローダウン", f"{res.max_dd_pips:.1f} pips")
                    st.line_chart(res.equity_curve, height=220)
                except Exception as e:
                    st.error(f"BT失敗: {e}")
        else:
            st.info("サイドバーで「バックテスト実行」をONにしてください。")

    st.divider()
    st.caption(
        "⚠️ 投資判断の補助情報です。最終判断はご自身の責任で。"
    )


if __name__ == "__main__":
    main()
