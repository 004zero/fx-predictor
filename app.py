"""FX/Gold/BTC 予想ダッシュボード - リアルタイム エントリー指示

起動:
    streamlit run app.py
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_fetcher import fetch_ohlc
from technical import add_all_indicators, signal_summary
from technical import atr as atr_func
from fundamental import FundamentalProvider
from predictor import train_and_predict, feature_importance
from signal_engine import integrate
from risk_manager import calc_plan
from entry_levels import compute_live_snapshot
from backtest import run_backtest
from settings import CFG


@st.cache_resource
def load_config():
    return CFG


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
@media (max-width: 768px) {
    .block-container {padding: 0.6rem 0.6rem 4rem 0.6rem !important; max-width: 100% !important;}
    h1 {font-size: 1.4rem !important;}
    h2 {font-size: 1.15rem !important;}
    h3 {font-size: 1.0rem !important;}
    [data-testid="stMetricLabel"] {font-size: 0.72rem !important;}
    [data-testid="stMetricValue"] {font-size: 1.05rem !important;}
    [data-testid="stMetricDelta"] {font-size: 0.7rem !important;}
    .stTabs [data-baseweb="tab-list"] {flex-wrap: wrap; gap: 4px;}
    .stTabs [data-baseweb="tab"] {font-size: 0.78rem; padding: 6px 10px;}
    button[kind="primary"], .stButton button {min-height: 44px;}
    div[data-testid="stHorizontalBlock"] {gap: 0.3rem !important;}
}
.block-container {padding-top: 0.8rem;}
[data-testid="stSidebarNav"] {display:none;}

.action-banner {
    padding: 18px 16px; border-radius: 14px; margin: 10px 0;
    font-size: 1.4rem; font-weight: 800; text-align: center;
    letter-spacing: 0.05em; box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.action-buy {background: linear-gradient(135deg,#0f4d20,#26a69a); color:#fff;}
.action-sell {background: linear-gradient(135deg,#6b1010,#ef5350); color:#fff;}
.action-wait {background: linear-gradient(135deg,#2a2f3a,#5f6b7a); color:#fff;}
.action-detail {
    font-size: 0.85rem; font-weight: 500; margin-top: 6px; opacity: 0.95;
}
.zone-card {
    border-radius: 10px; padding: 10px 12px; margin: 6px 0;
    border-left: 4px solid; background: rgba(255,255,255,0.04);
}
.zone-buy {border-color: #26a69a;}
.zone-sell {border-color: #ef5350;}
.zone-title {font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;}
.zone-range {font-family: monospace; font-size: 1.05rem;}
.zone-sub {font-size: 0.78rem; opacity: 0.75; margin-top: 4px;}
.zone-here {
    background: linear-gradient(90deg, rgba(38,166,154,0.25), transparent);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% {box-shadow: 0 0 0 0 rgba(38,166,154,0.7);}
    70% {box-shadow: 0 0 0 8px rgba(38,166,154,0);}
    100% {box-shadow: 0 0 0 0 rgba(38,166,154,0);}
}
</style>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0e1117">
"""


def chart(df: pd.DataFrame, title: str, snapshot, decimals: int) -> go.Figure:
    d = add_all_indicators(df).tail(200)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.19, 0.19],
        vertical_spacing=0.03,
        subplot_titles=(title, "MACD", "RSI"),
    )
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350", showlegend=False,
    ), row=1, col=1)
    for col, color in [("SMA20", "#42a5f5"), ("SMA50", "#ffa726"), ("SMA200", "#ab47bc")]:
        if col in d:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col,
                          line=dict(width=1, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BBu"], showlegend=False,
                              line=dict(width=1, dash="dot", color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BBl"], showlegend=False,
                              line=dict(width=1, dash="dot", color="gray"),
                              fill="tonexty", fillcolor="rgba(128,128,128,0.08)"), row=1, col=1)

    # Buy / Sell ゾーンを水平帯で表示
    for z in snapshot.zones:
        color = "rgba(38,166,154,0.18)" if z.kind == "buy" else "rgba(239,83,80,0.18)"
        line = "#26a69a" if z.kind == "buy" else "#ef5350"
        fig.add_hrect(y0=z.low, y1=z.high, fillcolor=color, line_width=0, row=1, col=1)
        fig.add_hline(y=(z.low + z.high) / 2, line_width=1, line_dash="dot",
                      line_color=line, row=1, col=1,
                      annotation_text=f"{'買' if z.kind=='buy' else '売'} {z.confidence}%",
                      annotation_position="right",
                      annotation_font_size=10)

    # 現在価格ライン
    last_price = snapshot.price
    fig.add_hline(y=last_price, line_width=1, line_color="#ffeb3b",
                  row=1, col=1,
                  annotation_text=f"現値 {last_price:.{decimals}f}",
                  annotation_position="left", annotation_font_size=10)

    fig.add_trace(go.Bar(x=d.index, y=d["MACDhist"], showlegend=False,
                          marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in d["MACDhist"].fillna(0)]), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], showlegend=False, line=dict(color="#42a5f5", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACDsig"], showlegend=False, line=dict(color="#ffa726", width=1)), row=2, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["RSI14"], showlegend=False, line=dict(color="#ab47bc", width=1)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=3, col=1)

    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=4, r=4, t=30, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def action_banner(action: str, bias: str, note: str):
    if "買い" in action:
        cls, icon = "action-buy", "🟢 BUY"
    elif "売り" in action:
        cls, icon = "action-sell", "🔴 SELL"
    else:
        cls, icon = "action-wait", "⚪ WAIT"
    st.markdown(
        f'<div class="action-banner {cls}">{icon} ｜ {action}'
        f'<div class="action-detail">{note}</div></div>',
        unsafe_allow_html=True,
    )


def zone_card(z, is_here: bool, decimals: int):
    cls = "zone-buy" if z.kind == "buy" else "zone-sell"
    here = " zone-here" if is_here else ""
    icon = "🟢" if z.kind == "buy" else "🔴"
    here_tag = "  📍 価格はここ！" if is_here else ""
    st.markdown(
        f'<div class="zone-card {cls}{here}">'
        f'<div class="zone-title">{icon} {z.label}　信頼度{z.confidence}%{here_tag}</div>'
        f'<div class="zone-range">エントリー: {z.low:.{decimals}f} 〜 {z.high:.{decimals}f}</div>'
        f'<div class="zone-sub">TP: {z.target:.{decimals}f}　SL: {z.stop:.{decimals}f}　RR: 1:{z.rr}</div>'
        f'<div class="zone-sub">{z.reason}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="FX予想", layout="centered", page_icon="📈",
                       initial_sidebar_state="collapsed")
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    cfg = load_config()
    fp = fundamental_provider(cfg)

    st.title("📈 FX/Gold/BTC 予想")

    # --- 銘柄&時間足クイック選択 (Python3.14互換性のためformat_func不使用) ---
    pair_label_to_key = {}
    for k, v in cfg["pairs"].items():
        if isinstance(v, dict):
            label = str(v.get("label", k))
        else:
            label = str(k)
        pair_label_to_key[label] = k
    pair_labels = list(pair_label_to_key.keys())

    interval_label_to_key = {}
    for k, v in cfg["intervals"].items():
        if isinstance(v, dict):
            label = str(v.get("label", k))
        else:
            label = str(k)
        interval_label_to_key[label] = k
    interval_labels = list(interval_label_to_key.keys())

    qc1, qc2 = st.columns(2)
    with qc1:
        selected_pair_label = st.selectbox(
            "銘柄",
            pair_labels,
            index=0,
            key="pair_label",
        )
    with qc2:
        selected_interval_label = st.selectbox(
            "時間足",
            interval_labels,
            index=min(2, len(interval_labels) - 1),
            key="interval_label",
        )
    pair = pair_label_to_key[selected_pair_label]
    interval = interval_label_to_key[selected_interval_label]

    pair_cfg = cfg["pairs"][pair]
    symbol = pair_cfg["symbol"]
    period = cfg["intervals"][interval]["period"]
    pip = float(pair_cfg["pip"])
    decimals = 5 if pip < 0.01 else (3 if pip < 1 else 1)

    # --- サイドバー ---
    with st.sidebar:
        st.header("⚙️ 設定")
        horizon = st.slider("予想バー数 (Nバー先)", 1, 24, 4)
        st.divider()
        st.subheader("💰 資金管理")
        bal = st.number_input("口座残高 (円)",
                                value=int(cfg["risk"]["account_balance_jpy"]), step=10000)
        risk_pct = st.slider("1トレード許容リスク %", 0.1, 5.0,
                              float(cfg["risk"]["risk_per_trade_pct"]), 0.1)
        rr = st.slider("リスクリワード比", 1.0, 5.0, 2.0, 0.1)
        st.divider()
        st.subheader("🎯 シグナルフィルタ")
        min_conf = st.slider(
            "最低信頼度 (%)", 50, 95, 70, 5,
            help="この信頼度未満のシグナルは表示しません。デフォルト70%以上のみ通知。",
        )
        st.divider()
        auto_refresh = st.toggle(
            "自動更新 (シグナル/ゾーン再計算)",
            value=False,
            help="ON にすると指定秒ごとに全体リロード。LIVEチャートは常時リアルタイムなのでOFF推奨。",
        )
        refresh_sec = st.slider("更新間隔(秒)", 15, 300,
                                 int(cfg.get("refresh", {}).get("auto_refresh_seconds", 60)), 15)
        st.divider()
        do_bt = st.checkbox("バックテスト実行", value=False)
        bt_threshold = st.slider("BT 閾値", 0.5, 0.9, 0.6, 0.01)
        if st.button("🔄 キャッシュクリア", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # --- データ取得 ---
    with st.spinner(f"{pair_cfg['label']} データ取得中..."):
        try:
            df = fetch_ohlc(symbol, interval=interval, period=period)
        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            return

    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    chg = last_close - prev_close
    chg_pct = chg / prev_close * 100

    # --- 現値カード ---
    c1, c2 = st.columns(2)
    c1.metric(pair_cfg["label"], f"{last_close:.{decimals}f}",
               f"{chg:+.{decimals}f} ({chg_pct:+.2f}%)")
    c2.metric("更新", df.index[-1].strftime("%m/%d %H:%M"),
               cfg["intervals"][interval]["label"])

    # --- 予想計算 ---
    tech = signal_summary(add_all_indicators(df).dropna())
    try:
        pred = train_and_predict(df, horizon=horizon)
        ml_ok = True
    except Exception as e:
        ml_ok = False
        pred = None

    with st.spinner("ファンダ取得中..."):
        # ファンダはFXペアにのみ適用 (BTC/GOLDは経済指標影響あるが羊飼いは為替主体)
        if pair in ("USDJPY", "EURUSD"):
            fund = fp.fundamental_bias(pair)
        else:
            fund = {"bias": 0, "risk_today": 0, "events": []}

    sig = integrate(tech, pred.proba_up if ml_ok else 0.5, fund)

    # --- リアルタイム エントリースナップショット ---
    snap = compute_live_snapshot(df, pair_cfg, sig.final_score, min_confidence=min_conf)

    # ★ メイン: 「今すぐどうする」 ★
    action_banner(snap.in_zone_action, snap.bias, snap.note)

    # --- 主要メトリクス 2x2 ---
    m1, m2 = st.columns(2)
    m1.metric("総合方向", snap.bias, f"score {sig.final_score:+.0f}")
    m2.metric("テクニカル", tech["verdict"], f"{tech['score']:+.0f}")
    m3, m4 = st.columns(2)
    if ml_ok:
        m3.metric("ML上昇確率", f"{pred.proba_up*100:.1f}%", f"精度{pred.accuracy*100:.1f}%")
    else:
        m3.metric("ML", "学習中")
    if pair in ("USDJPY", "EURUSD"):
        m4.metric("ファンダ", f"{sig.fund_score:+.0f}", f"当日リスク{sig.risk_today}")
    else:
        atr_val = float(atr_func(df).iloc[-1])
        m4.metric("ATR(14)", f"{atr_val:.{decimals}f}")

    if sig.risk_today >= 30:
        st.warning("⚠️ 本日は重要指標・要人発言あり。エントリーは指標前後を避けて。")

    # --- エントリーゾーン一覧 ---
    st.markdown(f"### 🎯 今のエントリーゾーン (信頼度{min_conf}%以上)")
    if not snap.zones:
        st.info(
            f"📭 信頼度{min_conf}%以上のシグナルなし。"
            f"様子見推奨。"
            f"（サイドバーで閾値を下げると条件が緩くなります）"
        )
    for z in snap.zones:
        is_here = (snap.in_zone is not None and z.label == snap.in_zone.label)
        zone_card(z, is_here, decimals)

    # 推奨ロット
    atr_val = float(atr_func(df).iloc[-1])
    plan_dir = "買い" if sig.final_score >= 0 else "売り"
    plan = calc_plan(pair_cfg, last_close, plan_dir, atr_val, bal, risk_pct, rr)
    st.markdown(
        f"💡 **推奨ロット ({plan_dir}):** {plan.units} {plan.units_label}　"
        f"SL/TP: {plan.sl_points:.0f}/{plan.tp_points:.0f}pt　"
        f"最大損失: {plan.risk_jpy:,.0f}円"
    )

    # --- タブ ---
    tab_live, tab_chart, tab_levels, tab_fund, tab_ml, tab_bt = st.tabs(
        ["🔴LIVE", "📊チャート", "📏レベル", "📰ファンダ", "🤖ML", "🧪BT"]
    )

    with tab_live:
        st.caption("🔴 ライブチャート (10秒ごと自動更新・5分足で直近フロー表示)")

        @st.fragment(run_every="10s")
        def live_chart_fragment():
            # 5分足の直近データを取得して描画 (LIVEっぽい表示)
            try:
                live_df = fetch_ohlc(symbol, interval="5m", period="5d")
            except Exception as e:
                st.error(f"ライブデータ取得失敗: {e}")
                return
            now_str = datetime.now().strftime("%H:%M:%S")
            cur = float(live_df["Close"].iloc[-1])
            prev = float(live_df["Close"].iloc[-2])
            chg = cur - prev
            chg_pct = chg / prev * 100
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("現値", f"{cur:.{decimals}f}", f"{chg:+.{decimals}f} ({chg_pct:+.3f}%)")
            lc2.metric("足", "5分", f"直近{len(live_df)}本")
            lc3.metric("更新", now_str)

            d = live_df.tail(120)
            fig = go.Figure(go.Candlestick(
                x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                name="価格",
            ))
            # サポート/レジスタンス水平線
            for r in snap.resistance[:3]:
                fig.add_hline(y=r, line_color="#ef5350", line_width=1, line_dash="dot",
                              annotation_text=f"R {r:.{decimals}f}", annotation_position="right",
                              annotation_font_size=10)
            for s in snap.support[:3]:
                fig.add_hline(y=s, line_color="#26a69a", line_width=1, line_dash="dot",
                              annotation_text=f"S {s:.{decimals}f}", annotation_position="right",
                              annotation_font_size=10)
            # 現値ライン
            fig.add_hline(y=cur, line_color="#ffeb3b", line_width=1,
                          annotation_text=f"現値 {cur:.{decimals}f}",
                          annotation_position="left", annotation_font_size=10)
            # ゾーン
            for z in snap.zones[:2]:
                color = "rgba(38,166,154,0.18)" if z.kind == "buy" else "rgba(239,83,80,0.18)"
                fig.add_hrect(y0=z.low, y1=z.high, fillcolor=color, line_width=0)
            fig.update_layout(
                height=460,
                xaxis_rangeslider_visible=False,
                margin=dict(l=4, r=4, t=20, b=4),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=10),
                showlegend=False,
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
            st.plotly_chart(fig, use_container_width=True,
                             config={"displayModeBar": False, "scrollZoom": False})

        live_chart_fragment()
        st.caption("💡 サポート(緑線)/レジスタンス(赤線)はこのアプリの計算値です")

    with tab_chart:
        st.caption("📊 エントリーゾーン (緑=買い帯/赤=売り帯) 付き静的チャート")
        st.plotly_chart(
            chart(df, f"{pair_cfg['label']} {cfg['intervals'][interval]['label']}",
                  snap, decimals),
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False},
        )

    with tab_levels:
        st.subheader("📏 サポート / レジスタンス")
        col_s, col_r = st.columns(2)
        with col_r:
            st.markdown("**🔴 レジスタンス (上値抵抗)**")
            if snap.resistance:
                for r in snap.resistance:
                    diff = r - snap.price
                    st.markdown(f"- `{r:.{decimals}f}` (+{diff:.{decimals}f})")
            else:
                st.write("該当なし")
        with col_s:
            st.markdown("**🟢 サポート (下値支持)**")
            if snap.support:
                for s in snap.support:
                    diff = snap.price - s
                    st.markdown(f"- `{s:.{decimals}f}` (-{diff:.{decimals}f})")
            else:
                st.write("該当なし")
        st.divider()
        with st.expander("判断根拠の詳細"):
            for r in sig.reason:
                st.markdown(f"- {r}")

    with tab_fund:
        if pair not in ("USDJPY", "EURUSD"):
            st.info("Gold / BTC は羊飼いブログのカバー外です。為替主体ペアで参照されます。")
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
            with st.expander("今週分すべて"):
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
            try:
                imp = feature_importance(df, horizon=horizon)
                st.bar_chart(imp, height=240)
            except Exception:
                pass
        else:
            st.info("MLモデル利用不可。")

    with tab_bt:
        if do_bt:
            with st.spinner("バックテスト中..."):
                try:
                    res = run_backtest(df, pair, horizon=horizon,
                                       train_window=min(800, max(300, len(df) // 2)),
                                       step=50, threshold=bt_threshold)
                    b1, b2 = st.columns(2)
                    b1.metric("勝率", f"{res.win_rate*100:.1f}%")
                    b2.metric("トレード数", res.trades)
                    b3, b4 = st.columns(2)
                    b3.metric("平均pt", f"{res.avg_pips:+.1f}")
                    b4.metric("合計pt", f"{res.total_pips:+.1f}")
                    st.metric("最大DD", f"{res.max_dd_pips:.1f}")
                    st.line_chart(res.equity_curve, height=220)
                except Exception as e:
                    st.error(f"BT失敗: {e}")
        else:
            st.info("サイドバーで「バックテスト実行」をONにしてください。")

    st.divider()
    st.caption(
        f"⚠️ 投資判断の補助情報です。最終判断はご自身で。"
        f"次回更新: {refresh_sec}秒後 (自動更新{'ON' if auto_refresh else 'OFF'})"
    )

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


if __name__ == "__main__":
    main()
