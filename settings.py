"""アプリ設定 (YAMLパース問題回避のためPython辞書で直接定義)"""

PAIRS = {
    "USDJPY": {
        "symbol": "USDJPY=X",
        "tv_symbol": "FX:USDJPY",
        "pip": 0.01,
        "point_jpy": 100,
        "label": "ドル円",
        "category": "fx",
    },
    "EURUSD": {
        "symbol": "EURUSD=X",
        "tv_symbol": "FX:EURUSD",
        "pip": 0.0001,
        "point_jpy": 150,
        "label": "ユーロドル",
        "category": "fx",
    },
    "GOLD": {
        "symbol": "GC=F",
        "tv_symbol": "OANDA:XAUUSD",
        "pip": 0.1,
        "point_jpy": 15,
        "label": "ゴールド (XAU/USD)",
        "category": "metal",
    },
    "BITCOIN": {
        "symbol": "BTC-USD",
        "tv_symbol": "BITSTAMP:BTCUSD",
        "pip": 1.0,
        "point_jpy": 150,
        "label": "ビットコイン",
        "category": "crypto",
    },
}

INTERVALS = {
    "5m":  {"period": "5d",  "label": "5分足"},
    "15m": {"period": "10d", "label": "15分足"},
    "1h":  {"period": "60d", "label": "1時間足"},
    "4h":  {"period": "60d", "label": "4時間足"},
    "1d":  {"period": "2y",  "label": "日足"},
}

RISK = {
    "account_balance_jpy": 1_000_000,
    "risk_per_trade_pct": 2.0,
}

FUNDAMENTAL = {
    "weekly_schedule_url": "https://kissfx.com/2025fx/",
    "blog_top_url": "https://kissfx.com/",
    "analysis_url": "https://kissfx.com/fxanalysis/",
    "cache_minutes": 30,
}

PREDICTION = {
    "lookback_bars": 500,
    "feature_window": 20,
    "horizon_bars": 4,
    "model": "random_forest",
    "test_size": 0.2,
}

REFRESH = {
    "auto_refresh_seconds": 60,
}


CFG = {
    "pairs": PAIRS,
    "intervals": INTERVALS,
    "risk": RISK,
    "fundamental": FUNDAMENTAL,
    "prediction": PREDICTION,
    "refresh": REFRESH,
}
