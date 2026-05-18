"""ファンダメンタルズ取得モジュール
羊飼いのFXブログ (kissfx.com) から今週の経済指標・要人発言・分析記事を取得する。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# 通貨キーワード -> 通貨コード
_CCY_MAP = {
    "米": "USD", "アメリカ": "USD", "ドル": "USD", "FRB": "USD", "FOMC": "USD",
    "日": "JPY", "日本": "JPY", "円": "JPY", "日銀": "JPY", "BOJ": "JPY",
    "欧": "EUR", "ユーロ圏": "EUR", "独": "EUR", "ドイツ": "EUR", "ECB": "EUR",
    "英": "GBP", "イギリス": "GBP", "ポンド": "GBP", "BOE": "GBP",
    "豪": "AUD", "豪州": "AUD", "RBA": "AUD",
    "NZ": "NZD", "ニュージーランド": "NZD",
    "加": "CAD", "カナダ": "CAD", "BOC": "CAD",
    "スイス": "CHF", "SNB": "CHF",
    "中国": "CNY", "中": "CNY",
}

_IMPORTANT_KEYWORDS = [
    "雇用統計", "CPI", "消費者物価", "PCE", "PPI", "GDP", "ISM", "PMI",
    "小売売上高", "FOMC", "政策金利", "議事要旨", "議事録",
    "パウエル", "植田", "ラガルド", "ベイリー", "鈴木", "神田",
    "日銀", "総裁", "副総裁", "理事", "総裁会議", "声明",
    "失業率", "求人", "ADP", "ベージュブック", "鉱工業生産",
    "貿易収支", "経常収支", "住宅", "中古住宅", "新築住宅",
    "ミシガン", "コンファレンスボード", "消費者信頼感",
]


@dataclass
class FxEvent:
    date: str          # YYYY-MM-DD
    weekday: str       # 月/火/...
    time: str          # HH:MM or "終日"
    currency: str
    name: str
    importance: int    # 1-3
    impact: str        # 想定インパクト方向 ("円安/円高/双方/不明")
    source_url: str


class FundamentalProvider:
    """羊飼いのFXブログから情報取得。失敗時は空の結果を返す。"""

    def __init__(self, weekly_url: str, blog_url: str, analysis_url: str, cache_minutes: int = 30):
        self.weekly_url = weekly_url
        self.blog_url = blog_url
        self.analysis_url = analysis_url
        self.cache_ttl = cache_minutes * 60
        self._cache: dict[str, tuple[float, object]] = {}

    # ---------- HTTP ----------
    def _get(self, url: str) -> Optional[str]:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return r.text
        except Exception:
            return None

    def _cached(self, key: str):
        v = self._cache.get(key)
        if v and time.time() - v[0] < self.cache_ttl:
            return v[1]
        return None

    def _store(self, key: str, val):
        self._cache[key] = (time.time(), val)

    # ---------- 解析 ----------
    @staticmethod
    def _detect_currency(text: str) -> str:
        for k, v in _CCY_MAP.items():
            if k in text:
                return v
        return "OTH"

    @staticmethod
    def _detect_importance(text: str) -> int:
        score = 0
        for kw in _IMPORTANT_KEYWORDS:
            if kw in text:
                score += 1
        if "雇用統計" in text or "FOMC" in text or "CPI" in text or "政策金利" in text:
            return 3
        if score >= 2:
            return 3
        if score == 1:
            return 2
        return 1

    @staticmethod
    def _detect_impact(text: str, ccy: str) -> str:
        # 簡易ヒューリスティック: 文中のニュアンスは判別困難なので双方扱い
        if "利上げ" in text:
            return f"{ccy}買い"
        if "利下げ" in text:
            return f"{ccy}売り"
        return "双方"

    # ---------- 今週の経済指標・要人発言 ----------
    def fetch_weekly_events(self) -> list[FxEvent]:
        cached = self._cached("weekly")
        if cached is not None:
            return cached

        # 羊飼いの週間スケジュールは年単位の集約ページ。最新の記事リンクを辿る。
        index_html = self._get(self.weekly_url)
        events: list[FxEvent] = []
        latest_url: Optional[str] = None
        if index_html:
            soup = BeautifulSoup(index_html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if re.search(r"\d{4}/\d{2}/\d{2}", href) and "kissfx" in href:
                    latest_url = href
                    break

        target_html = self._get(latest_url) if latest_url else None
        if not target_html:
            target_html = self._get(self.blog_url)
        if not target_html:
            self._store("weekly", events)
            return events

        soup = BeautifulSoup(target_html, "lxml")
        body = soup.find("article") or soup.body
        text = body.get_text("\n", strip=True) if body else ""

        # 行ベースで「月曜日」～「金曜日」のブロックを抽出
        weekday_pattern = re.compile(r"^(月|火|水|木|金)曜日?")
        time_pattern = re.compile(r"(\d{1,2}:\d{2})")
        today = datetime.now()
        # 直近月曜
        monday = today - timedelta(days=today.weekday())
        wk_map = {"月": 0, "火": 1, "水": 2, "木": 3, "金": 4}

        current_wd: Optional[str] = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = weekday_pattern.match(line)
            if m:
                current_wd = m.group(1)
                continue
            if current_wd and any(kw in line for kw in _IMPORTANT_KEYWORDS):
                tm = time_pattern.search(line)
                hhmm = tm.group(1) if tm else "終日"
                ccy = FundamentalProvider._detect_currency(line)
                imp = FundamentalProvider._detect_importance(line)
                impact = FundamentalProvider._detect_impact(line, ccy)
                d = monday + timedelta(days=wk_map[current_wd])
                events.append(FxEvent(
                    date=d.strftime("%Y-%m-%d"),
                    weekday=current_wd,
                    time=hhmm,
                    currency=ccy,
                    name=line[:120],
                    importance=imp,
                    impact=impact,
                    source_url=latest_url or self.blog_url,
                ))

        self._store("weekly", events)
        return events

    # ---------- 相場分析記事の見出し ----------
    def fetch_analysis_headlines(self, limit: int = 10) -> list[dict]:
        cached = self._cached("analysis")
        if cached is not None:
            return cached
        html = self._get(self.analysis_url)
        items: list[dict] = []
        if html:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.select("a"):
                href = a.get("href", "")
                title = a.get_text(strip=True)
                if (
                    href.startswith("https://kissfx.com/")
                    and len(title) > 8
                    and not href.endswith("/fxanalysis/")
                ):
                    items.append({"title": title, "url": href})
                    if len(items) >= limit:
                        break
        self._store("analysis", items)
        return items

    # ---------- ファンダメンタルスコア ----------
    def fundamental_bias(self, pair: str) -> dict:
        """通貨ペア (例 'USDJPY') に対する今週のファンダメンタルバイアスを算出。
        +100=買い / -100=売り。重要指標のリスク量を返す。"""
        if len(pair) != 6:
            return {"bias": 0, "risk_today": 0, "events": []}
        base, quote = pair[:3], pair[3:]
        events = self.fetch_weekly_events()
        today_str = datetime.now().strftime("%Y-%m-%d")
        bias = 0
        risk_today = 0
        related: list[dict] = []
        for ev in events:
            if ev.currency not in (base, quote):
                continue
            if ev.date == today_str:
                risk_today += ev.importance * 10
            related.append(asdict(ev))
            sign = 1 if ev.currency == base else -1
            if f"{ev.currency}買い" in ev.impact:
                bias += sign * 5 * ev.importance
            elif f"{ev.currency}売り" in ev.impact:
                bias -= sign * 5 * ev.importance
        bias = max(-100, min(100, bias))
        return {"bias": bias, "risk_today": risk_today, "events": related}
