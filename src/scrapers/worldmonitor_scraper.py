"""
Macro Data Aggregator — aggregates macro and market data from multiple
reliable public APIs (equivalent to what worldmonitor.app displays):

  • Yahoo Finance API  → live prices (EURUSD, GBPUSD, USDJPY, XAUUSD, BTC...)
  • FRED (St. Louis Fed) → Fed rate, CPI, unemployment, yields (no key needed)
  • World Bank API    → GDP growth, inflation by country
  • ECB press RSS     → ECB policy statements
  • Investing.com RSS → global economy news
  • MarketWatch RSS   → breaking market bulletins
"""

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import REQUEST_HEADERS, REQUEST_TIMEOUT

WORLDMONITOR_URL = "https://www.worldmonitor.app"

# Yahoo Finance symbol map
YF_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "XAUUSD": "GC=F",
    "BTCUSD": "BTC-USD",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "SP500": "^GSPC",
    "USOIL": "CL=F",
}

# FRED series (no API key needed)
FRED_SERIES = {
    "US Fed Funds Rate": "FEDFUNDS",
    "EUR Money Market Rate": "IRSTCI01EZM156N",
    "GBP Bank Rate": "IRSTCI01GBM156N",
    "JPY Policy Rate": "IRSTCI01JPM156N",
    "US CPI": "CPIAUCSL",
    "US Unemployment": "UNRATE",
    "US GDP Growth (QoQ ann.)": "A191RL1Q225SBEA",
    "US 10Y Yield": "DGS10",
    "US 2Y Yield": "DGS2",
    "US Core PCE": "PCEPILFE",
}


@dataclass
class WorldMonitorSnapshot:
    timestamp: str
    prices: dict
    central_bank_rates: dict
    economic_indicators: dict
    news_headlines: list
    market_context: dict
    raw_text: str


class WorldMonitorScraper:
    """Aggregates macro/market data from public APIs (Yahoo Finance, FRED, World Bank, RSS)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)

    # ── Yahoo Finance ─────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _fetch_yf(self, symbol: str) -> Optional[dict]:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=2d&interval=1d"
        r = self.session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        result = data.get("chart", {}).get("result", [])
        if result:
            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice")
            prev = meta.get("chartPreviousClose")
            chg = round((price - prev) / prev * 100, 4) if price and prev and prev != 0 else None
            return {
                "symbol": meta.get("symbol"),
                "price": price,
                "prev_close": prev,
                "change_pct": chg,
                "currency": meta.get("currency"),
                "market_time": datetime.fromtimestamp(
                    meta.get("regularMarketTime", 0)
                ).strftime("%Y-%m-%d %H:%M") if meta.get("regularMarketTime") else "",
            }
        return None

    def fetch_prices(self) -> dict:
        prices = {}
        for name, symbol in YF_SYMBOLS.items():
            try:
                data = self._fetch_yf(symbol)
                if data:
                    prices[name] = data
                time.sleep(0.25)
            except Exception as e:
                logger.warning(f"Price fetch failed {name}: {e}")
        return prices

    # ── FRED ──────────────────────────────────────────────────────────────────

    def _fetch_fred(self, series_id: str, obs: int = 3) -> Optional[dict]:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            r = self.session.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            lines = r.text.strip().splitlines()[1:]  # skip header
            recent = lines[-obs:]
            result = {}
            for line in recent:
                parts = line.split(",")
                if len(parts) == 2:
                    result[parts[0]] = parts[1]
            return result
        except Exception as e:
            logger.debug(f"FRED error {series_id}: {e}")
            return None

    def fetch_central_bank_rates(self) -> dict:
        rates = {}
        cb_series = {
            "USD_FED (Federal Reserve)": "FEDFUNDS",
            "EUR_ECB": "IRSTCI01EZM156N",
            "GBP_BOE (Bank of England)": "IRSTCI01GBM156N",
            "JPY_BOJ (Bank of Japan)": "IRSTCI01JPM156N",
            "CAD_BOC (Bank of Canada)": "IRSTCI01CAM156N",
            "AUD_RBA (Reserve Bank of Australia)": "IRSTCI01AUM156N",
            "CHF_SNB (Swiss National Bank)": "IRSTCI01CHM156N",
        }
        for name, series_id in cb_series.items():
            data = self._fetch_fred(series_id, obs=2)
            if data:
                items = list(data.items())
                latest = items[-1]
                prev = items[-2] if len(items) > 1 else None
                rates[name] = {
                    "rate": latest[1],
                    "date": latest[0],
                    "previous": prev[1] if prev else None,
                }
            time.sleep(0.3)
        return rates

    def fetch_economic_indicators(self) -> dict:
        indicators = {}
        series_map = {
            "US CPI (nivel)": "CPIAUCSL",
            "US Unemployment Rate (%)": "UNRATE",
            "US GDP Growth QoQ ann. (%)": "A191RL1Q225SBEA",
            "US 10Y Treasury Yield (%)": "DGS10",
            "US 2Y Treasury Yield (%)": "DGS2",
            "US Core PCE": "PCEPILFE",
            "US Initial Jobless Claims": "ICSA",
        }
        for name, series_id in series_map.items():
            data = self._fetch_fred(series_id, obs=3)
            if data:
                items = list(data.items())
                latest = items[-1]
                prev = items[-2] if len(items) > 1 else None
                indicators[name] = {
                    "value": latest[1],
                    "date": latest[0],
                    "previous": prev[1] if prev else None,
                }
            time.sleep(0.3)
        return indicators

    # ── News RSS ──────────────────────────────────────────────────────────────

    def _fetch_rss(self, url: str, max_items: int = 8) -> list[dict]:
        try:
            r = self.session.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            items = []
            for item in root.iter("item"):
                t = item.find("title")
                d = item.find("description")
                dt = item.find("pubDate")
                if t is not None and t.text:
                    items.append({
                        "title": t.text.strip(),
                        "description": (d.text or "")[:200].strip() if d is not None else "",
                        "date": dt.text.strip()[:25] if dt is not None and dt.text else "",
                    })
                if len(items) >= max_items:
                    break
            return items
        except Exception as e:
            logger.debug(f"RSS error {url}: {e}")
            return []

    def fetch_news_headlines(self) -> list[dict]:
        headlines = []
        feeds = [
            ("Investing.com Economy", "https://www.investing.com/rss/news_14.rss"),
            ("Investing.com Forex",   "https://www.investing.com/rss/news_1.rss"),
            ("ECB Press Releases",   "https://www.ecb.europa.eu/rss/press.html"),
            ("MarketWatch Bulletins", "https://feeds.content.dowjones.io/public/rss/mw_bulletins"),
        ]
        for source, feed_url in feeds:
            items = self._fetch_rss(feed_url, max_items=6)
            for item in items:
                item["source"] = source
                headlines.append(item)
            time.sleep(0.4)
        return headlines

    # ── World Bank ────────────────────────────────────────────────────────────

    def fetch_world_bank_inflation(self) -> dict:
        wb = {}
        try:
            r = self.session.get(
                "https://api.worldbank.org/v2/country/US;EU;GB;JP;AU;CA;NZ;CH"
                "/indicator/FP.CPI.TOTL.ZG?format=json&mrv=1",
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            if len(data) > 1 and data[1]:
                for entry in data[1]:
                    code = entry.get("country", {}).get("id", "")
                    val = entry.get("value")
                    year = entry.get("date", "")
                    if code and val is not None:
                        wb[f"Inflation {code}"] = {"value": f"{val:.1f}%", "year": year}
        except Exception as e:
            logger.debug(f"World Bank error: {e}")
        return wb

    # ── Master fetch ──────────────────────────────────────────────────────────

    def fetch_all(self) -> WorldMonitorSnapshot:
        logger.info("Aggregating macro data from Yahoo Finance, FRED, World Bank, RSS...")
        timestamp = datetime.utcnow().isoformat()

        prices = self.fetch_prices()
        logger.debug(f"Prices fetched: {len(prices)}")

        central_banks = self.fetch_central_bank_rates()
        logger.debug(f"Central bank rates: {len(central_banks)}")

        indicators = self.fetch_economic_indicators()
        logger.debug(f"Economic indicators: {len(indicators)}")

        news = self.fetch_news_headlines()
        logger.debug(f"News headlines: {len(news)}")

        # Separate market context from prices
        context_keys = {"VIX", "DXY", "US10Y", "SP500", "USOIL"}
        market_context = {k: v for k, v in prices.items() if k in context_keys}
        instrument_prices = {k: v for k, v in prices.items() if k not in context_keys}

        raw_text = self._build_summary(instrument_prices, central_banks, indicators, news, market_context)
        logger.info(f"Macro data ready: {len(raw_text)} chars")

        return WorldMonitorSnapshot(
            timestamp=timestamp,
            prices=instrument_prices,
            central_bank_rates=central_banks,
            economic_indicators=indicators,
            news_headlines=news,
            market_context=market_context,
            raw_text=raw_text,
        )

    def _build_summary(self, prices, banks, indicators, news, ctx) -> str:
        lines = []

        lines.append(f"=== DATI MACRO AGGREGATI — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ===\n")

        lines.append("── PREZZI LIVE STRUMENTI ──")
        for name, d in prices.items():
            if d:
                chg = d.get("change_pct")
                direction = "▲" if (chg or 0) > 0 else "▼" if (chg or 0) < 0 else "─"
                lines.append(f"  {name:10s} {d.get('price'):<12} {direction} {chg:+.2f}% [{d.get('market_time','')}]"
                             if chg is not None else f"  {name:10s} {d.get('price')}")

        if ctx:
            lines.append("\n── CONTESTO DI MERCATO ──")
            for name, d in ctx.items():
                if d:
                    chg = d.get("change_pct")
                    lines.append(f"  {name:8s} {d.get('price'):<12}" +
                                (f" {chg:+.2f}%" if chg is not None else ""))

        if banks:
            lines.append("\n── TASSI BANCHE CENTRALI (FRED) ──")
            for name, d in banks.items():
                prev = d.get("previous")
                delta = ""
                try:
                    diff = float(d["rate"]) - float(prev)
                    delta = f" (Δ {diff:+.2f})" if abs(diff) > 0.001 else " (invariato)"
                except Exception:
                    pass
                lines.append(f"  {name}: {d.get('rate')}%{delta}  [{d.get('date','')}]")

        if indicators:
            lines.append("\n── INDICATORI MACRO ──")
            for name, d in indicators.items():
                prev = d.get("previous")
                prev_str = f"  |  Prec: {prev}" if prev else ""
                lines.append(f"  {name}: {d.get('value')}  [{d.get('date','')}]{prev_str}")

        if news:
            lines.append("\n── ULTIME NOTIZIE ──")
            seen = set()
            for item in news[:18]:
                t = item.get("title", "")
                if t and t not in seen:
                    seen.add(t)
                    lines.append(f"  [{item.get('source','')}] {t}")

        return "\n".join(lines)

    def to_dict(self, snapshot: WorldMonitorSnapshot) -> dict:
        return asdict(snapshot)
