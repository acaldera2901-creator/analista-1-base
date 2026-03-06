"""
Economic Calendar Scraper — uses TradingView's public economic calendar API
as primary source (equivalent to ForexFactory) plus Investing.com RSS.

TradingView calendar endpoint is publicly accessible with proper headers.
Covers all major currencies: USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD.
"""

import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import REQUEST_HEADERS, REQUEST_TIMEOUT

FOREXFACTORY_URL = "https://www.forexfactory.com"

# TradingView calendar API
TV_CALENDAR_URL = "https://economic-calendar.tradingview.com/events"

# TradingView importance: 1=high, 0=medium, -1=low
IMPORTANCE_MAP = {1: "high", 0: "medium", -1: "low"}

# Currencies we care about (our instruments + Gold/BTC drivers)
MONITORED_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "CHF", "CAD", "NZD"}


@dataclass
class ForexFactorySnapshot:
    timestamp: str
    events_today: list
    events_week: list
    high_impact_upcoming: list
    recurring_themes: list
    currencies_in_focus: list
    raw_text: str


class ForexFactoryScraper:
    """Fetches economic calendar from TradingView API (public endpoint)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            **REQUEST_HEADERS,
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.tradingview.com/",
            "Origin": "https://www.tradingview.com",
        })

    # TradingView uses ISO country codes (2-letter), not currency codes
    TV_COUNTRIES = "US,EU,GB,JP,AU,CA,CH,NZ"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_calendar(self, from_dt: datetime, to_dt: datetime, min_importance: int = 1) -> list[dict]:
        """Fetch events from TradingView economic calendar."""
        params = {
            "from": from_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "to": to_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "countries": self.TV_COUNTRIES,
        }
        try:
            r = self.session.get(TV_CALENDAR_URL, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            events = data.get("result", [])
            logger.debug(f"TradingView calendar: {len(events)} events fetched")
            return events
        except Exception as e:
            logger.warning(f"TradingView calendar fetch error: {e}")
            raise

    def _normalize_event(self, ev: dict) -> dict:
        """Normalize a TradingView event to our standard format."""
        importance_raw = ev.get("importance", 1)
        impact = IMPORTANCE_MAP.get(importance_raw, "unknown")
        currency = ev.get("currency", "").upper()

        # Parse datetime
        date_str = ""
        time_str = ""
        try:
            dt_raw = ev.get("date", "")
            if dt_raw:
                dt = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
                # Convert to local display (keep UTC for now)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M")
        except Exception:
            pass

        actual = ev.get("actual")
        forecast = ev.get("forecast")
        previous = ev.get("previous")

        # Format values
        unit = ev.get("unit", "")
        def fmt_val(v):
            if v is None:
                return ""
            try:
                if unit == "%":
                    return f"{v:.2f}%"
                elif abs(float(v)) >= 1000:
                    return f"{v:,.1f}{unit}"
                return f"{v}{unit}"
            except Exception:
                return str(v)

        # Surprise direction
        surprise = "unknown"
        if actual is not None and forecast is not None:
            try:
                a, f = float(actual), float(forecast)
                if a > f:
                    surprise = "beat"
                elif a < f:
                    surprise = "miss"
                else:
                    surprise = "inline"
            except Exception:
                pass

        return {
            "id": str(ev.get("id", "")),
            "date": date_str,
            "time": time_str,
            "currency": currency,
            "impact": impact,
            "title": ev.get("title", ""),
            "indicator": ev.get("indicator", ""),
            "actual": fmt_val(actual),
            "forecast": fmt_val(forecast),
            "previous": fmt_val(previous),
            "surprise": surprise,
            "is_high_impact": impact == "high",
            "period": ev.get("period", ""),
            "source": ev.get("source", ""),
        }

    def _get_currencies_in_focus(self, events: list[dict]) -> list[str]:
        currencies = set()
        for ev in events:
            if ev.get("is_high_impact") and ev.get("currency") in MONITORED_CURRENCIES:
                currencies.add(ev["currency"])
        return sorted(currencies)

    def _extract_recurring_themes(self, events: list[dict]) -> list[str]:
        themes = []
        keywords = {
            "CPI": "Inflazione (CPI)",
            "PPI": "Inflazione Produttori (PPI)",
            "Non-Farm": "Non-Farm Payrolls (NFP)",
            "Nonfarm": "Non-Farm Payrolls (NFP)",
            "GDP": "PIL / GDP",
            "PMI": "Attività Economica (PMI)",
            "FOMC": "Decisione Fed (FOMC)",
            "ECB": "Decisione BCE",
            "BOE": "Decisione Bank of England",
            "BOJ": "Decisione Bank of Japan",
            "Retail Sales": "Vendite al Dettaglio",
            "Interest Rate": "Decisione Tassi",
            "Unemployment": "Mercato del Lavoro",
            "Employment": "Mercato del Lavoro",
            "Trade Balance": "Bilancia Commerciale",
            "Inflation": "Inflazione",
            "Housing": "Mercato Immobiliare",
            "ISM": "Sondaggi ISM",
            "Consumer Confidence": "Fiducia Consumatori",
        }
        found = set()
        for ev in events:
            title = ev.get("title", "") + " " + ev.get("indicator", "")
            for kw, theme in keywords.items():
                if kw.lower() in title.lower() and theme not in found:
                    found.add(theme)
                    themes.append(theme)
        return themes

    def _build_calendar_text(self, events_today: list, events_week: list) -> str:
        lines = [f"=== CALENDARIO ECONOMICO — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ===\n"]

        # Today's events
        today_high = [e for e in events_today if e.get("is_high_impact")]
        today_med = [e for e in events_today if e.get("impact") == "medium"]

        lines.append(f"── OGGI ({len(events_today)} eventi, {len(today_high)} HIGH IMPACT) ──")
        for ev in events_today:
            if ev.get("impact") in ("high", "medium"):
                actual_str = f"  → Actual: {ev['actual']}" if ev.get("actual") else ""
                surprise_str = f" [{ev['surprise'].upper()}]" if ev.get("surprise") not in ("unknown", "") else ""
                lines.append(
                    f"  {ev['time']} [{ev['currency']}] [{ev['impact'].upper()}] {ev['title']}"
                    f"  Prev: {ev.get('previous','')}  Forecast: {ev.get('forecast','')}"
                    f"{actual_str}{surprise_str}"
                )

        # This week — high impact only
        week_high = [e for e in events_week if e.get("is_high_impact") and e.get("currency") in MONITORED_CURRENCIES]
        if week_high:
            lines.append(f"\n── SETTIMANA — HIGH IMPACT ({len(week_high)} eventi) ──")
            for ev in week_high[:25]:
                lines.append(
                    f"  {ev['date']} {ev['time']} [{ev['currency']}] {ev['title']}"
                    f"  Forecast: {ev.get('forecast','')}  Prec: {ev.get('previous','')}"
                )

        return "\n".join(lines)

    def fetch_calendar(self) -> ForexFactorySnapshot:
        logger.info("Fetching economic calendar from TradingView...")
        timestamp = datetime.utcnow().isoformat()

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        week_end = today_start + timedelta(days=7)

        # Fetch today
        events_today_raw = []
        try:
            events_today_raw = self._fetch_calendar(today_start, today_end)
        except Exception as e:
            logger.error(f"Today calendar fetch failed: {e}")

        time.sleep(0.8)

        # Fetch this week
        events_week_raw = []
        try:
            events_week_raw = self._fetch_calendar(today_start, week_end)
        except Exception as e:
            logger.error(f"Week calendar fetch failed: {e}")

        # Normalize
        events_today = [self._normalize_event(e) for e in events_today_raw]
        events_week = [self._normalize_event(e) for e in events_week_raw]

        # High impact upcoming (next 48h)
        two_days_end = today_start + timedelta(days=2)
        high_impact = [
            ev for ev in events_week
            if ev.get("is_high_impact") and ev.get("currency") in MONITORED_CURRENCIES
        ]

        recurring_themes = self._extract_recurring_themes(events_week)
        currencies_in_focus = self._get_currencies_in_focus(events_week)
        raw_text = self._build_calendar_text(events_today, events_week)

        logger.info(
            f"Calendar: {len(events_today)} today, {len(events_week)} this week, "
            f"{len(high_impact)} high-impact, currencies: {currencies_in_focus}"
        )

        return ForexFactorySnapshot(
            timestamp=timestamp,
            events_today=events_today,
            events_week=events_week,
            high_impact_upcoming=high_impact,
            recurring_themes=recurring_themes,
            currencies_in_focus=currencies_in_focus,
            raw_text=raw_text,
        )

    def to_dict(self, snapshot: ForexFactorySnapshot) -> dict:
        return asdict(snapshot)
