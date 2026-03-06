"""
ForexFactory scraper — fetches the economic calendar, high-impact events,
and recurring news items from forexfactory.com
"""

import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import FOREXFACTORY_URL, REQUEST_HEADERS, REQUEST_TIMEOUT


@dataclass
class ForexEvent:
    date: str
    time: str
    currency: str
    impact: str          # low / medium / high / holiday
    title: str
    actual: str
    forecast: str
    previous: str
    surprise: str        # beat / miss / inline / unknown
    is_high_impact: bool


@dataclass
class ForexFactorySnapshot:
    timestamp: str
    events_today: list[dict]
    events_week: list[dict]
    high_impact_upcoming: list[dict]
    recurring_themes: list[str]
    currencies_in_focus: list[str]
    raw_text: str


# Map ForexFactory impact indicators to labels
IMPACT_MAP = {
    "red": "high",
    "orange": "medium",
    "yellow": "low",
    "gray": "holiday",
    "": "unknown",
}

# Currencies tied to our instruments
MONITORED_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "CHF", "CAD", "NZD", "XAU", "BTC"}


class ForexFactoryScraper:
    """Fetches the economic calendar from ForexFactory."""

    BASE_URL = FOREXFACTORY_URL

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)
        # ForexFactory needs these extra headers
        self.session.headers.update({
            "Referer": "https://www.forexfactory.com/",
            "Cache-Control": "no-cache",
        })

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=3, max=15))
    def _fetch(self, path: str = "/") -> Optional[BeautifulSoup]:
        url = self.BASE_URL + path
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.warning(f"ForexFactory fetch error for {url}: {e}")
            raise

    def _parse_impact(self, row) -> str:
        """Extract impact level from a calendar row."""
        # ForexFactory uses colored icons/spans for impact
        impact_el = row.find(class_=lambda c: c and "impact" in c.lower())
        if impact_el:
            # Check for icon classes or span colors
            for color, label in IMPACT_MAP.items():
                if color and color in str(impact_el):
                    return label
            text = impact_el.get_text(strip=True).lower()
            for color, label in IMPACT_MAP.items():
                if color in text:
                    return label
        return "unknown"

    def _parse_calendar_row(self, row) -> Optional[dict]:
        """Parse a single row from the FF calendar table."""
        try:
            cells = row.find_all(["td", "th"])
            if len(cells) < 5:
                return None

            # ForexFactory calendar columns:
            # date | time | currency | impact | event | actual | forecast | previous
            event = {
                "date": "",
                "time": "",
                "currency": "",
                "impact": "unknown",
                "title": "",
                "actual": "",
                "forecast": "",
                "previous": "",
                "surprise": "unknown",
                "is_high_impact": False,
            }

            # Extract text from each cell
            texts = [c.get_text(strip=True) for c in cells]

            # The calendar table structure varies; try to extract key fields
            if len(texts) >= 8:
                event["date"] = texts[0]
                event["time"] = texts[1]
                event["currency"] = texts[2].upper()
                event["impact"] = self._parse_impact(row)
                event["title"] = texts[4]
                event["actual"] = texts[5]
                event["forecast"] = texts[6]
                event["previous"] = texts[7]
            elif len(texts) >= 5:
                event["currency"] = texts[1].upper() if len(texts) > 1 else ""
                event["impact"] = self._parse_impact(row)
                event["title"] = texts[3] if len(texts) > 3 else texts[-1]

            event["is_high_impact"] = event["impact"] == "high"

            # Calculate surprise direction
            if event["actual"] and event["forecast"]:
                try:
                    actual = float(event["actual"].replace("%", "").replace("K", "").replace("M", ""))
                    forecast = float(event["forecast"].replace("%", "").replace("K", "").replace("M", ""))
                    if actual > forecast:
                        event["surprise"] = "beat"
                    elif actual < forecast:
                        event["surprise"] = "miss"
                    else:
                        event["surprise"] = "inline"
                except ValueError:
                    event["surprise"] = "unknown"

            if not event["title"]:
                return None
            return event
        except Exception as e:
            logger.debug(f"Row parse error: {e}")
            return None

    def _extract_calendar_text(self, soup: BeautifulSoup) -> str:
        """Extract clean calendar text for Claude."""
        if soup is None:
            return ""
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _get_currencies_in_focus(self, events: list[dict]) -> list[str]:
        """Return currencies with high-impact events."""
        currencies = set()
        for ev in events:
            if ev.get("is_high_impact") and ev.get("currency") in MONITORED_CURRENCIES:
                currencies.add(ev["currency"])
        return sorted(currencies)

    def _extract_recurring_themes(self, events: list[dict]) -> list[str]:
        """Identify recurring economic themes from event titles."""
        themes = []
        keywords = {
            "CPI": "Inflation",
            "PPI": "Producer Inflation",
            "NFP": "Non-Farm Payrolls",
            "GDP": "GDP Growth",
            "PMI": "Business Activity (PMI)",
            "FOMC": "Fed Policy Decision",
            "ECB": "ECB Policy Decision",
            "BOE": "Bank of England Decision",
            "BOJ": "Bank of Japan Decision",
            "Retail Sales": "Consumer Spending",
            "Interest Rate": "Central Bank Rate Decision",
            "Unemployment": "Labor Market",
            "Employment": "Labor Market",
            "Trade Balance": "Trade Balance",
            "Inflation": "Inflation",
        }
        found = set()
        for ev in events:
            title = ev.get("title", "")
            for keyword, theme in keywords.items():
                if keyword.lower() in title.lower() and theme not in found:
                    found.add(theme)
                    themes.append(theme)
        return themes

    def fetch_calendar(self) -> ForexFactorySnapshot:
        """Fetch today's and this week's economic calendar."""
        logger.info("Fetching ForexFactory calendar...")
        timestamp = datetime.utcnow().isoformat()
        all_text_parts = []
        events_today = []
        events_week = []

        # Fetch today's calendar
        try:
            soup_today = self._fetch("/calendar")
            if soup_today:
                text = self._extract_calendar_text(soup_today)
                all_text_parts.append(f"=== FOREXFACTORY CALENDAR (TODAY) ===\n{text}")

                cal_table = soup_today.find("table", class_=lambda c: c and "calendar" in str(c).lower())
                if cal_table:
                    rows = cal_table.find_all("tr")
                    for row in rows:
                        parsed = self._parse_calendar_row(row)
                        if parsed:
                            events_today.append(parsed)
        except Exception as e:
            logger.error(f"ForexFactory today fetch failed: {e}")

        time.sleep(1)

        # Fetch this week's calendar
        try:
            soup_week = self._fetch("/calendar?week=this")
            if soup_week:
                text = self._extract_calendar_text(soup_week)
                all_text_parts.append(f"\n=== FOREXFACTORY CALENDAR (THIS WEEK) ===\n{text}")

                cal_table = soup_week.find("table", class_=lambda c: c and "calendar" in str(c).lower())
                if cal_table:
                    rows = cal_table.find_all("tr")
                    for row in rows:
                        parsed = self._parse_calendar_row(row)
                        if parsed:
                            events_week.append(parsed)
        except Exception as e:
            logger.error(f"ForexFactory week fetch failed: {e}")

        # Identify high impact upcoming
        high_impact = [
            ev for ev in events_week
            if ev.get("is_high_impact") and ev.get("currency") in MONITORED_CURRENCIES
        ]

        raw_text = "\n".join(all_text_parts)
        logger.info(
            f"ForexFactory: {len(events_today)} events today, "
            f"{len(events_week)} this week, {len(high_impact)} high-impact"
        )

        return ForexFactorySnapshot(
            timestamp=timestamp,
            events_today=events_today,
            events_week=events_week,
            high_impact_upcoming=high_impact,
            recurring_themes=self._extract_recurring_themes(events_week),
            currencies_in_focus=self._get_currencies_in_focus(events_week),
            raw_text=raw_text,
        )

    def to_dict(self, snapshot: ForexFactorySnapshot) -> dict:
        return asdict(snapshot)
