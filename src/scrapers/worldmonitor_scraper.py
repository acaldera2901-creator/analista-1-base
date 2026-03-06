"""
WorldMonitor scraper — fetches macro data, economic indicators,
central bank rates, and market sentiment from worldmonitor.app
"""

import time
import json
import asyncio
from datetime import datetime
from typing import Any
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import WORLDMONITOR_URL, REQUEST_HEADERS, REQUEST_TIMEOUT


@dataclass
class EconomicIndicator:
    name: str
    country: str
    value: str
    previous: str
    change: str
    impact: str
    timestamp: str
    url: str


@dataclass
class CentralBankData:
    bank: str
    country: str
    rate: str
    last_change: str
    next_meeting: str
    bias: str  # hawkish / dovish / neutral


@dataclass
class WorldMonitorSnapshot:
    timestamp: str
    indicators: list[dict]
    central_banks: list[dict]
    market_sentiment: dict
    inflation_data: list[dict]
    gdp_data: list[dict]
    employment_data: list[dict]
    raw_text: str  # full page text for Claude to analyze


class WorldMonitorScraper:
    """Fetches macroeconomic data from worldmonitor.app."""

    BASE_URL = WORLDMONITOR_URL
    SECTIONS = {
        "main": "/",
        "rates": "/central-banks",
        "inflation": "/inflation",
        "gdp": "/gdp",
        "employment": "/employment",
        "calendar": "/economic-calendar",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch(self, path: str = "/") -> BeautifulSoup | None:
        url = self.BASE_URL + path
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.warning(f"WorldMonitor fetch error for {url}: {e}")
            raise

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text from page for Claude analysis."""
        if soup is None:
            return ""
        # Remove scripts, styles, navigation
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse multiple blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _parse_indicators(self, soup: BeautifulSoup) -> list[dict]:
        """Parse economic indicators from the main page."""
        indicators = []
        if soup is None:
            return indicators
        try:
            # Look for tables or data rows with economic data
            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows[1:]:  # skip header
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 3:
                        indicator = {
                            "name": cells[0].get_text(strip=True),
                            "value": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                            "previous": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                            "change": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                        }
                        if indicator["name"]:
                            indicators.append(indicator)
        except Exception as e:
            logger.debug(f"Indicator parse error: {e}")
        return indicators

    def _parse_central_banks(self, soup: BeautifulSoup) -> list[dict]:
        """Parse central bank rates."""
        banks = []
        if soup is None:
            return banks
        try:
            # Central bank rate entries
            rate_elements = soup.find_all(class_=lambda c: c and "rate" in c.lower())
            for el in rate_elements:
                text = el.get_text(strip=True)
                if text:
                    banks.append({"data": text})
        except Exception as e:
            logger.debug(f"Central bank parse error: {e}")
        return banks

    def fetch_all(self) -> WorldMonitorSnapshot:
        """Fetch all available data from WorldMonitor."""
        logger.info("Fetching data from WorldMonitor...")
        timestamp = datetime.utcnow().isoformat()

        all_text_parts = []
        indicators = []
        central_banks = []
        inflation_data = []
        gdp_data = []
        employment_data = []
        market_sentiment = {}

        # Fetch main page
        try:
            main_soup = self._fetch("/")
            if main_soup:
                text = self._extract_text_content(main_soup)
                all_text_parts.append(f"=== WORLDMONITOR MAIN PAGE ===\n{text}")
                indicators = self._parse_indicators(main_soup)
        except Exception as e:
            logger.error(f"Main page fetch failed: {e}")

        # Small delay between requests to be respectful
        time.sleep(1)

        # Fetch each section
        section_data = {
            "central-banks": (central_banks, "CENTRAL BANKS & INTEREST RATES"),
            "inflation": (inflation_data, "INFLATION DATA"),
            "gdp": (gdp_data, "GDP DATA"),
            "employment": (employment_data, "EMPLOYMENT DATA"),
        }

        for path_suffix, (data_list, label) in section_data.items():
            try:
                soup = self._fetch(f"/{path_suffix}")
                if soup:
                    text = self._extract_text_content(soup)
                    all_text_parts.append(f"\n=== {label} ===\n{text}")
                    parsed = self._parse_indicators(soup)
                    data_list.extend(parsed)
                time.sleep(0.8)
            except Exception as e:
                logger.warning(f"Section /{path_suffix} fetch failed: {e}")

        raw_text = "\n".join(all_text_parts)
        logger.info(f"WorldMonitor: collected {len(raw_text)} chars of data")

        return WorldMonitorSnapshot(
            timestamp=timestamp,
            indicators=indicators,
            central_banks=central_banks,
            market_sentiment=market_sentiment,
            inflation_data=inflation_data,
            gdp_data=gdp_data,
            employment_data=employment_data,
            raw_text=raw_text,
        )

    def to_dict(self, snapshot: WorldMonitorSnapshot) -> dict:
        return asdict(snapshot)
