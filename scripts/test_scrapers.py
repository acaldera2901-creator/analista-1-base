"""
Test script: verify scrapers work and print a summary of fetched data.

Usage:
    python scripts/test_scrapers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.scrapers.worldmonitor_scraper import WorldMonitorScraper
from src.scrapers.forexfactory_scraper import ForexFactoryScraper


def main():
    print("\n🌐 Testing WorldMonitor scraper...")
    try:
        wm = WorldMonitorScraper()
        snap = wm.fetch_all()
        print(f"  ✅ WorldMonitor OK")
        print(f"  Timestamp: {snap.timestamp}")
        print(f"  Raw text length: {len(snap.raw_text)} chars")
        print(f"  Indicators found: {len(snap.indicators)}")
        print(f"  Central banks found: {len(snap.central_banks)}")
        if snap.raw_text:
            preview = snap.raw_text[:300].replace('\n', ' ')
            print(f"  Preview: {preview}...")
    except Exception as e:
        print(f"  ❌ WorldMonitor FAILED: {e}")

    print("\n📅 Testing ForexFactory scraper...")
    try:
        ff = ForexFactoryScraper()
        snap = ff.fetch_calendar()
        print(f"  ✅ ForexFactory OK")
        print(f"  Timestamp: {snap.timestamp}")
        print(f"  Events today: {len(snap.events_today)}")
        print(f"  Events this week: {len(snap.events_week)}")
        print(f"  High impact upcoming: {len(snap.high_impact_upcoming)}")
        print(f"  Currencies in focus: {snap.currencies_in_focus}")
        print(f"  Recurring themes: {snap.recurring_themes}")
        if snap.high_impact_upcoming:
            ev = snap.high_impact_upcoming[0]
            print(f"  Next high-impact: [{ev.get('currency','')}] {ev.get('title','')} @ {ev.get('time','')}")
    except Exception as e:
        print(f"  ❌ ForexFactory FAILED: {e}")

    print("\n✅ Scraper test complete.")


if __name__ == "__main__":
    main()
