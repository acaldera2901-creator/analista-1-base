"""
Utility script: generate and print a morning briefing without starting the bot.
Useful for testing or running from a cron job manually.

Usage:
    python scripts/run_briefing_now.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.memory.memory_manager import MemoryManager
from src.agent.analyst_agent import FinancialAnalystAgent
from src.scrapers.worldmonitor_scraper import WorldMonitorScraper
from src.scrapers.forexfactory_scraper import ForexFactoryScraper


async def main():
    logger.info("Fetching data...")
    wm = WorldMonitorScraper()
    ff = ForexFactoryScraper()

    wm_snap = wm.fetch_all()
    ff_snap = ff.fetch_calendar()

    memory = MemoryManager()
    agent = FinancialAnalystAgent(memory)
    agent.update_market_data(wm_snap, ff_snap)

    logger.info("Generating briefing...")
    briefing = agent.generate_morning_briefing()
    print("\n" + "=" * 70)
    print(briefing)
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
