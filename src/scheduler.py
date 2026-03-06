"""
Scheduler — orchestrates all timed and periodic tasks:

- 07:30 → Morning briefing (data fetch + analysis + Telegram send)
- Every ALERT_CHECK_INTERVAL minutes → Check for high-impact events & alerts
- Every 4 hours → Refresh market data silently
- Weekly → Deep self-improvement review
"""

import asyncio
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from src.config import (
    ANALYST_TIMEZONE, MORNING_BRIEFING_TIME,
    ALERT_CHECK_INTERVAL, MIN_ALERT_IMPACT_VALUE, IMPACT_LEVELS,
)
from src.scrapers.worldmonitor_scraper import WorldMonitorScraper
from src.scrapers.forexfactory_scraper import ForexFactoryScraper
from src.agent.analyst_agent import FinancialAnalystAgent
from src.communication.telegram_bot import TelegramCommunicator
from src.memory.memory_manager import MemoryManager
from src.memory.self_improvement import SelfImprovementEngine


class AnalystScheduler:
    """Manages all scheduled tasks for the analyst agent."""

    def __init__(
        self,
        agent: FinancialAnalystAgent,
        communicator: TelegramCommunicator,
        memory: MemoryManager,
    ):
        self.agent = agent
        self.communicator = communicator
        self.memory = memory
        self.self_improvement = SelfImprovementEngine(memory)
        self.wm_scraper = WorldMonitorScraper()
        self.ff_scraper = ForexFactoryScraper()

        self.tz = pytz.timezone(ANALYST_TIMEZONE)
        self.scheduler = AsyncIOScheduler(timezone=self.tz)

        # Parse briefing time
        try:
            bh, bm = MORNING_BRIEFING_TIME.split(":")
            self._briefing_hour = int(bh)
            self._briefing_minute = int(bm)
        except ValueError:
            self._briefing_hour = 7
            self._briefing_minute = 30

        # Track sent alerts to avoid duplicates
        self._sent_alerts: set[str] = set()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Register all scheduled jobs."""

        # 1. Morning briefing (e.g. 07:25 fetch data, 07:30 send briefing)
        fetch_hour = self._briefing_hour
        fetch_minute = max(0, self._briefing_minute - 5)

        self.scheduler.add_job(
            self._job_refresh_data,
            trigger=CronTrigger(
                hour=fetch_hour, minute=fetch_minute,
                timezone=self.tz,
            ),
            id="pre_briefing_data_fetch",
            name="Pre-briefing data fetch",
        )

        self.scheduler.add_job(
            self._job_morning_briefing,
            trigger=CronTrigger(
                hour=self._briefing_hour,
                minute=self._briefing_minute,
                timezone=self.tz,
            ),
            id="morning_briefing",
            name="Morning briefing",
        )

        # 2. Proactive alert check (every N minutes during market hours)
        self.scheduler.add_job(
            self._job_check_alerts,
            trigger=IntervalTrigger(minutes=ALERT_CHECK_INTERVAL),
            id="alert_check",
            name="Proactive alert check",
        )

        # 3. Periodic data refresh (every 4 hours)
        self.scheduler.add_job(
            self._job_refresh_data,
            trigger=IntervalTrigger(hours=4),
            id="data_refresh",
            name="Periodic data refresh",
        )

        # 4. Weekly self-improvement review (Sunday at 06:00)
        self.scheduler.add_job(
            self._job_weekly_review,
            trigger=CronTrigger(
                day_of_week="sun", hour=6, minute=0,
                timezone=self.tz,
            ),
            id="weekly_review",
            name="Weekly self-improvement review",
        )

        logger.info(
            f"Scheduler configured: briefing at {self._briefing_hour:02d}:{self._briefing_minute:02d} "
            f"({ANALYST_TIMEZONE}), alert check every {ALERT_CHECK_INTERVAL}m"
        )

    def start(self) -> None:
        """Start the scheduler."""
        self.scheduler.start()
        logger.info("Scheduler started.")

    def stop(self) -> None:
        """Gracefully stop the scheduler."""
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")

    # ── Jobs ─────────────────────────────────────────────────────────────────

    async def _job_refresh_data(self) -> None:
        """Fetch fresh market data from both sources."""
        logger.info("[Scheduler] Refreshing market data...")
        try:
            # Run scrapers in thread pool (they're synchronous)
            loop = asyncio.get_event_loop()

            wm_snap = await loop.run_in_executor(None, self.wm_scraper.fetch_all)
            ff_snap = await loop.run_in_executor(None, self.ff_scraper.fetch_calendar)

            # Update agent's data
            self.agent.update_market_data(wm_snap, ff_snap)

            # Persist snapshots
            self.memory.save_market_data("worldmonitor", self.wm_scraper.to_dict(wm_snap))
            self.memory.save_market_data("forexfactory", self.ff_scraper.to_dict(ff_snap))

            self.communicator._last_data_update = datetime.now().strftime("%d/%m/%Y %H:%M")
            logger.info("[Scheduler] Market data refresh complete.")

        except Exception as e:
            logger.error(f"[Scheduler] Data refresh failed: {e}")

    async def _job_morning_briefing(self) -> None:
        """Send the daily morning briefing."""
        logger.info("[Scheduler] Sending morning briefing...")
        try:
            # Don't double-send if already sent today
            if self.memory.has_morning_briefing_today():
                logger.info("[Scheduler] Morning briefing already sent today, skipping.")
                return
            await self.communicator.send_morning_briefing()
        except Exception as e:
            logger.error(f"[Scheduler] Morning briefing job failed: {e}")
            await self.communicator.send_message(
                f"⚠️ Errore nel briefing mattutino: {str(e)[:200]}"
            )

    async def _job_check_alerts(self) -> None:
        """Check for market-moving events and send proactive alerts."""
        logger.debug("[Scheduler] Checking for alerts...")
        try:
            alerts = self.agent.check_for_alerts()
            for alert in alerts:
                alert_key = f"{alert['type']}_{datetime.now().strftime('%Y%m%d%H')}"
                if alert_key not in self._sent_alerts:
                    impact = IMPACT_LEVELS.get(alert.get("priority", "low"), 1)
                    if impact >= MIN_ALERT_IMPACT_VALUE:
                        await self.communicator.send_alert(
                            alert["type"], alert["trigger_data"]
                        )
                        self._sent_alerts.add(alert_key)
                        # Clean up old keys to prevent set from growing
                        if len(self._sent_alerts) > 500:
                            self._sent_alerts = set(list(self._sent_alerts)[-200:])
        except Exception as e:
            logger.error(f"[Scheduler] Alert check failed: {e}")

    async def _job_weekly_review(self) -> None:
        """Run the weekly deep self-improvement review."""
        logger.info("[Scheduler] Running weekly self-improvement review...")
        try:
            review = self.self_improvement.run_review()
            if review:
                summary = self.self_improvement.get_improvement_summary()
                await self.communicator.send_message(
                    f"🧠 *REVISIONE SETTIMANALE — Auto-Miglioramento*\n\n{summary}"
                )
        except Exception as e:
            logger.error(f"[Scheduler] Weekly review failed: {e}")

    # ── Manual triggers ───────────────────────────────────────────────────────

    async def trigger_data_refresh(self) -> None:
        """Manually trigger a data refresh."""
        await self._job_refresh_data()

    async def trigger_briefing(self) -> None:
        """Manually trigger the morning briefing."""
        await self._job_morning_briefing()
