"""
Scheduler — orchestrates all timed and periodic tasks:

- 07:25 → Pre-briefing data fetch
- 07:30 → Morning briefing
- Every ALERT_CHECK_INTERVAL minutes → Check for high-impact events & alerts
- 1 hour before any high-impact event → Pre-event alert
- 18:30 → Evening market recap
- Every 4 hours → Refresh market data silently
- Sunday 06:00 → Weekly self-improvement review
"""

import asyncio
from datetime import datetime, timedelta

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

        # Track sent alerts/notifications to avoid duplicates
        self._sent_alerts: set[str] = set()
        self._sent_pre_event_alerts: set[str] = set()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Register all scheduled jobs."""

        # 1. Pre-briefing data fetch (5 min before briefing)
        fetch_minute = max(0, self._briefing_minute - 5)
        self.scheduler.add_job(
            self._job_refresh_data,
            trigger=CronTrigger(
                hour=self._briefing_hour, minute=fetch_minute,
                timezone=self.tz,
            ),
            id="pre_briefing_data_fetch",
            name="Pre-briefing data fetch",
        )

        # 2. Morning briefing
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

        # 3. Proactive alert check (every N minutes)
        self.scheduler.add_job(
            self._job_check_alerts,
            trigger=IntervalTrigger(minutes=ALERT_CHECK_INTERVAL),
            id="alert_check",
            name="Proactive alert check",
        )

        # 4. Periodic data refresh (every 4 hours)
        self.scheduler.add_job(
            self._job_refresh_data,
            trigger=IntervalTrigger(hours=4),
            id="data_refresh",
            name="Periodic data refresh",
        )

        # 5. Evening recap (18:30 every day)
        self.scheduler.add_job(
            self._job_evening_recap,
            trigger=CronTrigger(
                hour=18, minute=30,
                timezone=self.tz,
            ),
            id="evening_recap",
            name="Evening market recap",
        )

        # 6. Pre-event alert checker (every 10 min to catch 1h-before windows)
        self.scheduler.add_job(
            self._job_pre_event_alerts,
            trigger=IntervalTrigger(minutes=10),
            id="pre_event_check",
            name="Pre-event 1h alert check",
        )

        # 7. Weekly self-improvement review (Sunday at 06:00)
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
            f"({ANALYST_TIMEZONE}), alert check every {ALERT_CHECK_INTERVAL}m, "
            f"evening recap at 18:30, pre-event checks every 10m"
        )

    def start(self) -> None:
        self.scheduler.start()
        logger.info("Scheduler started.")

    def stop(self) -> None:
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")

    # ── Jobs ─────────────────────────────────────────────────────────────────

    async def _job_refresh_data(self) -> None:
        """Fetch fresh market data from both sources."""
        logger.info("[Scheduler] Refreshing market data...")
        try:
            loop = asyncio.get_event_loop()
            wm_snap = await loop.run_in_executor(None, self.wm_scraper.fetch_all)
            ff_snap = await loop.run_in_executor(None, self.ff_scraper.fetch_calendar)

            self.agent.update_market_data(wm_snap, ff_snap)
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
            if self.memory.has_morning_briefing_today():
                logger.info("[Scheduler] Morning briefing already sent today, skipping.")
                return
            await self.communicator.send_morning_briefing()
        except Exception as e:
            logger.error(f"[Scheduler] Morning briefing job failed: {e}")
            await self.communicator.send_message(
                f"Errore nel briefing mattutino: {str(e)[:200]}"
            )

    async def _job_evening_recap(self) -> None:
        """Send an end-of-day market recap at 18:30."""
        logger.info("[Scheduler] Sending evening recap...")
        try:
            loop = asyncio.get_event_loop()

            def _generate():
                wm_text = self.agent._format_wm_data()[:3000]
                ff_text = self.agent._format_ff_data()[:2000]
                prompt = (
                    f"RECAP SERALE — {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
                    f"Dati mercato:\n{wm_text}\n\nEventi giornata:\n{ff_text}\n\n"
                    "Genera un RECAP SERALE sintetico (max 600 parole):\n"
                    "1. Come si e' chiusa la giornata per i principali strumenti\n"
                    "2. Sorprese o movimenti significativi\n"
                    "3. Setup e livelli chiave per domani\n"
                    "4. Outlook overnight"
                )
                system = self.agent._build_system_prompt()
                return self.agent._llm.call(prompt, system, max_tokens=2000)

            recap = await loop.run_in_executor(None, _generate)
            await self.communicator.send_message(
                f"RECAP SERALE - {datetime.now().strftime('%d/%m/%Y')}\n\n{recap}"
            )
            logger.info("[Scheduler] Evening recap sent.")
        except Exception as e:
            logger.error(f"[Scheduler] Evening recap failed: {e}")

    async def _job_pre_event_alerts(self) -> None:
        """Send alerts for high-impact events occurring in ~1 hour."""
        if not self.agent._last_ff_snapshot:
            return
        try:
            snap = self.agent._last_ff_snapshot
            now = datetime.now()

            for ev in snap.high_impact_upcoming:
                ev_time_str = ev.get("time", "")
                if not ev_time_str:
                    continue
                try:
                    ev_dt = datetime.strptime(ev_time_str, "%H:%M").replace(
                        year=now.year, month=now.month, day=now.day
                    )
                except ValueError:
                    continue

                diff_min = (ev_dt - now).total_seconds() / 60
                # Alert window: 50-70 minutes before event
                if 50 <= diff_min <= 70:
                    alert_key = f"pre_{ev.get('title', '')}_{now.strftime('%Y%m%d%H')}"
                    if alert_key not in self._sent_pre_event_alerts:
                        self._sent_pre_event_alerts.add(alert_key)
                        logger.info(f"[Scheduler] Pre-event alert: {ev.get('title', '')}")

                        loop = asyncio.get_event_loop()

                        def _generate(event=ev):
                            prompt = (
                                f"TRA CIRCA 1 ORA: {event.get('title', '')} "
                                f"[{event.get('currency', '')}]\n"
                                f"Atteso: {event.get('forecast', 'N/A')} | "
                                f"Precedente: {event.get('previous', 'N/A')}\n\n"
                                "Genera un PRE-EVENT BRIEF sintetico (max 300 parole):\n"
                                "- Cosa misura questo dato\n"
                                "- Consensus e aspettative\n"
                                "- Strumenti piu' impattati e direzione attesa\n"
                                "- Scenario beat vs miss\n"
                                "- Livelli chiave da monitorare"
                            )
                            system = self.agent._build_system_prompt()
                            return self.agent._llm.call(prompt, system, max_tokens=800)

                        brief = await loop.run_in_executor(None, _generate)
                        currency = ev.get("currency", "")
                        title = ev.get("title", "")
                        await self.communicator.send_message(
                            f"PRE-EVENT (1h) - {currency} {title}\n\n{brief}"
                        )

                # Cleanup old keys
                if len(self._sent_pre_event_alerts) > 200:
                    self._sent_pre_event_alerts = set(list(self._sent_pre_event_alerts)[-100:])

        except Exception as e:
            logger.error(f"[Scheduler] Pre-event alerts failed: {e}")

    async def _job_check_alerts(self) -> None:
        """Check for imminent events and surprising data releases."""
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
                        if len(self._sent_alerts) > 500:
                            self._sent_alerts = set(list(self._sent_alerts)[-200:])
        except Exception as e:
            logger.error(f"[Scheduler] Alert check failed: {e}")

    async def _job_weekly_review(self) -> None:
        """Run the weekly deep self-improvement review."""
        logger.info("[Scheduler] Running weekly self-improvement review...")
        try:
            loop = asyncio.get_event_loop()
            review = await loop.run_in_executor(None, self.self_improvement.run_review)
            if review:
                summary = self.self_improvement.get_improvement_summary()
                await self.communicator.send_message(
                    f"REVISIONE SETTIMANALE - Auto-Miglioramento\n\n{summary}"
                )
        except Exception as e:
            logger.error(f"[Scheduler] Weekly review failed: {e}")

    # ── Manual triggers ───────────────────────────────────────────────────────

    async def trigger_data_refresh(self) -> None:
        await self._job_refresh_data()

    async def trigger_briefing(self) -> None:
        await self._job_morning_briefing()
