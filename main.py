"""
Financial Analyst Agent — Hedge Fund
Entry point: starts the Telegram bot, scheduler, and initial data fetch.

Usage:
    python main.py

Environment:
    Copy .env.example → .env and fill in your API keys.
"""

import asyncio
import signal
import sys
from loguru import logger

from src.config import (
    GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ANALYST_TIMEZONE
)
from src.memory.memory_manager import MemoryManager
from src.agent.analyst_agent import FinancialAnalystAgent
from src.communication.telegram_bot import TelegramCommunicator
from src.scheduler import AnalystScheduler


# ── Logging setup ─────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)
logger.add(
    "logs/analyst_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="1 day",
    retention="30 days",
    compression="gz",
)


# ── Startup validation ────────────────────────────────────────────────────────

def validate_config() -> bool:
    """Check that required environment variables are set."""
    missing = []
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Copy .env.example to .env and fill in the values.")
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not validate_config():
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("FINANCIAL ANALYST AGENT — HEDGE FUND")
    logger.info(f"Timezone: {ANALYST_TIMEZONE}")
    logger.info("=" * 60)

    # Initialize core components
    memory = MemoryManager()
    agent = FinancialAnalystAgent(memory)
    communicator = TelegramCommunicator(agent, memory)
    scheduler = AnalystScheduler(agent, communicator, memory)

    # Build Telegram app
    app = communicator.build_application()

    # Setup & start scheduler
    scheduler.setup()
    scheduler.start()

    # Initial data fetch on startup
    logger.info("Performing initial data fetch...")
    try:
        await scheduler.trigger_data_refresh()
        logger.info("Initial data fetch complete.")
    except Exception as e:
        logger.warning(f"Initial data fetch error (continuing): {e}")

    # Send startup notification
    try:
        profile = memory.get_analyst_profile()
        await communicator.send_message(
            f"🟢 *Analista avviato*\n"
            f"Marco v{profile.get('version', 1)} è operativo.\n"
            f"Analisi effettuate: {profile.get('analysis_count', 0)}\n"
            f"Briefing mattutino programmato alle {MORNING_BRIEFING_TIME} ({ANALYST_TIMEZONE})\n\n"
            f"Scrivi qualsiasi cosa per iniziare un'analisi, oppure usa /briefing per il report immediato."
        )
    except Exception as e:
        logger.warning(f"Startup notification error: {e}")

    # Graceful shutdown handler
    stop_event = asyncio.Event()

    def _signal_handler(*_):
        logger.info("Shutdown signal received...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start Telegram bot
    logger.info("Starting Telegram bot (polling)...")
    async with app:
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        logger.info("Agent is live. Press Ctrl+C to stop.")
        await stop_event.wait()

        # Graceful shutdown
        logger.info("Shutting down...")
        scheduler.stop()
        await app.updater.stop()
        await app.stop()

    logger.info("Agent stopped cleanly.")


from src.config import MORNING_BRIEFING_TIME  # re-import for the startup message

if __name__ == "__main__":
    asyncio.run(main())
