"""
Telegram Bot — bidirectional communication layer between the analyst agent
and the hedge fund manager.

Features:
- Sends morning briefings, proactive alerts, and updates
- Handles user messages in real-time conversational mode
- Supports long messages via chunking (Telegram 4096 char limit)
- Commands: /start, /briefing, /reset, /status, /miglioramento
"""

import asyncio
import textwrap
from typing import Optional

from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode
from loguru import logger

from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.agent.analyst_agent import FinancialAnalystAgent
from src.memory.memory_manager import MemoryManager
from src.memory.self_improvement import SelfImprovementEngine

# Telegram message limit
TG_MAX_CHARS = 4000


def split_message(text: str, max_len: int = TG_MAX_CHARS) -> list[str]:
    """Split a long message into Telegram-sized chunks, respecting markdown."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > max_len:
            if current:
                chunks.append(current.rstrip())
            current = line
        else:
            current += line
    if current:
        chunks.append(current.rstrip())
    return chunks or [text[:max_len]]


class TelegramCommunicator:
    """Handles all Telegram bot interaction."""

    def __init__(self, agent: FinancialAnalystAgent, memory: MemoryManager):
        self.agent = agent
        self.memory = memory
        self.self_improvement = SelfImprovementEngine(memory)
        self.app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._allowed_chat_id = int(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID else None

    # ── Bot lifecycle ─────────────────────────────────────────────────────────

    def build_application(self) -> Application:
        """Build and configure the Telegram bot application."""
        self.app = (
            Application.builder()
            .token(TELEGRAM_BOT_TOKEN)
            .build()
        )

        # Register handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("briefing", self._cmd_briefing))
        self.app.add_handler(CommandHandler("reset", self._cmd_reset))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("miglioramento", self._cmd_self_improvement))
        self.app.add_handler(CommandHandler("help", self._cmd_help))

        # Handle regular text messages as analyst queries
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        return self.app

    # ── Auth check ────────────────────────────────────────────────────────────

    def _is_authorized(self, update: Update) -> bool:
        if self._allowed_chat_id is None:
            return True
        return update.effective_chat.id == self._allowed_chat_id

    # ── Command handlers ──────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        profile = self.memory.get_analyst_profile()
        msg = (
            f"*Benvenuto* — Sono **Marco**, il tuo Senior Speculative Financial Analyst.\n\n"
            f"Analisi effettuate: {profile.get('analysis_count', 0)}\n"
            f"Versione profilo: v{profile.get('version', 1)}\n\n"
            f"Puoi scrivermi direttamente per qualsiasi analisi o domanda di mercato.\n\n"
            f"*Comandi disponibili:*\n"
            f"/briefing — Genera briefing mattutino ora\n"
            f"/status — Stato del sistema\n"
            f"/miglioramento — Report auto-miglioramento\n"
            f"/reset — Resetta la conversazione\n"
            f"/help — Mostra questo messaggio\n\n"
            f"_Strumenti monitorati: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD, XAUUSD, BTCUSD_"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._cmd_start(update, context)

    async def _cmd_briefing(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text("⏳ Generando briefing mattutino, attendi...")
        try:
            briefing = self.agent.generate_morning_briefing()
            await self._send_long_message(update, briefing)
        except Exception as e:
            await update.message.reply_text(f"❌ Errore nella generazione del briefing: {e}")

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        self.agent.reset_conversation()
        await update.message.reply_text(
            "✅ Conversazione resettata. Nuova sessione avviata."
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        profile = self.memory.get_analyst_profile()
        today_analyses = self.memory.get_today_analyses()
        has_briefing = self.memory.has_morning_briefing_today()

        msg = (
            f"*STATUS SISTEMA*\n\n"
            f"Analista: {profile.get('name', 'Marco')}\n"
            f"Versione profilo: v{profile.get('version', 1)}\n"
            f"Analisi totali: {profile.get('analysis_count', 0)}\n"
            f"Analisi oggi: {len(today_analyses)}\n"
            f"Briefing mattutino: {'✅ Fatto' if has_briefing else '❌ Non ancora'}\n"
            f"Messaggi in sessione: {self.agent.get_conversation_length()}\n"
        )
        if self._last_data_update:
            msg += f"Ultimo aggiornamento dati: {self._last_data_update}"

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_self_improvement(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text("🧠 Avvio revisione auto-miglioramento...")
        try:
            review = self.self_improvement.run_review()
            if review:
                summary = self.self_improvement.get_improvement_summary()
                await self._send_long_message(update, f"*REPORT AUTO-MIGLIORAMENTO*\n\n{summary}")
            else:
                await update.message.reply_text("Dati insufficienti per la revisione. Serve almeno qualche analisi.")
        except Exception as e:
            await update.message.reply_text(f"❌ Errore revisione: {e}")

    # ── Message handler (conversational) ─────────────────────────────────────

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return

        user_text = update.message.text
        if not user_text:
            return

        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing",
        )

        try:
            # Collect streaming response
            full_response = ""
            for chunk in self.agent.chat(user_text):
                full_response += chunk

            await self._send_long_message(update, full_response)

            # Check if self-improvement should run after this analysis
            if self.self_improvement.should_run_review():
                asyncio.create_task(self._run_background_improvement(update))

        except Exception as e:
            logger.error(f"Message handler error: {e}")
            await update.message.reply_text(
                f"❌ Errore durante l'analisi: {str(e)[:200]}"
            )

    # ── Proactive sending (called by scheduler) ───────────────────────────────

    async def send_message(self, text: str) -> None:
        """Send a message to the configured chat (used by scheduler)."""
        if not self._allowed_chat_id:
            logger.warning("No TELEGRAM_CHAT_ID configured — cannot send proactive message")
            return
        try:
            bot = self.app.bot if self.app else Bot(token=TELEGRAM_BOT_TOKEN)
            chunks = split_message(text)
            for chunk in chunks:
                await bot.send_message(
                    chat_id=self._allowed_chat_id,
                    text=chunk,
                    parse_mode=ParseMode.MARKDOWN,
                )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    async def send_morning_briefing(self) -> None:
        """Generate and send morning briefing proactively."""
        logger.info("Sending scheduled morning briefing via Telegram...")
        try:
            intro = "☀️ *Buongiorno!* Ecco il briefing mattutino:"
            await self.send_message(intro)
            briefing = self.agent.generate_morning_briefing()
            await self.send_message(briefing)
        except Exception as e:
            logger.error(f"Morning briefing send error: {e}")
            await self.send_message(f"❌ Errore briefing mattutino: {e}")

    async def send_alert(self, alert_type: str, trigger_data: dict) -> None:
        """Generate and send a proactive alert."""
        logger.info(f"Sending proactive alert: {alert_type}")
        try:
            analysis = self.agent.generate_proactive_alert(alert_type, trigger_data)
            await self.send_message(f"🚨 *ALERT PROATTIVO*\n\n{analysis}")
        except Exception as e:
            logger.error(f"Alert send error: {e}")

    # ── Utility ───────────────────────────────────────────────────────────────

    async def _send_long_message(self, update: Update, text: str) -> None:
        """Send a potentially long message in chunks."""
        chunks = split_message(text)
        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    async def _run_background_improvement(self, update: Update) -> None:
        """Run self-improvement in background and notify if completed."""
        try:
            review = self.self_improvement.run_review()
            if review:
                summary = self.self_improvement.get_improvement_summary()
                await self.send_message(
                    f"🧠 *Auto-miglioramento completato*\n\n{summary}"
                )
        except Exception as e:
            logger.error(f"Background improvement error: {e}")

    _last_data_update: str = "Mai"
