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
import base64
import io
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
from telegram.error import BadRequest
from loguru import logger

from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GROQ_API_KEY
from src.agent.analyst_agent import FinancialAnalystAgent
from src.memory.memory_manager import MemoryManager
from src.memory.self_improvement import SelfImprovementEngine

# Telegram message limit
TG_MAX_CHARS = 4000


def split_message(text: str, max_len: int = TG_MAX_CHARS) -> list[str]:
    """Split a long message into Telegram-sized chunks."""
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


def _to_telegram(text: str) -> str:
    """
    Convert LLM markdown to clean plain text suitable for Telegram.
    Telegram MARKDOWN v1 does NOT support ###, **, ---  etc.
    We convert to readable plain text instead of risking parse errors.
    """
    import re
    # ### Heading → HEADING (uppercase for visual hierarchy)
    text = re.sub(r"^#{1,6}\s+(.+)$", lambda m: m.group(1).upper(), text, flags=re.MULTILINE)
    # **bold** or __bold__ → text (strip)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # *italic* → text
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    # _italic_ → text
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    # ```code``` or `code` → text
    text = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", text)
    # [link](url) → link
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # --- separators → blank line
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TelegramCommunicator:
    """Handles all Telegram bot interaction."""

    def __init__(self, agent: FinancialAnalystAgent, memory: MemoryManager):
        self.agent = agent
        self.memory = memory
        self.self_improvement = SelfImprovementEngine(memory)
        self.app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._allowed_chat_id = int(TELEGRAM_CHAT_ID) if TELEGRAM_CHAT_ID else None
        self._last_data_update: str = "Mai"

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

        # Handle photos (chart screenshots) — with or without caption
        self.app.add_handler(
            MessageHandler(filters.PHOTO, self._handle_photo)
        )

        return self.app

    # ── Auth check ────────────────────────────────────────────────────────────

    def _is_authorized(self, update: Update) -> bool:
        if self._allowed_chat_id is None:
            return True
        return update.effective_chat.id == self._allowed_chat_id

    # ── Safe send helpers ─────────────────────────────────────────────────────

    async def _safe_reply(self, update: Update, text: str) -> None:
        """Send reply as clean plain text (no parse_mode — always works)."""
        chunks = split_message(_to_telegram(text))
        for chunk in chunks:
            await update.message.reply_text(chunk)

    async def _safe_send(self, text: str) -> None:
        """Send proactive message as clean plain text."""
        if not self._allowed_chat_id:
            logger.warning("No TELEGRAM_CHAT_ID configured")
            return
        bot = self.app.bot if self.app else Bot(token=TELEGRAM_BOT_TOKEN)
        chunks = split_message(_to_telegram(text))
        for chunk in chunks:
            try:
                await bot.send_message(chat_id=self._allowed_chat_id, text=chunk)
            except Exception as e:
                logger.error(f"Telegram send error: {e}")

    # ── Command handlers ──────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        profile = self.memory.get_analyst_profile()
        msg = (
            f"*Benvenuto* - Sono *Marco*, il tuo Senior Speculative Financial Analyst.\n\n"
            f"Analisi effettuate: {profile.get('analysis_count', 0)}\n"
            f"Versione profilo: v{profile.get('version', 1)}\n\n"
            f"Puoi scrivermi direttamente per qualsiasi analisi o domanda di mercato.\n\n"
            f"*Comandi disponibili:*\n"
            f"/briefing - Genera briefing mattutino ora\n"
            f"/status - Stato del sistema\n"
            f"/miglioramento - Report auto-miglioramento\n"
            f"/reset - Resetta la conversazione\n"
            f"/help - Mostra questo messaggio\n\n"
            f"Strumenti monitorati: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD, XAUUSD, BTCUSD"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._cmd_start(update, context)

    async def _cmd_briefing(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Generando briefing mattutino, attendi...")
        try:
            loop = asyncio.get_event_loop()
            briefing = await loop.run_in_executor(
                None, self.agent.generate_morning_briefing
            )
            await self._safe_reply(update, briefing)
        except Exception as e:
            logger.error(f"Briefing command error: {e}")
            await update.message.reply_text(f"Errore nella generazione del briefing: {str(e)[:200]}")

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        self.agent.reset_conversation()
        await update.message.reply_text("Conversazione resettata. Nuova sessione avviata.")

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
            f"Briefing mattutino: {'Fatto' if has_briefing else 'Non ancora'}\n"
            f"Messaggi in sessione: {self.agent.get_conversation_length()}\n"
            f"Ultimo aggiornamento dati: {self._last_data_update}\n"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_self_improvement(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        await update.message.reply_text("Avvio revisione auto-miglioramento...")
        try:
            loop = asyncio.get_event_loop()
            review = await loop.run_in_executor(None, self.self_improvement.run_review)
            if review:
                summary = self.self_improvement.get_improvement_summary()
                await self._safe_reply(update, f"*REPORT AUTO-MIGLIORAMENTO*\n\n{summary}")
            else:
                await update.message.reply_text("Dati insufficienti per la revisione. Serve almeno qualche analisi.")
        except Exception as e:
            await update.message.reply_text(f"Errore revisione: {str(e)[:200]}")

    # ── Message handler (conversational) ─────────────────────────────────────

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return

        user_text = update.message.text
        if not user_text:
            return

        logger.info(f"Message received: {user_text[:80]}")

        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing",
        )

        try:
            # Run sync generator in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()

            def _collect_response():
                full = ""
                for chunk in self.agent.chat(user_text):
                    full += chunk
                return full

            full_response = await loop.run_in_executor(None, _collect_response)
            await self._safe_reply(update, full_response)

            # Check if self-improvement should run after this analysis
            if self.self_improvement.should_run_review():
                asyncio.create_task(self._run_background_improvement())

        except Exception as e:
            logger.error(f"Message handler error: {e}")
            await update.message.reply_text(
                f"Errore durante l'analisi: {str(e)[:200]}"
            )

    # ── Photo/chart handler ────────────────────────────────────────────────────

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Analyse a chart screenshot sent by the user via Groq Vision."""
        if not self._is_authorized(update):
            return

        caption = update.message.caption or "Analizza questo grafico. Cosa noti? Qual e' il tuo giudizio tecnico e speculativo?"
        logger.info(f"Photo received, caption: {caption[:80]}")

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )

        try:
            # Download highest-resolution photo
            photo = update.message.photo[-1]
            tg_file = await photo.get_file()
            img_bytes = await tg_file.download_as_bytearray()
            b64_image = base64.b64encode(img_bytes).decode("utf-8")

            loop = asyncio.get_event_loop()

            def _vision_call():
                return _groq_vision(
                    b64_image=b64_image,
                    user_text=caption,
                    system=self.agent._build_system_prompt(),
                )

            response = await loop.run_in_executor(None, _vision_call)
            await self._safe_reply(update, response)

        except Exception as e:
            logger.error(f"Photo handler error: {e}")
            await update.message.reply_text(
                f"Errore nell'analisi del grafico: {str(e)[:200]}"
            )

    # ── Proactive sending (called by scheduler) ───────────────────────────────

    async def send_message(self, text: str) -> None:
        """Send a message to the configured chat (used by scheduler)."""
        await self._safe_send(text)

    async def send_morning_briefing(self) -> None:
        """Generate and send morning briefing proactively."""
        logger.info("Sending scheduled morning briefing via Telegram...")
        try:
            await self.send_message("Buongiorno! Ecco il briefing mattutino:")
            loop = asyncio.get_event_loop()
            briefing = await loop.run_in_executor(
                None, self.agent.generate_morning_briefing
            )
            await self.send_message(briefing)
        except Exception as e:
            logger.error(f"Morning briefing send error: {e}")
            await self.send_message(f"Errore briefing mattutino: {str(e)[:200]}")

    async def send_alert(self, alert_type: str, trigger_data: dict) -> None:
        """Generate and send a proactive alert."""
        logger.info(f"Sending proactive alert: {alert_type}")
        try:
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(
                None,
                lambda: self.agent.generate_proactive_alert(alert_type, trigger_data),
            )
            await self.send_message(f"ALERT PROATTIVO\n\n{analysis}")
        except Exception as e:
            logger.error(f"Alert send error: {e}")

    # ── Utility ───────────────────────────────────────────────────────────────

    async def _run_background_improvement(self) -> None:
        """Run self-improvement in background and notify if completed."""
        try:
            loop = asyncio.get_event_loop()
            review = await loop.run_in_executor(None, self.self_improvement.run_review)
            if review:
                summary = self.self_improvement.get_improvement_summary()
                await self.send_message(f"Auto-miglioramento completato\n\n{summary}")
        except Exception as e:
            logger.error(f"Background improvement error: {e}")


# ── Groq Vision helper (module-level) ────────────────────────────────────────

def _groq_vision(b64_image: str, user_text: str, system: str) -> str:
    """
    Send a base64-encoded image + text to Groq's vision model.
    Uses meta-llama/llama-4-scout-17b-16e-instruct (supports vision).
    Falls back to plain text analysis if vision unavailable.
    """
    if not GROQ_API_KEY:
        return "Nessuna chiave Groq configurata per l'analisi visiva."

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
        )

        full_prompt = (
            "Sei Marco, Senior Speculative Financial Analyst di un hedge fund.\n"
            "Analizza il grafico finanziario nell'immagine seguendo questo schema:\n\n"
            "1. STRUTTURA E TIMEFRAME: Cosa mostri il grafico (strumento, TF, pattern visibili)\n"
            "2. TREND E MOMENTUM: Direzione dominante, forza del trend\n"
            "3. LIVELLI CHIAVE: Supporti, resistenze, zone demand/supply visibili\n"
            "4. PATTERN TECNICI: Formazioni candlestick, pattern grafici rilevanti\n"
            "5. BIAS OPERATIVO: Long/Short/Neutro con motivazione\n"
            "6. TRADE SETUP: Entry, target, invalidazione\n\n"
            f"Domanda/contesto utente: {user_text}"
        )

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            },
                        },
                        {"type": "text", "text": full_prompt},
                    ],
                },
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        return response.choices[0].message.content or "Nessuna risposta dal modello."

    except Exception as e:
        logger.error(f"Groq vision error: {e}")
        return f"Errore analisi visiva: {str(e)[:200]}"
