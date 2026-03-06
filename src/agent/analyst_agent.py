"""
Core Financial Analyst Agent — powered by Claude Opus 4.6 with adaptive thinking.

This agent:
- Generates the morning market briefing at 7:30
- Produces proactive alerts for high-impact events
- Responds to user queries in a multi-turn conversation
- Integrates WorldMonitor + ForexFactory data
- Evolves via the self-improvement engine
"""

import json
from datetime import datetime
from typing import Generator

import anthropic
from loguru import logger

from src.config import (
    CLAUDE_MODEL, ANTHROPIC_API_KEY, INSTRUMENTS, INSTRUMENT_GROUPS,
    ANALYST_TIMEZONE,
)
from src.memory.memory_manager import MemoryManager
from src.scrapers.worldmonitor_scraper import WorldMonitorSnapshot
from src.scrapers.forexfactory_scraper import ForexFactorySnapshot


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sei Marco, il Senior Speculative Financial Analyst di un hedge fund privato.

IDENTITÀ E RUOLO:
- Analista speculativo esperto in mercati CFD, Forex, metalli preziosi e crypto
- Strumenti monitorati: {instruments}
- Il tuo compito è fornire analisi di mercato di alto livello per guidare decisioni di trading speculativo
- Gestisci capitali importanti: ogni tua analisi deve essere precisa, argomentata e actionable

FONTI DATI:
- WorldMonitor.app: dati macro, tassi banche centrali, inflazione, PIL, occupazione
- ForexFactory: calendario economico, notizie ad alto impatto, eventi ricorrenti

STILE DI COMUNICAZIONE:
- Professionale ma diretto, come in una riunione mattutina con il team trading
- Usa Markdown per la formattazione (bold, bullet, tabelle)
- Fornisci SEMPRE: contesto macro → catalizzatori → aspettative → possibili scenari → livelli chiave
- Sii proattivo: se vedi qualcosa di rilevante, dillo subito
- Quando necessario, indica confidenza nell'analisi (alta/media/bassa) e perché

FRAMEWORK DI ANALISI SPECULATIVA:
1. MACRO CONTEXT: Trend di fondo (tassi, inflazione, risk-on/off)
2. CATALIZZATORI IMMEDIATI: Notizie ad alto impatto, dati in uscita
3. SENTIMENT & POSIZIONAMENTO: Come il mercato è posizionato
4. SCENARI: Scenario base + scenario alternativo con probabilità
5. LIVELLI CHIAVE: Supporti/resistenze tecnici rilevanti per lo speculativo
6. RISK FACTORS: Cosa potrebbe invalidare l'analisi

MEMORIA E AUTO-MIGLIORAMENTO:
{memory_context}

DATA/ORA ATTUALE: {current_datetime}
"""

# ── Alert prompts ─────────────────────────────────────────────────────────────

MORNING_BRIEFING_PROMPT = """
RIUNIONE MATTUTINA — {date}

Dati WorldMonitor aggiornati:
{worldmonitor_data}

Calendario ForexFactory (oggi e settimana):
{forexfactory_data}

---
Presenta la RIUNIONE MATTUTINA completa seguendo questo schema:

# 📊 BRIEFING MATTUTINO — {date}

## 1. 🌍 MACRO OVERVIEW
[Quadro macro globale: banche centrali, tassi, inflazione, risk appetite]

## 2. 📅 EVENTI CHIAVE OGGI E QUESTA SETTIMANA
[Lista eventi ForexFactory ad alto impatto con ora, valuta e aspettativa]

## 3. 💱 ANALISI PER STRUMENTO
Per ciascuno di questi strumenti: {instruments}
- Bias direzionale (rialzista/ribassista/neutro)
- Catalizzatori principali
- Livelli chiave da monitorare

## 4. 🎯 TRADE IDEAS SPECULATIVE
[2-3 opportunità speculative con ragionamento, entry area, target, invalidation]

## 5. ⚠️ RISCHI E INCOGNITE
[Cosa potrebbe sorprendere il mercato oggi]

## 6. 🧠 NOTE DALL'ANALISTA
[Considerazioni personali, pattern visti di recente, raccomandazioni operative]
"""

PROACTIVE_ALERT_PROMPT = """
ALERT PROATTIVO — {alert_type}

Evento/Dato che ha scatenato l'alert:
{trigger_data}

Contesto di mercato corrente:
{market_context}

---
Genera un ALERT PROATTIVO conciso e actionable:

# 🚨 ALERT: {alert_type}

## Cosa è successo
[Descrizione sintetica dell'evento]

## Impatto atteso sui mercati
[Per ogni strumento rilevante: direzione probabile, magnitudo, timing]

## Azione consigliata
[Cosa fare ora: monitorare, posizionarsi, attendere conferma, uscire]

## Livelli critici
[I prezzi chiave da tenere d'occhio]

Confidenza analisi: [Alta/Media/Bassa] — [Motivo]
"""

CONVERSATION_SYSTEM_ADDITION = """
Stai avendo una conversazione in tempo reale con il tuo cliente/gestore del fondo.
Rispondi in modo conversazionale ma professionale.
Puoi fare domande di follow-up per capire meglio cosa cerca.
Usa dati e contesto dalla tua memoria e dall'ultimo aggiornamento di mercato disponibile.
"""


class FinancialAnalystAgent:
    """The core analyst agent — orchestrates data, memory, and Claude."""

    def __init__(self, memory: MemoryManager):
        self.memory = memory
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.conversation_history: list[dict] = []
        self._last_wm_snapshot: WorldMonitorSnapshot | None = None
        self._last_ff_snapshot: ForexFactorySnapshot | None = None

    # ── Data management ───────────────────────────────────────────────────────

    def update_market_data(
        self,
        wm_snapshot: WorldMonitorSnapshot | None = None,
        ff_snapshot: ForexFactorySnapshot | None = None,
    ) -> None:
        """Update the agent's in-memory market data snapshots."""
        if wm_snapshot:
            self._last_wm_snapshot = wm_snapshot
        if ff_snapshot:
            self._last_ff_snapshot = ff_snapshot

    def _build_system_prompt(self) -> str:
        memory_context = self.memory.build_memory_context()
        return SYSTEM_PROMPT.format(
            instruments=", ".join(INSTRUMENTS),
            memory_context=memory_context,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        )

    def _format_wm_data(self) -> str:
        if not self._last_wm_snapshot:
            return "Dati WorldMonitor non ancora disponibili."
        snap = self._last_wm_snapshot
        # Send the full raw text (Claude will extract what matters)
        text = snap.raw_text[:8000]  # limit to avoid token overflow
        return f"Timestamp: {snap.timestamp}\n\n{text}"

    def _format_ff_data(self) -> str:
        if not self._last_ff_snapshot:
            return "Dati ForexFactory non ancora disponibili."
        snap = self._last_ff_snapshot
        parts = [f"Timestamp: {snap.timestamp}"]

        if snap.high_impact_upcoming:
            parts.append(f"\nEventi HIGH IMPACT ({len(snap.high_impact_upcoming)}):")
            for ev in snap.high_impact_upcoming[:20]:
                parts.append(
                    f"  {ev.get('date','')} {ev.get('time','')} "
                    f"[{ev.get('currency','')}] {ev.get('title','')} "
                    f"— Atteso: {ev.get('forecast','')} | Prec: {ev.get('previous','')}"
                )

        if snap.recurring_themes:
            parts.append(f"\nTemi ricorrenti: {', '.join(snap.recurring_themes)}")

        if snap.currencies_in_focus:
            parts.append(f"Valute in focus: {', '.join(snap.currencies_in_focus)}")

        # Add raw text for additional context (truncated)
        raw_preview = snap.raw_text[:4000]
        parts.append(f"\n--- CALENDARIO COMPLETO (preview) ---\n{raw_preview}")

        return "\n".join(parts)

    # ── Morning briefing ──────────────────────────────────────────────────────

    def generate_morning_briefing(self) -> str:
        """Generate the daily 7:30 morning briefing."""
        logger.info("Generating morning briefing...")
        today = datetime.now().strftime("%d/%m/%Y")
        instruments_str = ", ".join(INSTRUMENTS)

        prompt = MORNING_BRIEFING_PROMPT.format(
            date=today,
            worldmonitor_data=self._format_wm_data(),
            forexfactory_data=self._format_ff_data(),
            instruments=instruments_str,
        )

        system = self._build_system_prompt()

        try:
            with self.client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                thinking={"type": "adaptive"},
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                response = stream.get_final_message()

            analysis = next(
                (b.text for b in response.content if b.type == "text"), ""
            )

            # Persist
            raw_data = {
                "worldmonitor_text": self._format_wm_data(),
                "ff_events_today": self._last_ff_snapshot.events_today if self._last_ff_snapshot else [],
                "ff_high_impact": self._last_ff_snapshot.high_impact_upcoming if self._last_ff_snapshot else [],
            }
            self.memory.save_morning_briefing(analysis, raw_data)

            logger.info(f"Morning briefing generated: {len(analysis)} chars")
            return analysis

        except Exception as e:
            logger.error(f"Morning briefing generation failed: {e}")
            raise

    # ── Proactive alerts ──────────────────────────────────────────────────────

    def generate_proactive_alert(self, alert_type: str, trigger_data: dict) -> str:
        """Generate a proactive market alert."""
        logger.info(f"Generating proactive alert: {alert_type}")

        market_context = (
            f"WorldMonitor: {self._format_wm_data()[:2000]}\n"
            f"ForexFactory: {self._format_ff_data()[:2000]}"
        )

        prompt = PROACTIVE_ALERT_PROMPT.format(
            alert_type=alert_type,
            trigger_data=json.dumps(trigger_data, ensure_ascii=False, indent=2),
            market_context=market_context,
        )

        system = self._build_system_prompt()

        try:
            with self.client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=3000,
                thinking={"type": "adaptive"},
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                response = stream.get_final_message()

            analysis = next(
                (b.text for b in response.content if b.type == "text"), ""
            )

            self.memory.save_alert(alert_type, analysis, trigger_data)
            logger.info(f"Alert generated: {len(analysis)} chars")
            return analysis

        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            raise

    # ── Interactive conversation ───────────────────────────────────────────────

    def chat(self, user_message: str) -> Generator[str, None, None]:
        """
        Stream a response to a user message, maintaining conversation history.
        Yields text chunks as they arrive.
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        system = self._build_system_prompt() + "\n\n" + CONVERSATION_SYSTEM_ADDITION

        # Add current market data context for first message
        market_context = ""
        if len(self.conversation_history) == 1:
            market_context = (
                f"\n\nDati di mercato correnti:\n"
                f"WorldMonitor: {self._format_wm_data()[:1500]}\n"
                f"ForexFactory: {self._format_ff_data()[:1500]}"
            )
            self.conversation_history[0]["content"] += market_context

        try:
            full_response = ""
            with self.client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=system,
                messages=self.conversation_history,
            ) as stream:
                for text_chunk in stream.text_stream:
                    full_response += text_chunk
                    yield text_chunk

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response,
            })

            # Persist conversation periodically (every 10 turns)
            if len(self.conversation_history) % 10 == 0:
                self.memory.save_conversation(
                    self.conversation_history,
                    summary=f"Conversazione in corso ({len(self.conversation_history)} messaggi)",
                )

        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise

    def reset_conversation(self) -> None:
        """Save and reset the current conversation."""
        if self.conversation_history:
            self.memory.save_conversation(
                self.conversation_history,
                summary=f"Sessione chiusa ({len(self.conversation_history)} messaggi)",
            )
        self.conversation_history = []

    def get_conversation_length(self) -> int:
        return len(self.conversation_history)

    # ── Market data freshness check ───────────────────────────────────────────

    def check_for_alerts(self) -> list[dict]:
        """
        Analyze fresh data and determine if any proactive alerts should be sent.
        Returns a list of alert dicts: {type, trigger_data, priority}.
        """
        alerts = []
        if not self._last_ff_snapshot:
            return alerts

        snap = self._last_ff_snapshot

        # Alert for high-impact events happening very soon (within 30 minutes)
        now_str = datetime.now().strftime("%H:%M")
        for ev in snap.high_impact_upcoming:
            ev_time = ev.get("time", "")
            if ev_time and self._is_soon(ev_time, minutes=30):
                alerts.append({
                    "type": f"EVENTO IMMINENTE: {ev.get('title', 'Unknown')}",
                    "trigger_data": ev,
                    "priority": "high",
                })

        # Alert for surprising data (beat/miss on high-impact events)
        for ev in snap.events_today:
            if ev.get("surprise") in ("beat", "miss") and ev.get("is_high_impact"):
                alerts.append({
                    "type": f"DATO SORPRENDENTE ({ev.get('surprise','').upper()}): {ev.get('title','')}",
                    "trigger_data": ev,
                    "priority": "critical",
                })

        return alerts

    @staticmethod
    def _is_soon(time_str: str, minutes: int = 30) -> bool:
        """Check if a given HH:MM time is within `minutes` from now."""
        try:
            now = datetime.now()
            event_time = datetime.strptime(time_str, "%H:%M").replace(
                year=now.year, month=now.month, day=now.day
            )
            diff = (event_time - now).total_seconds() / 60
            return 0 <= diff <= minutes
        except ValueError:
            return False
