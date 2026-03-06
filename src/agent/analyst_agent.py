"""
Core Financial Analyst Agent.

Supports multiple LLM providers via src/llm_client.py:
  - Gemini 2.0 Flash  (GOOGLE_API_KEY)
  - Groq / Llama 3.3  (GROQ_API_KEY)
  - OpenRouter        (OPENROUTER_API_KEY)
  - Anthropic Claude  (ANTHROPIC_API_KEY)

Set LLM_PROVIDER=gemini|groq|openrouter|anthropic in .env to force a provider.
"""

import json
from datetime import datetime
from typing import Generator

from loguru import logger

from src.config import (
    INSTRUMENTS, INSTRUMENT_GROUPS, ANALYST_TIMEZONE,
)
from src.memory.memory_manager import MemoryManager
from src.scrapers.worldmonitor_scraper import WorldMonitorSnapshot
from src.scrapers.forexfactory_scraper import ForexFactorySnapshot
from src.llm_client import build_llm_client


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

# BRIEFING MATTUTINO — {date}

## 1. MACRO OVERVIEW
[Quadro macro globale: banche centrali, tassi, inflazione, risk appetite]

## 2. EVENTI CHIAVE OGGI E QUESTA SETTIMANA
[Lista eventi ForexFactory ad alto impatto con ora, valuta e aspettativa]

## 3. ANALISI PER STRUMENTO
Per ciascuno di questi strumenti: {instruments}
- Bias direzionale (rialzista/ribassista/neutro)
- Catalizzatori principali
- Livelli chiave da monitorare

## 4. TRADE IDEAS SPECULATIVE
[2-3 opportunità speculative con ragionamento, entry area, target, invalidation]

## 5. RISCHI E INCOGNITE
[Cosa potrebbe sorprendere il mercato oggi]

## 6. NOTE DALL'ANALISTA
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

# ALERT: {alert_type}

## Cosa e' successo
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
    """The core analyst agent — provider-agnostic via src/llm_client."""

    def __init__(self, memory: MemoryManager):
        self.memory = memory
        self._llm = build_llm_client()
        self.conversation_history: list[dict] = []
        self._last_wm_snapshot: WorldMonitorSnapshot | None = None
        self._last_ff_snapshot: ForexFactorySnapshot | None = None

    # ── Data management ───────────────────────────────────────────────────────

    def update_market_data(
        self,
        wm_snapshot: WorldMonitorSnapshot | None = None,
        ff_snapshot: ForexFactorySnapshot | None = None,
    ) -> None:
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
        return f"Timestamp: {snap.timestamp}\n\n{snap.raw_text[:8000]}"

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
                    f"-- Atteso: {ev.get('forecast','')} | Prec: {ev.get('previous','')}"
                )

        if snap.recurring_themes:
            parts.append(f"\nTemi ricorrenti: {', '.join(snap.recurring_themes)}")
        if snap.currencies_in_focus:
            parts.append(f"Valute in focus: {', '.join(snap.currencies_in_focus)}")

        parts.append(f"\n--- CALENDARIO COMPLETO (preview) ---\n{snap.raw_text[:4000]}")
        return "\n".join(parts)

    # ── Morning briefing ──────────────────────────────────────────────────────

    def generate_morning_briefing(self) -> str:
        logger.info("Generating morning briefing...")
        today = datetime.now().strftime("%d/%m/%Y")
        prompt = MORNING_BRIEFING_PROMPT.format(
            date=today,
            worldmonitor_data=self._format_wm_data(),
            forexfactory_data=self._format_ff_data(),
            instruments=", ".join(INSTRUMENTS),
        )
        system = self._build_system_prompt()
        try:
            analysis = self._llm.call(prompt, system, max_tokens=8192)
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
            analysis = self._llm.call(prompt, system, max_tokens=3000)
            self.memory.save_alert(alert_type, analysis, trigger_data)
            logger.info(f"Alert generated: {len(analysis)} chars")
            return analysis
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            raise

    # ── Interactive conversation ───────────────────────────────────────────────

    def chat(self, user_message: str) -> Generator[str, None, None]:
        """Stream a response, maintaining conversation history."""
        system = self._build_system_prompt() + "\n\n" + CONVERSATION_SYSTEM_ADDITION

        # Enrich first message with market data context
        if not self.conversation_history:
            market_context = (
                f"\n\nDati di mercato correnti:\n"
                f"WorldMonitor: {self._format_wm_data()[:1500]}\n"
                f"ForexFactory: {self._format_ff_data()[:1500]}"
            )
            user_message_full = user_message + market_context
        else:
            user_message_full = user_message

        self.conversation_history.append(
            self._llm.make_message("user", user_message_full)
        )

        try:
            full_response = ""
            for chunk in self._llm.stream(self.conversation_history, system, max_tokens=4096):
                full_response += chunk
                yield chunk

            assistant_role = self._llm.assistant_role()
            self.conversation_history.append(
                self._llm.make_message(assistant_role, full_response)
            )

            # Persist conversation periodically
            if len(self.conversation_history) % 10 == 0:
                history_dicts = [
                    {"role": m["role"], "content": m["content"]}
                    for m in self.conversation_history
                ]
                self.memory.save_conversation(
                    history_dicts,
                    summary=f"Conversazione in corso ({len(self.conversation_history)} messaggi)",
                )

        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise

    def reset_conversation(self) -> None:
        if self.conversation_history:
            history_dicts = [
                {"role": m["role"], "content": m["content"]}
                for m in self.conversation_history
            ]
            self.memory.save_conversation(
                history_dicts,
                summary=f"Sessione chiusa ({len(self.conversation_history)} messaggi)",
            )
        self.conversation_history = []

    def get_conversation_length(self) -> int:
        return len(self.conversation_history)

    # ── Alert check ───────────────────────────────────────────────────────────

    def check_for_alerts(self) -> list[dict]:
        alerts = []
        if not self._last_ff_snapshot:
            return alerts
        snap = self._last_ff_snapshot

        for ev in snap.high_impact_upcoming:
            ev_time = ev.get("time", "")
            if ev_time and self._is_soon(ev_time, minutes=30):
                alerts.append({
                    "type": f"EVENTO IMMINENTE: {ev.get('title', 'Unknown')}",
                    "trigger_data": ev,
                    "priority": "high",
                })

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
        try:
            now = datetime.now()
            event_time = datetime.strptime(time_str, "%H:%M").replace(
                year=now.year, month=now.month, day=now.day
            )
            diff = (event_time - now).total_seconds() / 60
            return 0 <= diff <= minutes
        except ValueError:
            return False
