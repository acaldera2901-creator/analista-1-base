"""
Self-Improvement Engine — periodically reviews past analyses,
identifies strengths/weaknesses, and updates the analyst profile.
Uses the shared LLM client (same provider as the rest of the system).
"""

import json
from datetime import datetime

from loguru import logger

from src.config import SELF_IMPROVEMENT_LOOKBACK
from src.memory.memory_manager import MemoryManager
from src.llm_client import build_llm_client


SELF_IMPROVEMENT_PROMPT = """
Sei il motore di auto-miglioramento di un analista finanziario speculativo AI.
Il tuo compito e' analizzare le analisi di mercato passate e identificare:

1. PUNTI DI FORZA: Dove l'analista ha ragionato bene?
2. PUNTI DEBOLI: Dove l'analisi era superficiale, imprecisa o mancava di elementi chiave?
3. PATTERN APPRESI: Quali correlazioni o pattern ricorrenti emergono nei dati di mercato?
4. RACCOMANDAZIONI: Come migliorare le future analisi?

Analisi passate da esaminare:
{analyses_text}

Profilo attuale dell'analista:
{analyst_profile}

Rispondi SEMPRE in formato JSON con questa struttura esatta:
{{
  "strengths": ["lista di punti di forza identificati"],
  "weaknesses": ["lista di punti deboli identificati"],
  "learned_patterns": ["lista di pattern di mercato appresi"],
  "recommendations": ["azioni concrete per migliorarsi"],
  "overall_assessment": "valutazione complessiva in 2-3 frasi",
  "confidence_in_review": "high/medium/low"
}}
"""


class SelfImprovementEngine:
    """Runs periodic meta-analysis reviews to improve analyst quality."""

    def __init__(self, memory: MemoryManager):
        self.memory = memory
        # Use the same LLM client as the rest of the system (not Gemini hardcoded)
        try:
            self._llm = build_llm_client()
        except Exception as e:
            logger.warning(f"SelfImprovementEngine: LLM not available: {e}")
            self._llm = None

    def should_run_review(self) -> bool:
        """Decide if a self-improvement review should run now."""
        from src.config import SELF_IMPROVEMENT_FREQUENCY
        profile = self.memory.get_analyst_profile()
        count = profile.get("analysis_count", 0)
        log = self.memory.get_self_improvement_log()

        if not log:
            return count >= 3

        last_review_count = log[-1].get("analysis_count_at_review", 0)
        return (count - last_review_count) >= SELF_IMPROVEMENT_FREQUENCY

    def run_review(self) -> dict:
        """Execute a self-improvement review using the shared LLM client."""
        if self._llm is None:
            logger.warning("Self-improvement: LLM client not available, skipping.")
            return {}

        logger.info("Running self-improvement review...")

        recent_analyses = self.memory.get_recent_analyses(n=SELF_IMPROVEMENT_LOOKBACK)
        if not recent_analyses:
            logger.info("No analyses to review yet.")
            return {}

        analyses_parts = []
        for i, rec in enumerate(recent_analyses, 1):
            ts = rec.get("timestamp", "")[:16]
            typ = rec.get("type", "unknown")
            analysis = rec.get("analysis", "")
            analyses_parts.append(f"[{i}] [{ts}] Tipo: {typ}\n{analysis[:600]}")

        analyses_text = "\n\n---\n\n".join(analyses_parts)
        analyst_profile = json.dumps(
            self.memory.get_analyst_profile(), ensure_ascii=False, indent=2
        )

        prompt = SELF_IMPROVEMENT_PROMPT.format(
            analyses_text=analyses_text,
            analyst_profile=analyst_profile,
        )

        try:
            system = (
                "Sei un sistema di meta-analisi. Rispondi SOLO con JSON valido, "
                "senza testo extra prima o dopo."
            )
            text = self._llm.call(prompt, system, max_tokens=1000)

            review = self._parse_json_response(text)
            if not review:
                logger.warning("Self-improvement: could not parse JSON response")
                return {}

            profile = self.memory.get_analyst_profile()
            current_strengths = profile.get("strengths", [])
            current_weaknesses = profile.get("weaknesses", [])
            current_patterns = self.memory.get_recurring_patterns().get("patterns", [])

            # Merge and deduplicate, keeping last 15 of each
            new_strengths = list(dict.fromkeys(review.get("strengths", []) + current_strengths))[:15]
            new_weaknesses = list(dict.fromkeys(review.get("weaknesses", []) + current_weaknesses))[:15]
            new_patterns = list(dict.fromkeys(review.get("learned_patterns", []) + current_patterns))[:20]

            self.memory.update_analyst_profile({
                "strengths": new_strengths,
                "weaknesses": new_weaknesses,
                "version": profile.get("version", 1) + 1,
            })
            self.memory.update_recurring_patterns(new_patterns)

            review["analysis_count_at_review"] = profile.get("analysis_count", 0)
            self.memory.save_self_improvement_review(review)

            logger.info(
                f"Self-improvement complete: {len(review.get('strengths', []))} strengths, "
                f"{len(review.get('weaknesses', []))} weaknesses, "
                f"{len(review.get('learned_patterns', []))} patterns"
            )
            return review

        except Exception as e:
            logger.error(f"Self-improvement review failed: {e}")
            return {}

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        import re

        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find first { ... } block (non-greedy to avoid capturing extra data)
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}

    def get_improvement_summary(self) -> str:
        """Return a human-readable summary of recent improvements."""
        log = self.memory.get_self_improvement_log()
        if not log:
            return "Nessuna revisione di auto-miglioramento ancora effettuata."

        last = log[-1]
        lines = [
            f"Ultima revisione: {last.get('timestamp', '')[:16]}",
            f"Valutazione: {last.get('overall_assessment', 'N/A')}",
            "",
            "Punti di forza:",
        ]
        for s in last.get("strengths", [])[:3]:
            lines.append(f"  + {s}")
        lines.append("\nAree di miglioramento:")
        for w in last.get("weaknesses", [])[:3]:
            lines.append(f"  - {w}")

        return "\n".join(lines)
