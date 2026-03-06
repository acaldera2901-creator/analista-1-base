"""
Self-Improvement Engine — periodically reviews past analyses,
identifies strengths/weaknesses, and updates the analyst profile.
Uses Claude to perform the meta-analysis.
"""

import json
from datetime import datetime

import anthropic
from loguru import logger

from src.config import CLAUDE_MODEL, SELF_IMPROVEMENT_LOOKBACK, ANTHROPIC_API_KEY
from src.memory.memory_manager import MemoryManager


SELF_IMPROVEMENT_PROMPT = """
Sei il motore di auto-miglioramento di un analista finanziario speculativo AI.
Il tuo compito è analizzare le analisi di mercato passate e identificare:

1. PUNTI DI FORZA: Dove l'analista ha ragionato bene?
2. PUNTI DEBOLI: Dove l'analisi era superficiale, imprecisa o mancava di elementi chiave?
3. PATTERN APPRESI: Quali correlazioni o pattern ricorrenti emergono nei dati di mercato?
4. RACCOMANDAZIONI: Come migliorare le future analisi?

Analisi passate da esaminare:
{analyses_text}

Profilo attuale dell'analista:
{analyst_profile}

Rispondi SEMPRE in formato JSON con questa struttura:
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
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def should_run_review(self) -> bool:
        """Decide if a self-improvement review should run now."""
        from src.config import SELF_IMPROVEMENT_FREQUENCY
        profile = self.memory.get_analyst_profile()
        count = profile.get("analysis_count", 0)
        log = self.memory.get_self_improvement_log()

        # Always run if no reviews yet
        if not log:
            return count >= 3

        # Run every N analyses
        last_review_count = log[-1].get("analysis_count_at_review", 0)
        return (count - last_review_count) >= SELF_IMPROVEMENT_FREQUENCY

    def run_review(self) -> dict:
        """Execute a self-improvement review using Claude."""
        logger.info("Running self-improvement review...")

        recent_analyses = self.memory.get_recent_analyses(n=SELF_IMPROVEMENT_LOOKBACK)
        if not recent_analyses:
            logger.info("No analyses to review yet.")
            return {}

        # Build analyses text
        analyses_parts = []
        for i, rec in enumerate(recent_analyses, 1):
            ts = rec.get("timestamp", "")[:16]
            typ = rec.get("type", "unknown")
            analysis = rec.get("analysis", "")
            analyses_parts.append(f"[{i}] [{ts}] Tipo: {typ}\n{analysis[:800]}")

        analyses_text = "\n\n---\n\n".join(analyses_parts)
        analyst_profile = json.dumps(self.memory.get_analyst_profile(), ensure_ascii=False, indent=2)

        prompt = SELF_IMPROVEMENT_PROMPT.format(
            analyses_text=analyses_text,
            analyst_profile=analyst_profile,
        )

        try:
            with self.client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                response = stream.get_final_message()

            text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )

            # Extract JSON from response
            review = self._parse_json_response(text)
            if not review:
                logger.warning("Self-improvement: could not parse JSON response")
                return {}

            # Update analyst profile with new insights
            profile = self.memory.get_analyst_profile()
            current_strengths = profile.get("strengths", [])
            current_weaknesses = profile.get("weaknesses", [])
            current_patterns = self.memory.get_recurring_patterns().get("patterns", [])

            new_strengths = list(set(current_strengths + review.get("strengths", [])))[-20:]
            new_weaknesses = list(set(current_weaknesses + review.get("weaknesses", [])))[-20:]
            new_patterns = list(set(current_patterns + review.get("learned_patterns", [])))[-30:]

            self.memory.update_analyst_profile({
                "strengths": new_strengths,
                "weaknesses": new_weaknesses,
                "version": profile.get("version", 1) + 1,
            })
            self.memory.update_recurring_patterns(new_patterns)

            # Save the review
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
        """Extract JSON from Claude's response."""
        import re
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in text
        match = re.search(r"\{.*\}", text, re.DOTALL)
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
            "Punti di forza emersi:",
        ]
        for s in last.get("strengths", [])[:3]:
            lines.append(f"  ✓ {s}")
        lines.append("\nAree di miglioramento:")
        for w in last.get("weaknesses", [])[:3]:
            lines.append(f"  ⚠ {w}")

        return "\n".join(lines)
