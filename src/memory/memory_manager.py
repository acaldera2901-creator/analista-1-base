"""
Memory Manager — persistent storage for analyses, market data,
analyst notes, and self-improvement records.

Structure:
  data/
  ├── analyses/
  │   ├── YYYY-MM-DD/
  │   │   ├── morning_briefing.json
  │   │   ├── alert_<timestamp>.json
  │   │   └── conversation_<timestamp>.json
  ├── market_data/
  │   ├── worldmonitor_<timestamp>.json
  │   └── forexfactory_<timestamp>.json
  ├── memory/
  │   ├── analyst_profile.json      ← evolving analyst identity & style
  │   ├── recurring_patterns.json   ← learned market patterns
  │   └── user_preferences.json     ← user communication preferences
  └── performance/
      ├── self_improvement_log.json
      └── accuracy_tracking.json
"""

import json
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.config import (
    ANALYSES_DIR, MARKET_DATA_DIR, MEMORY_DIR, PERFORMANCE_DIR,
    SELF_IMPROVEMENT_LOOKBACK,
)


class MemoryManager:
    """Handles all persistence for the financial analyst agent."""

    def __init__(self):
        self.analyses_dir = ANALYSES_DIR
        self.market_data_dir = MARKET_DATA_DIR
        self.memory_dir = MEMORY_DIR
        self.performance_dir = PERFORMANCE_DIR
        self._init_memory_files()

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_memory_files(self):
        """Create base memory files if they don't exist."""
        files = {
            self.memory_dir / "analyst_profile.json": self._default_analyst_profile(),
            self.memory_dir / "recurring_patterns.json": {"patterns": [], "last_updated": ""},
            self.memory_dir / "user_preferences.json": {"timezone": "Europe/Rome", "language": "it", "verbosity": "detailed"},
            self.performance_dir / "self_improvement_log.json": {"reviews": []},
            self.performance_dir / "accuracy_tracking.json": {"predictions": []},
        }
        for path, default in files.items():
            if not path.exists():
                self._write_json(path, default)

    def _default_analyst_profile(self) -> dict:
        return {
            "name": "Marco",
            "role": "Senior Speculative Financial Analyst",
            "hedge_fund": "Client Hedge Fund",
            "strengths": [],
            "weaknesses": [],
            "learned_preferences": [],
            "analysis_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "version": 1,
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_json(path: Path, default: Any = None) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {path}: {e}")
            return default

    def _today_dir(self) -> Path:
        d = self.analyses_dir / date.today().isoformat()
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Save analyses ─────────────────────────────────────────────────────────

    def save_morning_briefing(self, analysis: str, raw_data: dict) -> Path:
        """Persist the morning briefing analysis."""
        today_dir = self._today_dir()
        path = today_dir / "morning_briefing.json"
        payload = {
            "type": "morning_briefing",
            "timestamp": datetime.utcnow().isoformat(),
            "date": date.today().isoformat(),
            "analysis": analysis,
            "raw_data_summary": {
                "worldmonitor_chars": len(raw_data.get("worldmonitor_text", "")),
                "forexfactory_events_today": len(raw_data.get("ff_events_today", [])),
                "ff_high_impact": len(raw_data.get("ff_high_impact", [])),
            },
        }
        self._write_json(path, payload)
        self._increment_analysis_count()
        logger.info(f"Morning briefing saved to {path}")
        return path

    def save_alert(self, alert_type: str, analysis: str, trigger_data: dict) -> Path:
        """Persist a proactive alert."""
        today_dir = self._today_dir()
        ts = datetime.utcnow().strftime("%H%M%S")
        path = today_dir / f"alert_{ts}_{alert_type}.json"
        payload = {
            "type": "alert",
            "alert_type": alert_type,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": analysis,
            "trigger": trigger_data,
        }
        self._write_json(path, payload)
        self._increment_analysis_count()
        return path

    def save_conversation(self, messages: list[dict], summary: str = "") -> Path:
        """Persist a user conversation session."""
        today_dir = self._today_dir()
        ts = datetime.utcnow().strftime("%H%M%S")
        uid = str(uuid.uuid4())[:8]
        path = today_dir / f"conversation_{ts}_{uid}.json"
        payload = {
            "type": "conversation",
            "timestamp": datetime.utcnow().isoformat(),
            "message_count": len(messages),
            "summary": summary,
            "messages": messages,
        }
        self._write_json(path, payload)
        return path

    def save_market_data(self, source: str, data: dict) -> Path:
        """Persist raw market data snapshots."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self.market_data_dir / f"{source}_{ts}.json"
        self._write_json(path, data)
        return path

    # ── Read analyses ─────────────────────────────────────────────────────────

    def get_recent_analyses(self, n: int = SELF_IMPROVEMENT_LOOKBACK) -> list[dict]:
        """Return the N most recent analysis records across all types."""
        records = []
        # Walk date directories newest-first
        date_dirs = sorted(self.analyses_dir.glob("????-??-??"), reverse=True)
        for day_dir in date_dirs:
            for json_file in sorted(day_dir.glob("*.json"), reverse=True):
                data = self._read_json(json_file)
                if data and "analysis" in data:
                    records.append(data)
                    if len(records) >= n:
                        return records
        return records

    def get_today_analyses(self) -> list[dict]:
        """Return all analyses from today."""
        today_dir = self._today_dir()
        records = []
        for json_file in sorted(today_dir.glob("*.json")):
            data = self._read_json(json_file)
            if data:
                records.append(data)
        return records

    def has_morning_briefing_today(self) -> bool:
        """Check if morning briefing was already sent today."""
        path = self._today_dir() / "morning_briefing.json"
        return path.exists()

    # ── Analyst profile ───────────────────────────────────────────────────────

    def get_analyst_profile(self) -> dict:
        return self._read_json(
            self.memory_dir / "analyst_profile.json",
            default=self._default_analyst_profile(),
        )

    def update_analyst_profile(self, updates: dict) -> None:
        profile = self.get_analyst_profile()
        profile.update(updates)
        profile["last_updated"] = datetime.utcnow().isoformat()
        self._write_json(self.memory_dir / "analyst_profile.json", profile)

    def _increment_analysis_count(self) -> None:
        profile = self.get_analyst_profile()
        profile["analysis_count"] = profile.get("analysis_count", 0) + 1
        profile["last_updated"] = datetime.utcnow().isoformat()
        self._write_json(self.memory_dir / "analyst_profile.json", profile)

    # ── Recurring patterns ────────────────────────────────────────────────────

    def get_recurring_patterns(self) -> dict:
        return self._read_json(
            self.memory_dir / "recurring_patterns.json",
            default={"patterns": [], "last_updated": ""},
        )

    def update_recurring_patterns(self, patterns: list[str]) -> None:
        data = {"patterns": patterns, "last_updated": datetime.utcnow().isoformat()}
        self._write_json(self.memory_dir / "recurring_patterns.json", data)

    # ── Self-improvement log ──────────────────────────────────────────────────

    def save_self_improvement_review(self, review: dict) -> None:
        path = self.performance_dir / "self_improvement_log.json"
        log = self._read_json(path, default={"reviews": []})
        log["reviews"].append({
            "timestamp": datetime.utcnow().isoformat(),
            **review,
        })
        self._write_json(path, log)
        logger.info("Self-improvement review saved.")

    def get_self_improvement_log(self) -> list[dict]:
        path = self.performance_dir / "self_improvement_log.json"
        return self._read_json(path, default={"reviews": []}).get("reviews", [])

    # ── Accuracy tracking ─────────────────────────────────────────────────────

    def save_prediction(self, instrument: str, direction: str, confidence: str, context: str) -> None:
        path = self.performance_dir / "accuracy_tracking.json"
        data = self._read_json(path, default={"predictions": []})
        data["predictions"].append({
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "instrument": instrument,
            "direction": direction,
            "confidence": confidence,
            "context": context,
            "outcome": None,  # updated later
        })
        self._write_json(path, data)

    def get_predictions(self, unresolved_only: bool = False) -> list[dict]:
        path = self.performance_dir / "accuracy_tracking.json"
        preds = self._read_json(path, default={"predictions": []}).get("predictions", [])
        if unresolved_only:
            return [p for p in preds if p.get("outcome") is None]
        return preds

    # ── Context builder for Claude ────────────────────────────────────────────

    def build_memory_context(self) -> str:
        """Build a compact memory context string (< 300 tokens) for the system prompt."""
        profile = self.get_analyst_profile()
        patterns = self.get_recurring_patterns()

        parts = [
            f"Analista: {profile.get('name', 'Marco')} | "
            f"Analisi completate: {profile.get('analysis_count', 0)}"
        ]

        prefs = profile.get("learned_preferences", [])
        if prefs:
            parts.append(f"Preferenze utente: {', '.join(prefs[:3])}")

        pats = patterns.get("patterns", [])
        if pats:
            parts.append(f"Pattern ricorrenti: {', '.join(pats[:3])}")

        recent = self.get_recent_analyses(n=2)
        if recent:
            last = recent[0]
            ts = last.get("timestamp", "")[:16]
            preview = last.get("analysis", "")[:200].replace("\n", " ")
            parts.append(f"Ultima analisi ({ts}): {preview}...")

        return "\n".join(parts)

    # ── Active conversation persistence (cross-session memory) ────────────────

    def save_active_conversation(self, messages: list[dict]) -> None:
        """Persist current conversation to disk so it survives restarts."""
        path = self.memory_dir / "active_conversation.json"
        self._write_json(path, {
            "timestamp": datetime.utcnow().isoformat(),
            "messages": messages,
        })

    def get_active_conversation(self, max_age_hours: float = 4.0) -> list[dict]:
        """
        Load the last conversation if it was saved within max_age_hours.
        Returns empty list if too old or not found.
        """
        path = self.memory_dir / "active_conversation.json"
        data = self._read_json(path)
        if not data or not data.get("messages"):
            return []
        try:
            ts = datetime.fromisoformat(data["timestamp"])
            age_hours = (datetime.utcnow() - ts).total_seconds() / 3600
            if age_hours > max_age_hours:
                logger.info(f"Active conversation too old ({age_hours:.1f}h), starting fresh.")
                return []
        except (ValueError, KeyError):
            return []
        msgs = data["messages"]
        logger.info(f"Restored {len(msgs)} messages from previous session.")
        return msgs
