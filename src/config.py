"""Central configuration module — reads from .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Analyst settings ──────────────────────────────────────────────────────────
ANALYST_TIMEZONE: str = os.getenv("ANALYST_TIMEZONE", "Europe/Rome")
MORNING_BRIEFING_TIME: str = os.getenv("MORNING_BRIEFING_TIME", "07:30")
ALERT_CHECK_INTERVAL: int = int(os.getenv("ALERT_CHECK_INTERVAL", "15"))  # minutes
MIN_ALERT_IMPACT: str = os.getenv("MIN_ALERT_IMPACT", "high")

# ── Instruments ───────────────────────────────────────────────────────────────
_instruments_raw = os.getenv(
    "INSTRUMENTS",
    "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD,NZDUSD,AUDJPY,EURJPY,GBPJPY,XAUUSD,XAGUSD,BTCUSD,ETHUSD",
)
INSTRUMENTS: list[str] = [i.strip() for i in _instruments_raw.split(",")]

INSTRUMENT_GROUPS = {
    "forex_majors": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"],
    "forex_crosses": ["AUDJPY", "EURJPY", "GBPJPY"],
    "commodities": ["XAUUSD", "XAGUSD"],
    "crypto": ["BTCUSD", "ETHUSD"],
}

# ── LLM Provider ──────────────────────────────────────────────────────────────
# Set LLM_PROVIDER to force a provider: gemini | groq | openrouter | anthropic
# If not set, auto-detected from available API keys (priority: gemini > groq > openrouter > anthropic)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")

# Gemini
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Groq (free, fast — https://console.groq.com)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# OpenRouter (routes to many models — https://openrouter.ai)
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")

# Anthropic
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ── Storage paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
ANALYSES_DIR = Path(os.getenv("ANALYSES_DIR", str(DATA_DIR / "analyses")))
MARKET_DATA_DIR = Path(os.getenv("MARKET_DATA_DIR", str(DATA_DIR / "market_data")))
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", str(DATA_DIR / "memory")))
PERFORMANCE_DIR = Path(os.getenv("PERFORMANCE_DIR", str(DATA_DIR / "performance")))
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs")))

# Create directories if they don't exist
for _dir in [DATA_DIR, ANALYSES_DIR, MARKET_DATA_DIR, MEMORY_DIR, PERFORMANCE_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Web scraping ──────────────────────────────────────────────────────────────
WORLDMONITOR_URL = "https://www.worldmonitor.app"
FOREXFACTORY_URL = "https://www.forexfactory.com"
REQUEST_TIMEOUT = 30  # seconds
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ── Impact levels ─────────────────────────────────────────────────────────────
IMPACT_LEVELS = {"low": 1, "medium": 2, "high": 3, "critical": 4}
MIN_ALERT_IMPACT_VALUE = IMPACT_LEVELS.get(MIN_ALERT_IMPACT.lower(), 3)

# ── Self-improvement ──────────────────────────────────────────────────────────
# How many past analyses to include in the self-improvement review
SELF_IMPROVEMENT_LOOKBACK = 10
# Run self-improvement review every N analyses
SELF_IMPROVEMENT_FREQUENCY = 5
