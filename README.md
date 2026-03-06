# 📊 Financial Analyst Agent — Hedge Fund

**Marco** — Senior Speculative Financial Analyst AI per il tuo hedge fund.

Analista speculativo potenziato da Claude Opus 4.6 con pensiero adattivo, che monitora i principali strumenti CFD (Forex, XAUUSD, BTCUSD), aggrega dati macro da WorldMonitor e il calendario economico da ForexFactory, e comunica via **Telegram**.

---

## ✨ Funzionalità

| Feature | Descrizione |
|---------|-------------|
| 🌅 **Briefing Mattutino** | Ogni giorno alle 07:30 ricevi l'analisi completa del mercato |
| 🚨 **Alert Proattivi** | Marco ti contatta quando succede qualcosa ad alto impatto |
| 💬 **Chat Bidirezionale** | Scrivi a Marco quando vuoi per analisi e confronto |
| 🧠 **Auto-Miglioramento** | Rivede le sue analisi passate e migliora continuamente |
| 💾 **Memoria Persistente** | Ricorda tutte le analisi, i pattern appresi, le tue preferenze |
| 📁 **Struttura Drive** | Cartelle organizzate sincronizzabili su Google Drive |

### Strumenti monitorati
`EURUSD` `GBPUSD` `USDJPY` `AUDUSD` `USDCHF` `USDCAD` `NZDUSD` `XAUUSD` `BTCUSD`

### Fonti dati
- **[WorldMonitor.app](https://www.worldmonitor.app)** — Tassi banche centrali, inflazione, PIL, occupazione, sentiment macro
- **[ForexFactory.com](https://www.forexfactory.com)** — Calendario economico, eventi ricorrenti, notizie ad alto impatto

---

## 🚀 Setup rapido

### 1. Prerequisiti

```bash
python 3.11+
pip install -r requirements.txt
```

### 2. Configurazione

```bash
cp .env.example .env
```

Modifica `.env` con le tue chiavi:

```env
ANTHROPIC_API_KEY=sk-ant-...        # Da console.anthropic.com
TELEGRAM_BOT_TOKEN=123456:ABC...    # Da @BotFather su Telegram
TELEGRAM_CHAT_ID=123456789          # Il tuo chat ID (vedi sotto)
ANALYST_TIMEZONE=Europe/Rome
MORNING_BRIEFING_TIME=07:30
```

#### Come ottenere il TELEGRAM_CHAT_ID
1. Crea un bot con @BotFather → ottieni il token
2. Avvia il bot con `/start`
3. Vai su `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Copia il valore `message.chat.id`

### 3. Setup cartella Drive

```bash
python scripts/setup_drive_folder.py
```

Questo crea la struttura `HedgeFund-Analista/` da caricare su Drive.

### 4. Avvio

```bash
python main.py
```

Marco si avvia, scarica i dati di mercato e ti manda un messaggio su Telegram.

---

## 📱 Comandi Telegram

| Comando | Azione |
|---------|--------|
| `/start` | Presentazione e lista comandi |
| `/briefing` | Genera il briefing adesso (senza aspettare le 7:30) |
| `/status` | Stato del sistema e statistiche |
| `/miglioramento` | Report di auto-miglioramento |
| `/reset` | Resetta la conversazione corrente |
| _(qualsiasi testo)_ | Domanda/analisi libera a Marco |

---

## 🏗️ Architettura

```
analista-1-base/
├── main.py                          # Entry point
├── src/
│   ├── config.py                    # Configurazione centralizzata
│   ├── agent/
│   │   └── analyst_agent.py         # Core agent (Claude Opus 4.6)
│   ├── scrapers/
│   │   ├── worldmonitor_scraper.py  # WorldMonitor.app scraper
│   │   └── forexfactory_scraper.py  # ForexFactory.com scraper
│   ├── memory/
│   │   ├── memory_manager.py        # Persistenza analisi e dati
│   │   └── self_improvement.py      # Motore auto-miglioramento
│   ├── communication/
│   │   └── telegram_bot.py          # Bot Telegram bidirezionale
│   └── scheduler.py                 # Job scheduler (APScheduler)
├── data/
│   ├── analyses/                    # Analisi salvate per data
│   ├── market_data/                 # Snapshot dati di mercato
│   ├── memory/                      # Profilo analista e pattern
│   └── performance/                 # Log auto-miglioramento
├── scripts/
│   ├── run_briefing_now.py          # Genera briefing standalone
│   ├── test_scrapers.py             # Testa i scraper
│   └── setup_drive_folder.py        # Crea struttura Drive
└── logs/                            # Log rotanti giornalieri
```

### Flusso del Briefing Mattutino

```
07:25 → Fetch WorldMonitor + ForexFactory
07:30 → Claude Opus 4.6 analizza i dati (adaptive thinking)
      → Genera briefing strutturato
      → Salva in data/analyses/YYYY-MM-DD/morning_briefing.json
      → Invia su Telegram
```

### Flusso Auto-Miglioramento

```
Ogni 5 analisi → Claude esamina le ultime 10 analisi
               → Identifica punti di forza e debolezza
               → Aggiorna il profilo analista
               → Salva i pattern appresi
               → Notifica su Telegram
```

---

## 🔧 Script utili

```bash
# Testa i scraper (senza avviare il bot)
python scripts/test_scrapers.py

# Genera un briefing manuale
python scripts/run_briefing_now.py

# Crea struttura Google Drive
python scripts/setup_drive_folder.py
```

---

## 🛡️ Note di sicurezza

- Le chiavi API sono in `.env` (non committato in git)
- Il bot risponde solo al `TELEGRAM_CHAT_ID` configurato
- I dati di mercato vengono salvati localmente in `data/`

---

## 📈 Auto-Miglioramento

Marco analizza periodicamente le sue analisi passate usando Claude con extended thinking per:
- Identificare pattern di mercato ricorrenti
- Migliorare la precisione delle aspettative
- Adattare il suo stile comunicativo alle tue preferenze
- Costruire una memoria cumulativa del comportamento dei mercati

Il profilo evolve in `data/memory/analyst_profile.json`.

---

*Powered by [Claude Opus 4.6](https://www.anthropic.com) + APScheduler + python-telegram-bot*
