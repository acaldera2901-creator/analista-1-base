"""
Creates the recommended Google Drive folder structure for the analyst.
Run this once to set up the folder skeleton, then upload the `data/` directory.

Folder structure:
  📁 HedgeFund-Analista/
  ├── 📁 Analisi/
  │   ├── 📁 2025/
  │   │   ├── 📁 01-Gennaio/
  │   │   └── ...
  ├── 📁 Dati-Mercato/
  ├── 📁 Memoria-Analista/
  ├── 📁 Performance/
  └── 📁 Log/

Usage:
    python scripts/setup_drive_folder.py
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

MONTHS_IT = [
    "01-Gennaio", "02-Febbraio", "03-Marzo", "04-Aprile",
    "05-Maggio", "06-Giugno", "07-Luglio", "08-Agosto",
    "09-Settembre", "10-Ottobre", "11-Novembre", "12-Dicembre",
]

BASE = Path("./HedgeFund-Analista")
CURRENT_YEAR = datetime.now().year


def create_structure():
    dirs = [
        BASE / "Analisi" / str(CURRENT_YEAR) / month
        for month in MONTHS_IT
    ] + [
        BASE / "Analisi" / str(CURRENT_YEAR + 1),  # next year placeholder
        BASE / "Dati-Mercato" / "WorldMonitor",
        BASE / "Dati-Mercato" / "ForexFactory",
        BASE / "Memoria-Analista",
        BASE / "Performance" / "Auto-Miglioramento",
        BASE / "Performance" / "Tracking-Previsioni",
        BASE / "Log",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        # Create a .gitkeep so empty dirs are tracked
        (d / ".gitkeep").touch()

    # Create README
    readme = BASE / "README.md"
    readme.write_text(f"""# HedgeFund — Analista Finanziario AI

Struttura dati del sistema analista Marco.

## Cartelle

| Cartella | Contenuto |
|----------|-----------|
| `Analisi/` | Briefing mattutini, alert proattivi, conversazioni |
| `Dati-Mercato/` | Snapshot grezzi da WorldMonitor e ForexFactory |
| `Memoria-Analista/` | Profilo analista, pattern appresi, preferenze |
| `Performance/` | Log auto-miglioramento, tracking previsioni |
| `Log/` | Log applicazione |

## Aggiornamento

Questa cartella viene aggiornata automaticamente dall'agente.
Per sincronizzare con Google Drive, usa Google Drive Desktop o `rclone`.

Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}
""", encoding="utf-8")

    print(f"✅ Struttura cartelle creata in: {BASE.resolve()}")
    print(f"   Cartelle create: {len(dirs)}")
    print(f"\n📤 Ora puoi caricare '{BASE}/' su Google Drive")


if __name__ == "__main__":
    create_structure()
