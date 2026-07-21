# SETUP — Einmalige Einrichtung (macOS)

Diese Anleitung richtet sich an dich als Einsteiger. Einmal durcharbeiten, danach brauchst du nur noch den Abschnitt „Täglicher Ablauf".

## Was wird hier eigentlich eingerichtet?

- **Python**: die Programmiersprache, in der fast alle Projekte laufen. Ist bei dir schon installiert (Version 3.13).
- **Virtuelle Umgebung (venv)**: ein isolierter Ordner (`.venv/`), in den alle benötigten Bibliotheken installiert werden — so bleibt dein System-Python sauber und alle Projekte nutzen dieselben, getesteten Versionen.
- **Jupyter**: eine Umgebung, in der Code in „Notebooks" (`.ipynb`-Dateien) zellenweise ausgeführt wird — Code, Erklärungstext und Grafiken in einem Dokument. Viele Lernprojekte hier sind Notebooks.

## 1. Einmalige Einrichtung

Öffne das Programm **Terminal** (Spotlight: `⌘ + Leertaste`, dann „Terminal" tippen) und führe diese Befehle aus:

```bash
# 1. In den Repo-Ordner wechseln
cd ~/Documents/xtAI_Learn-Repository

# 2. Virtuelle Umgebung anlegen (erzeugt den Ordner .venv/)
python3 -m venv .venv

# 3. Umgebung aktivieren (muss man in jedem neuen Terminal-Fenster wiederholen)
source .venv/bin/activate

# 4. Alle Bibliotheken installieren (dauert ein paar Minuten)
pip install -r requirements.txt
```

Wenn die Umgebung aktiv ist, steht `(.venv)` am Anfang deiner Terminal-Zeile.

## 2. Täglicher Ablauf

```bash
cd ~/Documents/xtAI_Learn-Repository
source .venv/bin/activate
jupyter lab
```

`jupyter lab` öffnet automatisch deinen Browser mit einer Dateiübersicht. Dort navigierst du z. B. zu `modules/01-introduction-in-ai/projects/01-basic/` und öffnest das Notebook per Doppelklick.

- **Zelle ausführen**: `Shift + Enter`
- **Alles neu ausführen**: Menü *Run → Run All Cells*
- **Jupyter beenden**: Browser-Tab schließen, im Terminal `Ctrl + C` drücken und mit `y` bestätigen.

## 3. Python-Skripte (Projekte ohne Notebook)

Manche Projekte sind normale `.py`-Dateien. Die führst du im Terminal aus (bei aktivierter venv):

```bash
python modules/01-introduction-in-ai/projects/02-medium/tictactoe.py
```

## 4. Wenn etwas nicht klappt

- `command not found: jupyter` → venv vergessen zu aktivieren (`source .venv/bin/activate`).
- `ModuleNotFoundError: No module named ...` → gleiches Problem, oder `pip install -r requirements.txt` erneut ausführen.
- Notebook zeigt komische Ergebnisse nach Herumprobieren → *Kernel → Restart Kernel and Run All Cells* setzt alles zurück.

## 5. Was liegt wo?

- `modules/` — ein Ordner pro Modul, jeweils mit Lernskript (`README.md`) und `projects/`
- `progress.md` — was fertig ist und was als Nächstes gebaut wird
- `module-liste.md` — alle Module des Studiengangs in Lernreihenfolge
- `requirements.txt` — die Liste aller Python-Bibliotheken
