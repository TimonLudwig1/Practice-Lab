# CLAUDE.md — Build Guide for the Internship Prep Lab

You (Claude) build practice material in this repository. The content source is `INTERNSHIP_PREP_PROJEKTE.md` (the catalog). Progress is tracked exclusively in `PROGRESS.md`.

## Session Start (Always the Same)

1. Read ONLY the header of `PROGRESS.md` (the “NEXT ACTION” pointer and legend).
2. Read ONLY the catalog entry for the relevant project.
3. Build exactly this one unit. Nothing more.
4. After completion: update `PROGRESS.md`, then create a Git commit. Only then may you begin the next unit.

## What Each Project Contains

Each project gets its own folder following the pattern `phase-N/NN-project-slug/` (for example, `phase-1/01-window-functions/`) containing:

- `README.md` — An original project brief based on the catalog rather than copied from it: learning objective, step-by-step tasks, expected results/self-checks, and a stretch goal. Like every Markdown file in this lab, it consists of two complete blocks: the English primary version first, followed immediately by the German translation. Do not mix both languages within individual sections. Do not use emojis.
- **Practice file(s) as stubs:** A notebook or script with structure and TODO markers. The user writes the core implementation; only setup, data loading, and validation cells are provided.
- `loesung/` — A complete and TESTED solution for every practice file.
- `daten/generate_data.py` — If data is required: a generator with a fixed seed. Generated files are NOT committed (review or extend `.gitignore`).

**Provided-code rule** (as in the other labs): Stretch goals NEVER receive starter code, only a task description. Units marked “Capstone” or “final” (Phase 7) receive no code stub at all—only a README specification and `loesung/`.

## Format by Phase (Not Everything Is a Notebook)

- **Phase 1 (SQL):** A notebook using DuckDB (the `duckdb` Python package) OR `.sql` files with a runner script. The shared e-commerce database lives centrally in `shared/ecommerce/` (build it once and use it for every Phase 1 project).
- **Phase 2 (Git):** NO notebook. A `README.md` with step-by-step instructions and, where useful, a `setup_scenario.sh` script that creates a practice repository in the described state (for example, conflict scenarios or a bug for bisect). The solution is `loesung/WALKTHROUGH.md`, containing the exact commands and expected output.
- **Phase 3 (Code Quality):** A Python package using a `src/` layout and `pytest` tests. The solution is a complete package in `loesung/`.
- **Phase 4 (dbt/Airflow):** Real project structures (`dbt_project/`, `dags/`). The stub is a project scaffold with TODO comments in models and DAGs. Verify Airflow DAGs with `airflow dags test <dag_id> <date>` (do not start a permanent scheduler or long-running processes).
- **Phase 5 (S3/Parquet):** Scripts. Only document MinIO (Docker command in the README); the solution must also run WITHOUT Docker by providing and testing a local-directory fallback with identical path semantics.
- **Phase 6 (BI/Communication):** 6.1 uses a runnable Streamlit app (Metabase only as a documented alternative). 6.2/6.3 are writing and analysis exercises: README, templates, and a sample solution in Markdown.
- **Phase 7 (Capstone):** README specification and `loesung/` only (see the provided-code rule).

## Technical Guardrails

- **CPU-friendly and suitable for a MacBook:** No compute-intensive training and no permanently running processes. Everything must finish within seconds or a few minutes.
- **No cloud dependency:** Everything must run locally. Network access is allowed only for package installation.
- **Dependencies:** Maintain them centrally in `requirements.txt` (duckdb, dbt-duckdb, apache-airflow, pandas, pytest, streamlit, boto3, ruff, pre-commit). Before use, check whether they are installed; record missing packages as blockers in `PROGRESS.md` rather than guessing.
- **Testing requirement:** Nothing is marked complete unless it has been executed. Run notebooks headlessly with nbconvert; run scripts/packages with pytest or directly; run dbt with `dbt build`; test DAGs with `airflow dags test`.
- **References to other labs** (the catalog occasionally points to projects in ds-practice-lab and elsewhere): Use referenced material when it is available in the workspace. Otherwise, build a small standalone substitute inside the project folder and note this in the README. Never block solely because another lab is unavailable.
- **Language and structure of every Markdown file:** The lab is English-first. Every `.md` file—regardless of its filename or directory—must contain one complete English block first and then a complete German translation with the same structure and content. Do not interleave languages section by section.
- **Language of project files:** Notebooks, including Markdown cells, docstrings, code comments, TODO markers, code, and identifiers must be written exclusively in English. Do not use emojis.

## Role as a Tutor (When the User Is Solving a Project Rather Than Asking You to Build It)

Let the user work first, then help: provide hints before solutions, and provide a solution only when explicitly requested or by referring to `loesung/`. Take interpretation questions at the end of each project seriously—a project is not complete without a written interpretation.

---

# Deutsche Fassung

# CLAUDE.md — Bauanleitung Internship Prep Lab

Du (Claude) baust in diesem Repository Übungsmaterial. Die inhaltliche Quelle ist `INTERNSHIP_PREP_PROJEKTE.md` (der Katalog). Der Fortschritt steht ausschließlich in `PROGRESS.md`.

## Session-Start (immer gleich)

1. Lies NUR den Kopf von `PROGRESS.md` (Zeiger „NÄCHSTE AKTION“ + Legende).
2. Lies im Katalog NUR den Eintrag des betroffenen Projekts.
3. Baue genau diese eine Einheit. Nicht mehr.
4. Nach Fertigstellung: `PROGRESS.md` aktualisieren, dann Git-Commit. Erst danach ggf. die nächste Einheit beginnen.

## Was pro Projekt entsteht

Jedes Projekt bekommt einen eigenen Ordner nach dem Muster `phase-N/NN-projekt-slug/` (z. B. `phase-1/01-window-functions/`) mit:

- `README.md` — Aufgabenstellung in eigenen Worten (Katalogtext ausformulieren, nicht kopieren): Lernziel, Schritt-für-Schritt-Aufgaben, erwartete Ergebnisse/Selbstchecks, Stretch-Goal. Wie jede Markdown-Datei in diesem Lab besteht sie aus zwei vollständigen Blöcken: zuerst die primäre englische Fassung, direkt danach die deutsche Übersetzung. Die Sprachen nicht innerhalb einzelner Abschnitte vermischen. Keine Emojis.
- **Übungsdatei(en) als Stub:** Notebook oder Skript mit Struktur + TODO-Markierungen. Der Kern ist vom Nutzer zu schreiben; vorgegeben sind nur Setup, Datenladen und Prüf-Zellen.
- `loesung/` — vollständig gelöste und GETESTETE Version jeder Übungsdatei.
- `daten/generate_data.py` — falls Daten nötig: Generator mit festem Seed. Erzeugte Dateien werden NICHT eingecheckt (`.gitignore` beachten/ergänzen).

**Code-Vorgabe-Regel** (wie in den anderen Labs): Stretch-Goals erhalten NIE vorgegebenen Code, nur die Aufgabenbeschreibung. Bei mit „Capstone“ oder „final“ markierten Einheiten (Phase 7) gibt es keinerlei Code-Stub — nur README-Spezifikation + `loesung/`.

## Format je Phase (nicht alles ist ein Notebook)

- **Phase 1 (SQL):** Notebook mit DuckDB (`duckdb` Python-Paket) ODER `.sql`-Dateien + Runner-Skript. Gemeinsame E-Commerce-DB liegt zentral in `shared/ecommerce/` (einmal bauen, alle Phase-1-Projekte nutzen sie).
- **Phase 2 (Git):** KEIN Notebook. `README.md` als Schritt-für-Schritt-Anleitung + ggf. `setup_szenario.sh`, das ein Übungsrepo mit dem beschriebenen Zustand erzeugt (z. B. Konflikt-Szenarien, Bug für bisect). Lösung = `loesung/WALKTHROUGH.md` mit den exakten Befehlen und erwartetem Output.
- **Phase 3 (Code-Qualität):** Python-Paket-Struktur (`src/`-Layout), `pytest`-Tests. Lösung = vollständiges Paket in `loesung/`.
- **Phase 4 (dbt/Airflow):** Echte Projektstruktur (`dbt_project/`, `dags/`). Stub = Projektgerüst mit TODO-Kommentaren in Modellen/DAGs. Airflow-DAGs werden per `airflow dags test <dag_id> <datum>` verifiziert (kein dauerhafter Scheduler, keine langen Prozesse starten).
- **Phase 5 (S3/Parquet):** Skripte. MinIO nur dokumentieren (Docker-Befehl in README); die Lösung muss auch OHNE Docker laufen — dafür einen Fallback auf ein lokales Verzeichnis mit identischer Pfadlogik einbauen und testen.
- **Phase 6 (BI/Kommunikation):** 6.1 Streamlit-App als lauffähige Variante (Metabase nur als dokumentierte Alternative). 6.2/6.3 sind Schreib-/Analyseübungen: README + Vorlagen + Musterlösung als Markdown.
- **Phase 7 (Capstone):** Nur README-Spezifikation + `loesung/` (siehe Code-Vorgabe-Regel).

## Technische Leitplanken

- **CPU-freundlich, MacBook-tauglich:** keine rechenintensiven Trainings, keine Prozesse, die dauerhaft laufen. Alles muss in Sekunden bis wenigen Minuten durchlaufen.
- **Keine Cloud-Abhängigkeit:** alles lokal lauffähig. Netzwerkzugriffe nur für Paketinstallation.
- **Abhängigkeiten:** zentral in `requirements.txt` pflegen (duckdb, dbt-duckdb, apache-airflow, pandas, pytest, streamlit, boto3, ruff, pre-commit). Vor Nutzung prüfen, ob installiert; fehlende Pakete in `PROGRESS.md` als Blocker notieren statt raten.
- **Testpflicht:** Nichts wird als fertig markiert, was nicht ausgeführt wurde. Notebooks headless via nbconvert ausführen; Skripte/Pakete via pytest bzw. direktem Lauf; dbt via `dbt build`; DAGs via `airflow dags test`.
- **Querverweise auf andere Labs** (der Katalog verweist teils auf Projekte aus ds-practice-lab etc.): Wenn das referenzierte Material im Workspace verfügbar ist, nutze es. Wenn nicht, baue ein kleines eigenständiges Ersatzbeispiel im Projektordner und vermerke das in der README. Niemals blockieren, weil ein anderes Lab fehlt.
- **Sprache und Struktur jeder Markdown-Datei:** Das Lab ist primär englischsprachig. Jede `.md`-Datei — unabhängig von Dateiname und Verzeichnis — enthält zuerst einen vollständigen englischen Block und danach eine vollständige deutsche Übersetzung mit derselben Struktur und denselben Inhalten. Die Sprachen nicht abschnittsweise verschachteln.
- **Sprache der Projektdateien:** Notebooks einschließlich Markdown-Zellen, Docstrings, Code-Kommentare, TODO-Markierungen, Code und Bezeichner ausschließlich auf Englisch. Keine Emojis.

## Rolle als Tutor (falls der Nutzer ein Projekt BEARBEITET statt bauen lässt)

Erst arbeiten lassen, dann helfen: Hinweise vor Lösungen, Lösung nur auf ausdrückliche Anfrage oder Verweis auf `loesung/`. Interpretationsfragen am Ende jedes Projekts ernst nehmen — ohne schriftliche Interpretation gilt ein Projekt nicht als abgeschlossen.
