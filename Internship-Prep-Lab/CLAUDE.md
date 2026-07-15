# CLAUDE.md — Bauanleitung Internship Prep Lab

Du (Claude) baust in diesem Repository Übungsmaterial. Die inhaltliche Quelle ist `INTERNSHIP_PREP_PROJEKTE.md` (der Katalog). Der Fortschritt steht ausschließlich in `PROGRESS.md`.

## Session-Start (immer gleich)
1. Lies NUR den Kopf von `PROGRESS.md` (Zeiger „NÄCHSTE AKTION" + Legende).
2. Lies im Katalog NUR den Eintrag des betroffenen Projekts.
3. Baue genau diese eine Einheit. Nicht mehr.
4. Nach Fertigstellung: `PROGRESS.md` aktualisieren, dann Git-Commit. Erst danach ggf. die nächste Einheit beginnen.

## Was pro Projekt entsteht
Jedes Projekt bekommt einen eigenen Ordner nach dem Muster `phase-N/NN-projekt-slug/` (z. B. `phase-1/01-window-functions/`) mit:

- `README.md` — Aufgabenstellung in eigenen Worten (Katalogtext ausformulieren, nicht kopieren): Lernziel, Schritt-für-Schritt-Aufgaben, erwartete Ergebnisse/Selbstchecks, Stretch-Goal. Deutsch, ohne Emojis.
- **Übungsdatei(en) als Stub**: Notebook oder Skript mit Struktur + TODO-Markierungen. Der Kern ist vom Nutzer zu schreiben; vorgegeben sind nur Setup, Datenladen und Prüf-Zellen.
- `loesung/` — vollständig gelöste und GETESTETE Version jeder Übungsdatei.
- `daten/generate_data.py` — falls Daten nötig: Generator mit festem Seed. Erzeugte Dateien werden NICHT eingecheckt (`.gitignore` beachten/ergänzen).

**Code-Vorgabe-Regel** (wie in den anderen Labs): Stretch-Goals erhalten NIE vorgegebenen Code, nur die Aufgabenbeschreibung. Bei mit „Capstone" oder „final" markierten Einheiten (Phase 7) gibt es keinerlei Code-Stub — nur README-Spezifikation + `loesung/`.

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
- **Sprache:** Erklärtexte Deutsch, Code/Bezeichner Englisch, keine Emojis.

## Rolle als Tutor (falls der Nutzer ein Projekt BEARBEITET statt bauen lässt)
Erst arbeiten lassen, dann helfen: Hinweise vor Lösungen, Lösung nur auf ausdrückliche Anfrage oder Verweis auf `loesung/`. Interpretationsfragen am Ende jedes Projekts ernst nehmen — ohne schriftliche Interpretation gilt ein Projekt nicht als abgeschlossen.
