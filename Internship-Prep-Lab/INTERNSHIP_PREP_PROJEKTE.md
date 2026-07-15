# Internship Prep Lab — Der Praxiskatalog

> **Ziel:** Die Lücke zwischen "Ich kann ein Modell in einem Notebook trainieren" und "Ich kann in einem Data-Science-Team produktiv mitarbeiten" schließen. Dieses Lab ergänzt die bestehenden Labs (Statistik, ML/DS, Module) gezielt um das, was dort fehlt: Analytics-SQL, Data Engineering, Cloud-Konzepte, BI, kollaboratives Git und produktionsreifen Code.
>
> **Referenzprofil:** Data-Science-Praktikum in der Industrie. Typischer Stack: Python, SQL, AWS (Redshift/Athena/Glue/S3), Airflow, dbt, Looker, Git.

## Wie du diesen Katalog benutzt

- Die Projekte sind in **7 Phasen** organisiert. Phasen 1–3 sind Grundhandwerk (vor jeder Bewerbung sinnvoll), Phasen 4–6 sind der Tooling-Kern, Phase 7 ist das Portfolio-Finale.
- Alles läuft **lokal und CPU-freundlich**. Cloud-Dienste werden durch lokale Äquivalente ersetzt, die dieselben Konzepte und teils dieselben APIs nutzen:

| Industrie-Tool | Lokales Äquivalent hier | Warum das trägt |
|---|---|---|
| Redshift / Athena | DuckDB | Gleiche SQL-Konzepte (Window Functions, CTEs), Athena ist wie DuckDB ein Query-Layer über Dateien |
| S3 | MinIO (Docker) oder lokales Verzeichnis + boto3 | MinIO spricht die S3-API; boto3-Code ist identisch |
| Glue / Data Catalog | Parquet + Hive-Partitionierung | Das Konzept (Schema + Partitionen auf Dateien) ist dasselbe |
| Airflow | Airflow standalone (läuft lokal) | Ist das echte Tool |
| dbt | dbt-core + DuckDB-Adapter | Ist das echte Tool, nur mit lokalem Backend |
| Looker | Metabase (Docker) oder Streamlit | Semantik: Metriken definieren, Dashboards bauen |

- **Regel Nr. 1:** Jedes Projekt wird wie ein Arbeitsauftrag behandelt: eigener Branch, sauberer Commit-Verlauf, kurze schriftliche Zusammenfassung ("Was habe ich gebaut, welche Entscheidung ermöglicht es?").
- **Regel Nr. 2:** Business-Framing zählt. Ein Ergebnis ohne Empfehlung ("was sollte das Team jetzt tun?") gilt nicht als abgeschlossen.
- **Regel Nr. 3:** Erst selbst versuchen, dann Lösung anschauen.

**Empfohlener Stack:** Python, DuckDB, dbt-core, Apache Airflow, pandas, pytest, Streamlit oder Metabase, Docker (optional, für MinIO/Metabase), Git + GitHub.

---

## Phase 1 — Analytics-SQL (Die Sprache des Teams)

SQL ist im Praktikum Alltagssprache. Die bestehenden Labs streifen SQL nur; hier wird es auf Analytics-Niveau gehoben. Alle Projekte auf einer gemeinsamen synthetischen E-Commerce-Datenbank (Generator-Skript mit festem Seed: users, orders, order_items, products, sessions, marketing_spend).

### 1.1 Window Functions bis zum Reflex
- **Konzepte:** ROW_NUMBER, RANK, LAG/LEAD, gleitende Aggregationen, PARTITION BY
- **Aufgabe:** Beantworte 15 realistische Analystenfragen ausschließlich in SQL (DuckDB): Top-3-Produkte je Kategorie und Monat, Umsatzveränderung je Kunde zum Vormonat, rollierender 7-Tage-Umsatz, Zeit zwischen erster und zweiter Bestellung, Anteil Wiederkäufer je Kohorte.
- **Daten:** Generiert (E-Commerce-DB)
- **Stretch:** Löse fünf der Fragen zusätzlich in pandas und vergleiche Lesbarkeit und Laufzeit. Formuliere eine Regel, wann SQL und wann pandas das richtige Werkzeug ist.

### 1.2 Kohortenanalyse und Retention
- **Konzepte:** Kohorten, Retention-Matrix, CTEs, bedingte Aggregation
- **Aufgabe:** Baue in reinem SQL eine monatliche Retention-Matrix (Kohorte = Monat der Erstbestellung). Visualisiere sie als Heatmap (matplotlib). Interpretiere: Welche Kohorte fällt ab, und welche Hypothesen erklären das?
- **Daten:** Generiert (mit eingebautem Kohorteneffekt, z. B. eine schlechte Akquise-Kampagne)
- **Stretch:** Parametrisiere die Analyse als wiederverwendbare SQL-Vorlage (Kohorte wahlweise nach Woche/Monat/Marketingkanal).

### 1.3 Der Metrik-Layer: Eine Kennzahl, eine Wahrheit
- **Konzepte:** Metrikdefinitionen, Grain, Fallstricke bei Joins (Fan-out), Deduplizierung
- **Aufgabe:** Definiere fünf Kernmetriken (Umsatz, AOV, Conversion, aktive Nutzer, CAC) jeweils schriftlich (Definition, Grain, Ausschlüsse) und als SQL-View. Baue dann drei absichtlich falsche Varianten (z. B. Umsatz mit Join-Fan-out doppelt gezählt) und dokumentiere, woran man den Fehler erkennt.
- **Daten:** Generiert
- **Stretch:** Schreibe Datenqualitäts-Checks als SQL (Nulls, Duplikate, Referenzintegrität), die bei Verletzung eine Fehlermeldung ausgeben.

### 1.4 Query-Performance verstehen
- **Konzepte:** EXPLAIN, Spaltenformate, Partitionierung, Predicate Pushdown
- **Aufgabe:** Exportiere die E-Commerce-DB als CSV und als partitioniertes Parquet (nach Jahr/Monat). Miss dieselben fünf Queries auf beiden Formaten (DuckDB über Dateien, wie Athena über S3). Erkläre die Unterschiede mit EXPLAIN.
- **Daten:** Generiert (skaliere den Generator auf einige Millionen Zeilen)
- **Stretch:** Finde die Partitionierungsstrategie, die deine häufigste Query am stärksten beschleunigt, und begründe sie.

---

## Phase 2 — Git und Kollaboration (Arbeiten wie im Team)

Git kennst du als Speicherwerkzeug; im Team ist es ein Kommunikationswerkzeug. Diese Phase simuliert Teamabläufe, auch wenn du allein arbeitest.

### 2.1 Feature-Branch-Workflow durchspielen
- **Konzepte:** Branches, Pull Requests, Reviews, sauberer Verlauf
- **Aufgabe:** Nimm ein bestehendes Projekt aus einem deiner Labs. Setze drei Änderungen ausschließlich über Feature-Branches und Pull Requests auf GitHub um (inkl. PR-Beschreibung: Was, Warum, Wie getestet). Reviewe jeden PR selbst schriftlich, bevor du mergst.
- **Daten:** Bestehendes Projekt
- **Stretch:** Nutze `git rebase -i`, um einen unordentlichen Branch (5 chaotische Commits) vor dem Merge auf 2 sinnvolle Commits zu bereinigen.

### 2.2 Merge-Konflikte provozieren und lösen
- **Konzepte:** Konflikte, `git log`/`blame`/`bisect`, Wiederherstellung
- **Aufgabe:** Erzeuge in einem Testrepo gezielt drei Konfliktszenarien (gleiche Zeile auf zwei Branches, verschobene Datei, gelöschte vs. geänderte Datei) und löse sie. Verstecke dann in einem von 20 Commits einen Bug und finde ihn mit `git bisect`.
- **Daten:** Testrepo
- **Stretch:** Simuliere ein "Ich habe auf main committet und gepusht, was nun?"-Szenario und dokumentiere den sauberen Rückweg (revert vs. reset und wann welches).

### 2.3 Repo-Hygiene und Automatisierung
- **Konzepte:** pre-commit, Formatierung, Linting, CI
- **Aufgabe:** Richte in einem Lab pre-commit mit ruff (Lint + Format) ein. Ergänze eine GitHub Action, die bei jedem Push Tests und Linting ausführt. Dokumentiere den Setup-Weg in einer CONTRIBUTING.md.
- **Daten:** Bestehendes Lab
- **Stretch:** Lass die CI zusätzlich ein Jupyter-Notebook headless ausführen (nbconvert) und bei Fehlern fehlschlagen.

---

## Phase 3 — Produktionsreifer Python-Code (Vom Notebook zum Modul)

### 3.1 Notebook-Refactoring
- **Konzepte:** Funktionen/Module, Projektstruktur, Konfiguration, Docstrings, Type Hints
- **Aufgabe:** Nimm ein fertiges Analyse-Notebook aus deinen Labs und refaktoriere es in ein Python-Paket (`src/`-Layout): Datenladen, Transformation, Auswertung als getrennte Module; Parameter in einer config-Datei; ein schlanker CLI-Einstiegspunkt (`python -m projekt run`). Das Notebook bleibt als dünne Präsentationsschicht, die nur noch Funktionen aufruft.
- **Daten:** Bestehendes Projekt
- **Stretch:** Ergänze Logging (logging-Modul, sinnvolle Level) statt print und begründe, was auf INFO vs. DEBUG gehört.

### 3.2 Testen von Datencode
- **Konzepte:** pytest, Fixtures, Parametrisierung, Testen von Transformationen
- **Aufgabe:** Schreibe für das refaktorierte Paket aus 3.1 eine Testsuite: Unit-Tests für Transformationen (kleine handgebaute DataFrames als Fixtures), Edge Cases (leerer Input, Nulls, Duplikate), ein Integrationstest über die ganze Pipeline. Ziel: Du traust dich, den Code umzubauen, weil die Tests dich fangen.
- **Daten:** Aus 3.1
- **Stretch:** Miss Coverage und entscheide bewusst, welche 20 % du NICHT testest und warum.

### 3.3 Reproduzierbare Umgebungen
- **Konzepte:** Virtuelle Umgebungen, Lockfiles, Seeds, deterministische Ergebnisse
- **Aufgabe:** Mache ein Projekt vollständig reproduzierbar: gepinnte Abhängigkeiten (requirements.txt aus pip freeze oder uv/pip-tools), fixierte Seeds, ein `make setup && make run` (oder Shell-Skript), das auf einem frischen Rechner identische Zahlen liefert. Verifiziere durch Löschen und Neuaufsetzen der Umgebung.
- **Daten:** Bestehendes Projekt
- **Stretch:** Verpacke dasselbe Projekt in ein minimales Dockerfile und vergleiche beide Ansätze.

---

## Phase 4 — Data Pipelines (dbt und Airflow)

Der Kern des fehlenden Stacks. Grundlage ist wieder die E-Commerce-DB aus Phase 1.

### 4.1 dbt-Grundlagen: Rohdaten zu Modellen
- **Konzepte:** dbt-core, Staging/Marts, ref(), Jinja, dbt-Tests, Dokumentation
- **Aufgabe:** Baue ein dbt-Projekt auf DuckDB: Staging-Modelle (Umbenennen, Typisieren, Deduplizieren) und darauf Marts (dim_customers, fct_orders, tägliche Umsatz-Marts). Ergänze dbt-Tests (unique, not_null, relationships) und generiere die Doku-Site (`dbt docs`).
- **Daten:** E-Commerce-DB aus Phase 1
- **Stretch:** Baue eine inkrementelle Materialisierung für die Faktentabelle und zeige, dass ein zweiter Lauf nur neue Daten verarbeitet.

### 4.2 Airflow: Orchestrierung verstehen
- **Konzepte:** DAGs, Tasks, Abhängigkeiten, Scheduling, Retries, Idempotenz
- **Aufgabe:** Installiere Airflow lokal (standalone-Modus). Baue einen DAG mit vier Tasks: Rohdaten generieren (simulierter täglicher Ingest) → Qualitätscheck → dbt run → dbt test. Konfiguriere Retries und eine Benachrichtigung (Log reicht) bei Fehlschlag. Lass ihn drei "Tage" per Backfill laufen.
- **Daten:** Aus 4.1
- **Stretch:** Baue einen Task absichtlich nicht-idempotent (Anhängen statt Überschreiben), zeige den Schaden bei einem Retry, und repariere ihn. Idempotenz ist das wichtigste Pipeline-Konzept überhaupt.

### 4.3 Der Datenqualitäts-Wachhund
- **Konzepte:** Data Quality, Erwartungen, Alarmierung, Dead-Letter-Denken
- **Aufgabe:** Erweitere die Pipeline: Der Ingest liefert an "Tag 4" absichtlich kaputte Daten (Schema-Drift: umbenannte Spalte; inhaltlich: negative Preise, Duplikate). Die Pipeline soll das erkennen, den Lauf stoppen, die schlechten Zeilen in eine Quarantäne-Tabelle schreiben und einen verständlichen Fehlerreport loggen.
- **Daten:** Aus 4.2, mit Fehler-Injektor
- **Stretch:** Unterscheide zwischen "Pipeline stoppen" (Schema kaputt) und "warnen, aber durchlaufen" (2 % Ausreißer) und implementiere beide Pfade.

---

## Phase 5 — Cloud-Konzepte lokal (Der S3/Athena-Layer)

### 5.1 Objektspeicher mit boto3
- **Konzepte:** S3-API, Buckets, Keys, Upload/Download, Präfix-Layout
- **Aufgabe:** Starte MinIO per Docker (oder nutze zur Not ein lokales Verzeichnis mit identischer Ordnerlogik). Schreibe ein Skript, das die täglichen Ingest-Dateien aus Phase 4 nach dem Muster `raw/orders/year=2026/month=07/day=11/orders.parquet` hochlädt und wieder listet/liest — mit boto3, exakt wie gegen echtes S3.
- **Daten:** Aus Phase 4
- **Stretch:** Implementiere ein "neueste Partition finden"-Utility, wie man es in fast jedem Data-Team braucht.

### 5.2 Query über Dateien: Dein eigenes Athena
- **Konzepte:** Externe Tabellen, Partition Pruning, Schema-on-Read
- **Aufgabe:** Frage die Parquet-Dateien aus 5.1 direkt mit DuckDB ab (`read_parquet` mit Glob und Hive-Partitionierung). Zeige mit Zeitmessung, dass ein Filter auf die Partitionsspalte nur die betroffenen Dateien liest. Schreibe eine halbe Seite: Was ist bei Athena/Glue konzeptionell identisch, was anders?
- **Daten:** Aus 5.1
- **Stretch:** Verbinde 4 und 5: Die Airflow-Pipeline schreibt nach "S3", dbt liest von dort. Damit steht die Architektur der Ausschreibung im Kleinen.

---

## Phase 6 — BI und Kommunikation (Der Looker-Layer)

### 6.1 Dashboard mit Semantik
- **Konzepte:** BI-Tools, Self-Service, Metrik-Konsistenz
- **Aufgabe:** Starte Metabase per Docker und verbinde es mit deiner DuckDB/den Marts (alternativ: Streamlit-Dashboard). Baue ein Executive-Dashboard mit den fünf Metriken aus 1.3: Trend, Vergleich zum Vormonat, Aufriss nach Kanal. Jede Kachel beantwortet eine konkrete Frage — keine Deko-Charts.
- **Daten:** Marts aus Phase 4
- **Stretch:** Lass eine zweite Person (oder dich nach einer Woche) das Dashboard ohne Erklärung benutzen und notiere, wo es missverständlich ist. Iteriere einmal.

### 6.2 Das Entscheidungs-Memo
- **Konzepte:** Stakeholder-Kommunikation, Empfehlung, Unsicherheit ehrlich kommunizieren
- **Aufgabe:** Nimm eine abgeschlossene Analyse (z. B. die Kohortenanalyse aus 1.2 oder einen A/B-Test aus deinen anderen Labs) und schreibe ein einseitiges Memo an eine fiktive Produktmanagerin: Kernaussage in zwei Sätzen zuerst, dann Evidenz, dann Empfehlung mit Risiken. Keine Methodendetails im Hauptteil (Anhang erlaubt).
- **Daten:** Bestehende Analyse
- **Stretch:** Schreibe dieselbe Analyse einmal für die PM und einmal für einen Senior Data Scientist und mache die Unterschiede explizit.

### 6.3 Analyse unter Zeitdruck
- **Konzepte:** Priorisierung, 80/20, Ad-hoc-Anfragen
- **Aufgabe:** Simuliere eine typische Praktikumsanfrage: "Der Umsatz in Kanal X ist letzte Woche eingebrochen — warum? Brauchen wir bis 15 Uhr." Gib dir 90 Minuten Timer auf der E-Commerce-DB (baue vorher per Skript eine Ursache ein, ohne sie dir anzusehen). Liefere: 3 Bullet Points + 1 Chart + nächster Schritt.
- **Daten:** Generiert mit verdeckter Ursache
- **Stretch:** Führe Protokoll, wohin deine 90 Minuten flossen, und leite ab, was du beim nächsten Mal überspringst.

---

## Phase 7 — Capstone: Die Mini-Datenplattform (Portfolio-Stück)

### 7.1 End-to-End: Von Rohdaten zur Empfehlung
- **Konzepte:** Alles zusammen
- **Aufgabe:** Baue in einem eigenen Repo den kompletten Fluss: täglicher simulierter Ingest → "S3" (MinIO/Parquet-Partitionen) → Airflow-DAG orchestriert Qualitätschecks und dbt → Marts in DuckDB → Dashboard (Metabase/Streamlit) → darauf aufbauend EIN Modellprojekt mit Business-Bezug aus der Ausschreibung (Wahl: CLV-Schätzung, Churn-Ranking oder Nachfrage-Forecast; das Modell darfst du aus deinen bestehenden Labs wiederverwenden — hier zählt die Integration) → einseitiges Entscheidungs-Memo.
- **Anforderungen:** README mit Architekturdiagramm, `make demo` (oder Skript), das alles von null startet; Tests für die Kerntransformationen; sauberer Branch/PR-Verlauf.
- **Stretch:** Schreibe die README so, dass sie in einem Bewerbungsgespräch als Gesprächsgrundlage dient: Welche Trade-offs hast du wo entschieden und warum?

---

## Empfohlener Lernpfad

- **Minimalpfad vor einer Bewerbung (ca. 4–6 Wochen nebenbei):** 1.1 → 1.2 → 2.1 → 3.1 → 4.1 → 4.2 → 6.2
- **Vollständig:** Phase 1 → 2 → 3 → 4 → 5 → 6 → 7. Innerhalb einer Phase ist die Reihenfolge frei, die Nummerierung ist eine sinnvolle Voreinstellung.
- Phasen 1–3 kannst du parallel zu deinen bestehenden Labs laufen lassen; Phasen 4–7 bauen aufeinander auf.

## Abgrenzung zu den bestehenden Labs

Dieses Lab enthält bewusst KEINE Statistik-, ML- oder Modellierungstheorie — die liegt in `statistics_practice_lab`, `ds-practice-lab`, `ML_DS_ITOP_Learn-Repository` und `xtAI_Learn-Repository`. Wo ein Projekt ein Modell braucht (7.1), wird es von dort wiederverwendet. Umgekehrt gilt: Wer ein Modellprojekt aus den anderen Labs "praktikumsreif" machen will, wendet die Phasen 2–3 dieses Labs darauf an.
