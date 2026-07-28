# Internship Prep Lab — Project Catalog

> **Goal:** Close the gap between “I can train a model in a notebook” and “I can contribute productively to a data science team.” This lab complements the existing labs (statistics, ML/DS, and modules) with the areas they do not cover: analytics SQL, data engineering, cloud concepts, BI, collaborative Git, and production-ready code.
>
> **Reference role:** Data science internship in industry. Typical stack: Python, SQL, AWS (Redshift/Athena/Glue/S3), Airflow, dbt, Looker, Git.

## How to Use This Catalog

- The projects are organized into **7 phases**. Phases 1–3 cover essential foundations (useful before any application), Phases 4–6 form the tooling core, and Phase 7 is the portfolio finale.
- Everything runs **locally and is CPU-friendly**. Cloud services are replaced with local equivalents that use the same concepts and, in some cases, the same APIs:

| Industry Tool | Local Equivalent Used Here | Why It Transfers |
|---|---|---|
| Redshift / Athena | DuckDB | The same SQL concepts (window functions, CTEs); like Athena, DuckDB can query files directly |
| S3 | MinIO (Docker) or a local directory + boto3 | MinIO implements the S3 API; the boto3 code is identical |
| Glue / Data Catalog | Parquet + Hive partitioning | The underlying concept (schema + file partitions) is the same |
| Airflow | Airflow standalone (runs locally) | It is the real tool |
| dbt | dbt-core + DuckDB adapter | It is the real tool with a local backend |
| Looker | Metabase (Docker) or Streamlit | The semantics transfer: define metrics and build dashboards |

- **Rule No. 1:** Treat every project like a work assignment: use a dedicated branch, maintain a clean commit history, and write a short summary (“What did I build, and what decision does it enable?”).
- **Rule No. 2:** Business framing matters. A result without a recommendation (“What should the team do now?”) is not complete.
- **Rule No. 3:** Try it yourself before looking at the solution.

**Recommended stack:** Python, DuckDB, dbt-core, Apache Airflow, pandas, pytest, Streamlit or Metabase, Docker (optional, for MinIO/Metabase), Git + GitHub.

---

## Phase 1 — Analytics SQL (The Team's Language)

SQL is an everyday language during an internship. The existing labs only touch on SQL; this phase develops it to an analytics level. All projects use one shared synthetic e-commerce database generated with a fixed seed: users, orders, order_items, products, sessions, and marketing_spend.

### 1.1 Window Functions Until They Become Instinctive

- **Concepts:** ROW_NUMBER, RANK, LAG/LEAD, rolling aggregations, PARTITION BY
- **Task:** Answer 15 realistic analyst questions using only SQL (DuckDB): top three products by category and month, each customer's revenue change from the previous month, rolling seven-day revenue, time between the first and second order, and the share of repeat buyers by cohort.
- **Data:** Generated (e-commerce database)
- **Stretch:** Solve five questions in pandas as well and compare readability and runtime. Formulate a rule for when SQL is the right tool and when pandas is.

### 1.2 Cohort Analysis and Retention

- **Concepts:** Cohorts, retention matrix, CTEs, conditional aggregation
- **Task:** Build a monthly retention matrix entirely in SQL (cohort = month of first order). Visualize it as a heatmap (matplotlib). Interpret the result: Which cohort drops off, and which hypotheses could explain it?
- **Data:** Generated (with a built-in cohort effect, such as a poor acquisition campaign)
- **Stretch:** Parameterize the analysis as a reusable SQL template that can define cohorts by week, month, or marketing channel.

### 1.3 The Metric Layer: One Metric, One Truth

- **Concepts:** Metric definitions, grain, join pitfalls (fan-out), deduplication
- **Task:** Define five core metrics (revenue, AOV, conversion, active users, CAC) both in writing (definition, grain, exclusions) and as SQL views. Then build three intentionally incorrect variants (for example, revenue counted twice due to join fan-out) and document how to detect each error.
- **Data:** Generated
- **Stretch:** Write data quality checks in SQL (nulls, duplicates, referential integrity) that produce an error message when a rule is violated.

### 1.4 Understanding Query Performance

- **Concepts:** EXPLAIN, columnar formats, partitioning, predicate pushdown
- **Task:** Export the e-commerce database as CSV and as Parquet partitioned by year/month. Benchmark the same five queries on both formats (DuckDB over files, analogous to Athena over S3). Explain the differences with EXPLAIN.
- **Data:** Generated (scale the generator to several million rows)
- **Stretch:** Find the partitioning strategy that accelerates your most frequent query the most and justify it.

---

## Phase 2 — Git and Collaboration (Working as a Team)

You may know Git as a storage tool; within a team, it is a communication tool. This phase simulates team workflows even when you work alone.

### 2.1 Practice the Feature Branch Workflow

- **Concepts:** Branches, pull requests, reviews, clean history
- **Task:** Choose an existing project from one of your labs. Implement three changes exclusively through feature branches and GitHub pull requests (including a PR description: what, why, and how it was tested). Write a self-review for every PR before merging it.
- **Data:** Existing project
- **Stretch:** Use `git rebase -i` to turn a messy branch with five chaotic commits into two meaningful commits before merging.

### 2.2 Create and Resolve Merge Conflicts

- **Concepts:** Conflicts, `git log`/`blame`/`bisect`, recovery
- **Task:** Deliberately create three conflict scenarios in a test repository (the same line changed on two branches, a moved file, and a deleted-versus-modified file) and resolve them. Then hide a bug in one of 20 commits and locate it with `git bisect`.
- **Data:** Test repository
- **Stretch:** Simulate an “I committed and pushed directly to main—what now?” scenario and document the clean recovery path (revert vs. reset and when to use each).

### 2.3 Repository Hygiene and Automation

- **Concepts:** pre-commit, formatting, linting, CI
- **Task:** Configure pre-commit with ruff (lint + format) in one lab. Add a GitHub Action that runs tests and linting on every push. Document the setup process in `CONTRIBUTING.md`.
- **Data:** Existing lab
- **Stretch:** Make CI execute a Jupyter notebook headlessly with nbconvert and fail when the notebook fails.

---

## Phase 3 — Production-Ready Python Code (From Notebook to Module)

### 3.1 Notebook Refactoring

- **Concepts:** Functions/modules, project structure, configuration, docstrings, type hints
- **Task:** Take a completed analysis notebook from your labs and refactor it into a Python package (`src/` layout): separate modules for data loading, transformation, and analysis; parameters in a configuration file; and a small CLI entry point (`python -m project run`). Keep the notebook as a thin presentation layer that only calls functions.
- **Data:** Existing project
- **Stretch:** Replace print statements with logging (using the logging module and meaningful levels) and explain what belongs at INFO versus DEBUG.

### 3.2 Testing Data Code

- **Concepts:** pytest, fixtures, parametrization, testing transformations
- **Task:** Write a test suite for the refactored package from 3.1: unit tests for transformations (small hand-built DataFrames as fixtures), edge cases (empty input, nulls, duplicates), and one integration test for the complete pipeline. The goal is to refactor confidently because the tests catch mistakes.
- **Data:** From 3.1
- **Stretch:** Measure coverage and consciously decide which 20% you will NOT test and why.

### 3.3 Reproducible Environments

- **Concepts:** Virtual environments, lockfiles, seeds, deterministic results
- **Task:** Make a project fully reproducible: pinned dependencies (`requirements.txt` from pip freeze or uv/pip-tools), fixed seeds, and a `make setup && make run` command (or shell script) that produces identical results on a fresh computer. Verify this by deleting and recreating the environment.
- **Data:** Existing project
- **Stretch:** Package the same project in a minimal Dockerfile and compare both approaches.

---

## Phase 4 — Data Pipelines (dbt and Airflow)

This is the core of the missing stack. The Phase 1 e-commerce database serves as the foundation again.

### 4.1 dbt Fundamentals: From Raw Data to Models

- **Concepts:** dbt-core, staging/marts, ref(), Jinja, dbt tests, documentation
- **Task:** Build a dbt project on DuckDB: staging models (rename, cast, deduplicate), followed by marts (dim_customers, fct_orders, daily revenue marts). Add dbt tests (unique, not_null, relationships) and generate the documentation site (`dbt docs`).
- **Data:** E-commerce database from Phase 1
- **Stretch:** Build an incremental materialization for the fact table and demonstrate that a second run processes only new data.

### 4.2 Airflow: Understanding Orchestration

- **Concepts:** DAGs, tasks, dependencies, scheduling, retries, idempotency
- **Task:** Install Airflow locally (standalone mode). Build a DAG with four tasks: generate raw data (simulated daily ingest) → quality check → dbt run → dbt test. Configure retries and a failure notification (a log message is sufficient). Backfill three “days.”
- **Data:** From 4.1
- **Stretch:** Deliberately make one task non-idempotent (append instead of overwrite), demonstrate the damage caused by a retry, and fix it. Idempotency is the most important pipeline concept of all.

### 4.3 The Data Quality Watchdog

- **Concepts:** Data quality, expectations, alerting, dead-letter thinking
- **Task:** Extend the pipeline: On “day 4,” the ingest deliberately produces broken data (schema drift through a renamed column; data issues through negative prices and duplicates). The pipeline must detect this, stop the run, write bad rows to a quarantine table, and log an understandable error report.
- **Data:** From 4.2, with an error injector
- **Stretch:** Distinguish between “stop the pipeline” (broken schema) and “warn but continue” (2% outliers), and implement both paths.

---

## Phase 5 — Cloud Concepts Locally (The S3/Athena Layer)

### 5.1 Object Storage with boto3

- **Concepts:** S3 API, buckets, keys, upload/download, prefix layout
- **Task:** Start MinIO with Docker (or use a local directory with the same folder structure as a fallback). Write a script that uploads the daily ingest files from Phase 4 using a path such as `raw/orders/year=2026/month=07/day=11/orders.parquet`, then lists and reads them—with boto3, exactly as it would against real S3.
- **Data:** From Phase 4
- **Stretch:** Implement a utility that finds the newest partition, as needed in almost every data team.

### 5.2 Querying Files: Your Own Athena

- **Concepts:** External tables, partition pruning, schema-on-read
- **Task:** Query the Parquet files from 5.1 directly with DuckDB (`read_parquet` using a glob and Hive partitioning). Use timings to show that a filter on the partition column reads only the relevant files. Write half a page explaining what is conceptually identical in Athena/Glue and what differs.
- **Data:** From 5.1
- **Stretch:** Connect Phases 4 and 5: the Airflow pipeline writes to “S3,” and dbt reads from it. This recreates the job posting's architecture at a small scale.

---

## Phase 6 — BI and Communication (The Looker Layer)

### 6.1 Dashboard with a Semantic Layer

- **Concepts:** BI tools, self-service, metric consistency
- **Task:** Start Metabase with Docker and connect it to your DuckDB database/marts (alternative: a Streamlit dashboard). Build an executive dashboard with the five metrics from 1.3: trend, comparison to the previous month, and breakdown by channel. Every tile must answer a concrete question—no decorative charts.
- **Data:** Marts from Phase 4
- **Stretch:** Ask another person (or yourself after one week) to use the dashboard without an explanation, note where it is confusing, and iterate once.

### 6.2 The Decision Memo

- **Concepts:** Stakeholder communication, recommendations, communicating uncertainty honestly
- **Task:** Choose a completed analysis (such as the cohort analysis from 1.2 or an A/B test from another lab) and write a one-page memo for a fictional product manager: begin with the key message in two sentences, then provide evidence, followed by a recommendation and risks. Keep methodological details out of the main body (an appendix is allowed).
- **Data:** Existing analysis
- **Stretch:** Write the same analysis once for the product manager and once for a senior data scientist, making the differences explicit.

### 6.3 Analysis Under Time Pressure

- **Concepts:** Prioritization, 80/20, ad hoc requests
- **Task:** Simulate a typical internship request: “Revenue in channel X dropped last week—why? We need an answer by 3 p.m.” Set a 90-minute timer and work on the e-commerce database (first inject a cause with a script without inspecting it). Deliver three bullet points, one chart, and a next step.
- **Data:** Generated with a hidden cause
- **Stretch:** Record where your 90 minutes went and determine what you would skip next time.

---

## Phase 7 — Capstone: The Mini Data Platform (Portfolio Project)

### 7.1 End to End: From Raw Data to a Recommendation

- **Concepts:** Everything combined
- **Task:** Build the complete flow in a dedicated repository: simulated daily ingest → “S3” (MinIO/Parquet partitions) → an Airflow DAG orchestrating quality checks and dbt → marts in DuckDB → dashboard (Metabase/Streamlit) → ONE model project with business relevance to the job posting (choose CLV estimation, churn ranking, or demand forecasting; you may reuse the model from an existing lab—the integration is what matters here) → a one-page decision memo.
- **Requirements:** A README with an architecture diagram; `make demo` (or a script) that starts everything from scratch; tests for the core transformations; and a clean branch/PR history.
- **Stretch:** Write the README so it can serve as a conversation starter in an interview: Which trade-offs did you make, where, and why?

---

## Recommended Learning Path

- **Minimum path before applying (about 4–6 weeks part-time):** 1.1 → 1.2 → 2.1 → 3.1 → 4.1 → 4.2 → 6.2
- **Complete path:** Phase 1 → 2 → 3 → 4 → 5 → 6 → 7. The order within each phase is flexible; the numbering is a useful default.
- You can work through Phases 1–3 alongside your existing labs; Phases 4–7 build on one another.

## Scope Relative to the Existing Labs

This lab deliberately contains NO statistics, ML, or modeling theory—that material lives in `statistics_practice_lab`, `ds-practice-lab`, `ML_DS_ITOP_Learn-Repository`, and `xtAI_Learn-Repository`. When a project requires a model (7.1), reuse one from those labs. Conversely, to make a model project from another lab “internship-ready,” apply Phases 2–3 of this lab to it.

---

# Deutsche Fassung

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
