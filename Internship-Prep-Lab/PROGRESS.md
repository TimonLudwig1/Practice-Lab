# PROGRESS — Internship Prep Lab

**Dies ist die einzige Fortschrittsquelle. Bei Session-Start nur diesen Kopf + Legende lesen.**

▶ **NÄCHSTE AKTION:** Fundament bauen: `shared/ecommerce/generate_data.py` (E-Commerce-DB: users, orders, order_items, products, sessions, marketing_spend; fester Seed; Umfang parametrisierbar) — danach Projekt 1.1.

## Legende
- `[ ]` TODO
- `[~]` IN ARBEIT (höchstens EIN Task gleichzeitig)
- `[x]` FERTIG-GETESTET (unveränderlich — nicht erneut öffnen oder verbessern)
- `[!]` BLOCKIERT (Grund in Klammern)

Reihenfolge: Fundament → Phase 1 → 2 → 3 → 4 → 5 → 6 → 7. Innerhalb einer Phase der Nummerierung folgen.

## Fundament
- [ ] `shared/ecommerce/` — Datengenerator + kurze Schema-Doku
- [ ] `requirements.txt` + Root-`.gitignore` (daten/, *.duckdb, *.parquet, *.csv, .venv/)

## Phase 1 — Analytics-SQL
- [ ] 1.1 Window Functions (`phase-1/01-window-functions/`)
- [ ] 1.2 Kohortenanalyse & Retention (`phase-1/02-kohorten-retention/`)
- [ ] 1.3 Metrik-Layer (`phase-1/03-metrik-layer/`)
- [ ] 1.4 Query-Performance (`phase-1/04-query-performance/`)

## Phase 2 — Git & Kollaboration
- [ ] 2.1 Feature-Branch-Workflow (`phase-2/01-feature-branch-workflow/`)
- [ ] 2.2 Merge-Konflikte & bisect (`phase-2/02-konflikte-bisect/`)
- [ ] 2.3 Repo-Hygiene & CI (`phase-2/03-repo-hygiene-ci/`)

## Phase 3 — Produktionsreifer Python-Code
- [ ] 3.1 Notebook-Refactoring (`phase-3/01-notebook-refactoring/`)
- [ ] 3.2 Testen von Datencode (`phase-3/02-testen/`)
- [ ] 3.3 Reproduzierbare Umgebungen (`phase-3/03-reproduzierbarkeit/`)

## Phase 4 — Data Pipelines
- [ ] 4.1 dbt-Grundlagen (`phase-4/01-dbt-grundlagen/`)
- [ ] 4.2 Airflow-Orchestrierung (`phase-4/02-airflow/`)
- [ ] 4.3 Datenqualitäts-Wachhund (`phase-4/03-datenqualitaet/`)

## Phase 5 — Cloud-Konzepte lokal
- [ ] 5.1 Objektspeicher mit boto3 (`phase-5/01-objektspeicher/`)
- [ ] 5.2 Query über Dateien (`phase-5/02-query-ueber-dateien/`)

## Phase 6 — BI & Kommunikation
- [ ] 6.1 Dashboard mit Semantik (`phase-6/01-dashboard/`)
- [ ] 6.2 Entscheidungs-Memo (`phase-6/02-entscheidungs-memo/`)
- [ ] 6.3 Analyse unter Zeitdruck (`phase-6/03-analyse-zeitdruck/`)

## Phase 7 — Capstone
- [ ] 7.1 Mini-Datenplattform (`phase-7/01-mini-datenplattform/`) — keine Code-Vorgabe, nur Spezifikation + loesung/

---

## Build-Log (append-only)

<!-- Nach jedem fertigen Task eine Zeile anhängen: -->
<!-- - YYYY-MM-DD — <Einheit> — <erzeugte Pfade> — <Testnachweis> -->
