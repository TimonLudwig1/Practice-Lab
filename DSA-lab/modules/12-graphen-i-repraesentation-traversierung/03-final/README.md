# 03-final — Abhängigkeitsanalyse einer Datenpipeline

Dieses Abschlussprojekt modelliert das Denkmodell hinter einem Airflow-DAG:
Tasks sind Knoten, und eine Kante `A -> B` bedeutet, dass B erst nach erfolgreichem
Abschluss von A starten darf.

Die Pipeline umfasst 24 semantisch benannte Schritte von Ingest und Validierung
über Feature Engineering und Modelltraining bis zur Veröffentlichung. Der feste
Seed `1203` bestimmt Laufzeiten und optionale zusätzliche Abhängigkeiten. Dadurch
ist der Datensatz synthetisch, aber jeder Lauf exakt reproduzierbar.

## Fragestellungen

1. **Ist die Pipeline wirklich ein DAG?**
   Kahn verarbeitet alle Knoten oder meldet einen Zyklus.
2. **In welcher Reihenfolge darf sie laufen?**
   Eine stabile topologische Sortierung liefert eine gültige Reihenfolge.
3. **Welche Tasks können parallel laufen?**
   Der längste Kantenabstand von einer Quelle gruppiert Tasks in Wellen.
4. **Was bestimmt die minimale Gesamtdauer?**
   Dynamische Programmierung auf der Toposort findet den laufzeitgewichteten
   kritischen Pfad.
5. **Welche Knoten sind besonders kritisch?**
   Eine BFS ab jedem möglichen Ausfall zählt dessen direkte und transitive
   Nachfolger — den Blast Radius.

## Zwei Bedeutungen von „kritisch"

Das Projekt trennt zwei häufig vermischte Konzepte:

- **laufzeitkritisch:** Der Task liegt auf dem längsten gewichteten Pfad. Eine
  Verzögerung kann den frühestmöglichen Pipeline-Abschluss verschieben.
- **ausfallkritisch:** Vom Task sind viele Nachfolger erreichbar. Sein Ausfall
  blockiert einen großen Teil der Pipeline.

Ein Task kann in einer Kategorie sehr kritisch und in der anderen unauffällig
sein. Diese Trennung ist für reale Orchestrierung und Alarmpriorisierung wichtig.

## Ausfallanalyse

Für einen ausgefallenen Task arbeitet die Analyse wie folgt:

```text
failed_task
    |
    +-- direkte Nachfolger
            |
            +-- deren Nachfolger
                    |
                    +-- ... bis keine neuen Tasks erreichbar sind
```

Die BFS markiert jeden erreichbaren Nachfolger höchstens einmal. Daher kostet
eine einzelne Analyse `O(V + E)`. Die vollständige Rangliste simuliert jeden
Task einmal und benötigt `O(V * (V + E))` — für 24 Tasks gut nachvollziehbar,
für sehr große DAGs wäre eine spezialisierte Vorberechnung sinnvoll.

## Dateien

- `pipeline_analysis.py` — DAG-Modell, Seed-Generator und Algorithmen
- `reporting.py` — reproduzierbare CSV- und Markdown-Ausgabe
- `run_analysis.py` — vollständiger Analyse-Lauf
- `test_pipeline_analysis.py` — Invarianten-, Algorithmus- und Ausgabetests
- `results/task_metrics.csv` — Scheduling- und Reichweitenmetriken je Task
- `results/pipeline_edges.csv` — generierte Abhängigkeiten
- `results/failure_impacts.csv` — Ausfallrangliste für alle Tasks
- `results/pipeline_report.md` — lesbare Zusammenfassung und Interpretation

## CSV-Schema

`task_metrics.csv` enthält unter anderem:

- topologische Position und Ausführungswelle,
- frühesten Start und Abschluss,
- In-/Out-Degree,
- Anzahl transitiver Upstream-/Downstream-Tasks,
- Zugehörigkeit zum kritischen Pfad.

`failure_impacts.csv` enthält für jeden angenommenen Ausfall:

- direkt blockierte Tasks,
- alle transitiv blockierten Nachfolger,
- Gesamtzahl und Anteil nicht verfügbarer Tasks,
- verbleibende unbeeinträchtigte Tasks.

## Ausführen

Im Projektordner:

```bash
python3 run_analysis.py
python3 -m pytest -q
```

Der Analyse-Lauf überschreibt ausschließlich die vier reproduzierbaren Dateien
im Ordner `results/`.
