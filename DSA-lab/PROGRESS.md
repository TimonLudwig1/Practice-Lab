# PROGRESS.md — DSA Practice Lab (Build-Tracker)

Diese Datei ist die **einzige Wahrheit** über den Build-Fortschritt. Sie trackt die
Content-*Generierung* durch Claude Code, nicht den Lernfortschritt des Nutzers.

---

## NÄCHSTE AKTION

> **Modul 13 (Graphen II: Kürzeste Wege, Union-Find & MST): Projekt 02-medium erstellen.**
> Dazu ausschließlich den Abschnitt „Modul 13" sowie die Arbeitsanweisungen
> aus `Moduluebersicht_DSA.md` lesen.

*(Dieser Block wird nach jedem abgeschlossenen Task aktualisiert und zeigt immer
genau einen nächsten Schritt.)*

---

## Regeln für Claude Code (strikt einhalten)

1. **Session-Start = nur diese Datei lesen.** Danach den Abschnitt des in
   „NÄCHSTE AKTION" genannten Moduls aus `Moduluebersicht_DSA.md`. Kein `ls -R`,
   kein Repo-Scan, keine fertigen Ordner öffnen.
2. **Ein Task pro Arbeitszyklus.** Ein Task ist genau eine Zelle der Statusspalten
   unten (Theorie ODER ein einzelnes Projekt ODER der Test-Durchlauf). Nach jedem
   Task: Tracker aktualisieren, `git commit`, dann weiter oder Stopp.
3. **FERTIG ist unveränderlich.** Abgeschlossene Theorie-Skripte, Projekte, Daten
   und Tests werden niemals erneut geöffnet, gelesen, ausgeführt oder „verbessert" —
   außer der Nutzer verlangt es ausdrücklich.
4. **Reihenfolge einhalten.** Module strikt in der Reihenfolge 01 bis 17 bauen.
   Innerhalb eines Moduls: Theorie, dann 01-basic, 02-medium, 03-final, dann Test-Durchlauf.
5. **Test-Durchlauf** = alle Code-Dateien des Moduls einmal ausführen bzw. pytest
   laufen lassen, Fehler beheben, Ergebnis hier notieren. Danach gilt das Modul als FERTIG.
6. **Inhalte in Dateien, nicht in den Chat.**
7. **Git-Commit nach jedem Task** mit sprechender Message, z. B.
   `modul-04: 02-medium fertig`.

### Statuswerte
`offen` → `in Arbeit` → `fertig` (pro Zelle). Ein Modul ist **FERTIG**, wenn alle
fünf Zellen `fertig` sind.

---

## Modul-Tracker

| Nr. | Modul | Theorie | 01-basic | 02-medium | 03-final | Getestet | Formate | Notizen |
|---|---|---|---|---|---|---|---|---|
| 01 | Komplexitätsanalyse & Big-O | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV | FERTIG: 35/35 Unit Tests; alle 9 Code-Dateien und drei vollständigen Benchmark-/Auditläufe am 2026-07-16 erfolgreich |
| 02 | Arrays & Strings | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV, PNG | FERTIG: 102/102 Tests; alle 10 Python-Dateien, Experimente, Generator, Demo und vollständiger NumPy-Benchmark am 2026-07-16 erfolgreich |
| 03 | Rekursion & Divide and Conquer | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV | FERTIG: 68/68 Tests; alle 10 Python-Dateien, Theoriebeispiele, Demos, Generator und vollständiger Dateisystem-Benchmark am 2026-07-16 erfolgreich |
| 04 | Linked Lists | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV | FERTIG: 131/131 Tests; alle 9 Python-Dateien, Demos und vollständige Cache-Simulation am 2026-07-17 erfolgreich |
| 05 | Stacks & Queues | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV, PNG | FERTIG: 112/112 Tests; alle 11 Python-Dateien, Demos, Generator und vollständige 250-Job-Simulation am 2026-07-19 erfolgreich |
| 06 | Hashing & Hash Maps | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV | FERTIG: 113/113 Tests; alle 13 Python-Dateien, 12 Theorieblöcke, Demos, Benchmarks, Generator und vollständige 4.000-Zeilen-Log-Pipeline am 2026-07-19 erfolgreich |
| 07 | Sortieralgorithmen | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV, PNG | FERTIG: 165/165 Tests; alle 12 Python-Dateien, 15 Theorieblöcke, Demo, vollständiger 120-Punkt-Benchmark und externe 5.000-Zeilen-Sortierung am 2026-07-19 erfolgreich |
| 08 | Binary Search & Suchvarianten | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV, PNG | FERTIG: 195/195 Tests; alle 11 Python-Dateien, 17 Theorieblöcke, beide Demos, Generator und vollständige 5.000-Zeilen-Schwellenpipeline am 2026-07-19 erfolgreich |
| 09 | Two Pointers & Sliding Window | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV, PNG | FERTIG: 141/141 Tests; alle 10 Python-Dateien, 13 Theorieblöcke, Benchmark, Demo, Generator und vollständige 12.000-Punkte-Anomaliepipeline am 2026-07-19 erfolgreich |
| 10 | Bäume & Binary Search Trees | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV | FERTIG: 120/120 Tests; alle 10 Python-Dateien, 16 Theoriebeispiele, beide Demos, Generator und vollständiger Decision-Tree-Vergleich mit 94,0 % Testgenauigkeit am 2026-07-20 erfolgreich |
| 11 | Heaps & Priority Queues | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV, PNG | FERTIG: 148/148 Tests; alle 11 Python-Dateien, 17 Theoriebeispiele, beide Demos, 18 Benchmark-Fälle, Generator und vollständige 300-Job-Scheduler-Simulation am 2026-07-20 erfolgreich |
| 12 | Graphen I: Repräsentation & Traversierung | fertig | fertig | fertig | fertig | fertig | Markdown, Python-Skripte, CSV | FERTIG: 155/155 Tests; alle 10 Python-Dateien, 16 Theorieblöcke, beide Demos und vollständige Seed-1203-Pipeline mit 24 Tasks, 39 Kanten und 113-Minuten-Kritischem-Pfad am 2026-07-20 erfolgreich |
| 13 | Graphen II: Kürzeste Wege, Union-Find & MST | fertig | fertig | offen | offen | offen | Markdown, Python-Skripte | Theorie mit 19 Beispielen; gewichtete Graphklasse, Dijkstra mit Trace und unveränderlichem Ergebnis sowie Union-Find mit Path Compression, Union by Rank und Größen; Demo und 58 Tests erfolgreich |
| 14 | Greedy-Algorithmen | offen | offen | offen | offen | offen | — | — |
| 15 | Backtracking | offen | offen | offen | offen | offen | — | — |
| 16 | Dynamic Programming | offen | offen | offen | offen | offen | — | — |
| 17 | Capstone: DSA im Data-Science-Kontext | offen | offen | offen | offen | offen | — | — |

---

## Abschluss-Checkliste (erst nach Modul 17)

- [ ] `ABSCHLUSS.md` im Repo-Root erstellt (reines Inventar, keine Wiederholung)
- [ ] Eintrag in der Übersichtstabelle des `Practice-Lab`-Dach-Repos ergänzt (macht der Nutzer selbst)

---

## Session-Log

*(Pro Session eine Zeile: Datum, erledigte Tasks, besondere Entscheidungen.
Kurz halten — dieses Log ist Gedächtnisstütze, kein Bericht.)*

| Datum | Erledigt | Notizen |
|---|---|---|
| 2026-07-16 | Modul 01: Theorie | Umfassendes Theorie-Skript erstellt; Python-Beispiele syntaktisch geprüft. |
| 2026-07-16 | Modul 01: 01-basic | Laufzeit-Labor mit fünf Kurven, adaptivem Benchmark, CSV und Log-Log-Plot; 9 Tests erfolgreich. |
| 2026-07-16 | Modul 01: 02-medium | Zehn Detektivfälle mit versteckten Kosten, 70-Punkt-Benchmark und normalisiertem Plot; 15 Tests erfolgreich. |
| 2026-07-16 | Modul 01: 03-final | CSV-Pipeline-Audit mit Seed, Ergebnisgleichheit und Vorher/Nachher-Report; 11 Tests, bis 139,97x Speedup. |
| 2026-07-16 | Modul 01: Test-Durchlauf | 35/35 Tests; alle 9 Code-Dateien, Generatoren und drei vollständigen Benchmark-/Auditläufe erfolgreich. |
| 2026-07-16 | Modul 02: Theorie | Umfassendes Array-/String-Skript erstellt; alle extrahierten Python-Beispiele gemeinsam ausgeführt. |
| 2026-07-16 | Modul 02: 01-basic | Dynamisches Array auf festem ctypes-Puffer, Resize-Tracking, CSV und Kostenplot; 18 Tests erfolgreich. |
| 2026-07-16 | Modul 02: 02-medium | Zehn Array-/String-Patterns mit Invarianten, Randfällen und Demo; 53 Tests erfolgreich. |
| 2026-07-16 | Modul 02: 03-final | Reproduzierbares Zeitreihen-Toolkit, 31 Tests und NumPy-Benchmark mit 100.000 Messwerten erfolgreich. |
| 2026-07-16 | Modul 02: Test-Durchlauf | 102/102 Tests; alle 10 Python-Dateien und vollständigen Experimente/Benchmarks erfolgreich. |
| 2026-07-16 | Modul 03: Theorie | Umfassendes Rekursions-/Divide-and-Conquer-Skript erstellt; alle Python-Beispiele gemeinsam ausgeführt. |
| 2026-07-16 | Modul 03: 01-basic | Rekursions-Visualizer mit vier Funktionen, Call-Tree-Decorator und vollständiger Demo; 24 Tests erfolgreich. |
| 2026-07-16 | Modul 03: 02-medium | Instrumentierter D&C-Werkzeugkasten mit drei Rekursionsbäumen und Demo; 25 Tests erfolgreich. |
| 2026-07-16 | Modul 03: 03-final | Rekursive/iterative Dateisystem-Analyse auf 117 Verzeichnissen und 326 Dateien; 19 Tests und Benchmark erfolgreich. |
| 2026-07-16 | Modul 03: Test-Durchlauf | 68/68 Tests; alle 10 Python-Dateien, Theoriebeispiele, Demos und vollständiger Benchmark erfolgreich. |
| 2026-07-17 | Modul 04: Theorie | Umfassendes Linked-Lists-Skript erstellt; alle Python-Beispiele gemeinsam ausgeführt. |
| 2026-07-17 | Modul 04: 01-basic | Vollständige Singly Linked List mit Invariantenprüfung und Operationsdemo; 43 pytest-Fälle erfolgreich. |
| 2026-07-17 | Modul 04: 02-medium | Reversal, Runner-Mitte, Floyd-Metadaten und stabiler In-place-Merge mit Traces; 36 pytest-Fälle erfolgreich. |
| 2026-07-17 | Modul 04: 03-final | O(1)-LRU-Cache mit Sentinel-Liste und Hash Map; 52/52 Tests sowie Standardsimulation mit 77,40 % Hit-Rate und 4,41x Speedup erfolgreich. |
| 2026-07-17 | Modul 04: Test-Durchlauf | 131/131 Tests; alle 9 Python-Dateien, beide Demos und vollständige Cache-Simulation erfolgreich. |
| 2026-07-17 | Modul 05: Theorie | Umfassendes Stacks-&-Queues-Skript erstellt; alle enthaltenen Python-Beispiele gemeinsam erfolgreich ausgeführt. |
| 2026-07-17 | Modul 05: 01-basic | Arraybasierter Stack und fixer Queue-Ringpuffer mit LIFO/FIFO-Demo, Fehlerbehandlung und Wrap-around; 24 Tests erfolgreich. |
| 2026-07-17 | Modul 05: 02-medium | Ausdrucks-Rechner mit Tokenizer, Klammervalidierung, Shunting-Yard, Postfix-Auswertung und Trace-Demo; 59 Tests erfolgreich. |
| 2026-07-17 | Modul 05: 03-final | Reproduzierbare Job-Queue-Simulation mit FIFO/Priorität, 250 Jobs, segmentierten Kennzahlen, CSV-Bericht und PNG-Vergleich; 29 Tests erfolgreich. |
| 2026-07-19 | Modul 05: Test-Durchlauf | 112/112 Tests; alle 11 Python-Dateien, beide Demos, Generator und vollständige 250-Job-Simulation erfolgreich. |
| 2026-07-19 | Modul 06: Theorie | Umfassendes Hashing-/Hash-Map-Skript erstellt; alle 12 enthaltenen Python-Beispiele gemeinsam erfolgreich ausgeführt. |
| 2026-07-19 | Modul 06: 01-basic | Generische Chained Hash Map mit Rehashing, Invarianten, Kollisionsdemo und Benchmark gegen fixe Kapazität; 32 Tests erfolgreich. |
| 2026-07-19 | Modul 06: 02-medium | Two Sum, Anagramme, erstes eindeutiges Zeichen und Stream-Duplikate jeweils hash-basiert/naiv mit Benchmark; 47 Tests erfolgreich. |
| 2026-07-19 | Modul 06: 03-final | Reproduzierbare Log-Engine mit Deduplizierung, Top-K, Sessionisierung und äquivalenter Sortier-Referenz auf 4.000 Zeilen; 34 Tests erfolgreich. |
| 2026-07-19 | Modul 06: Test-Durchlauf | 113/113 Tests; alle 13 Python-Dateien, 12 Theorieblöcke, Demos, Benchmarks und vollständige 4.000-Zeilen-Log-Pipeline erfolgreich. |
| 2026-07-19 | Modul 07: Theorie | Umfassendes Sortier-Skript mit Hand-Traces, Vergleichstabelle, unteren Schranken und Timsort erstellt; alle 15 Python-Beispiele erfolgreich. |
| 2026-07-19 | Modul 07: 01-basic | Bubble, Selection, Insertion, Merge und Quick Sort mit Metriken, Traces und Seed-basierten Property-Tests; 73 Tests erfolgreich. |
| 2026-07-19 | Modul 07: 02-medium | Systematischer Benchmark von sechs Verfahren über vier Eingabeformen und fünf Größen mit CSV, Vier-Panel-Plot und Theoriebericht; 55 Tests erfolgreich. |
| 2026-07-19 | Modul 07: 03-final | Stabile externe CSV-Sortierung unter künstlicher 250-Zeilen-Speichergrenze mit 20 Runs, begrenztem k-Way-Merge, Integritätsprüfung und Laufbericht; 37 Tests erfolgreich. |
| 2026-07-19 | Modul 07: Test-Durchlauf | 165/165 Tests; alle 12 Python-Dateien, 15 Theorieblöcke, Demo, vollständiger Benchmark und externe Sortierpipeline erfolgreich. |
| 2026-07-19 | Modul 08: Theorie | Umfassendes Binary-Search-Skript mit Intervall-Simulationen, Korrektheitsinvarianten, Suchvarianten und Antwortsuche erstellt; alle 17 Python-Beispiele erfolgreich. |
| 2026-07-19 | Modul 08: 01-basic | Exakte Suche, First/Last Occurrence und Insert Position mit zwei Intervallverträgen, Invarianten-Traces und bisect-Referenzprüfung; 65 Tests erfolgreich. |
| 2026-07-19 | Modul 08: 02-medium | Gemeinsame first-true-Suche für minimale Versandkapazität, Integer-Wurzel und k-kleinstes Matrixelement mit Monotonie-Traces und 1.050 Referenzfällen; 81 Tests erfolgreich. |
| 2026-07-19 | Modul 08: 03-final | Reproduzierbare Schwellenoptimierung auf 5.000 Klassifikator-Scores mit FPR-Nebenbedingung, Binary Search, Rastervergleich, CSV-Bericht und Metrikplot; 49 Tests erfolgreich. |
| 2026-07-19 | Modul 08: Test-Durchlauf | 195/195 Tests; alle 11 Python-Dateien, 17 Theorieblöcke, beide Demos, Generator und vollständige Schwellenpipeline erfolgreich. |
| 2026-07-19 | Modul 09: Theorie | Umfassendes Two-Pointers-/Sliding-Window-Skript mit Zeiger- und Fenster-Simulationen, Invarianten, Amortisierung und Rolling-Statistiken erstellt; alle 13 Python-Beispiele erfolgreich. |
| 2026-07-19 | Modul 09: 01-basic | Je drei Two-Pointers-/Sliding-Window-Aufgaben mit Brute-Force-Referenzen, 2.400 Property-Fällen und Sechs-Fälle-Benchmark; 57 Tests erfolgreich. |
| 2026-07-19 | Modul 09: 02-medium | Minimales Deckungsfenster, maximale Einsen mit k Flips und Anagrammfenster mit expliziten Hash-Map-/Counter-Invarianten und 2.200 Referenzfällen; 40 Tests erfolgreich. |
| 2026-07-19 | Modul 09: 03-final | Reproduzierbare Streaming-Anomalieerkennung auf 12.000 Messwerten mit vorherigem 120er-Fenster, naiver Referenz, CSV-Berichten und PNG-Plot; 44 Tests, 20/20 Treffer ohne Fehlalarm und 3,84x Speedup erfolgreich. |
| 2026-07-19 | Modul 09: Test-Durchlauf | 141/141 Tests; alle 10 Python-Dateien, Benchmark, Demo, Generator und vollständige 12.000-Punkte-Anomaliepipeline mit 20/20 Treffern ohne Fehlalarm erfolgreich. |
| 2026-07-19 | Modul 10: Theorie | Umfassendes Baum-/BST-Skript mit gemeinsamem Traversierungsbeispiel, Suchinvariante, drei Delete-Fällen, Balancekonzepten und Data-Science-Bezügen erstellt; alle 16 Python-Beispiele erfolgreich. |
| 2026-07-20 | Modul 10: 01-basic | Vollständiger BST-Eigenbau mit Insert, Suche, Delete, Min/Max, vier Traversierungen und ASCII-Visualisierung; Demo, Syntaxprüfung und 33 pytest-Fälle erfolgreich. |
| 2026-07-20 | Modul 10: 02-medium | Rekursiver Aufgabenkatalog für Höhe, linearen Balance-Check, BST-Validierung, allgemeinen LCA und JSON-Serialisierung; Demo, Syntaxprüfung und 49 Tests erfolgreich. |
| 2026-07-20 | Modul 10: 03-final | Eigener Decision-Tree-Klassifikator mit Gini, rekursiven Splits, Max-Tiefe und Seed-Datensatz; 38 Tests, vollständiger Vergleich und für beide Modelle 94,0 % Testgenauigkeit bei 0/300 abweichenden Vorhersagen. |
| 2026-07-20 | Modul 10: Test-Durchlauf | 120/120 Tests; alle 10 Python-Dateien, beide Demos, Generator und vollständiger Decision-Tree-Vergleich mit erneut 94,0 % Testgenauigkeit erfolgreich. |
| 2026-07-20 | Modul 11: Theorie | Umfassendes Heap-/Priority-Queue-Skript mit Push-/Pop-/Heapify-Simulationen, O(n)-Argument, Heap Sort, heapq und drei Kernmustern erstellt; alle 17 Python-Beispiele gemeinsam erfolgreich ausgeführt. |
| 2026-07-20 | Modul 11: 01-basic | Arraybasierter Min-Heap mit Push, Pop, Peek, Bottom-up-Heapify und Heap Sort; Demo, Syntaxprüfung und 51 Tests inklusive Property-Vergleichen gegen heapq und sorted erfolgreich. |
| 2026-07-20 | Modul 11: 02-medium | Deterministisches Top-K Frequent, iteratorbasiertes k-Way-Merge und Running Median mit zwei Heaps; Demo, Syntaxprüfung, 64 Tests und 18 Benchmark-Vergleiche bis 125,62x erfolgreich. |
| 2026-07-20 | Modul 11: 03-final | Nicht-präemptiver FIFO-/Priority-Scheduler auf 300 Seed-Jobs mit Klassenmetriken, zwei CSV-Berichten und PNG-Plot; 33 Tests erfolgreich, kritischer Mean Wait von 552,88 auf 10,00 gesenkt. |
| 2026-07-20 | Modul 11: Test-Durchlauf | 148/148 Tests; alle 11 Python-Dateien, beide Demos, 18 Benchmark-Fälle, Generator und vollständige 300-Job-Scheduler-Simulation erfolgreich. |
| 2026-07-20 | Modul 12: Theorie | Umfassendes Graph-Skript zu Adjazenzliste/-matrix, BFS/DFS, Komponenten, Zykluserkennung, topologischer Sortierung und Bipartitheit erstellt; alle 16 Python-Beispiele gemeinsam erfolgreich ausgeführt. |
| 2026-07-20 | Modul 12: 01-basic | Generische deterministische Graphklasse mit gerichteter/ungerichteter Adjazenzliste, BFS, rekursiver/iterativer DFS, Zusammenhangskomponenten und Invariantenprüfung; Demo, Syntaxprüfung und 49 Tests erfolgreich. |
| 2026-07-20 | Modul 12: 02-medium | Eigenständiger Graph-Aufgabenkatalog für gerichtete/ungerichtete Zyklen, Kahn-/DFS-Toposort, Bipartitheit und 4-fach verbundene Inseln mit dokumentierten Traces; Demo und 56 Tests erfolgreich. |
| 2026-07-20 | Modul 12: 03-final | Reproduzierbare Seed-1203-Abhängigkeitsanalyse mit 24 Tasks, 39 DAG-Kanten, Toposort, Ausführungswellen, 113-Minuten-Kritischem-Pfad und Ausfallrangliste; vier Ergebnisdateien und 50 Tests erfolgreich. |
| 2026-07-20 | Modul 12: Test-Durchlauf | 155/155 Tests; alle 10 Python-Dateien, 16 Theorieblöcke, beide Demos und vollständige Seed-1203-Pipeline mit 24 Tasks, 39 Kanten und 113-Minuten-Kritischem-Pfad erfolgreich. |
| 2026-07-20 | Modul 13: Theorie | Umfassendes Skript zu gewichteten Graphen, Dijkstra mit vollständiger Handrechnung, Bellman-Ford, Union-Find mit beiden Optimierungen, Kruskal, Prim und A*-Einordnung erstellt; alle 19 Python-Beispiele gemeinsam erfolgreich ausgeführt. |
| 2026-07-20 | Modul 13: 01-basic | Gewichtete Graphklasse, Dijkstra mit Heap-Tie-Breaker, Relaxierungs-Trace und Pfadrekonstruktion sowie Union-Find mit Path Compression, Union by Rank, Komponentengrößen und Invarianten; Demo und 58 Tests erfolgreich. |
