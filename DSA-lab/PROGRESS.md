# PROGRESS.md — DSA Practice Lab (Build-Tracker)

Diese Datei ist die **einzige Wahrheit** über den Build-Fortschritt. Sie trackt die
Content-*Generierung* durch Claude Code, nicht den Lernfortschritt des Nutzers.

---

## NÄCHSTE AKTION

> **Modul 05 (Stacks & Queues): Test-Durchlauf durchführen.**
> Dazu ausschließlich den Abschnitt „Modul 05" sowie die Arbeitsanweisungen
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
| 05 | Stacks & Queues | fertig | fertig | fertig | fertig | offen | Markdown, Python-Skripte, CSV, PNG | Drei Projekte fertig; Job-Queue-Vergleich auf 250 Jobs zeigt Fairness-Trade-off zwischen FIFO und stabiler Priorisierung; bisher 24 + 59 + 29 Tests erfolgreich |
| 06 | Hashing & Hash Maps | offen | offen | offen | offen | offen | — | — |
| 07 | Sortieralgorithmen | offen | offen | offen | offen | offen | — | — |
| 08 | Binary Search & Suchvarianten | offen | offen | offen | offen | offen | — | — |
| 09 | Two Pointers & Sliding Window | offen | offen | offen | offen | offen | — | — |
| 10 | Bäume & Binary Search Trees | offen | offen | offen | offen | offen | — | — |
| 11 | Heaps & Priority Queues | offen | offen | offen | offen | offen | — | — |
| 12 | Graphen I: Repräsentation & Traversierung | offen | offen | offen | offen | offen | — | — |
| 13 | Graphen II: Kürzeste Wege, Union-Find & MST | offen | offen | offen | offen | offen | — | — |
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
