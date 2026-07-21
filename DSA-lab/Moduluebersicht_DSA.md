# Modulübersicht — DSA Practice Lab (Data Structures & Algorithms)

Diese Datei ist der **inhaltliche Leitfaden** für die Claude-Code-Session, die das
`dsa-practice-lab` aufbaut. Sie definiert alle Module, deren Theorieinhalte,
Qualifikationsziele und die drei Lernprojekte pro Modul.

**Wichtig für Token-Effizienz:** Pro Arbeitszyklus wird nur der Abschnitt des
aktuell zu bauenden Moduls gelesen — niemals die gesamte Datei erneut, niemals
Abschnitte bereits fertiger Module.

---

## Arbeitsanweisungen für die Claude-Code-Session

### Sprache und Stil
- Alle Dokumentation auf **Deutsch**, englische Fachbegriffe wo Standard
  (z. B. Linked List, Hash Map, Dynamic Programming — nicht eindeutschen).
- Code und Docstrings auf **Englisch**.
- Professioneller Ton, **keine Emojis**.

### Didaktisches Prinzip: Simulation vor Formel
Jedes Theorie-Skript folgt strikt der Drei-Ebenen-Struktur:
1. **Intuition** — Alltagsanalogie oder konkretes Problem, das die Struktur/den
   Algorithmus motiviert. Warum existiert das Konzept? Welches Problem löst es?
2. **Simulation** — Schritt-für-Schritt-Nachvollzug am konkreten Beispiel:
   Zustände einer Datenstruktur nach jeder Operation zeigen, Algorithmen von Hand
   auf kleinen Eingaben durchspielen (Trace-Tabellen), Laufzeitverhalten durch
   kleine Python-Experimente messbar machen (z. B. `time.perf_counter` über
   wachsende Eingabegrößen).
3. **Formalisierung** — Erst jetzt: formale Definition, Pseudocode,
   Komplexitätsanalyse (Big-O), Invarianten, Korrektheitsargument.

### Struktur pro Modul
```
modules/NN-modul-slug/
├── THEORY.md            # umfassendes Theorie-Skript (Intuition → Simulation → Formalisierung)
├── data/                 # nur falls das Modul Datensätze braucht
│   └── generate_data.py  # fixer Random Seed, Begründung als Kommentar
└── projects/
    ├── 01-basic/         # Einstieg: die Struktur / den Algorithmus selbst bauen
    ├── 02-medium/        # Anwendung: Struktur einsetzen, Varianten, Kanten- und Fehlerfälle
    └── 03-final/         # Praxisnah: realistisches Problem, wenn möglich mit Data-Science-Bezug
```

- **THEORY.md** ist umfassend und erklärend, nicht stichpunktartig. Ziel ist
  *Verständnis*. Alle im Modulabschnitt gelisteten Inhalte werden abgedeckt.
- **Projektformat** frei wählbar (Jupyter Notebook, Python-Skripte, Markdown-Walkthrough) —
  je nachdem, was didaktisch am besten passt. Entscheidung kurz in der Projekt-README begründen.
- Jedes Projekt bekommt eine eigene `README.md` mit: Ziel, Aufgabenstellung,
  Hinweisen und expliziten **„Fertig, wenn …"-Kriterien** (keine Zeitangaben).
- **Eigenimplementierung vor Bibliothek:** In `01-basic` wird die Struktur bzw. der
  Algorithmus grundsätzlich selbst implementiert (keine `collections`/`heapq`/
  `networkx`-Abkürzung). In `02-medium` und `03-final` dürfen Standardbibliotheken
  genutzt werden, wo es dem Lernziel dient — der Vergleich Eigenbau vs. Bibliothek
  ist ausdrücklich erwünscht.
- Lösungen enthalten lauffähigen, getesteten Code. Wo sinnvoll: kleine
  `assert`-basierte Tests oder pytest-Dateien.

### Arbeitsregeln (Kurzfassung, Details in PROGRESS.md)
1. Session-Start: nur `PROGRESS.md` lesen, dann den Abschnitt des dort genannten Moduls in dieser Datei.
2. Ein Task pro Arbeitszyklus (Theorie ODER ein Projekt), danach Tracker aktualisieren und committen.
3. Fertige Inhalte werden niemals erneut geöffnet oder verbessert.
4. Inhalte gehören in Dateien, nicht in Chat-Antworten.
5. Immer erst ein Modul komplett fertigstellen (Theorie + 3 Projekte + getestet), dann das nächste.

---

## Modulplan

Die Reihenfolge ist didaktisch: Jedes Modul setzt nur Inhalte vorangegangener Module voraus.

| Nr. | Modul | Block |
|---|---|---|
| 01 | Komplexitätsanalyse & Big-O | A — Fundamente |
| 02 | Arrays & Strings | A — Fundamente |
| 03 | Rekursion & Divide and Conquer | A — Fundamente |
| 04 | Linked Lists | B — Lineare Strukturen |
| 05 | Stacks & Queues | B — Lineare Strukturen |
| 06 | Hashing & Hash Maps | B — Lineare Strukturen |
| 07 | Sortieralgorithmen | C — Algorithmen-Grundtechniken |
| 08 | Binary Search & Suchvarianten | C — Algorithmen-Grundtechniken |
| 09 | Two Pointers & Sliding Window | C — Algorithmen-Grundtechniken |
| 10 | Bäume & Binary Search Trees | D — Hierarchische & vernetzte Strukturen |
| 11 | Heaps & Priority Queues | D — Hierarchische & vernetzte Strukturen |
| 12 | Graphen I: Repräsentation & Traversierung | D — Hierarchische & vernetzte Strukturen |
| 13 | Graphen II: Kürzeste Wege, Union-Find & MST | D — Hierarchische & vernetzte Strukturen |
| 14 | Greedy-Algorithmen | E — Entwurfsparadigmen |
| 15 | Backtracking | E — Entwurfsparadigmen |
| 16 | Dynamic Programming | E — Entwurfsparadigmen |
| 17 | Capstone: DSA im Data-Science-Kontext | F — Abschluss |

---

## Modul 01 — Komplexitätsanalyse & Big-O

**Inhalte (THEORY.md):**
- Warum Effizienz zählt: dasselbe Problem, drei Lösungen, drastisch verschiedene Laufzeiten
- Empirische Laufzeitmessung in Python (Messreihen über wachsende Eingabegrößen, Plot der Wachstumskurven)
- Wachstumsklassen: O(1), O(log n), O(n), O(n log n), O(n²), O(2ⁿ), O(n!) — mit je einem konkreten Algorithmusbeispiel
- Formale Definition der O-Notation, Ω und Θ (kurz), Best/Average/Worst Case
- Speicherkomplexität (Space Complexity)
- Amortisierte Analyse am Beispiel dynamischer Arrays (intuitiv, kein voller Beweisapparat)
- Typische Analysefehler (verschachtelte Schleifen, versteckte Kosten von Slicing und `in` bei Listen)

**Qualifikationsziele — fertig, wenn der Lernende:**
- die Laufzeitklasse einer gegebenen Python-Funktion durch Code-Lektüre bestimmen kann,
- Wachstumsverhalten empirisch messen und mit der theoretischen Klasse abgleichen kann,
- erklären kann, warum O(n log n) für großes n O(n²) schlägt, auch wenn Konstanten dagegen sprechen.

**Projekte:**
- **01-basic — Laufzeit-Labor:** Messskript, das für 5–6 vorgegebene Funktionen (konstant bis quadratisch) Laufzeiten über wachsende n misst und als Log-Log-Plot visualisiert. Der Lernende ordnet anschließend jede Kurve ihrer Big-O-Klasse zu.
- **02-medium — Komplexitäts-Detektiv:** Sammlung von 10 Python-Funktionen mit teils versteckten Kosten (Slicing, String-Konkatenation in Schleifen, `list.insert(0, x)`). Aufgabe: Laufzeitklasse vorhersagen, dann messen, Abweichungen erklären.
- **03-final — Performance-Audit eines Datenpipeline-Skripts:** Ein bewusst ineffizient geschriebenes Datenverarbeitungsskript (synthetischer CSV-Datensatz, fixer Seed) analysieren, Engpässe identifizieren, refactoren und Speedup dokumentieren (Vorher/Nachher-Messung).

---

## Modul 02 — Arrays & Strings

**Inhalte (THEORY.md):**
- Statische vs. dynamische Arrays; wie Python-Listen intern funktionieren (Over-Allocation, amortisiertes Append)
- Kostenmodell der wichtigsten Listenoperationen (Index-Zugriff, Append, Insert, Delete, Suche)
- Mehrdimensionale Arrays; Bezug zu NumPy (Contiguous Memory, warum Vektorisierung schnell ist — konzeptionell)
- Strings als immutable Sequenzen; Kosten von Konkatenation, `join` vs. `+=`
- Grundmuster: In-Place-Operationen, Prefix Sums, Matrix-Traversierung
- Häufige Interview-Patterns auf Arrays (Rotation, Deduplizierung, Merge zweier sortierter Arrays)

**Qualifikationsziele — fertig, wenn der Lernende:**
- für jede Standard-Listenoperation die Laufzeit nennen und begründen kann,
- Prefix Sums als Technik erkennen und anwenden kann,
- erklären kann, warum NumPy-Arrays für numerische Daten schneller sind als Listen.

**Projekte:**
- **01-basic — Dynamisches Array selbst bauen:** Eine `DynamicArray`-Klasse auf Basis eines fixen Puffers implementieren (Append mit Verdopplung, Insert, Delete, Resize). Wachstum des Puffers protokollieren und amortisierte Kosten sichtbar machen.
- **02-medium — Array-Pattern-Katalog:** 8–10 klassische Array-/String-Aufgaben (Rotation in-place, Merge sortierter Arrays, Prefix-Sum-Bereichsabfragen, Anagramm-Check) mit Tests lösen.
- **03-final — Zeitreihen-Toolkit ohne Pandas:** Auf einem synthetischen Sensordaten-Array (fixer Seed) gleitende Durchschnitte, Bereichssummen via Prefix Sums und Ausreißererkennung implementieren — nur mit Listen/Arrays. Abschluss: Vergleichsmessung gegen die NumPy-Äquivalente.

---

## Modul 03 — Rekursion & Divide and Conquer

**Inhalte (THEORY.md):**
- Rekursion als Selbstähnlichkeit: Basisfall, Rekursionsfall, Fortschrittsgarantie
- Der Call Stack: Simulation eines Rekursionsablaufs Frame für Frame (Trace-Diagramme)
- Rekursionsbäume zeichnen und Kosten daran ablesen
- Rekursion vs. Iteration; wann Rekursion natürlich ist, Stack-Limits in Python
- Divide and Conquer als Paradigma (Teilen, Erobern, Kombinieren) am Beispiel Merge-Prinzip
- Master-Theorem intuitiv (Fälle anhand von Rekursionsbäumen plausibilisieren, keine formale Herleitung)
- Memoization als Vorgriff auf Dynamic Programming (nur anreißen)

**Qualifikationsziele — fertig, wenn der Lernende:**
- eine rekursive Funktion von Hand tracen kann (Stack-Zustände notieren),
- für rekursive Algorithmen einen Rekursionsbaum aufstellen und die Laufzeit daraus ableiten kann,
- entscheiden kann, wann eine iterative Umformung sinnvoll ist.

**Projekte:**
- **01-basic — Rekursions-Visualizer:** Kleine rekursive Funktionen (Fakultät, Fibonacci, Summe, Potenz) implementieren und einen Decorator schreiben, der Aufrufe eingerückt protokolliert, sodass der Call Tree im Terminal sichtbar wird.
- **02-medium — Divide-and-Conquer-Werkzeugkasten:** Binäre Potenzierung, Maximum-Subarray (D&C-Variante) und Zählen von Inversionen implementieren; für jedes Verfahren den Rekursionsbaum dokumentieren und die Laufzeit begründen.
- **03-final — Rekursive Dateisystem-Analyse:** Ein synthetisch generierter, verschachtelter Ordnerbaum (Generator-Skript mit Seed) wird rekursiv ausgewertet: Gesamtgrößen, Tiefenstatistiken, Suche nach Mustern. Vergleich rekursive vs. iterative Lösung (expliziter Stack).

---

## Modul 04 — Linked Lists

**Inhalte (THEORY.md):**
- Motivation: Was können Linked Lists, was Arrays nicht können (und umgekehrt)
- Singly Linked List: Knoten, Referenzen, Kopf; Simulation aller Grundoperationen mit Zeigerdiagrammen
- Doubly Linked List, Sentinel-Knoten als Vereinfachungstechnik
- Kostenvergleich Array vs. Linked List (Tabelle: Zugriff, Insert, Delete, Suche)
- Klassische Techniken: Runner-Technik (schneller/langsamer Zeiger), In-Place-Reversal, Zyklenerkennung (Floyd)
- Wo Linked Lists real vorkommen (LRU-Caches, Deques, Speicherverwaltung)

**Qualifikationsziele — fertig, wenn der Lernende:**
- alle Grundoperationen fehlerfrei implementieren kann (inkl. Kantenfälle: leere Liste, ein Element),
- Zeigermanipulationen auf Papier vorzeichnen kann, bevor er sie codiert,
- begründen kann, wann eine Linked List einem Array vorzuziehen ist.

**Projekte:**
- **01-basic — Singly Linked List von Grund auf:** Vollständige Implementierung (Append, Prepend, Insert, Delete, Suche, Länge, `__repr__`) mit pytest-Suite, die alle Kantenfälle abdeckt.
- **02-medium — Zeiger-Klassiker:** Reversal in-place, Mitte finden (Runner), Zyklenerkennung nach Floyd, Merge zweier sortierter Listen — jeweils mit Zeigerdiagramm in der Dokumentation.
- **03-final — LRU-Cache:** Einen Least-Recently-Used-Cache aus Doubly Linked List + Hash Map bauen (O(1) für Get und Put), anschließend als Funktions-Cache für eine teure, simulierte Datenabfrage einsetzen und Hit-Rate messen.

---

## Modul 05 — Stacks & Queues

**Inhalte (THEORY.md):**
- LIFO und FIFO als Denkmodelle; Alltagsanalogien und Simulation der Operationen
- Implementierungsvarianten: Stack auf Array, Queue auf Linked List, Ringpuffer, zwei Stacks als Queue
- Deque als Verallgemeinerung; `collections.deque` und warum Listen als Queues ineffizient sind
- Anwendungsmuster: Klammerprüfung, Undo-Mechanismen, Expression Evaluation, Monotonic Stack
- Queues in realen Systemen (Task Queues, Message Queues, BFS-Vorgriff)

**Qualifikationsziele — fertig, wenn der Lernende:**
- Stack und Queue jeweils selbst implementieren und die Laufzeiten aller Operationen begründen kann,
- erkennt, wann ein Problem nach LIFO- oder FIFO-Verarbeitung verlangt,
- das Monotonic-Stack-Muster auf neue Aufgaben übertragen kann.

**Projekte:**
- **01-basic — Stack & Queue selbst bauen:** Stack (arraybasiert) und Queue (Ringpuffer mit fixer Kapazität) implementieren, inklusive Fehlerbehandlung (Underflow/Overflow) und Tests.
- **02-medium — Ausdrucks-Rechner:** Einen Parser/Evaluator für arithmetische Ausdrücke bauen (Infix zu Postfix via Shunting-Yard, dann Auswertung per Stack), inkl. Klammervalidierung.
- **03-final — Job-Queue-Simulation:** Eine Warteschlangen-Simulation für eingehende Analyse-Jobs (synthetischer Ankunftsprozess, fixer Seed): FIFO vs. Prioritätsregeln vergleichen, Wartezeiten statistisch auswerten und visualisieren.

---

## Modul 06 — Hashing & Hash Maps

**Inhalte (THEORY.md):**
- Das Kernversprechen: Suche in O(1) — und warum das fast zu gut klingt, um wahr zu sein
- Hash-Funktionen: Anforderungen, einfache Beispiele, Simulation der Verteilung von Schlüsseln auf Buckets
- Kollisionen und Auflösung: Chaining vs. Open Addressing (Linear Probing), Simulation beider Verfahren
- Load Factor und Rehashing; warum Worst Case O(n) bleibt
- Python-Interna: `dict` und `set` (konzeptionell), Hashability, warum Listen keine Keys sein können
- Anwendungsmuster: Frequency Counting, Deduplizierung, Lookup-Tabellen, Zweisummen-Muster

**Qualifikationsziele — fertig, wenn der Lernende:**
- eine eigene Hash Map mit Kollisionsbehandlung implementieren kann,
- die Auswirkung des Load Factors auf die Performance empirisch zeigen kann,
- Hash-basierte Lösungsmuster in Aufgaben erkennt (Trade-off Speicher gegen Zeit).

**Projekte:**
- **01-basic — Hash Map von Grund auf:** Implementierung mit Chaining (Put, Get, Delete, Rehashing bei Load Factor > 0.75), Messreihe: Performance bei wachsender Füllung mit und ohne Rehashing.
- **02-medium — Hash-Pattern-Katalog:** Klassiker mit Hash Maps lösen: Two Sum, Gruppierung von Anagrammen, erste eindeutige Zeichen, Duplikaterkennung in Datenströmen — mit Laufzeitvergleich gegen naive Lösungen.
- **03-final — Log-Analyse-Engine:** Synthetische Serverlogs (Generator mit Seed) auswerten: Top-K-IPs, Session-Zuordnung, Duplikatfilterung — vollständig hash-basiert, mit Messung gegen eine sortierbasierte Alternative.

---

## Modul 07 — Sortieralgorithmen

**Inhalte (THEORY.md):**
- Warum Sortieren das Lehrbuchproblem schlechthin ist (Vergleichbarkeit der Paradigmen)
- Elementare Verfahren: Bubble, Selection, Insertion Sort — je mit Schritt-für-Schritt-Simulation auf einem kleinen Array
- Effiziente Verfahren: Merge Sort und Quicksort (Partitionierung im Detail, Pivot-Wahl, Worst Case)
- Heap Sort (Verweis auf Modul 11, hier nur Prinzip)
- Nicht-vergleichende Verfahren: Counting Sort, Bucket Sort, Radix Sort — und wann sie erlaubt sind
- Stabilität, In-Place-Eigenschaft, untere Schranke Ω(n log n) für Vergleichssortierung (intuitives Argument)
- Timsort: was Python tatsächlich tut und warum

**Qualifikationsziele — fertig, wenn der Lernende:**
- mindestens fünf Sortierverfahren implementieren und deren Laufzeit/Stabilität/Speicherbedarf gegenüberstellen kann,
- die Wahl eines Verfahrens für ein gegebenes Szenario begründen kann (fast sortiert, viele Duplikate, begrenzte Wertebereiche),
- die Ω(n log n)-Schranke intuitiv erklären kann.

**Projekte:**
- **01-basic — Sortier-Werkstatt:** Bubble, Selection, Insertion, Merge und Quicksort implementieren; jede Implementierung gegen `sorted()` testen (Property-Tests mit Zufallsarrays, fixer Seed).
- **02-medium — Sortier-Benchmark:** Systematischer Benchmark aller Verfahren über verschiedene Eingabetypen (zufällig, fast sortiert, umgekehrt, viele Duplikate) mit Visualisierung; Interpretation der Ergebnisse gegen die Theorie.
- **03-final — Externe Sortierung:** Einen Datensatz sortieren, der „nicht in den Speicher passt" (künstliche Speichergrenze): Chunk-basiertes Sortieren + k-Way-Merge implementieren — das Grundprinzip verteilter Sortierung in Datenpipelines.

---

## Modul 08 — Binary Search & Suchvarianten

**Inhalte (THEORY.md):**
- Vom Telefonbuch zur Halbierungsidee; Simulation der Intervallverkleinerung Schritt für Schritt
- Die Invariante als Korrektheitswerkzeug; die klassischen Off-by-One-Fallen (`left <= right` vs. `left < right`)
- Varianten: erstes/letztes Vorkommen, Insert-Position, `bisect`-Modul
- Binary Search auf der Antwort (Suchraum ist kein Array, sondern ein Wertebereich) — mit Beispielen
- Suche in rotierten sortierten Arrays; Exponential Search
- Ausblick: Binary Search als Grundidee hinter Hyperparameter-Tuning-Strategien

**Qualifikationsziele — fertig, wenn der Lernende:**
- Binary Search fehlerfrei aus dem Kopf implementieren kann (inkl. Begründung der Invariante),
- die Technik „Binary Search auf der Antwort" auf neue Probleme übertragen kann,
- erkennt, wann Sortieren + Suchen schneller ist als lineare Suche.

**Projekte:**
- **01-basic — Binary Search sauber bauen:** Standard-Binary-Search plus Varianten (first/last occurrence, insert position) mit expliziter Invarianten-Dokumentation und Tests für alle Kantenfälle.
- **02-medium — Suche auf der Antwort:** Drei Probleme dieser Klasse lösen (z. B. minimale Kapazität für Auslieferung in D Tagen, Quadratwurzel ohne `math.sqrt`, kth-kleinstes Element in Matrix), jeweils mit Begründung der Monotonie.
- **03-final — Schwellenwert-Optimierung für einen Klassifikator:** Für einen simulierten Score-Datensatz (Seed) per Binary Search den Klassifikations-Schwellenwert finden, der eine Zielmetrik-Nebenbedingung erfüllt (z. B. maximaler Recall bei Precision ≥ 0.9) — statt naiver Rastersuche.

---

## Modul 09 — Two Pointers & Sliding Window

**Inhalte (THEORY.md):**
- Warum diese Muster existieren: von O(n²)-Brute-Force zu O(n) durch Ausnutzung von Struktur
- Two Pointers: gegenläufig (sortierte Arrays, Paarsuche) und gleichläufig (In-Place-Filterung) — je mit Zeiger-Simulation
- Sliding Window: festes Fenster (gleitende Summen) und variables Fenster (Wachsen/Schrumpfen mit Bedingung)
- Fenster-Zustandsverwaltung mit Hash Maps (z. B. längster Substring ohne Wiederholung)
- Abgrenzung: wann welches Muster greift, typische Erkennungssignale in Aufgabenstellungen
- Direkter Data-Science-Bezug: Rolling Windows in Zeitreihenanalyse

**Qualifikationsziele — fertig, wenn der Lernende:**
- beide Muster sicher erkennt und die Laufzeitverbesserung gegenüber Brute Force herleiten kann,
- variable Fenster mit Bedingungslogik korrekt implementiert (Wachsen/Schrumpfen ohne Endlosschleifen),
- die Muster auf Zeitreihenprobleme überträgt.

**Projekte:**
- **01-basic — Muster-Grundübungen:** Je drei kleine Aufgaben pro Muster (Paarsumme im sortierten Array, Container mit meistem Wasser, feste Fenstersumme, längster Substring ohne Duplikate) mit Brute-Force-Vergleichslösung und Laufzeitmessung.
- **02-medium — Fenster mit Zustand:** Aufgaben mit komplexerem Fensterzustand: minimales Fenster mit allen Zielzeichen, maximale Anzahl Einsen mit k Flips, Anagramm-Fenster — inkl. sauberer Zustandsinvarianten in der Dokumentation.
- **03-final — Streaming-Anomalieerkennung:** Über einem simulierten Metrik-Stream (Seed) rollierende Statistiken (Mittel, Standardabweichung) in O(1) pro Schritt pflegen und Anomalien markieren; Vergleich gegen die naive Neuberechnung pro Fenster.

---

## Modul 10 — Bäume & Binary Search Trees

**Inhalte (THEORY.md):**
- Hierarchien als Datenmodell: vom Dateisystem zum Entscheidungsbaum
- Terminologie (Wurzel, Blatt, Tiefe, Höhe) und rekursive Natur von Bäumen
- Traversierungen: Preorder, Inorder, Postorder, Level-Order — jede per Simulation an demselben Beispielbaum
- Binary Search Tree: Suchinvariante, Insert, Suche, Delete (drei Fälle, ausführlich simuliert)
- Degeneration zum „Linked-List-Baum" und warum Balance zählt; AVL/Red-Black nur konzeptionell (Rotationen als Idee, keine volle Implementierung gefordert)
- Bäume im Data-Science-Alltag: Entscheidungsbäume, hierarchische Indizes, JSON/XML-Strukturen

**Qualifikationsziele — fertig, wenn der Lernende:**
- alle vier Traversierungen rekursiv und Level-Order iterativ implementieren kann,
- BST-Operationen inklusive Delete korrekt umsetzt und die Invariante jederzeit begründen kann,
- erklären kann, warum unbalancierte BSTs zu O(n) degenerieren und was Balancierung dagegen tut.

**Projekte:**
- **01-basic — BST von Grund auf:** Vollständiger Binary Search Tree (Insert, Suche, Delete, Min/Max, alle Traversierungen) mit pytest-Suite und ASCII-Visualisierung der Baumstruktur.
- **02-medium — Baum-Aufgabenkatalog:** Höhe, Balance-Check, Validierung eines BST, Lowest Common Ancestor, Serialisierung/Deserialisierung — mit Rekursionsbaum-Skizzen in der Dokumentation.
- **03-final — Entscheidungsbaum von Hand:** Einen einfachen Decision-Tree-Klassifikator (Gini-Impurity, rekursives Splitten, Max-Tiefe) auf einem synthetischen Datensatz (Seed) selbst implementieren und gegen `sklearn.tree.DecisionTreeClassifier` vergleichen — die Brücke zwischen DSA und ML.

---

## Modul 11 — Heaps & Priority Queues

**Inhalte (THEORY.md):**
- Motivation: „Gib mir immer das Wichtigste zuerst" — warum sortierte Listen dafür zu teuer sind
- Der Binary Heap als Array: Eltern-Kind-Arithmetik, Heap-Eigenschaft
- Sift-Up und Sift-Down mit Schritt-für-Schritt-Simulation; Heapify in O(n) (intuitives Argument)
- Heap Sort; `heapq` in Python (Min-Heap, Max-Heap-Trick, Tupel als Prioritäten)
- Kernmuster: Top-K-Probleme, k-Way-Merge, Median-Erhaltung mit zwei Heaps
- Priority Queues in Algorithmen (Dijkstra-Vorgriff) und Systemen (Scheduler)

**Qualifikationsziele — fertig, wenn der Lernende:**
- einen Binary Heap arraybasiert selbst implementieren kann (Push, Pop, Heapify),
- Top-K-Probleme mit dem richtigen Heap-Typ (Min vs. Max) und der richtigen Größe löst,
- die O(n)-Heapify-Konstruktion der naiven O(n log n)-Variante gegenüberstellen kann.

**Projekte:**
- **01-basic — Binary Heap selbst bauen:** Min-Heap arraybasiert implementieren (Push, Pop, Peek, Heapify), gegen `heapq` property-testen, Heap Sort darauf aufsetzen.
- **02-medium — Heap-Pattern-Katalog:** Top-K häufigste Elemente, k-Way-Merge sortierter Listen, laufender Median mit zwei Heaps — mit Laufzeitanalysen.
- **03-final — Task-Scheduler mit Prioritäten:** Simulation eines Job-Schedulers (Jobs mit Priorität, Ankunftszeit, Dauer; Seed): Priority-Queue-basierte Abarbeitung gegen FIFO vergleichen, Kennzahlen (Wartezeit nach Prioritätsklasse) auswerten und visualisieren.

---

## Modul 12 — Graphen I: Repräsentation & Traversierung

**Inhalte (THEORY.md):**
- Graphen als universelles Modell: soziale Netzwerke, Abhängigkeiten, Straßennetze, Feature-Beziehungen
- Terminologie (gerichtet/ungerichtet, gewichtet, Grad, Pfad, Zyklus, Zusammenhang)
- Repräsentationen: Adjazenzliste vs. Adjazenzmatrix — Kostenvergleich und Wahlkriterien
- BFS mit Queue: Simulation Ebene für Ebene; kürzeste Pfade in ungewichteten Graphen
- DFS rekursiv und iterativ: Simulation mit explizitem Stack; Entdeckungs-/Abschlusszeiten (konzeptionell)
- Anwendungen: Zusammenhangskomponenten, Zyklenerkennung, Topologische Sortierung (Kahn und DFS-basiert)
- Bipartitheitstest per Färbung

**Qualifikationsziele — fertig, wenn der Lernende:**
- eine Graphklasse mit Adjazenzliste implementieren und BFS/DFS fehlerfrei darauf ausführen kann,
- für ein gegebenes Problem entscheidet, ob BFS oder DFS das richtige Werkzeug ist,
- topologische Sortierung implementieren und ihre Voraussetzung (DAG) prüfen kann.

**Projekte:**
- **01-basic — Graph-Grundgerüst:** Graphklasse (Adjazenzliste, gerichtet/ungerichtet), BFS und DFS (rekursiv + iterativ), Zusammenhangskomponenten — mit Tests auf handkonstruierten Beispielgraphen.
- **02-medium — Graph-Aufgabenkatalog:** Zyklenerkennung (gerichtet und ungerichtet), topologische Sortierung, Bipartitheitstest, Inselzählung auf Grid-Graphen — je mit Traversierungs-Trace in der Dokumentation.
- **03-final — Abhängigkeitsanalyse einer Datenpipeline:** Einen synthetischen DAG von Pipeline-Tasks (Seed) analysieren: gültige Ausführungsreihenfolge (topologische Sortierung), kritische Knoten, Auswirkung von Task-Ausfällen (Erreichbarkeitsanalyse) — das Denkmodell hinter Airflow-DAGs.

---

## Modul 13 — Graphen II: Kürzeste Wege, Union-Find & MST

**Inhalte (THEORY.md):**
- Gewichtete Graphen: warum BFS nicht mehr reicht
- Dijkstra mit Priority Queue: vollständige Simulation an einem Beispielgraphen (Distanztabelle Schritt für Schritt); Voraussetzung nichtnegativer Gewichte
- Bellman-Ford: Idee der Kantenrelaxierung, negative Zyklen erkennen (kompakter als Dijkstra behandelt)
- Union-Find (Disjoint Set Union): Simulation von Union und Find, Path Compression und Union by Rank intuitiv
- Minimale Spannbäume: Kruskal (mit Union-Find) und Prim (mit Heap) — Gegenüberstellung
- Einordnung: Wann welches Verfahren; Ausblick A* (nur Idee der Heuristik)

**Qualifikationsziele — fertig, wenn der Lernende:**
- Dijkstra mit Heap implementieren und an einem Beispiel von Hand nachrechnen kann,
- Union-Find mit beiden Optimierungen implementiert und dessen Rolle in Kruskal erklärt,
- für ein Szenario das passende Kürzeste-Wege- bzw. MST-Verfahren begründet wählt.

**Projekte:**
- **01-basic — Dijkstra & Union-Find bauen:** Beide Strukturen von Grund auf implementieren, Dijkstra auf einem handkonstruierten Graphen gegen die Handrechnung aus der Theorie testen.
- **02-medium — MST-Werkstatt:** Kruskal und Prim implementieren, auf denselben Zufallsgraphen (Seed) anwenden, Ergebnisse und Laufzeiten vergleichen.
- **03-final — Routing auf einem synthetischen Straßennetz:** Ein generiertes Gitter-/Zufallsstraßennetz mit Gewichten (Seed): kürzeste Routen berechnen, Auswirkungen von „Sperrungen" analysieren und die Ergebnisse visualisieren (matplotlib/networkx nur zur Darstellung, Algorithmen selbst implementiert).

---

## Modul 14 — Greedy-Algorithmen

**Inhalte (THEORY.md):**
- Die Greedy-Idee: lokal optimal entscheiden und nie zurückblicken
- Wann Greedy funktioniert — und die Beweislast: Gegenbeispiele konstruieren als Kernkompetenz
- Klassiker mit Simulation: Interval Scheduling (Auswahl nach frühestem Ende), Münzwechsel (wann Greedy scheitert), fraktionales Rucksackproblem
- Huffman-Codierung als konstruktives Greedy-Beispiel (mit Heap)
- Exchange-Argument als Beweisidee (intuitiv, an einem Beispiel)
- Abgrenzung zu Dynamic Programming: dieselben Probleme, andere Garantien

**Qualifikationsziele — fertig, wenn der Lernende:**
- für eine Greedy-Strategie prüfen kann, ob sie korrekt ist (Gegenbeispiel oder Plausibilisierung),
- Interval Scheduling und Huffman implementieren kann,
- artikulieren kann, woran man Greedy-taugliche Probleme erkennt.

**Projekte:**
- **01-basic — Greedy-Klassiker:** Interval Scheduling, fraktionaler Rucksack und Münzwechsel implementieren; für Münzwechsel ein Münzsystem konstruieren, bei dem Greedy scheitert, und das Scheitern demonstrieren.
- **02-medium — Huffman-Kompressor:** Vollständige Huffman-Codierung (Frequenzanalyse, Baumbau per Heap, En-/Dekodierung) für Textdateien; Kompressionsraten auf verschiedenen Texten messen.
- **03-final — Meeting-Raum-Planer:** Für einen synthetischen Kalender-Datensatz (Seed) die minimale Raumzahl bestimmen (Sweep-Line/Heap) und Belegungspläne erzeugen; Ergebnisqualität gegen eine naive Zuordnung vergleichen.

---

## Modul 15 — Backtracking

**Inhalte (THEORY.md):**
- Systematisches Ausprobieren mit Rückzug: der Suchbaum als mentales Modell
- Anatomie eines Backtracking-Algorithmus: Wahl, Constraint-Prüfung, Rekursion, Rückgängigmachen
- Simulation an N-Damen: Suchbaum zeichnen, Pruning sichtbar machen
- Standardprobleme: Permutationen, Kombinationen, Subsets, Sudoku, Wortsuche im Grid
- Pruning-Strategien und ihre Wirkung (Messung: Knoten im Suchbaum mit/ohne Pruning)
- Komplexität: warum Backtracking exponentiell bleibt und wann das akzeptabel ist

**Qualifikationsziele — fertig, wenn der Lernende:**
- das Backtracking-Template sicher auf neue Probleme anwenden kann (Wahl/Constraint/Undo sauber trennen),
- Pruning-Bedingungen formulieren und ihren Effekt messen kann,
- Permutationen/Kombinationen/Subsets ohne Nachschlagen generieren kann.

**Projekte:**
- **01-basic — Generatoren-Trio:** Permutationen, Kombinationen und Subsets per Backtracking generieren, gegen `itertools` property-testen, Suchbaumgröße protokollieren.
- **02-medium — Constraint-Löser:** N-Damen und Sudoku-Löser implementieren; für beide die Anzahl besuchter Knoten mit und ohne Pruning messen und dokumentieren.
- **03-final — Feature-Subset-Suche:** Für ein kleines synthetisches Regressionsproblem (Seed) per Backtracking mit Pruning (Branch and Bound auf Basis einer Schranke) die beste Feature-Teilmenge unter einer Budgetbeschränkung finden — Vergleich gegen erschöpfende Suche und Greedy-Forward-Selection.

---

## Modul 16 — Dynamic Programming

**Inhalte (THEORY.md):**
- Der Kern: überlappende Teilprobleme + optimale Substruktur; Fibonacci als Einstiegs-Simulation (Rekursionsbaum mit Doppelberechnungen sichtbar machen)
- Memoization (Top-Down) vs. Tabulation (Bottom-Up): dieselbe Idee, zwei Richtungen — beide simuliert
- Das DP-Vorgehensmodell: Zustand definieren, Rekurrenz aufstellen, Basisfälle, Auswertungsreihenfolge, Speicheroptimierung
- 1D-Klassiker: Climbing Stairs, House Robber, Coin Change, Longest Increasing Subsequence
- 2D-Klassiker: 0/1-Rucksack (Tabelle Zelle für Zelle simuliert), Edit Distance, Longest Common Subsequence, Grid-Pfade
- Rekonstruktion der Lösung aus der Tabelle
- DP im Data-Science-Umfeld: Edit Distance in der Textverarbeitung, Sequence Alignment, Dynamic Time Warping (Ausblick)

**Qualifikationsziele — fertig, wenn der Lernende:**
- für ein neues Problem Zustand und Rekurrenz eigenständig formulieren kann (das eigentliche Lernziel),
- ein Problem sowohl top-down als auch bottom-up lösen und die Tabelle von Hand füllen kann,
- Speicheroptimierung (Zeilenreduktion) anwenden kann, wo die Rekurrenz es erlaubt.

**Projekte:**
- **01-basic — Vom Rekursionsbaum zur Tabelle:** Fibonacci, Climbing Stairs und Coin Change je dreifach lösen (naiv rekursiv, memoisiert, tabuliert) mit Laufzeitmessung; Rekursionsbaum-Visualisierung aus Modul 03 wiederverwenden.
- **02-medium — DP-Klassiker-Katalog:** 0/1-Rucksack, Longest Common Subsequence, Edit Distance und Longest Increasing Subsequence mit Tabellen-Dokumentation und Lösungsrekonstruktion.
- **03-final — Fuzzy-Matching für Datenbereinigung:** Ein Deduplizierungs-Tool für einen synthetischen Kundendatensatz mit Tippfehlern (Seed): Edit-Distance-basiertes Matching mit Schwellenwert, Blocking zur Beschleunigung, Auswertung von Precision/Recall gegen die bekannte Ground Truth.

---

## Modul 17 — Capstone: DSA im Data-Science-Kontext

**Inhalte (THEORY.md — hier kompakter, eher Einordnung als neue Theorie):**
- Synthese: Welche Struktur/welches Paradigma für welches Problem — Entscheidungsleitfaden als Referenz
- Wo DSA in DS-Werkzeugen steckt: Hashing in Joins und GroupBy, Sortierung in Indizes, Bäume in Modellen, Graphen in Pipelines, Heaps in Top-K-Queries
- Interview-Vorbereitung: typischer Ablauf technischer Interviews, lautes Denken, Muster-Erkennung, Komplexität kommunizieren
- Kein neuer Stoff — Verweise auf die Module 01–16 statt Wiederholung

**Qualifikationsziele — fertig, wenn der Lernende:**
- für unbekannte Probleme systematisch Struktur- und Algorithmuswahl begründen kann,
- alle drei Capstone-Projekte eigenständig und mit sauberer Komplexitätsanalyse gelöst hat.

**Projekte:**
- **01-basic — Mock-Interview-Set:** 10 gemischte Aufgaben quer durch alle Module (ohne Modulzuordnung im Aufgabentext), Musterlösungen mit Erklärung, welches Muster jeweils greift und woran man es erkennt.
- **02-medium — Mini-DataFrame-Engine:** Eine stark vereinfachte DataFrame-Klasse selbst bauen: Filter, hashbasiertes GroupBy mit Aggregation, sortierbasierter und hashbasierter Join, Top-K per Heap — mit Komplexitätsangabe pro Operation und Benchmark gegen Pandas auf einem synthetischen Datensatz (Seed).
- **03-final — Empfehlungs-Pipeline End-to-End:** Auf einem synthetischen Nutzer-Interaktions-Datensatz (Seed): Ähnlichkeitsgraph aufbauen (Hashing für Feature-Lookup), Nachbarschaften per BFS begrenzen, Top-K-Empfehlungen per Heap, Deduplizierung, Laufzeitbudget einhalten und dokumentieren — ein Projekt, das mindestens fünf Module kombiniert und portfoliotauglich in einer README präsentiert wird.

---

## Hinweis zum Abschluss

Nach Fertigstellung aller 17 Module: kurze `ABSCHLUSS.md` im Repo-Root erzeugen mit
Gesamtübersicht (Module, Projektanzahl, verwendete Formate) — keine inhaltliche
Wiederholung, nur Inventar.
