# Projekt 02-medium: Sortier-Benchmark

## Ziel

Dieses Projekt vergleicht sechs Sortierverfahren systematisch über vier
Eingabeformen: zufällig, fast sortiert, umgekehrt und mit vielen Duplikaten.
Gemessen werden Medianlaufzeit und deterministische Vergleichszahl. Dadurch
lassen sich Hardwareeffekte von algorithmischen Wachstumsmustern unterscheiden.

Das Projekt verwendet Python-Skripte, CSV, Matplotlib und einen generierten
Markdown-Bericht. Diese Pipeline ist einem Notebook vorzuziehen, weil jeder Lauf
alle Ergebnisprüfungen, Messungen und Artefakte reproduzierbar neu erzeugt.

## Untersuchte Verfahren

- Bubble Sort mit Early Exit
- Selection Sort
- Insertion Sort
- Merge Sort
- 3-Wege-Quicksort
- Pythons `sorted()` beziehungsweise Timsort

Die Implementierungen sind bewusst eigenständig, damit das Benchmark-Projekt
isoliert ausführbar bleibt. Jede Messung wird gegen `sorted()` validiert, bevor
ihre Zeit übernommen wird.

## Eingabeformen

| Typ | Konstruktion | erwartete Beobachtung |
|---|---|---|
| `random` | gleichverteilte Integer | allgemeines Wachstum |
| `nearly_sorted` | sortiert, etwa 5 % zufällige Swaps | Vorteil adaptiver Verfahren |
| `reversed` | streng absteigend | Worst Case für Insertion, viele Swaps |
| `many_duplicates` | nur acht mögliche Werte | Vorteil der 3-Wege-Partitionierung |

Ein fixer Seed sorgt dafür, dass jedes Verfahren dieselben Werte und Swaps
erhält. Die Standardgrößen sind 100, 200, 400, 800 und 1.600.

## Messprotokoll

Für jede Kombination aus Algorithmus, Eingabeform und Größe:

1. wird die Referenz mit `sorted()` berechnet,
2. läuft der Algorithmus dreimal auf derselben unveränderten Eingabe,
3. wird jedes Resultat mit der Referenz verglichen,
4. muss die Vergleichszahl über alle Wiederholungen identisch sein,
5. wird der Median der Laufzeiten gespeichert.

So entstehen `6 × 4 × 5 = 120` Messzeilen. Der Median reduziert den Einfluss
einzelner Störungen, ersetzt aber keine kontrollierte Low-Level-Mikrobenchmark.

## Visualisierung lesen

Der Plot besitzt ein Panel pro Eingabeform. Beide Achsen sind logarithmisch:

- Eine Verdopplung von n hat überall denselben horizontalen Abstand.
- Quadratisches Wachstum vervierfacht die Arbeit ungefähr.
- O(n log n) wächst deutlich flacher.
- Ein niedriger Einzelpunkt ist weniger wichtig als die Form der ganzen Kurve.

Die Vergleichszahlen im CSV helfen, wenn Python-Overhead die Millisekundenkurven
bei kleinen n überlagert.

## Theoriebezug

Selection Sort führt unabhängig von der Ordnung n(n-1)/2 Vergleiche aus.
Insertion Sort hängt von der Zahl der Inversionen ab und reagiert stark auf die
Eingabeform. Merge Sort bleibt O(n log n). Der 3-Wege-Quicksort trennt Werte
kleiner, gleich und größer als das Pivot, sodass große Gleichheitsgruppen nicht
rekursiv weiterbearbeitet werden. Timsort erkennt vorhandene Runs und nutzt fast
sortierte Daten besonders gut.

## Ausführen

Im Projektordner:

```bash
python run_benchmark.py
python -m pytest -q
```

Erzeugte Artefakte:

```text
output/
├── benchmark.csv
├── sorting_benchmark.png
└── REPORT.md
```

## Hinweise

- Verwende für alle Algorithmen exakt dieselbe Eingabe, nicht nur dieselbe
  Verteilung.
- Kopiere Eingaben intern; eine Mutation würde spätere Messungen bevorteilen.
- Prüfe Korrektheit in jeder Wiederholung.
- Interpretiere Laufzeiten nicht ohne Vergleichszahlen und Wachstumskurven.
- Ein Quicksort mit einfachem Rand-Pivot wäre auf sortierten Eingaben unfair
  degeneriert; die 3-Wege-Mittel-Pivot-Variante isoliert hier den
  Paradigmenvergleich besser.

## Fertig, wenn …

- alle sechs Verfahren jede Eingabeform korrekt sortieren,
- Workloads mit festem Seed reproduzierbar sind,
- alle 120 Standardmessungen Ergebnisgleichheit bestätigen,
- Medianlaufzeit und Vergleichszahl im CSV stehen,
- der Plot vier lesbare Panels mit logarithmischen Achsen enthält,
- der Bericht Beobachtungen mit Best/Average/Worst Case verbindet,
- die vollständige Standardpipeline ohne Fehler läuft und
- alle Tests mit `python -m pytest -q` erfolgreich sind.
