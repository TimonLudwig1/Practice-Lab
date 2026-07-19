# Ergebnisbericht: Sortier-Benchmark

Alle Algorithmen wurden vor jeder Zeitübernahme gegen `sorted()` geprüft. Die
Tabelle zeigt den größten Lauf mit n = 1600; Zeiten sind Medianwerte
aus den im CSV dokumentierten Wiederholungen.

| Eingabe | schnellste gemessene Variante | Median ms | Insertion-Vergleiche | Selection-Vergleiche |
|---|---|---:|---:|---:|
| random | python_timsort | 0.962 | 645017 | 1279200 |
| nearly_sorted | python_timsort | 0.592 | 84317 | 1279200 |
| reversed | python_timsort | 0.195 | 1279200 | 1279200 |
| many_duplicates | quick_3way | 0.353 | 560015 | 1279200 |

## Interpretation gegen die Theorie

Selection Sort führt für dieselbe Größe unabhängig von der Eingabeform stets
nahezu n(n-1)/2 Vergleiche aus. Seine Kurven reagieren deshalb kaum auf bereits
vorhandene Ordnung. Bubble Sort kann nur dann früh abbrechen, wenn ein kompletter
Pass ohne Tausch bleibt.

Insertion Sort nutzt vorhandene Ordnung direkt: Seine Arbeit hängt von der Zahl
der Inversionen ab. Auf umgekehrten Eingaben nähert es sich dem quadratischen
Worst Case, während es auf fast sortierten Folgen deutlich weniger Vergleiche
benötigt.

Merge Sort bleibt über alle Formen bei O(n log n). Der 3-Wege-Quicksort isoliert
gleiche Werte gemeinsam und vermeidet deshalb die typische Degeneration bei
vielen Duplikaten. Python-Timsort erkennt natürliche Runs und ist besonders für
fast sortierte reale Daten optimiert.

Absolute Millisekunden hängen von Hardware und Python-Version ab. Für die
algorithmische Bewertung sind Kurvenform und Vergleichszahlen belastbarer als
ein einzelner Geschwindigkeitsfaktor.
