# Benchmark-Bericht: Muster-Grundübungen

Alle Fälle verwenden Seed `90901`, drei Wiederholungen und den Median. Vor der
Messung wird die Ergebnisgleichheit jeder Referenz-/Musterlösung geprüft.

| Aufgabe | n | Brute Force (ms) | Muster (ms) | Faktor |
|---|---:|---:|---:|---:|
| pair_sum | 2.500 | 70.253 | 0.115 | 608.47x |
| max_container | 2.200 | 117.037 | 0.272 | 431.08x |
| in_place_filter | 100.000 | 2.288 | 3.965 | 0.58x |
| rolling_sums | 5.000 | 9.548 | 0.207 | 46.13x |
| longest_unique_substring | 1.500 | 96.911 | 0.131 | 742.14x |
| minimum_length | 2.500 | 71.209 | 0.172 | 413.81x |

Die quadratischen Referenzen für Paarsumme, Wassercontainer, eindeutigen
Substring und unerreichbare Zielsumme zeigen den strukturellen Vorteil der
gerichteten Zeigerbewegung. Bei Rolling Sums wird jedes überlappende Fenster
nicht erneut summiert, sondern mit Eintritt und Austritt aktualisiert.

Die In-Place-Filterung ist bewusst kein asymptotischer Zeitgewinn: Beide
Varianten sind `O(n)`. Der Schreibzeiger reduziert den zusätzlichen Speicher von
`O(n)` auf `O(1)` abgesehen von der übergebenen Liste. Kleine Laufzeitunterschiede
sind hier Implementierungsdetails und nicht das Lernziel.
