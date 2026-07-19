# Benchmark-Bericht: Muster-Grundübungen

Alle Fälle verwenden Seed `90901`, drei Wiederholungen und den Median. Vor der
Messung wird die Ergebnisgleichheit jeder Referenz-/Musterlösung geprüft.

| Aufgabe | n | Brute Force (ms) | Muster (ms) | Faktor |
|---|---:|---:|---:|---:|
| pair_sum | 2.500 | 68.800 | 0.117 | 587.62x |
| max_container | 2.200 | 117.398 | 0.255 | 460.53x |
| in_place_filter | 100.000 | 2.238 | 3.817 | 0.59x |
| rolling_sums | 5.000 | 9.671 | 0.227 | 42.52x |
| longest_unique_substring | 1.500 | 95.504 | 0.144 | 661.31x |
| minimum_length | 2.500 | 68.191 | 0.184 | 370.35x |

Die quadratischen Referenzen für Paarsumme, Wassercontainer, eindeutigen
Substring und unerreichbare Zielsumme zeigen den strukturellen Vorteil der
gerichteten Zeigerbewegung. Bei Rolling Sums wird jedes überlappende Fenster
nicht erneut summiert, sondern mit Eintritt und Austritt aktualisiert.

Die In-Place-Filterung ist bewusst kein asymptotischer Zeitgewinn: Beide
Varianten sind `O(n)`. Der Schreibzeiger reduziert den zusätzlichen Speicher von
`O(n)` auf `O(1)` abgesehen von der übergebenen Liste. Kleine Laufzeitunterschiede
sind hier Implementierungsdetails und nicht das Lernziel.
