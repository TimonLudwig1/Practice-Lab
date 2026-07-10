# Projekt 02 (medium) — Bootstrap & Permutationstest: Statistik aus dem Computer

**Format:** Jupyter Notebook (`bootstrap_permutation.ipynb`).
**Warum dieses Format?** Resampling-Verfahren versteht man über die Histogramme der Bootstrap- und Nullverteilungen — Code und Grafik müssen direkt nebeneinander stehen.

**Daten: synthetisch, mit bekannter Wahrheit — das ist hier der Kern der Didaktik.** Ein A/B-Test (2×4.000 Besucher, Konversion + Bestellwerte, fester Seed 42) wird in der ersten Zelle erzeugt. Weil die wahren Effekte im Generator stehen, kannst du am Ende prüfen, ob deine Tests die Wahrheit gefunden haben — und erlebst kontrolliert den wichtigsten Fall der Praxis: **einen echten Effekt, den der Test mangels Power nicht findet.**

## Ziel

Bootstrap-Konfidenzintervalle (Median, Median-Differenz) und einen Permutationstest von Hand implementieren, auf beide A/B-Fragen anwenden und die Ergebnisse korrekt interpretieren: Bestellwert-Effekt signifikant (p ≈ 0,005), Konversions-Effekt trotz realer Existenz nicht nachweisbar (p ≈ 0,19) → „nicht signifikant ≠ kein Effekt".

## Vorwissen

- Modul-Skript Abschnitte 3.1 und 3.2; Modul 02 Abschnitt 3.1 (p-Werte, KIs)
- numpy-Grundlagen (`choice`, `permutation`, `percentile`)

## Aufgaben

1. Notebook durcharbeiten; zu implementieren sind `bootstrap_verteilung`, das Differenz-KI, `permutationstest` und dessen Anwendung auf die Konversionsraten.
2. Mini-Checks müssen `True` ergeben.
3. Abschluss: 3-Sätze-Empfehlung ans Produktteam formulieren (Muster ausklappbar).

## Was am Ende funktionieren soll

- Bootstrap-95%-KI des B-Medians ≈ [53, 68] €; KI der Differenz ≈ [3,5, 20,5] € (enthält die 0 nicht, enthält aber den wahren Wert 6 €).
- Permutationstest Bestellwert: p < 0,02; Konversion: p > 0,05.
- Du kannst erklären, warum der Konversionstest trotz echten Effekts scheitert (Power).

## Lösung

Vollständig ausgeführte Musterlösung: [`loesung/loesung.ipynb`](loesung/loesung.ipynb).
