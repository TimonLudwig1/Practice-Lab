# T03 – Sampling Variation vs. Bias bei einer Zufriedenheitsumfrage

## Ausgangslage

Eine Plattform möchte die durchschnittliche Kundenzufriedenheit messen. Bei einer Zufallsstichprobe hat jede Person dieselbe Auswahlchance. Bei einer freiwilligen In-App-Umfrage antworten dagegen besonders zufriedene Kundinnen und Kunden häufiger. Du vergleichst, wie sich beide Designs über viele Wiederholungen verhalten.

## Lernziele

Nach dem Projekt kannst du:

- Sampling Variation als Streuung eines Schätzers über Stichproben definieren,
- Bias als systematische Abweichung vom wahren Parameter messen,
- Bias und Präzision in einer gemeinsamen Grafik auseinanderhalten,
- erklären, warum eine große verzerrte Stichprobe sehr präzise falsch sein kann.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Berechne den wahren Mittelwert der Zufriedenheit in der Population.
2. Untersuche grafisch, wie die Teilnahmechance mit der Zufriedenheit zusammenhängt.
3. Simuliere für \(n=100\) und \(n=1000\) jeweils 2.000 Mittelwerte aus einer gleichgewichteten Zufallsstichprobe.
4. Wiederhole die Simulation mit Auswahlwahrscheinlichkeiten proportional zur Teilnahmechance.
5. Berechne pro Design und \(n\): mittleren Schätzwert, Bias, Standardabweichung und RMSE.
6. Stelle die Stichprobenverteilungen dar und markiere den wahren Mittelwert.
7. Erkläre getrennt, was eine Vergrößerung von \(n\) mit Variation und mit Bias macht.

## Ausführen

```bash
python3 exercises/T03-sampling-variation-vs-bias/starter.py
```

Die Musterlösung erzeugt `results/bias_variation_summary.csv`, `results/selection_mechanism.png` und `results/bias_vs_variation.png`:

```bash
python3 exercises/T03-sampling-variation-vs-bias/solution.py
```

## Denkfragen

- Welches Design ist erwartungstreu und woran erkennst du das?
- Kann ein Schätzer zugleich geringe Varianz und großen Bias haben?
- Warum behebt eine Million freiwillige Antworten nicht automatisch Selection Bias?
