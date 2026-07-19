# Laufbericht: Streaming-Anomalieerkennung

## Szenario

Der Generator erzeugt mit Seed `90903` insgesamt 12.000
Minutenmesswerte mit langsamem periodischem Verlauf, Gaußrauschen und
20 weit auseinanderliegenden injizierten Ausreißern.

Jeder neue Wert wird ausschließlich gegen die **vorherigen 120 Werte**
bewertet. Damit gibt es weder Look-ahead noch eine Aufnahme des zu prüfenden
Werts in seine eigene Baseline. Eine Anomalie liegt bei
`|z| > 5.0` vor.

## Ergebnis

- True Positives: 20
- False Positives: 0
- False Negatives: 0
- Precision: 1.0000
- Recall: 1.0000

## Laufzeitvergleich

- Naive Neuberechnung, Median: 59.638 ms
- O(1)-Rolling-Update, Median: 15.151 ms
- Beschleunigung: 3.94x

Beide Verfahren erzeugen dieselben Anomalieflags sowie innerhalb enger Toleranz
dieselben Mittelwerte und Standardabweichungen. Die Referenz summiert für jeden
Punkt ein Fenster der Breite `k` neu und kostet `O(nk)`. Die Streaming-Variante
entfernt einen Wert aus Summe und Quadratsumme und fügt einen Wert hinzu. Damit
kostet jeder Schritt `O(1)` und der gesamte Lauf `O(n)`.

Die Varianzformel `E[x²] - E[x]²` ist schnell, kann bei sehr großen fast gleichen
Zahlen aber Auslöschung zeigen. Die Implementierung klemmt winzige negative
Rundungsreste auf null; für numerisch extreme Produktionsdaten wären stabilere
Online-Verfahren oder periodische Rekalibrierung zu prüfen.
