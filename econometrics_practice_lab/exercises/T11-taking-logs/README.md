# T11 – Schiefe Unternehmensdaten logarithmieren

## Ausgangslage

Beschäftigtenzahl und Umsatz von 300 synthetischen Unternehmen sind stark rechtsschief: Viele Firmen sind klein, wenige sehr groß. Gewinne können dagegen negativ sein. Du untersuchst, was eine Log-Transformation mit Verteilung, Abständen und zulässigem Wertebereich macht.

## Lernziele

Nach dem Projekt kannst du:

- rechtsschiefe Variablen diagnostizieren und logarithmieren,
- additive Abstände auf der Log-Skala als multiplikative Verhältnisse verstehen,
- den Definitionsbereich des natürlichen Logarithmus prüfen,
- `log(x)` und `log1p(x)` bewusst statt austauschbar einsetzen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Beschreibe Mittelwert, Median, Quantile und Schiefe von Beschäftigtenzahl, Umsatz und Gewinn.
2. Erzeuge `log_employees` und `log_revenue` mit dem natürlichen Logarithmus. Prüfe vorher strikt positive Werte.
3. Vergleiche Histogramme und Schiefe vor und nach der Transformation.
4. Wähle zwei Firmen, deren Umsatz sich ungefähr verdoppelt. Zeige, dass ihre Log-Umsätze sich ungefähr um \(\log(2)\) unterscheiden.
5. Erkläre, weshalb gleiche Abstände auf der Log-Skala gleichen Verhältnissen statt gleichen Eurobeträgen entsprechen.
6. Zähle Unternehmen mit Gewinn \(\leq0\). Was geschieht bei blindem `np.log(profit_eur)`?
7. Berechne `log_profit` nur für positive Gewinne und diskutiere, welche Auswahl dadurch entsteht.
8. Erkläre, wann `np.log1p(x)` für Nullen technisch sinnvoll sein kann und warum es negative Gewinne unter −1 nicht löst.

## Ausführen

```bash
python3 exercises/T11-taking-logs/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/transformation_summary.csv`, `results/log_domain_summary.csv` und `results/log_distributions.png`:

```bash
python3 exercises/T11-taking-logs/solution.py
```

## Denkfragen

- Verändert Logarithmieren die Rangfolge positiver Beobachtungen?
- Warum wird aus einem extrem großen absoluten Abstand auf der Rohskala oft ein moderater Abstand auf der Log-Skala?
- Darf man negative Beobachtungen ohne inhaltliche Begründung löschen, nur um ein Log-Modell schätzen zu können?
