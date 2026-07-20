# T14 – Heteroskedastizität im Haushaltskonsum

## Ausgangslage

Bei einkommensschwachen Haushalten liegen Konsumausgaben relativ eng beieinander, während sie bei hohen Einkommen stark variieren. Der mittlere Zusammenhang bleibt linear, aber die bedingte Fehlervarianz wächst mit dem Einkommen. Das erzeugt die typische Trichterform der Heteroskedastizität.

## Lernziele

Nach dem Projekt kannst du:

- Heteroskedastizität in Rohdaten und Residuenplots erkennen,
- konstante bedingte Varianz von einem korrekten bedingten Mittelwert unterscheiden,
- konventionelle und heteroskedastizitätsrobuste Standardfehler vergleichen,
- den Breusch–Pagan-Test als ergänzende, nicht alleinige Diagnose einsetzen.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Zeichne Konsum gegen Einkommen und beschreibe Mittelwerts- und Varianzstruktur getrennt.
2. Schätze `consumption_eur ~ income_eur` mit OLS.
3. Zeichne Residuen gegen Fits. Teile die Fits in zehn Gruppen und berechne je Gruppe die Residuenstandardabweichung.
4. Formuliere die verletzte Gauss–Markov-Annahme als \(\operatorname{Var}(u\mid X)=\sigma^2\).
5. Berechne konventionelle und HC1-robuste Standardfehler und 95%-Konfidenzintervalle für die Steigung.
6. Führe einen Breusch–Pagan-Test durch und interpretiere Nullhypothese sowie p-Wert.
7. Erkläre, weshalb Heteroskedastizität bei \(E[u\mid X]=0\) nicht automatisch den OLS-Punktschätzer verzerrt.
8. Erkläre, weshalb falsche Standardfehler trotzdem t-Tests und Konfidenzintervalle unzuverlässig machen.

## Ausführen

```bash
python3 exercises/T14-heteroscedasticity/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/inference_comparison.csv`, `results/residual_scale_by_decile.csv` und `results/heteroscedasticity_diagnostics.png`:

```bash
python3 exercises/T14-heteroscedasticity/solution.py
```

## Denkfragen

- Ist ein gekrümmter Residuenverlauf dasselbe Problem wie ein Trichter?
- Reparieren robuste Standardfehler eine falsch spezifizierte Regressionsfunktion?
- Warum sollte eine grafische Diagnose nicht durch einen einzelnen Test ersetzt werden?
