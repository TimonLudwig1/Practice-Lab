# T50 – IV-Schätzung als Kovarianzquotient

## Ausgangslage

Ein Stipendienindex beeinflusst Bildungsjahre. Fähigkeit ist unbeobachtet und erhöht sowohl Bildung als auch Lohn, weshalb selbst ein OLS-Modell mit Berufserfahrung verzerrt bleibt. Der Stipendienindex ist nach Kontrolle für Berufserfahrung exogen. Dieses Abschlussprojekt verbindet First Stage, Reduced Form, 2SLS und den Kovarianzquotienten.

## Lernziele

- den einfachen IV-Schätzer als `Cov(Z,Y) / Cov(Z,X)` herleiten und berechnen,
- bei exogenen Kontrollvariablen residualisierte Kovarianzen verwenden,
- First-Stage-/Reduced-Form-Ratio und Two-Stage Least Squares reproduzieren,
- Invarianz gegenüber affiner Skalierung des Instruments prüfen,
- OLS-Bias trotz beobachteter Kontrollen erkennen,
- IV-Unsicherheit per Bootstrap quantifizieren.

## Aufgaben

1. Schätze ein OLS-Lohnmodell mit Bildung und Berufserfahrung.
2. Berechne zunächst den unkonditionalen Kovarianzquotienten.
3. Residualisiere Instrument, Bildung und Lohn jeweils auf Berufserfahrung.
4. Berechne `Cov(Z̃,Ỹ) / Cov(Z̃,X̃)` und alternativ das Verhältnis der kontrollierten Regressionskoeffizienten.
5. Führe 2SLS explizit in zwei Stufen durch und vergleiche den Bildungskoeffizienten.
6. Skaliere das Instrument affin und prüfe, dass sich der IV-Schätzer nicht ändert.
7. Erzeuge ein Bootstrap-Konfidenzintervall für den kontrollierten IV-Schätzer.
8. Erkläre, weshalb die Second-Stage-OLS-Standardfehler mit vorhergesagtem Treatment nicht einfach übernommen werden sollten.

## Ausführen

```bash
python3 exercises/T50-iv-covariance-ratio/starter.py
python3 exercises/T50-iv-covariance-ratio/solution.py
```

Die Lösung erzeugt `data/education_iv_data.csv`, Ergebnistabellen und `results/iv_covariance_ratio.png`.

## Denkfragen

- Warum müssen bei Kontrollen alle drei Größen `Z`, `X` und `Y` residualisiert werden?
- Weshalb ändert eine Multiplikation des Instruments sowohl First Stage als auch Reduced Form, aber nicht ihr Verhältnis?
- Auf welche Population bezieht sich der IV-Effekt bei einem binären Instrument und heterogenen Wirkungen?
