# T44 – Two-Way Fixed Effects im Filialpanel

## Ausgangslage

Ein neues Beratungskonzept startet in der Hälfte der Filialen ab Woche 6. Die ausgewählten Filialen besitzen dauerhaft höhere Standortqualität. Gleichzeitig schwankt der Umsatz in jeder Woche durch gemeinsame Nachfrageimpulse. Ein Two-Way-Fixed-Effects-Modell kontrolliert beides: Filial- und Wochen-Fixed-Effects.

## Lernziele

- ein DiD-Modell mit individuellen und zeitlichen Fixed Effects schätzen,
- verstehen, warum nur eine der beiden FE-Dimensionen unzureichend sein kann,
- den TWFE-Koeffizienten durch doppeltes Demeaning reproduzieren,
- unter gleichzeitigem Treatmentbeginn die Verbindung zum klassischen DiD herstellen,
- Standardfehler auf Ebene der Behandlungseinheit clustern.

## Aufgaben

1. Erzeuge den aktiven Treatmentindikator `did = treated × post`.
2. Schätze nacheinander ein naives Modell, ein Modell nur mit Filial-FE, nur mit Wochen-FE und mit beiden FE-Arten.
3. Vergleiche Koeffizienten und geclusterte Konfidenzintervalle.
4. Entferne von Outcome und `did` Filialmittel und Wochenmittel und addiere das Gesamtmittel zurück.
5. Berechne den TWFE-Koeffizienten aus den doppelt zentrierten Variablen per Hand.
6. Berechne den klassischen Gruppen-Zeit-DiD-Schätzer und vergleiche ihn mit TWFE.
7. Erkläre, welche Verzerrungsquelle jeweils im Filial-FE- und im Wochen-FE-Modell verbleibt.

## Ausführen

```bash
python3 exercises/T44-did-two-way-fixed-effects/starter.py
python3 exercises/T44-did-two-way-fixed-effects/solution.py
```

Die Lösung erzeugt `data/store_twfe_panel.csv`, Ergebnistabellen und `results/two_way_fixed_effects.png`.

## Denkfragen

- Warum werden `treated` und `post` im TWFE-Modell nicht separat benötigt?
- Unter welchen komplizierteren Treatmentstrukturen ist ein einzelner TWFE-Koeffizient schwieriger zu interpretieren?
- Weshalb sollte bei wenigen Behandlungseinheiten eine gewöhnliche Cluster-Näherung kritisch betrachtet werden?
