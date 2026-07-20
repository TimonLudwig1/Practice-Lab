# T43 – DiD mit Zeit-Fixed-Effects

## Ausgangslage

Ein regionales Gutscheinprogramm startet ab Monat 7. Gleichzeitig schwankt die Kundennachfrage durch Feiertage, Wetter und Saison. Diese Schocks betreffen alle Regionen, sind aber nicht durch einen einzigen Vorher-/Nachher-Sprung angemessen beschrieben. Monats-Fixed-Effects erlauben für jeden Zeitpunkt ein eigenes gemeinsames Niveau.

## Lernziele

- gemeinsame Zeitschocks von einem Treatment-Effekt trennen,
- einen einzelnen Post-Dummy mit vollständigen Zeit-Fixed-Effects vergleichen,
- erkennen, welche Zeitvariation Zeit-Fixed-Effects absorbieren,
- Zeit-Fixed-Effects aus einer Regression rekonstruieren und normalisieren,
- DiD-Inferenz bei Paneldaten nach Einheit clustern.

## Aufgaben

1. Visualisiere die monatlichen Mittelwerte beider Gruppen.
2. Schätze ein naives Modell nur mit dem aktiven Treatmentindikator `did`.
3. Ergänze Treatment- und Post-Dummy als klassisches DiD-Modell.
4. Ersetze den Post-Dummy durch vollständige Monats-Fixed-Effects.
5. Vergleiche DiD-Koeffizienten, Konfidenzintervalle und Residuenstreuung.
6. Rekonstruiere die Monats-Fixed-Effects relativ zum Referenzmonat und zentriere sie.
7. Vergleiche in der Simulation geschätzte und wahre gemeinsame Monatsschocks.

## Ausführen

```bash
python3 exercises/T43-did-time-fixed-effects/starter.py
python3 exercises/T43-did-time-fixed-effects/solution.py
```

Die Lösung erzeugt `data/regional_voucher_panel.csv`, Ergebnistabellen und `results/time_fixed_effects_did.png`.

## Denkfragen

- Warum ist ein Post-Dummy weniger flexibel als Zeit-Fixed-Effects?
- Welche Variable wird bei gleichzeitigem Treatmentbeginn durch die Zeit-Fixed-Effects absorbiert?
- Helfen Zeit-Fixed-Effects gegen einen Schock, der ausschließlich die Treatmentgruppe trifft?
