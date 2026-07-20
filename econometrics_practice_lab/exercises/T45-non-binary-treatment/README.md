# T45 – Nicht-binäre Förderung und Behandlungsintensität

## Ausgangslage

Kommunen erhalten ab 2022 unterschiedlich hohe Förderbeträge pro Einwohner für lokale Investitionen; einige erhalten nichts. Eine binäre Treatmentvariable würde 10 € und 100 € Förderung gleichsetzen. Ein lineares Intensitätsmodell nutzt mehr Information, setzt aber voraus, dass jede zusätzliche Einheit überall denselben Effekt besitzt. In den simulierten Daten gibt es abnehmende Grenzerträge.

## Lernziele

- binäre Behandlung von diskreter oder kontinuierlicher Intensität unterscheiden,
- ein DiD-Modell mit dosisabhängigem Treatment formulieren,
- einen linearen Dosis-Koeffizienten in sinnvollen Einheiten interpretieren,
- funktionale Form mit quadratischem und kategorialem Modell untersuchen,
- marginale Effekte bei einer nichtlinearen Dosis-Wirkungs-Beziehung berechnen,
- zusätzliche Identifikationsannahmen bei nicht-binärer Behandlung benennen.

## Aufgaben

1. Beschreibe die Verteilung der Förderintensität und definiere Kontroll-, niedrige, mittlere und hohe Dosis.
2. Erzeuge `post × dose` und schätze ein Two-Way-FE-Modell mit linearem Dosiseffekt.
3. Interpretiere den Koeffizienten als Effekt zusätzlicher 10 € je Einwohner.
4. Ergänze einen quadratischen Dosisterm und teste ihn auf Signifikanz.
5. Berechne vorhergesagte Gesamteffekte und marginale Effekte bei 20 €, 50 € und 80 €.
6. Schätze alternativ gruppenspezifische Post-Effekte für niedrige, mittlere und hohe Förderung.
7. Vergleiche Modellfit und Residuenmuster entlang der Dosis.
8. Diskutiere, warum Paralleltrends nun für verschiedene Intensitätsniveaus plausibel sein müssen.

## Ausführen

```bash
python3 exercises/T45-non-binary-treatment/starter.py
python3 exercises/T45-non-binary-treatment/solution.py
```

Die Lösung erzeugt `data/municipal_funding_panel.csv`, mehrere Ergebnistabellen und `results/non_binary_treatment.png`.

## Denkfragen

- Welche Information geht bei einer bloßen Dummy-Codierung `Förderung ja/nein` verloren?
- Bedeutet ein positiver linearer Koeffizient, dass Verdopplung der Dosis den Effekt verdoppelt?
- Wie könnte die Förderhöhe selbst endogen gewählt worden sein?
