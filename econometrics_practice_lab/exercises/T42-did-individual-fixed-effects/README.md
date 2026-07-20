# T42 – DiD mit individuellen Fixed Effects

## Ausgangslage

Ein Zuschuss für Wärmepumpen wird Haushalten mit dauerhaft hohem Stromverbrauch angeboten. Unbeobachtete Gebäudeeffizienz erzeugt deshalb große, zeitlich konstante Unterschiede zwischen den Haushalten. Individuelle Fixed Effects absorbieren diese unterschiedlichen Ausgangsniveaus; der Programmeffekt wird aus Veränderungen desselben Haushalts identifiziert.

## Lernziele

- ein DiD-Modell mit Haushalts-Fixed-Effects schätzen,
- die Dummy-Regression durch Within-Transformation reproduzieren,
- verstehen, warum zeitinvariante Treatment- und Haushaltsmerkmale absorbiert werden,
- einen falsch spezifizierten Pooled-Schätzer mit dem FE-Schätzer vergleichen,
- nach Haushalten geclusterte Standardfehler verwenden.

## Aufgaben

1. Prüfe, ob jeder Haushalt in allen acht Monaten beobachtet wird.
2. Erzeuge `did = treated × post`.
3. Schätze zunächst ein Pooled-Modell ohne Haushaltskontrolle.
4. Schätze anschließend `consumption ~ post + did + household fixed effects`.
5. Zentriere Outcome, `post` und `did` innerhalb jedes Haushalts und reproduziere den FE-Koeffizienten.
6. Zeige rechnerisch, dass sich `treated` und Gebäudeeffizienz durch Demeaning nicht mehr verändern.
7. Prüfe den Rang einer Designmatrix, die gleichzeitig alle Haushaltsdummies und ein zeitinvariantes Haushaltsmerkmal enthält.

## Ausführen

```bash
python3 exercises/T42-did-individual-fixed-effects/starter.py
python3 exercises/T42-did-individual-fixed-effects/solution.py
```

Die Lösung erzeugt `data/household_energy_panel.csv`, Ergebnistabellen und `results/individual_fe_did.png`.

## Denkfragen

- Welche Variation identifiziert den DiD-Koeffizienten im FE-Modell?
- Warum kann der Haupteffekt `treated` nicht zusätzlich zu Haushalts-Fixed-Effects geschätzt werden?
- Kontrollieren individuelle Fixed Effects auch zeitvariable unbeobachtete Confounder?
