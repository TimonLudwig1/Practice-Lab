# T34 – Firmen über acht Quartale

## Ausgangslage

Für 80 Firmen werden Investitionen, Beschäftigung und Produktivität über acht Quartale beobachtet. Einige Firmenquartale fehlen. Bevor ein Panelmodell geschätzt wird, müssen Entity- und Zeitindex, Duplikate, Panelbalance und die Herkunft der Variation verstanden werden.

## Lernziele

- Paneldaten von Querschnitt und Zeitreihe unterscheiden,
- Entity- und Zeitdimension eindeutig definieren,
- balancierte und unbalancierte Panels diagnostizieren,
- Within- und Between-Variation empirisch trennen,
- fehlende Perioden und doppelte Entity-Zeit-Schlüssel erkennen,
- individuelle Verläufe statt nur gepoolter Verteilungen lesen.

## Aufgaben

1. Prüfe, ob `(firm_id, quarter)` jede Zeile eindeutig identifiziert.
2. Bestimme Zahl der Firmen, Quartale, Beobachtungen und theoretisch möglichen Zellen.
3. Berechne fehlende Firmenquartale sowie minimale, mittlere und maximale Periodenzahl pro Firma.
4. Erstelle für jede Firma eine Beobachtungsmatrix über die acht Quartale.
5. Zerlege für `investment_million_eur`, `employees` und `productivity_index` die Streuung des Firmenmittels und die Streuung um das jeweilige Firmenmittel.
6. Zeichne die Produktivitätspfade ausgewählter Firmen. Welche Information würde ein reiner Querschnitt verlieren?
7. Erkläre, welche Arten von Änderungen ein Fixed-Effects-Modell später verwenden kann.

## Ausführen

```bash
python3 exercises/T34-panel-data/starter.py
python3 exercises/T34-panel-data/solution.py
```

Die Lösung erzeugt `data/unbalanced_firm_panel.csv`, drei Ergebnistabellen und `results/panel_structure.png`.

## Denkfragen

- Kann ein Panel viele Zeilen, aber wenig Within-Variation besitzen?
- Wann ist ein fehlendes Firmenquartal problematisch für die Interpretation?
- Welche Variable ist definitionsgemäß rein zeitinvariant und hätte daher keine Within-Variation?
