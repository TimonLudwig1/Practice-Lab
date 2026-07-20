# T41 – Warum ein Post-only-Vergleich täuscht

## Ausgangslage

Schulen mit anfangs niedrigen Mathematikergebnissen erhalten ein Förderprogramm. Nach dem Programm schneiden sie noch immer schlechter ab als die Kontrollschulen. Ein reiner Vergleich nach der Behandlung würde deshalb einen negativen „Effekt“ nahelegen – obwohl sich ihre Ergebnisse stärker verbessert haben.

## Lernziele

- einen Post-Treatment-Gruppenvergleich von DiD unterscheiden,
- zeitinvariante Gruppenunterschiede als Quelle von Selection Bias erkennen,
- Baseline-, Post- und Veränderungsvergleiche gemeinsam interpretieren,
- das unbeobachtete Gegenfaktum der Treatmentgruppe rekonstruieren,
- erkennen, welche Information bei reinen Post-Daten unwiederbringlich fehlt.

## Aufgaben

1. Berechne den Gruppenunterschied vor und nach der Förderung.
2. Schätze mit den Post-Daten allein `score ~ treated`.
3. Berechne für jede Schule die Veränderung und vergleiche deren Mittelwerte.
4. Schätze das vollständige DiD-Modell mit Treatment-, Post- und Interaktionsterm.
5. Zerlege den Post-only-Vergleich in Baselineunterschied, gemeinsame Veränderung und Treatment-Effekt.
6. Zeichne beobachtete Gruppenmittel und das DiD-Gegenfaktum.
7. Erkläre, warum zusätzliche Post-Beobachtungen das Fehlen einer Baseline nicht automatisch reparieren.

## Ausführen

```bash
python3 exercises/T41-did-vs-post-only/starter.py
python3 exercises/T41-did-vs-post-only/solution.py
```

Die Lösung erzeugt `data/school_panel.csv`, Ergebnistabellen und `results/post_only_vs_did.png`.

## Denkfragen

- Kann der Post-only-Unterschied das falsche Vorzeichen besitzen?
- Welche Annahme müsste man treffen, um ihn trotzdem kausal zu interpretieren?
- Was leistet DiD, wenn sich unbeobachtete Gruppenunterschiede im Zeitverlauf verändern?
