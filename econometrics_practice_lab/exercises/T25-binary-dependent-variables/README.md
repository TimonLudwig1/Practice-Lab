# T25 – Kreditausfälle als binäre Outcomes

## Ausgangslage

Eine Bank beobachtet, ob ein Kredit innerhalb eines Jahres ausfällt (`defaulted = 1`) oder vollständig bedient wird (`defaulted = 0`). Für einzelne Personen gibt es nur diese beiden Ergebnisse. Über vergleichbare Personen hinweg ist der Mittelwert des Outcomes jedoch eine geschätzte Ausfallwahrscheinlichkeit.

## Lernziele

Nach dem Projekt kannst du:

- ein binäres Outcome als Bernoulli-Variable beschreiben,
- erklären, warum der Mittelwert einer 0/1-Variable einem Anteil entspricht,
- individuelle Realisationen von bedingten Wahrscheinlichkeiten unterscheiden,
- die Varianzformel \(p(1-p)\) empirisch überprüfen,
- Unsicherheit von Gruppenanteilen mit Konfidenzintervallen darstellen,
- erkennen, warum hohe Klassifikationsgenauigkeit bei seltenen Ereignissen irreführend sein kann.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Prüfe, ob `defaulted` ausschließlich die Werte null und eins enthält.
2. Berechne Mittelwert, Ereignisanteil und Varianz mit Divisor \(n\). Zeige, dass Mittelwert und Anteil sowie empirische Varianz und \(\bar y(1-\bar y)\) übereinstimmen.
3. Berechne die Ausfallquote getrennt nach der Zahl früherer Zahlungsverzüge und konstruiere Wilson-Konfidenzintervalle.
4. Teile die unbeobachtete, im synthetischen Datensatz bekannte Risikowahrscheinlichkeit in Dezile. Vergleiche die durchschnittliche Wahrscheinlichkeit mit der beobachteten Ausfallquote.
5. Berechne die Accuracy eines Modells, das für alle Kredite die Mehrheitsklasse vorhersagt. Warum ist diese Zahl keine ausreichende Qualitätsmetrik?
6. Vergleiche den Brier Score einer konstanten Basisprognose mit dem Brier Score der datengenerierenden Wahrscheinlichkeit.
7. Erstelle eine Grafik, die binäre Einzeloutcomes, Gruppenanteile, Bernoulli-Varianz und die Überlappung der Risikoverteilungen gemeinsam erklärt.

## Ausführen

```bash
python3 exercises/T25-binary-dependent-variables/starter.py
```

Die Musterlösung erzeugt den Datensatz sowie `results/event_summary.csv`, `results/risk_deciles.csv`, `results/late_payment_rates.csv` und `results/binary_outcomes.png`:

```bash
python3 exercises/T25-binary-dependent-variables/solution.py
```

## Denkfragen

- Warum kann eine Person mit geschätzten 80 Prozent Risiko trotzdem nicht ausfallen?
- Bei welchem Wert von \(p\) ist die Bernoulli-Varianz maximal?
- Warum werden Konfidenzintervalle bei kleinen Untergruppen breiter?
- Welche zusätzliche Information geht verloren, wenn ein kontinuierliches Risiko in ein 0/1-Outcome umgewandelt wird?
