# T38 – First Differences bei Haushaltseinkommen

## Ausgangslage

Haushalte werden sechs Jahre lang beobachtet. Dauerhafte finanzielle Stabilität beeinflusst sowohl das Einkommensniveau als auch den Konsum. First Differences ersetzen Niveaus durch Veränderungen zwischen aufeinanderfolgenden Jahren und eliminieren dadurch den zeitinvarianten Haushaltseffekt.

## Lernziele

- Panelvariablen korrekt innerhalb einer Einheit differenzieren,
- zeigen, warum \(\Delta\alpha_i=0\),
- ein First-Difference-Modell ohne Niveaueffekte schätzen,
- Pooled OLS, Within und First Differences vergleichen,
- die Äquivalenz von Within und FD bei genau zwei Perioden nachvollziehen,
- den Verlust einer Beobachtung je Einheit und die veränderte Fehlerstruktur verstehen.

## Aufgaben

1. Sortiere nach Haushalt und Jahr und berechne Einkommens- und Konsumdifferenzen.
2. Prüfe, dass die erste Beobachtung jedes Haushalts nach dem Differenzieren fehlt.
3. Schätze Pooled OLS, Within und First Differences mit nach Haushalt geclusterten Standardfehlern.
4. Vergleiche die Schätzwerte mit dem wahren marginalen Konsumeffekt von 0,6.
5. Untersuche die Korrelation der zeitinvarianten Stabilität mit Einkommensniveau und Einkommensänderung.
6. Beschränke das Panel auf die ersten zwei Jahre und verifiziere die exakte Gleichheit von Within- und FD-Koeffizient.
7. Erkläre, warum FE und FD bei mehr als zwei Perioden meist nicht numerisch identisch sind.

## Ausführen

```bash
python3 exercises/T38-first-differences/starter.py
python3 exercises/T38-first-differences/solution.py
```

Die Lösung erzeugt `data/household_income_panel.csv`, Ergebnistabellen und `results/first_differences.png`.

## Denkfragen

- Weshalb verstärkt Differenzieren möglicherweise Messfehler?
- Welche serielle Fehlerstruktur entsteht aus unabhängigen Level-Fehlern?
- Wann kann First Differences gegenüber Within günstiger sein?
