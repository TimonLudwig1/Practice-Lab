# T28 – Marginale Effekte bei Jobangeboten

## Ausgangslage

Ein Recruitingprogramm modelliert, ob Bewerbende ein Jobangebot erhalten. Die Logit-Wahrscheinlichkeit hängt von Vorbereitungsstunden, Berufserfahrung und der Teilnahme an einem Mentoringprogramm ab. Weil Logit nichtlinear ist, hängt die Prozentpunktwirkung einer erklärenden Variable vom Ausgangsniveau aller Kovariaten ab.

## Lernziele

Nach dem Projekt kannst du:

- marginale Effekte eines kontinuierlichen Regressors als \(\beta_j p_i(1-p_i)\) berechnen,
- bei einer Dummy-Variable eine diskrete Wahrscheinlichkeitsänderung statt einer Ableitung verwenden,
- Average Marginal Effects (AME) und Marginal Effects at the Mean (MEM) unterscheiden,
- erklären, warum ein Logit-Koeffizient kein konstanter Prozentpunkteffekt ist,
- individuelle Heterogenität marginaler Effekte darstellen,
- marginale Effekte mit Konfidenzintervallen kommunizieren.

## Aufgaben

Bearbeite die `TODO`-Stellen in `starter.py`.

1. Schätze ein Logit-Modell für `received_offer` mit `preparation_hours`, `experience_years` und `mentor_program`.
2. Berechne für jede Person den marginalen Effekt einer zusätzlichen Vorbereitungsstunde.
3. Berechne für jede Person den diskreten Mentoringeffekt als \(\hat p(D=1)-\hat p(D=0)\).
4. Bilde aus beiden individuellen Größen jeweils den AME.
5. Berechne den MEM an den Mittelwerten der kontinuierlichen Variablen. Begründe, warum die „durchschnittliche Person“ nicht zwingend eine reale Beobachtung ist.
6. Vergleiche deine Handrechnung mit `statsmodels.get_margeff` und gib Standardfehler sowie 95%-Konfidenzintervalle aus.
7. Zeichne, wie Wahrscheinlichkeit und marginaler Effekt über die Vorbereitungsstunden variieren.
8. Erkläre, weshalb marginale Effekte nahe \(p=0{,}5\) typischerweise größer und nahe null oder eins kleiner sind.

## Ausführen

```bash
python3 exercises/T28-marginal-effects/starter.py
```

Die Musterlösung erzeugt `data/job_offers.csv` sowie `results/coefficients.csv`, `results/marginal_effect_summary.csv`, `results/individual_marginal_effects.csv` und `results/marginal_effects.png`:

```bash
python3 exercises/T28-marginal-effects/solution.py
```

## Denkfragen

- Warum unterscheiden sich AME und MEM selbst ohne Interaktionsterme?
- Ist der AME einer Dummy-Variable eine Ableitung?
- Welche Einheit hat der marginale Effekt von `preparation_hours`?
- Wie verändert sich der marginale Effekt, wenn derselbe Logit-Koeffizient auf Personen mit sehr unterschiedlichen Ausgangsrisiken angewendet wird?
