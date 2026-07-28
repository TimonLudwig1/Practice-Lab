# 05 — Sample Bias Laboratory 🎯 `[from your lectures]`

Difficulty: 🟡 Intermediate | Topic: Statistics — Sampling & Selection Bias

## 🎯 Project Goal
Build a simulation that demonstrates, quantifies, and partially corrects sample bias: you create a synthetic "ground truth" population, draw samples with different (flawed) sampling schemes, and measure how badly each one distorts your estimates.

## 📊 Dataset Description
You generate it yourself — that's the point: only with a synthetic population do you *know* the true values your samples should recover. Create a population of ~200,000 people with correlated attributes, e.g. age, income, smartphone ownership, daily screen time, and political/consumer preference. Build in realistic dependencies (income depends on age; smartphone ownership lower for older people; screen time depends on age and smartphone ownership).

The notebook contains a starting snippet for the population generator.

## 💡 Suggested Approach (high-level)
1. Generate the population and record the true population means/proportions — your gold standard.
2. Implement a **simple random sample** as the unbiased reference. Show sampling error shrinking as n grows (this is your sanity check).
3. Implement at least three biased sampling schemes and compare each against the truth:
   - **Convenience sample** (e.g., online survey → only smartphone owners can respond)
   - **Voluntary response / self-selection** (response probability depends on the quantity being measured, e.g. people with extreme opinions respond more)
   - **Survivorship-style filter** (e.g., only sampling people still subscribed/alive in the system)
4. For each scheme: repeat the draw many times, plot the sampling distribution of the estimate vs the true value. Bias = systematic offset, not random scatter — your plots should show exactly that distinction.
5. Attempt a correction: **post-stratification or inverse-probability weighting** on the convenience sample using known population age structure. How much of the bias can you remove? What stays unfixable, and why?
6. Bonus: recreate a mini "Literary Digest 1936" — show that a biased sample of 50,000 loses against an unbiased sample of 500.

## 🏁 Success Criteria
- A figure per sampling scheme showing the sampling distribution against the true value
- A summary table: scheme → bias, standard error, RMSE of the estimate
- Weighting correction implemented and its effect quantified
- A short written conclusion: which kinds of bias can sample size fix? (Hint: none of them — be able to argue this.)

## 🔗 Useful References
- `numpy.random.default_rng` — modern NumPy random API
- Look up: *post-stratification*, *inverse probability weighting*, *Horvitz-Thompson estimator*
- The Literary Digest 1936 poll disaster (any article) — your bonus task in real life
- `pandas.cut` for building age strata

---

# Deutsche Übersetzung

# 05 — Labor zur Stichprobenverzerrung 🎯 `[aus deinen Vorlesungen]`

Schwierigkeit: 🟡 Mittel | Thema: Statistik — Stichproben- und Auswahlverzerrung

## 🎯 Projektziel
Erstelle eine Simulation, die Stichprobenverzerrung sichtbar macht, quantifiziert und teilweise korrigiert. Dazu erzeugst du eine synthetische Grundgesamtheit, ziehst Stichproben mit verschiedenen fehlerhaften Auswahlverfahren und misst, wie stark diese deine Schätzungen verzerren.

## 📊 Beschreibung des Datensatzes
Du erzeugst den Datensatz selbst. Nur bei einer synthetischen Grundgesamtheit kennst du die wahren Werte, die deine Stichproben wiedergeben sollten. Erzeuge etwa 200.000 Personen mit zusammenhängenden Merkmalen wie Alter, Einkommen, Smartphone-Besitz, täglicher Bildschirmzeit sowie politischen oder konsumbezogenen Präferenzen. Baue realistische Abhängigkeiten ein, etwa Einkommen in Abhängigkeit vom Alter und Bildschirmzeit in Abhängigkeit von Alter und Smartphone-Besitz.

Das Notebook enthält einen Ausgangspunkt für den Generator der Grundgesamtheit.

## 💡 Empfohlenes Vorgehen
1. Erzeuge die Grundgesamtheit und notiere ihre wahren Mittelwerte und Anteile als Referenzwerte.
2. Implementiere eine **einfache Zufallsstichprobe** als unverzerrte Referenz. Zeige, wie der Stichprobenfehler mit wachsendem n abnimmt.
3. Implementiere mindestens drei verzerrte Auswahlverfahren und vergleiche sie mit der Wahrheit:
   - **Gelegenheitsstichprobe**, beispielsweise eine Online-Umfrage, an der nur Smartphone-Besitzer teilnehmen
   - **Freiwillige Teilnahme beziehungsweise Selbstselektion**, bei der die Antwortwahrscheinlichkeit von der gemessenen Größe abhängt
   - **Survivorship-Filter**, bei dem nur Personen betrachtet werden, die noch abonniert oder im System aktiv sind
4. Wiederhole für jedes Verfahren die Stichprobenziehung häufig und zeichne die Verteilung der Schätzwerte sowie den wahren Wert. Verzerrung ist eine systematische Verschiebung und nicht bloß zufällige Streuung.
5. Versuche die Gelegenheitsstichprobe durch **Poststratifizierung oder inverse Wahrscheinlichkeitsgewichtung** anhand der bekannten Altersstruktur zu korrigieren. Untersuche, welcher Teil der Verzerrung bestehen bleibt und warum.
6. Bonus: Stelle im Kleinen die „Literary Digest“-Umfrage von 1936 nach und zeige, dass eine verzerrte Stichprobe von 50.000 Personen gegen eine unverzerrte Stichprobe von 500 verlieren kann.

## 🏁 Erfolgskriterien
- Eine Abbildung je Auswahlverfahren, welche die Stichprobenverteilung dem wahren Wert gegenüberstellt
- Eine Übersichtstabelle mit Verfahren, Verzerrung, Standardfehler und RMSE der Schätzung
- Eine implementierte Gewichtungskorrektur mit quantifiziertem Effekt
- Ein kurzes Fazit dazu, welche Arten von Verzerrung durch größere Stichproben behoben werden können – keine von ihnen, was du begründen können solltest

## 🔗 Nützliche Quellen
- `numpy.random.default_rng` — moderne NumPy-API für Zufallszahlen
- Suchbegriffe: *post-stratification*, *inverse probability weighting*, *Horvitz-Thompson estimator*
- Die gescheiterte „Literary Digest“-Umfrage von 1936 als reales Beispiel
- `pandas.cut` zum Erzeugen von Altersgruppen
