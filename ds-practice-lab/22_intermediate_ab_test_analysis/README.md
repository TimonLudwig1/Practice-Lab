# 22 — Experimentation: A/B Test Analysis 🧪

Difficulty: 🟡 Intermediate | Topic: Experimentation & Inference

## 🎯 Project Goal
Analyze a real mobile-game A/B test end to end and deliver a **ship / don't-ship recommendation** that you can defend. The skill here is turning a noisy experiment into an honest decision: effect size, uncertainty, and the difference between "statistically significant" and "actually matters".

## 📊 Dataset Description
**Cookie Cats A/B test** — ~90k players randomly assigned to `gate_30` (control) vs `gate_40` (treatment): the level at which a progression gate appears. Columns: `userid`, `version`, `sum_gamerounds`, `retention_1` (day-1 return), `retention_7` (day-7 return).

Download from Kaggle: *"Mobile Games A/B Testing - Cookie Cats"* → save `cookie_cats.csv` into `data/raw/`. (Small file; the notebook assumes that path.)

## 💡 Suggested Approach (high-level)
1. **Sanity-check the randomization first** before looking at any outcome: are the two groups balanced in size? Any duplicate users? This is the step people skip and regret.
2. Pick your primary metric deliberately (`retention_7` is the business-relevant one) and treat `retention_1` and `sum_gamerounds` as secondary. Decide this *before* peeking.
3. For the binary retention metrics: estimate each group's rate, the **difference**, and a confidence interval for the difference. Run a two-proportion test (look up `statsmodels.stats.proportion.proportions_ztest`). Report the effect size, not just the p-value.
4. `sum_gamerounds` is heavily skewed with extreme outliers — a plain t-test on the mean is fragile. Look at the distribution, consider a bootstrap CI or a rank-based test, and explain your choice.
5. Quantify uncertainty visually: a bootstrap distribution of the retention-7 difference makes the result far more honest than a single p-value.
6. Watch for the **multiple-comparisons** trap: you're testing several metrics. Note how that affects your confidence.
7. Write the verdict: ship or not, the estimated effect with its interval, and what you'd want before committing (sample size / power, longer horizon, guardrail metrics).

## 🏁 Success Criteria
- Randomization / balance check done and reported before any outcome analysis
- Primary metric chosen and justified up front
- Difference in retention-7 with a confidence interval **and** a significance test — effect size emphasized over the bare p-value
- Skewed `sum_gamerounds` handled with a method you justify (not a naive mean t-test)
- One bootstrap-based figure of the treatment effect's uncertainty
- A clear, defensible ship/don't-ship recommendation in plain language

## 🔗 Useful References
- [statsmodels proportions_ztest](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html)
- Look up: *two-proportion z-test*, *bootstrap confidence interval*, *statistical power*, *multiple comparisons / family-wise error*, *practical vs statistical significance*
- This pairs naturally with project 02 (`hypothesis_testing`) — same tools, real decision.

---

# Deutsche Übersetzung

# 22 — Experimente: Analyse eines A/B-Tests 🧪

Schwierigkeit: 🟡 Mittel | Thema: Experimente und Inferenz

## 🎯 Projektziel
Analysiere einen echten A/B-Test eines mobilen Spiels vollständig und gib eine begründete **Einführen-/Nicht-einführen-Empfehlung** ab. Im Mittelpunkt steht die Übersetzung eines verrauschten Experiments in eine ehrliche Entscheidung unter Berücksichtigung von Effektstärke, Unsicherheit und dem Unterschied zwischen statistischer Signifikanz und praktischer Relevanz.

## 📊 Beschreibung des Datensatzes
Der **Cookie-Cats-A/B-Test** enthält etwa 90.000 zufällig `gate_30` als Kontrolle oder `gate_40` als Behandlung zugewiesene Spieler. Die Variante bestimmt, bei welchem Level eine Fortschrittssperre erscheint. Spalten: `userid`, `version`, `sum_gamerounds`, `retention_1` und `retention_7`.

Lade „Mobile Games A/B Testing - Cookie Cats“ von Kaggle herunter und speichere `cookie_cats.csv` unter `data/raw/`. Das Notebook erwartet diesen Pfad.

## 💡 Empfohlenes Vorgehen
1. **Prüfe zuerst die Randomisierung**, bevor du Ergebnisse betrachtest: Sind die Gruppen ähnlich groß und gibt es doppelte Nutzer? Dieser oft übersprungene Schritt kann die gesamte Analyse ungültig machen.
2. Lege die primäre Metrik bewusst fest; `retention_7` ist geschäftlich besonders relevant. Behandle `retention_1` und `sum_gamerounds` als sekundär und entscheide dies vor der Auswertung.
3. Schätze für die binären Bindungsmetriken die Rate je Gruppe, ihre **Differenz** und ein Konfidenzintervall. Führe einen Test zweier Anteile mit `statsmodels.stats.proportion.proportions_ztest` durch und berichte die Effektstärke statt nur des p-Werts.
4. `sum_gamerounds` ist stark schief und enthält extreme Ausreißer. Ein einfacher t-Test des Mittelwerts ist empfindlich. Untersuche die Verteilung und begründe ein Bootstrap-Konfidenzintervall oder einen rangbasierten Test.
5. Visualisiere die Unsicherheit, etwa durch eine Bootstrap-Verteilung der Differenz in der 7-Tage-Bindung.
6. Beachte das Problem **multipler Vergleiche**, da mehrere Metriken getestet werden, und beschreibe den Einfluss auf deine Sicherheit.
7. Formuliere das Urteil mit geschätztem Effekt, Intervall und benötigten nächsten Schritten wie größerer Stichprobe, Power, längerem Horizont oder Guardrail-Metriken.

## 🏁 Erfolgskriterien
- Prüfung und Dokumentation von Randomisierung und Gruppenausgleich vor jeder Ergebnisanalyse
- Vorab gewählte und begründete primäre Metrik
- Differenz der 7-Tage-Bindung mit Konfidenzintervall und Signifikanztest, wobei die Effektstärke betont wird
- Begründeter Umgang mit der schiefen Variable `sum_gamerounds` statt eines naiven Mittelwert-t-Tests
- Eine Bootstrap-Abbildung zur Unsicherheit des Behandlungseffekts
- Klare, in einfacher Sprache begründete Empfehlung für oder gegen die Einführung

## 🔗 Nützliche Quellen
- [statsmodels proportions_ztest](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html)
- Suchbegriffe: *two-proportion z-test*, *bootstrap confidence interval*, *statistical power*, *multiple comparisons / family-wise error*, *practical vs statistical significance*
- Dieses Lab ergänzt Projekt 02 zu Hypothesentests: dieselben Werkzeuge, aber eine reale Entscheidung.
