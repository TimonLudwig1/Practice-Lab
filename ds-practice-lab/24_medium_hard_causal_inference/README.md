# 24 — Causal Inference: Did the Treatment Actually Work?

Difficulty: 🟠 Medium-Hard | Topic: Causal Inference

## 🎯 Project Goal
Estimate the causal effect of a job-training program on real earnings using **observational** data, where treated and control groups are not comparable to begin with. Recover a credible average treatment effect and state honestly how much to trust it.

## 📊 Dataset + Evaluation Metric
- **Dataset:** LaLonde / NSW job-training data. Use the **experimental** sample (randomized) and the **observational** sample (NSW treated vs PSID/CPS controls) — both are widely mirrored as `nsw.dta` / `lalonde.csv` (e.g. the `dowhy`, `causaldata`, or R `MatchIt` mirrors). Outcome: `re78` (1978 real earnings). Treatment: `treat`. Covariates: age, education, race, marital status, `re74`, `re75`.
- **Evaluation metric:** there is no single accuracy score. The benchmark is the **experimental ATE (~$1,800 increase in earnings)**. Your observational estimate is "good" to the extent it recovers a similar effect once confounding is addressed — and you are graded on the rigor of the argument, not the closeness alone.

## 🏁 Success Criteria
- The naive difference in means computed first, and an explanation of why it is badly biased here (show the covariate imbalance)
- A stated identification strategy with its assumptions made explicit (unconfoundedness / overlap) and overlap checked, not assumed
- At least two estimators that adjust for confounding, compared against each other and against the experimental benchmark
- A sensitivity / robustness check that asks how strong an unmeasured confounder would have to be to overturn the conclusion
- A final ATE with an uncertainty interval and a one-paragraph verdict on how much you'd trust it for a real funding decision

Relevant techniques (look them up yourself): confounding and back-door adjustment, propensity scores, matching, inverse-probability weighting (IPW), doubly-robust / AIPW estimators, common-support / overlap diagnostics, sensitivity analysis for unmeasured confounding.

---

# Deutsche Übersetzung

# 24 — Kausale Inferenz: Hat die Maßnahme tatsächlich gewirkt?

Schwierigkeit: 🟠 Mittel bis anspruchsvoll | Thema: Kausale Inferenz

## 🎯 Projektziel
Schätze anhand von **Beobachtungsdaten** den kausalen Effekt eines beruflichen Trainingsprogramms auf das tatsächliche Einkommen, obwohl Behandlungs- und Kontrollgruppe anfangs nicht vergleichbar sind. Ermittle einen glaubwürdigen durchschnittlichen Behandlungseffekt und bewerte ehrlich, wie vertrauenswürdig er ist.

## 📊 Datensatz und Bewertungsmaßstab
- **Datensatz:** LaLonde-/NSW-Daten zum beruflichen Trainingsprogramm. Verwende sowohl die randomisierte experimentelle Stichprobe als auch die Beobachtungsstichprobe aus NSW-Behandelten und PSID-/CPS-Kontrollen. Sie sind häufig als `nsw.dta` oder `lalonde.csv` gespiegelt, etwa bei `dowhy`, `causaldata` oder R `MatchIt`. Ergebnisvariable ist `re78`, Behandlung `treat`; Kovariaten sind Alter, Bildung, ethnische Zugehörigkeit, Familienstand, `re74` und `re75`.
- **Bewertungsmaßstab:** Es gibt keine einzelne Accuracy. Referenz ist der **experimentelle ATE von ungefähr 1.800 US-Dollar zusätzlichem Einkommen**. Die Beobachtungsschätzung ist insofern gut, als sie nach Berücksichtigung von Confounding einen ähnlichen Effekt ergibt. Bewertet wird vor allem die methodische Begründung.

## 🏁 Erfolgskriterien
- Zuerst berechnete naive Mittelwertdifferenz und Erklärung ihrer starken Verzerrung anhand der ungleichen Kovariaten
- Klar formulierte Identifikationsstrategie mit expliziten Annahmen zu Unconfoundedness und Overlap sowie tatsächlicher Prüfung der Überlappung
- Mindestens zwei Verfahren zur Confounder-Bereinigung, untereinander und mit der experimentellen Referenz verglichen
- Sensitivitäts- oder Robustheitsprüfung zur erforderlichen Stärke eines unbeobachteten Confounders, um die Schlussfolgerung umzukehren
- Abschließender ATE mit Unsicherheitsintervall und ein Absatz dazu, wie stark du ihm bei einer realen Finanzierungsentscheidung vertrauen würdest

Relevante Verfahren zum selbstständigen Nachschlagen: Confounding und Backdoor-Anpassung, Propensity Scores, Matching, inverse Wahrscheinlichkeitsgewichtung, doppelt robuste beziehungsweise AIPW-Schätzer, Common-Support- und Overlap-Diagnostik sowie Sensitivitätsanalyse für unbeobachtetes Confounding.
