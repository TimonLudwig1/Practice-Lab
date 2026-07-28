# 25 — Survival Analysis: Time-to-Event, Not Just Yes/No

Difficulty: 🟠 Medium-Hard | Topic: Survival Analysis

## 🎯 Project Goal
Model **how long** until employees leave, not merely whether they leave. Handle censoring correctly (people still employed haven't "not-churned" — their event simply hasn't happened yet), and turn the model into something an HR team could act on.

## 📊 Dataset + Evaluation Metric
- **Dataset:** IBM HR Analytics Employee Attrition (~1470 rows, on Kaggle as `WA_Fn-UseC_-HR-Employee-Attrition.csv`). Duration = `YearsAtCompany`; event = `Attrition == "Yes"`; everyone with `Attrition == "No"` is **right-censored**. Covariates: age, role, income, overtime, satisfaction scores, etc.
- **Evaluation metric:** **concordance index (C-index)** on a held-out set for any fitted survival model; the Kaplan–Meier and curve work is graded on correctness and interpretation.

## 🏁 Success Criteria
- Censoring identified and encoded correctly (duration + event indicator), with one sentence on why treating censored rows as "did not churn" would bias the analysis
- Kaplan–Meier survival curves overall and split by at least two groups (e.g. overtime, department), with a log-rank test for whether the curves differ
- A Cox proportional-hazards model fit with interpreted hazard ratios (which factors raise/lower the churn hazard, and by how much)
- The proportional-hazards assumption actually checked, not assumed — and a response if it's violated
- Held-out C-index reported, plus one concrete, individual-level output (e.g. predicted survival curve / median time-to-attrition for a given employee profile)

Relevant techniques (look them up yourself): right-censoring, Kaplan–Meier estimator, log-rank test, Cox proportional-hazards regression, hazard ratios, proportional-hazards assumption checks (Schoenfeld residuals), concordance index, the `lifelines` library.

---

# Deutsche Übersetzung

# 25 — Überlebenszeitanalyse: Zeit bis zum Ereignis statt nur Ja oder Nein

Schwierigkeit: 🟠 Mittel bis anspruchsvoll | Thema: Überlebenszeitanalyse

## 🎯 Projektziel
Modelliere, **wie lange** es dauert, bis Beschäftigte das Unternehmen verlassen, statt nur die Kündigung vorherzusagen. Behandle Zensierung korrekt: Noch beschäftigte Personen haben nicht dauerhaft „nicht gekündigt“; das Ereignis wurde lediglich noch nicht beobachtet. Übersetze das Modell anschließend in eine für ein HR-Team nutzbare Form.

## 📊 Datensatz und Bewertungsmetrik
- **Datensatz:** IBM HR Analytics Employee Attrition mit etwa 1.470 Zeilen, auf Kaggle als `WA_Fn-UseC_-HR-Employee-Attrition.csv`. Dauer ist `YearsAtCompany`, das Ereignis `Attrition == "Yes"`; alle Fälle mit `Attrition == "No"` sind **rechtszensiert**. Kovariaten sind unter anderem Alter, Rolle, Einkommen, Überstunden und Zufriedenheitswerte.
- **Bewertungsmetrik:** **Konkordanzindex (C-Index)** auf einem zurückgehaltenen Satz für jedes angepasste Überlebenszeitmodell. Kaplan-Meier- und Kurvenanalysen werden nach Korrektheit und Interpretation bewertet.

## 🏁 Erfolgskriterien
- Korrekte Erkennung und Kodierung der Zensierung durch Dauer und Ereignisindikator sowie Erklärung, warum die Behandlung zensierter Zeilen als „nicht gekündigt“ die Analyse verzerren würde
- Kaplan-Meier-Überlebenskurven insgesamt und für mindestens zwei Gruppen, etwa Überstunden oder Abteilung, mit Log-Rank-Test auf Unterschiede
- Cox-Proportional-Hazards-Modell mit interpretierten Hazard Ratios: welche Faktoren erhöhen oder senken die Kündigungsrate und um wie viel?
- Tatsächliche Prüfung der Proportional-Hazards-Annahme samt Reaktion bei Verletzung
- C-Index auf zurückgehaltenen Daten sowie eine konkrete Ausgabe für eine einzelne Person, etwa Überlebenskurve oder mediane Zeit bis zur Kündigung

Relevante Verfahren zum selbstständigen Nachschlagen: Rechtszensierung, Kaplan-Meier-Schätzer, Log-Rank-Test, Cox-Proportional-Hazards-Regression, Hazard Ratios, Prüfung der Proportional-Hazards-Annahme mit Schoenfeld-Residuen, Konkordanzindex und die Bibliothek `lifelines`.
