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
