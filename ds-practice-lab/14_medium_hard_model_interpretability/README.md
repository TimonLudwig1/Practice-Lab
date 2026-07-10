# 14 — Model Interpretability: Opening the Black Box

Difficulty: 🟠 Medium-Hard | Topic: Model Evaluation & Interpretability

## 🎯 Project Goal
Train a strong gradient-boosting model on the Adult census income dataset, then explain it at three levels — global (what drives the model), local (why this person was classified this way), and critical (where do the explanations and the data mislead us?).

## 📊 Dataset + Evaluation Metric
- **Dataset:** UCI Adult ("Census Income") — ~48k rows, predict income >50k from 14 demographic/employment features. Load via `sklearn.datasets.fetch_openml("adult", version=2)`.
- **Evaluation metric:** ROC-AUC ≥ 0.90 for the model itself; the project is then graded on the quality of the interpretation work, not the score.

## 🏁 Success Criteria
- A tuned gradient-boosting model with proper validation
- Global interpretation: permutation importance AND SHAP summary plot, compared against each other — and against the model's built-in impurity-based importances, with an explanation of why they disagree
- Dependence analysis: partial dependence / SHAP dependence plots for the 3 most important features, including at least one surprising shape explained
- Local interpretation: full SHAP breakdown for 3 individuals (one clear positive, one clear negative, one borderline)
- Critical section: identify at least one feature whose "importance" is an artifact (proxy/leakage/encoding effect) and one fairness concern in this dataset, each backed by evidence

Relevant techniques (look them up yourself): permutation importance, SHAP values, partial dependence plots, correlated-feature pitfalls in importance measures, proxy variables.
