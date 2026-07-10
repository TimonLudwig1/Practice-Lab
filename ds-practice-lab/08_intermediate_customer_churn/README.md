# 08 — Customer Churn: Feature Engineering & Classification 📞

Difficulty: 🟡 Intermediate | Topic: Feature Engineering + Tabular Classification

## 🎯 Project Goal
Predict which telco customers will cancel their contract, with the emphasis on **feature engineering and a clean preprocessing pipeline** — the part of ML that actually moves the needle on tabular data.

## 📊 Dataset Description
**IBM Telco Customer Churn** — 7,043 customers, 20 columns: demographics, contract type, services booked, monthly/total charges. Target: `Churn` (Yes/No), ~26% positive. The notebook contains a loading snippet pulling the CSV directly from IBM's GitHub; save a copy to `data/raw/`.

Known quirk: `TotalCharges` is read as a string and contains blanks — your first cleaning task.

## 💡 Suggested Approach (high-level)
1. EDA focused on the target: churn rate per contract type, tenure, payment method. Two or three sharp matplotlib figures beat twenty lazy ones.
2. Clean the data (the `TotalCharges` issue, target encoding Yes/No → 1/0).
3. Build a `ColumnTransformer`: one-hot encode categoricals, scale numerics — wrapped with the model in a single `Pipeline` so nothing leaks.
4. Engineer features beyond the raw columns, e.g.: tenure buckets, number of booked add-on services, average charge per service, "is on month-to-month contract AND has high charges" interactions. Measure whether each idea actually helps (cross-validation, not a single split).
5. Compare logistic regression vs a tree ensemble (e.g., `RandomForestClassifier` or `HistGradientBoostingClassifier`).
6. The dataset is imbalanced — accuracy is misleading. Evaluate with ROC-AUC and PR-AUC; look at `class_weight="balanced"`.
7. Translate the model into business terms: at which probability threshold would you trigger a retention offer, if an offer costs 10€ and a saved customer is worth 200€?

## 🏁 Success Criteria
- Leak-free pipeline (all preprocessing inside the Pipeline/ColumnTransformer)
- ≥3 engineered features with measured impact (CV score with vs without)
- ROC-AUC ≥ 0.84 on a held-out test set
- A threshold recommendation derived from the cost/benefit numbers above, not from the 0.5 default

## 🔗 Useful References
- [sklearn ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [sklearn: cross_val_score](https://scikit-learn.org/stable/modules/cross_validation.html)
- Look up: *PR curve vs ROC curve for imbalanced data*, *cost-sensitive thresholding*
