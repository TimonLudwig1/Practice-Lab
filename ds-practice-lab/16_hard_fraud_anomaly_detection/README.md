# 16 — Fraud Detection Under Extreme Class Imbalance

Difficulty: 🔴 Hard | Topic: Anomaly Detection / Imbalanced Classification

## 🎯 Project Goal
Detect fraudulent credit card transactions in a dataset where only 0.17% of examples are positive, comparing supervised and unsupervised approaches under a realistic operational constraint: a human review team that can inspect at most 200 flagged transactions per day.

## 📊 Dataset Description
**Credit Card Fraud Detection (ULB)** — 284,807 transactions over 2 days, 492 frauds. Features V1–V28 are PCA-anonymized, plus `Time` and `Amount`.
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (free Kaggle account required).

## 📏 Evaluation Metric
- Primary: **AUPRC** (area under the precision-recall curve) — ROC-AUC is misleading at this imbalance and you should be able to explain why
- Operational: **precision and recall within the top-K alert budget** (K derived from the 200/day review capacity scaled to the dataset's time span)
- Split must respect time: train on the earlier portion, evaluate on the later portion

## 🏁 Success Criteria
- At least one supervised approach and one unsupervised/anomaly approach (trained without labels), evaluated on the identical time-based split
- AUPRC ≥ 0.80 for the best model
- A top-K alert analysis: with the review budget, how many frauds are caught, how much fraud value (€ via `Amount`) is recovered, and what is the analyst precision?
- An explicit comparison: when would the unsupervised approach be the right choice despite weaker metrics?
- A short note on why resampling techniques must never touch the evaluation set
