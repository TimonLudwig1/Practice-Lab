# 04 — Your First Classification: Heart Disease ❤️

Difficulty: 🟢 Beginner | Topic: Tabular Classification (scikit-learn)

## 🎯 Project Goal
Predict whether a patient has heart disease from 13 clinical features, and learn why **accuracy alone is a trap** in classification.

## 📚 What You'll Learn
- Classification vs regression — same workflow, different target and metrics
- `LogisticRegression` and `KNeighborsClassifier`
- Why kNN *requires* feature scaling (`StandardScaler`) and Pipelines to do it safely
- The confusion matrix: TP, FP, TN, FN — and what each costs in a medical context
- Precision, recall, F1 — and choosing the right one for the problem
- Predicted probabilities (`predict_proba`) vs hard predictions

## 🗂️ Dataset Description
**UCI Heart Disease (Cleveland)** — 303 patients, 13 features (age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, …). Target: presence of heart disease (0 = no, 1–4 = yes; you'll binarize it).

Download (the notebook's first cell does this for you):

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"]
df = pd.read_csv(url, names=cols, na_values="?")
```

If the UCI server is down, search for "heart disease cleveland processed" — many mirrors exist; save the file to `data/raw/`.

## 🚀 Getting Started
```bash
cd 04_beginner_heart_disease_classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/solution_template.ipynb
```

## 📋 Step-by-Step Guide
1. **Load & clean** — load with the snippet above, check `na` counts, drop the few rows with missing `ca`/`thal`, and binarize the target (`(df["target"] > 0).astype(int)`). *Why binarize:* the original 0–4 scale mixes "how severe" into "yes/no"; we want a clean binary question first.
2. **Explore class balance** — `value_counts(normalize=True)`. *Why:* if 54% of patients are healthy, then a model that always says "healthy" is already 54% accurate. That number is your accuracy baseline.
3. **Split** — train/test split, `stratify=y`. *Why stratify:* keeps the disease ratio identical in both sets, which matters a lot with only 300 rows.
4. **Logistic regression** — fit, predict, and compute accuracy. Then immediately compute the **confusion matrix**. *Why:* accuracy compresses 4 numbers (TP/FP/TN/FN) into 1 and hides the difference between "missed a sick patient" and "false alarm".
5. **Precision, recall, F1** — use `classification_report`. Then answer in writing: *in a medical screening context, which error type is worse — false positive or false negative? Which metric should we optimize?*
6. **kNN with a Pipeline** — `make_pipeline(StandardScaler(), KNeighborsClassifier())`. Try k = 3, 5, 11. *Why scaling:* kNN computes distances; without scaling, cholesterol (~250) drowns out sex (0/1). *Why a Pipeline:* it guarantees the scaler is fit only on training data — fitting it on everything would leak test information.
7. **Probabilities** — `predict_proba` gives you P(disease). Print the 5 patients the model is *least sure* about (closest to 0.5). *Why:* in the real world these are the patients you'd send for additional tests.
8. **Compare & conclude** — which model wins on recall? Write a 3-sentence recommendation.

## ✅ Completion Checklist
- [ ] I binarized the target and can explain why
- [ ] I know the accuracy a do-nothing model would get
- [ ] I can label all four cells of a confusion matrix
- [ ] I can explain precision vs recall with this dataset's context
- [ ] I used a Pipeline and can explain what data leakage it prevents
- [ ] I found the patients with predictions closest to 0.5
- [ ] I picked a final model based on a *justified* metric, not just accuracy

## 💡 Hints & Tips
- `ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)` plots the matrix in one line.
- `classification_report(y_test, y_pred)` prints precision/recall/F1 per class.
- For the "least sure" patients: `proba = model.predict_proba(X_test)[:, 1]`, then sort by `np.abs(proba - 0.5)`.
- LogisticRegression may warn about convergence — `max_iter=1000` fixes it.
- Increase k in kNN and watch the decision get smoother: small k = flexible (overfits), large k = rigid (underfits). Same trade-off as the tree depth in project 03.

## 🔗 Further Reading
- [scikit-learn: Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Google ML Crash Course: Classification metrics](https://developers.google.com/machine-learning/crash-course/classification)
- [UCI Heart Disease dataset page](https://archive.ics.uci.edu/dataset/45/heart+disease)
