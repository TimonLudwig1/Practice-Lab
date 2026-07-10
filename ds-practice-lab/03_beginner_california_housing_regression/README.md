# 03 — Your First Regression: California Housing 🏠

Difficulty: 🟢 Beginner | Topic: Tabular Regression (scikit-learn)

## 🎯 Project Goal
Build, evaluate, and interpret your first end-to-end regression model: predicting median house values in California districts.

## 📚 What You'll Learn
- The canonical ML workflow: **split → fit → predict → evaluate** (you'll use this in every future project)
- Why we need a train/test split (overfitting and honest evaluation)
- `LinearRegression` and `DecisionTreeRegressor` from scikit-learn
- Regression metrics: MAE, RMSE, R² — and what each one actually tells you
- Reading model coefficients (and why correlation ≠ causation — a preview of project 06!)
- A baseline model, and why every project needs one

## 🗂️ Dataset Description
**California Housing** (1990 census) — 20,640 districts with 8 numeric features (median income, house age, average rooms, population, lat/long, …). Target: median house value in $100k units. Built into scikit-learn:

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
```

## 🚀 Getting Started
```bash
cd 03_beginner_california_housing_regression
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/solution_template.ipynb
```

## 📋 Step-by-Step Guide
1. **Load & explore** — `.describe()`, histograms of the target and `MedInc`. *Why:* you'll spot that the target is capped at 5.0 ($500k) — real datasets have quirks like this, and knowing them changes how you read your errors.
2. **Split features/target, then train/test** — `X`, `y`, then `train_test_split(X, y, test_size=0.2, random_state=42)`. *Why:* the test set simulates "data the model has never seen". Touch it only once, at the very end. `random_state` makes your results reproducible.
3. **Build a baseline** — a "model" that always predicts the training mean (`DummyRegressor`). *Why:* metrics are meaningless in a vacuum; "RMSE = 0.74" only matters relative to "the dumbest possible model gets 1.15".
4. **Fit LinearRegression** — `model.fit(X_train, y_train)`, then predict on the test set. Evaluate with MAE, RMSE, R² (`sklearn.metrics`). *Why each metric:* MAE = average error in target units, RMSE punishes big misses harder, R² = fraction of variance explained.
5. **Look at the coefficients** — pair `model.coef_` with feature names. Which feature has the biggest effect? Careful: features are on different scales, so raw coefficients are NOT directly comparable. *Why:* this plants the seed for feature scaling and, later, omitted variable bias.
6. **Fit a DecisionTreeRegressor** — first with default settings, then with `max_depth=8`. Compare train vs test scores for both. *Why:* the default tree will get ~perfect train scores and worse test scores — that's overfitting, seen with your own eyes.
7. **Plot predictions vs actuals** — scatter of `y_test` vs predictions, plus the diagonal "perfect prediction" line. *Why:* one plot reveals patterns the metrics hide (e.g., what happens at the $500k cap?).
8. **Write your verdict** — which model would you ship, and what is its expected error in dollars?

## ✅ Completion Checklist
- [ ] I can explain why we hold out a test set
- [ ] I built a dummy baseline and beat it
- [ ] I computed MAE, RMSE, and R² and can explain each in one sentence
- [ ] I saw overfitting happen (tree: train score ≫ test score)
- [ ] I plotted predictions vs actuals and spotted the cap artifact
- [ ] I can say why a big raw coefficient doesn't automatically mean an important feature

## 💡 Hints & Tips
- RMSE: `mean_squared_error(y_true, y_pred)` then `np.sqrt(...)`, or `root_mean_squared_error` in newer sklearn versions.
- Coefficients as a readable table: `pd.Series(model.coef_, index=X.columns).sort_values()`.
- Diagonal line for the pred-vs-actual plot: `ax.plot([0, 5], [0, 5], "r--")`.
- The target unit is $100,000 — an MAE of 0.53 means you're off by ~$53k on average.
- Don't tune the tree endlessly; the point here is *seeing* the train/test gap, not winning Kaggle.

## 🔗 Further Reading
- [scikit-learn: Getting Started](https://scikit-learn.org/stable/getting_started.html)
- [Underfitting vs Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
- [StatQuest: Linear Regression (video)](https://www.youtube.com/watch?v=nk2CQITm_eo)
