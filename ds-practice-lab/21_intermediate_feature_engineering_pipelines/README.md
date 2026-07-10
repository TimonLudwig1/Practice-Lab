# 21 — Feature Engineering & scikit-learn Pipelines 🏗️

Difficulty: 🟡 Intermediate | Topic: Feature Engineering

## 🎯 Project Goal
Predict house sale prices on the Ames Housing dataset, but treat the **feature engineering and the pipeline** as the real deliverable — not the model. The goal is to build one leak-free `Pipeline` that takes raw, messy columns (numeric + categorical + missing values) all the way to a prediction, and to prove the engineering decisions paid off.

## 📊 Dataset Description
**Ames Housing** — ~1460 rows, 79 explanatory features describing residential homes in Ames, Iowa; target is `SalePrice`. Richer and messier than the Boston/California sets: ordinal quality ratings, many categoricals, and meaningful missing values (e.g. `PoolQC = NaN` means "no pool", not "unknown").

Load via `sklearn.datasets.fetch_openml("house_prices", as_frame=True)` (no manual download needed). The notebook has a loading snippet.

## 💡 Suggested Approach (high-level)
1. Audit the columns first: split into numeric vs categorical, and within categorical separate **true nominal** from **ordinal** (quality grades like `Ex > Gd > TA > Fa > Po`). Note which "missing" values are structurally meaningful.
2. Build a `ColumnTransformer`: impute + scale numerics, impute + one-hot nominals, and **ordinally encode** the quality columns in the right order (don't let one-hot throw away the ordering you know exists).
3. Wrap the whole thing in a `Pipeline` ending in a model. Everything — imputation, encoding, scaling — must live *inside* the pipeline so it gets refit on each CV fold. This is the anti-leakage discipline of the project.
4. Engineer a handful of *derived* features (e.g. total square footage, house age at sale, total bathrooms, a "has-garage" flag). Add them via a custom transformer or `FunctionTransformer` so they also live inside the pipeline.
5. Consider the target: `SalePrice` is right-skewed. Look up `TransformedTargetRegressor` and log-transform the target the principled way.
6. Compare honestly with `cross_val_score`: baseline (median predictor) → numeric-only model → full engineered pipeline. Show each step earns its complexity.

## 🏁 Success Criteria
- A single `Pipeline` / `ColumnTransformer` object that goes from the raw dataframe to a prediction with **no manual preprocessing outside it**
- Ordinal columns encoded with their real ordering, structural-missing handled deliberately (one sentence per decision)
- At least 3 engineered features, added inside the pipeline
- Cross-validated RMSE on **log(SalePrice)**, reported for baseline vs numeric-only vs full pipeline, showing the engineering helped
- One short paragraph: which engineered feature mattered most, and how you know

## 🔗 Useful References
- [ColumnTransformer + mixed types guide](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)
- [TransformedTargetRegressor docs](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html)
- Look up: *data leakage in cross-validation*, *ordinal vs one-hot encoding*, *FunctionTransformer*, *custom sklearn transformers (`BaseEstimator`, `TransformerMixin`)*
