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

---

# Deutsche Übersetzung

# 21 — Feature Engineering und scikit-learn-Pipelines 🏗️

Schwierigkeit: 🟡 Mittel | Thema: Feature Engineering

## 🎯 Projektziel
Prognostiziere Immobilienverkaufspreise im Ames-Housing-Datensatz, behandle aber **Feature Engineering und Pipeline** als eigentliches Ergebnis. Baue eine Pipeline ohne Datenleck, die rohe, unordentliche numerische und kategoriale Spalten mit fehlenden Werten bis zur Vorhersage verarbeitet, und belege den Nutzen deiner Entscheidungen.

## 📊 Beschreibung des Datensatzes
**Ames Housing** enthält etwa 1.460 Zeilen und 79 erklärende Merkmale zu Wohnhäusern in Ames, Iowa; die Zielvariable ist `SalePrice`. Der Datensatz ist komplexer als Boston oder California Housing: ordinale Qualitätsstufen, zahlreiche kategoriale Merkmale und inhaltlich bedeutsame fehlende Werte. So bedeutet `PoolQC = NaN` „kein Pool“ und nicht „unbekannt“.

Lade die Daten ohne manuellen Download über `sklearn.datasets.fetch_openml("house_prices", as_frame=True)`. Das Notebook enthält ein Beispiel.

## 💡 Empfohlenes Vorgehen
1. Untersuche zuerst alle Spalten. Trenne numerische und kategoriale Merkmale und innerhalb der kategorialen Merkmale echte nominale von **ordinalen** Qualitätsstufen wie `Ex > Gd > TA > Fa > Po`. Dokumentiere strukturell bedeutsame fehlende Werte.
2. Erstelle einen `ColumnTransformer`: Imputation und Skalierung numerischer Spalten, Imputation und One-Hot-Kodierung nominaler Spalten sowie **ordinale Kodierung** der Qualitätsmerkmale in richtiger Reihenfolge.
3. Verbinde alles mit einem Modell in einer `Pipeline`. Imputation, Kodierung und Skalierung müssen innerhalb der Pipeline liegen, damit sie in jedem CV-Fold neu angepasst werden und kein Datenleck entsteht.
4. Erzeuge abgeleitete Merkmale wie Gesamtwohnfläche, Alter beim Verkauf, Gesamtzahl der Bäder oder ein Garagen-Flag. Füge sie über einen eigenen Transformer oder `FunctionTransformer` innerhalb der Pipeline ein.
5. Da `SalePrice` rechtsschief ist, untersuche `TransformedTargetRegressor` für eine methodisch korrekte Log-Transformation der Zielvariable.
6. Vergleiche per `cross_val_score` die Median-Baseline, ein Modell nur mit numerischen Merkmalen und die vollständige Pipeline. Zeige, dass jede zusätzliche Komplexität einen messbaren Nutzen besitzt.

## 🏁 Erfolgskriterien
- Ein einzelnes `Pipeline`-/`ColumnTransformer`-Objekt, das den rohen DataFrame **ohne manuelle Vorverarbeitung außerhalb der Pipeline** in Vorhersagen umwandelt
- Ordinale Spalten in echter Reihenfolge kodiert und strukturell fehlende Werte bewusst behandelt; je Entscheidung ein Begründungssatz
- Mindestens drei innerhalb der Pipeline erzeugte Merkmale
- Kreuzvalidierter RMSE auf **log(SalePrice)** für Baseline, rein numerisches Modell und vollständige Pipeline, wobei das Engineering eine Verbesserung zeigt
- Kurzer Absatz dazu, welches erzeugte Merkmal am wichtigsten war und woran du dies erkennst

## 🔗 Nützliche Quellen
- [Anleitung zu ColumnTransformer und gemischten Datentypen](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)
- [Dokumentation zu TransformedTargetRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html)
- Suchbegriffe: *data leakage in cross-validation*, *ordinal vs one-hot encoding*, *FunctionTransformer*, *custom sklearn transformers (`BaseEstimator`, `TransformerMixin`)*
