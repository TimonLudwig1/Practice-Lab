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

---

# Deutsche Übersetzung

# 03 — Deine erste Regression: Immobilienpreise in Kalifornien 🏠

Schwierigkeit: 🟢 Einsteiger | Thema: Regression mit tabellarischen Daten (scikit-learn)

## 🎯 Projektziel
Erstelle, bewerte und interpretiere dein erstes vollständiges Regressionsmodell zur Vorhersage mittlerer Immobilienpreise in kalifornischen Bezirken.

## 📚 Das lernst du
- Den grundlegenden ML-Ablauf: **aufteilen → trainieren → vorhersagen → bewerten**
- Warum Trainings- und Testdaten getrennt werden müssen
- `LinearRegression` und `DecisionTreeRegressor` aus scikit-learn
- Die Regressionsmetriken MAE, RMSE und R² und ihre jeweilige Aussage
- Interpretation von Modellkoeffizienten sowie den Unterschied zwischen Korrelation und Kausalität
- Die Bedeutung eines Baseline-Modells

## 🗂️ Beschreibung des Datensatzes
**California Housing** aus der Volkszählung von 1990 enthält 20.640 Bezirke mit acht numerischen Merkmalen, darunter mittleres Einkommen, Alter der Häuser, durchschnittliche Zimmerzahl, Bevölkerung sowie Längen- und Breitengrad. Die Zielvariable ist der mittlere Immobilienwert in Einheiten von 100.000 US-Dollar. Der Datensatz ist in scikit-learn enthalten:

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
```

## 🚀 Erste Schritte
```bash
cd 03_beginner_california_housing_regression
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/solution_template.ipynb
```

## 📋 Schritt-für-Schritt-Anleitung
1. **Laden und untersuchen** — Verwende `.describe()` und erstelle Histogramme der Zielvariable und von `MedInc`. *Warum:* Du erkennst, dass die Zielvariable bei 5,0 beziehungsweise 500.000 US-Dollar gedeckelt ist. Solche Besonderheiten beeinflussen die Interpretation der Fehler.
2. **Merkmale und Zielvariable sowie Trainings- und Testdaten aufteilen** — Erzeuge `X` und `y` und verwende `train_test_split(X, y, test_size=0.2, random_state=42)`. *Warum:* Der Testsatz simuliert unbekannte Daten und sollte erst am Ende verwendet werden. `random_state` macht Ergebnisse reproduzierbar.
3. **Baseline erstellen** — Verwende mit `DummyRegressor` ein Modell, das stets den Mittelwert der Trainingsdaten vorhersagt. *Warum:* Eine Metrik ist nur im Vergleich aussagekräftig; ein RMSE von 0,74 ist erst dann gut, wenn ein einfaches Modell beispielsweise 1,15 erreicht.
4. **Lineare Regression trainieren** — Führe `model.fit(X_train, y_train)` aus, erzeuge Vorhersagen für den Testsatz und berechne MAE, RMSE und R². MAE ist der mittlere Fehler in der Einheit der Zielvariable, RMSE gewichtet große Fehler stärker und R² beschreibt den erklärten Varianzanteil.
5. **Koeffizienten betrachten** — Ordne `model.coef_` den Merkmalen zu. Da die Merkmale unterschiedliche Skalen besitzen, sind Rohkoeffizienten nicht direkt miteinander vergleichbar.
6. **DecisionTreeRegressor trainieren** — Verwende zunächst die Standardeinstellungen und anschließend `max_depth=8`. Vergleiche jeweils die Ergebnisse auf Trainings- und Testdaten. *Warum:* Der Standardbaum erreicht fast perfekte Trainingswerte, aber schwächere Testwerte – ein sichtbares Beispiel für Overfitting.
7. **Vorhersagen und tatsächliche Werte darstellen** — Erstelle ein Streudiagramm von `y_test` und den Vorhersagen sowie eine diagonale Linie für perfekte Vorhersagen. *Warum:* Die Grafik zeigt Muster, die einzelne Kennzahlen verbergen, etwa den Effekt der Preisobergrenze.
8. **Fazit formulieren** — Entscheide, welches Modell du einsetzen würdest, und beschreibe den erwarteten Fehler in US-Dollar.

## ✅ Checkliste zum Abschluss
- [ ] Ich kann erklären, warum ein Testsatz zurückgehalten wird.
- [ ] Ich habe eine Dummy-Baseline erstellt und übertroffen.
- [ ] Ich habe MAE, RMSE und R² berechnet und kann jede Metrik erklären.
- [ ] Ich habe Overfitting anhand des Abstands zwischen Trainings- und Testergebnis beobachtet.
- [ ] Ich habe Vorhersagen gegen tatsächliche Werte aufgetragen und den Effekt der Obergrenze erkannt.
- [ ] Ich kann erklären, warum ein großer Rohkoeffizient nicht automatisch ein wichtiges Merkmal bezeichnet.

## 💡 Hinweise und Tipps
- Berechne den RMSE mit `mean_squared_error(y_true, y_pred)` und anschließend `np.sqrt(...)` oder mit `root_mean_squared_error` in neueren scikit-learn-Versionen.
- Eine gut lesbare Koeffiziententabelle erhältst du mit `pd.Series(model.coef_, index=X.columns).sort_values()`.
- Zeichne die Diagonale im Ist-gegen-Prognose-Diagramm mit `ax.plot([0, 5], [0, 5], "r--")`.
- Die Zielvariable ist in Einheiten von 100.000 US-Dollar angegeben. Ein MAE von 0,53 entspricht daher einem mittleren Fehler von etwa 53.000 US-Dollar.
- Optimiere den Baum nicht endlos; Ziel dieses Labs ist es, den Unterschied zwischen Trainings- und Testleistung zu erkennen.

## 🔗 Weiterführende Informationen
- [scikit-learn: Getting Started](https://scikit-learn.org/stable/getting_started.html)
- [Underfitting vs Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
- [StatQuest: Linear Regression (Video)](https://www.youtube.com/watch?v=nk2CQITm_eo)
