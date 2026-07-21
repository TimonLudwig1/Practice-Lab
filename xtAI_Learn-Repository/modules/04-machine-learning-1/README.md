# Module 04 — Machine Learning 1

> **Language note.** This document is bilingual. The English version comes first; the German version (*deutsche Fassung*) follows below the horizontal rule.

**What is this about?** Now learning from data becomes systematic: how do you build, evaluate and improve predictive models — honestly, reproducibly and without deceiving yourself? You learn the most important model families of classical supervised learning (kNN, linear models with regularisation, decision trees, ensembles), but above all the **craft around them**: validation, metrics, pipelines, hyperparameter tuning. This is the module whose content is needed most often in practice.

**Prerequisites:** modules 02/03 (pandas, the train/test idea, regression, leakage). Module 01 (naive Bayes) helps.

**To do beforehand:** modules 01–03.

---

## Learning objectives

After this module you will be able to:

- formulate the core of supervised learning precisely (hypothesis, loss function, generalisation) and explain **overfitting/underfitting** via the **bias-variance tradeoff**,
- evaluate models correctly: train/validation/test, **k-fold cross-validation**, and justify why the test set is touched only once,
- choose and interpret the right **metrics** (confusion matrix, precision/recall/F1, ROC-AUC; MAE/RMSE/$R^2$) — especially with **imbalanced classes**,
- explain **kNN, logistic regression (with regularisation), decision trees, random forests and gradient boosting** (how do they learn? when do you use what?) and apply them with scikit-learn,
- build clean **pipelines** (scaling, one-hot encoding) that prevent leakage constructively,
- tune **hyperparameters** systematically (grid/random search with CV) and read **learning curves**,
- inspect models: feature importances, permutation importance — and their pitfalls.

---

## 1. Basics

### 1.1 What does "learning" mean formally?

Given: training data $(x_1, y_1), \dots, (x_n, y_n)$ from an unknown distribution. Wanted: a function $h$ (a hypothesis) from a model family $\mathcal{H}$ that minimises the **expected error on new data** — not the training error! Formally one minimises the training loss as a substitute:

$$\hat{h} = \arg\min_{h \in \mathcal{H}} \frac{1}{n} \sum_i L(h(x_i), y_i) \; (+ \text{regularisation})$$

with a **loss function** $L$ (e.g. squared error for regression, log loss for classification). The whole art of machine learning lies in the gap between the training loss and the true error — the **generalisation gap**.

**The three ingredients of every ML method:** (1) the model family $\mathcal{H}$ (which functions are possible at all?), (2) the loss function (what does "good" mean?), (3) the optimisation procedure (how do you find the best $h$?). When you meet a new method, always ask these three questions.

### 1.2 The first model: k-nearest neighbours (kNN)

**Idea:** to classify a new point, look at the $k$ nearest training points and take their majority class (or, for regression, their mean). No "training" in the proper sense — the model *is* the data set.

kNN is didactic gold, because you can see everything important on it:

- **The hyperparameter $k$ controls the complexity**: $k = 1$ gives an extremely flexible, jagged decision boundary (every training point gets its way — a risk of overfitting). $k = n$ always gives the most frequent class (maximally rigid — underfitting). The optimum lies in between.
- **Scaling is mandatory**: distances mix units! Without standardisation the feature with the largest numbers dominates (grams beat millimetres — the same lesson as with PCA in module 03).
- **Curse of dimensionality**: in high dimensions all points are roughly equally far apart — "closeness" loses its meaning. That is why kNN scales badly with many features.

### 1.3 Overfitting, underfitting and the bias-variance tradeoff

The expected prediction error of a model can be decomposed (for squared loss):

$$\text{error} = \underbrace{\text{bias}^2}_{\text{systematically off}} + \underbrace{\text{variance}}_{\text{unstable depending on the training data}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

- **High bias (underfitting):** the model is too simple for the truth (a straight line for a parabola). Symptom: training *and* test error high and close together.
- **High variance (overfitting):** the model is so flexible that it also learns the noise of the particular sample. Symptom: training error small, test error markedly larger.

> **Intuition:** imagine training the same model on 100 different samples. Bias = how far the *average* of the 100 models lies from the truth. Variance = how strongly the 100 models *scatter* among themselves. Simple models: all similar, all similarly wrong. Complex models: right on average, wildly different individually.

**Knobs for the complexity**: $k$ in kNN, tree depth, polynomial degree, regularisation strength, ... The right setting is never found on the training data (there the most complex model always wins), but by **validation**.

### 1.4 Honest evaluation: train / validation / test and cross-validation

Three roles that must not be mixed:

1. **Training**: learn the parameters.
2. **Validation**: choose hyperparameters, compare models.
3. **Test**: measure the final performance ONCE, right at the end.

Why is train/test not enough? Whoever compares 50 model variants on the test set has turned the test set into a validation set — the best of the 50 numbers is biased upwards (the multiple testing logic from module 02!).

**k-fold cross-validation (CV)** uses the data more efficiently: split the training data into $k$ (usually 5 or 10) blocks; train $k$ times on $k{-}1$ blocks and validate on the remaining one; average the results. Advantages: every observation is validated once, and you additionally get the **spread** of the estimate (a model choice based on a 0.5 % difference with a 2 % spread is reading tea leaves). For classification use **stratification** (equal class shares in every fold); for time series keep the temporal order (module 03!); for grouped data (several rows per patient!) split by group — otherwise leakage.

### 1.5 Metrics: how do you measure "good"?

**Classification — the confusion matrix** is the basis of everything:

|  | predicted + | predicted − |
|--|--|--|
| **actual +** | true positive (TP) | false negative (FN) |
| **actual −** | false positive (FP) | true negative (TN) |

- **Accuracy** $= (TP{+}TN)/n$ — misleading with imbalanced classes (module 02: guessing "ham" 87 % of the time!)
- **Precision** $= TP/(TP{+}FP)$ — when there is an alarm, how often is it justified? (the cost of false alarms)
- **Recall** $= TP/(TP{+}FN)$ — how much of the real thing is found? (the cost of missing)
- **F1** = the harmonic mean of precision and recall — a compromise number
- **ROC curve and AUC**: almost all classifiers output scores/probabilities; the decision threshold is freely choosable. The ROC curve shows the true positive rate against the false positive rate across all thresholds; the area under it (**AUC**) measures, independently of the threshold, how well the model separates positive from negative (0.5 = a coin flip, 1.0 = perfect). Under strong imbalance the **precision-recall curve** is more informative.

**Which metric?** This is not a statistical question but a **question of the costs of the application**: for a spam filter, false positives are expensive (a genuine mail is gone → keep precision high); for cancer screening, false negatives are (a case missed → keep recall high). Settle this trade-off *before* modelling.

**Regression:** MAE (robust, directly interpretable), RMSE (penalises large errors more strongly), $R^2$ (explained variance, comparison against "always guess the mean").

---

## 2. Intermediate

### 2.1 Linear models with regularisation

You know linear/logistic regression from module 03. What is new is **regularisation**: a penalty term against overly large coefficients,

$$\min_\beta \; \text{loss}(\beta) + \alpha \cdot \text{penalty}(\beta)$$

- **Ridge (L2)**: the penalty $\sum \beta_j^2$ — shrinks all coefficients evenly towards 0. The first choice with many correlated features (it stabilises the multicollinearity from module 03).
- **Lasso (L1)**: the penalty $\sum |\beta_j|$ — pushes unimportant coefficients to *exactly* 0 and therefore performs automatic feature selection.
- $\alpha$ controls the bias-variance tradeoff directly: $\alpha = 0$ gives OLS (full variance), $\alpha \to \infty$ gives the null model (full bias). $\alpha$ is chosen by CV.
- **Scaling is mandatory** (the penalty compares coefficient magnitudes — unfair with different units). In sklearn the regularisation parameter of `LogisticRegression` is incidentally called `C` = $1/\alpha$ (large = little regularisation) — a popular confusion.

### 2.2 Decision trees

**Idea:** ask yes/no questions about the features one after another ("temp above 12 degrees C?") until a decision is reached. Learning is greedy: at every node the split is chosen that separates the classes best — measured by impurity (**Gini index** $1 - \sum_c p_c^2$ or entropy).

- **Strengths:** interpretable (the tree *is* the explanation), no scaling needed, categorical and numerical features mixed, non-linear relationships and interactions automatically.
- **Weakness:** single trees overfit brutally (fully grown they memorise every training row) and are **unstable** — a small change in the data gives a completely different tree (high variance!). Remedies: limit the depth (`max_depth`), a minimum leaf size — or better: ensembles.

### 2.3 Ensembles: the wisdom of many models

**Random forest (bagging):** train many (e.g. 300) deep trees, each on a bootstrap sample (module 03!) of the data and with a random selection of features per split; average the predictions. The randomness **decorrelates** the trees, and the average of many decorrelated high-variance models has **low variance** at an almost unchanged bias. The result: robust, needing little tuning, an excellent default choice for tabular data.

**Gradient boosting:** build shallow trees **one after another**, each correcting the errors (more precisely: the gradient of the loss) of the sum so far:

$$F_{m}(x) = F_{m-1}(x) + \eta \cdot h_m(x), \qquad h_m \approx -\nabla L(F_{m-1})$$

With a small learning rate $\eta$ and enough trees, a very accurate model emerges — gradient boosting variants (XGBoost, LightGBM, sklearn's `HistGradientBoosting`) **win almost everything on tabular data** (against deep learning too!). The price: more hyperparameters, and it overfits without care (learning rate, number of trees, depth via CV/early stopping).

**Bagging vs. boosting in one sentence:** bagging averages independent complex models (lowering variance), boosting adds dependent simple models (lowering bias).

### 2.4 Pipelines: preprocessing without self-deception

Real data need preprocessing: scale numerical features (`StandardScaler`), one-hot encode categorical ones (`OneHotEncoder`), impute missing values. The leakage rule from module 03 ("learn everything learnable on train only") applies to **every** one of these steps — the mean used for imputation is learned too!

The sklearn **`Pipeline`** (+ `ColumnTransformer` for "scale column A, encode column B") enforces this automatically: `fit` learns all steps only on the training data of the respective (CV) split, `transform` applies them to validation/test. **Pipelines are not a nice-to-have but the only way to combine CV and preprocessing correctly.**

### 2.5 Hyperparameter tuning

- **Grid search**: try all combinations of a grid via CV (`GridSearchCV`). Complete, but exponentially expensive.
- **Random search**: random combinations — usually better at the same budget, because important parameters are sampled more finely (the classic result of Bergstra & Bengio).
- Afterwards: measure the final performance **on the untouched test set** — the best CV number itself is slightly optimistic (it was, after all, selected).
- **Learning curves** (performance vs. amount of training data) answer the question "do I need more data or a different model?": if the train and validation curves converge at a low level, it is a bias problem (more data will not help). A large gap means a variance problem (more data or more regularisation help).

### 2.6 Imbalanced classes

At ratios of 1:99 (fraud, disease, churn) special treatment is needed:

1. **The right metric** (PR-AUC, recall at a fixed precision — never accuracy),
2. **Move the threshold**: the 0.5 threshold is a convention, not a law — choose it by weighing costs on the validation set,
3. **Class weights** (`class_weight="balanced"`): errors on the rare class count more,
4. Resampling (over-/undersampling, SMOTE) — with care and only on the training data.

---

## 3. Advanced topics

### 3.1 Understanding models: importances and their traps

- **Impurity-based feature importance** (the random forest default): measures how much a feature reduces impurity. Traps: it favours features with many possible split values; it is computed on the training data (overfitting rubs off).
- **Permutation importance**: shuffle a feature in the *validation data* and measure the drop in performance — model-agnostic and more honest. Trap: with correlated features the importance is shared unpredictably ("my feature is unimportant" can mean "its twin already carries the same information").
- **Partial dependence plots**: the average course of the prediction as you push a feature through its range — this shows the *shape* of the relationship (e.g. the comfort temperature parabola from module 03).
- Outlook: SHAP values (a game-theoretically founded decomposition of contributions per prediction) — the standard in practice, the details go beyond this module.

**Warning:** all of these tools describe **what the model uses** — not what acts causally in the world.

### 3.2 Calibration: is 80 % really 80 %?

Many models output "probabilities" that are none: of all predictions with score 0.8, about 80 % should actually be positive (**calibration**). Random forests are often too cautious (scores squeezed towards 0.5), boosting often too self-confident; logistic regression is usually well calibrated out of the box. Check with the **reliability diagram** (predicted vs. observed frequency in score bins), repair with Platt scaling / isotonic regression (`CalibratedClassifierCV`). This matters everywhere scores feed into decisions with thresholds or expected values (risk scores, prices).

### 3.3 What this module deliberately leaves out

- **Neural networks and deep learning** → module 05 (Machine Learning 2). A mnemonic until then: on structured tabular data, gradient boosting models are usually at least on a par; deep learning dominates on raw data (images, audio, text).
- **Unsupervised learning** (clustering, density estimation) → module 05.
- **SVMs and kernel methods**: less often the first choice today, but conceptually important (maximum margin, the kernel trick) — a brief contact in the medium project, the theory in modules 06/07.
- **AutoML**: automated tuning/model selection — useful, but it replaces none of the honest evaluation rules of this module.

---

## 4. Summary / cheat sheet

**The framework**
- Learning = minimise the loss on train, the goal = error on new data; the gap is called generalisation
- Error = bias² + variance + noise; simple = bias, flexible = variance
- Symptoms: train bad and test bad → underfitting; train good, test bad → overfitting

**Evaluation**
- Train (parameters) / validation (choices) / test (once!); k-fold CV stratified; time → temporal, groups → by group
- Classification: confusion matrix; precision = TP/(TP+FP), recall = TP/(TP+FN), F1; ROC-AUC (threshold-free); imbalanced → PR curve, never accuracy
- The metric is a question of the costs of the application, fixed in advance

**Models**
- kNN: small k → variance, large k → bias; scale!; suffers in high dimensions
- Ridge (L2) shrinks, lasso (L1) selects; α by CV; scale; sklearn's `C` = 1/α
- Tree: greedy splits by Gini/entropy; interpretable but unstable → limit the depth
- Random forest: bagging + feature randomness = variance down; a robust default
- Gradient boosting: sequential error correction, a small learning rate; the champion on tabular data
- Naive Bayes (module 01) and logistic regression: fast, strong baselines

**Workflow**
- Pipeline + ColumnTransformer: preprocessing is learned inside the CV → no leakage
- GridSearchCV/RandomizedSearchCV; afterwards test ONCE
- Learning curve: a gap means variance (data/regularisation), no progress means bias (change the model)
- Imbalanced: metric, threshold, class_weight — in that order

**Interpretation**
- Permutation importance beats impurity importance; correlation dilutes both
- PDP for shapes; check calibration when scores steer decisions
- What the model uses is not causality

---

## 5. Self-test

<details><summary><b>1. Your model: training error 2 %, test error 19 %. Diagnosis and three possible remedies?</b></summary>

Classic **overfitting** (high variance): the model is learning the noise of the training data. Remedies: reduce complexity (a smaller tree, a larger $k$, stronger regularisation $\alpha$ up / `C` down), more training data, ensembles with bagging (random forest), reduce the number of features. And check whether leakage is flattering the training performance.
</details>

<details><summary><b>2. Why may hyperparameters not be chosen on the test set — and what is the solution when data are scarce?</b></summary>

Whoever compares many variants on the test set and takes the best is partly selecting the random noise of the test set — the reported performance is biased upwards and the test set is "used up". Solution: cross-validation on the training data for all decisions; the test set is used exactly once, at the end.
</details>

<details><summary><b>3. Fraud detection: 0.5 % positives. Model A: accuracy 99.5 %. Model B: accuracy 97 %, recall 80 %, precision 12 %. Which is probably more useful, and why?</b></summary>

Model A probably reaches 99.5 % by "never saying fraud" (recall about 0) — useless. Model B finds 80 % of the fraud cases; the low precision (12 %) means many false alarms, which may be acceptable depending on the cost of checking. Evaluation has to use PR metrics and the real costs of FP (an investigation) vs. FN (the damage) — not accuracy.
</details>

<details><summary><b>4. Why does kNN need scaling but a decision tree does not?</b></summary>

kNN compares **distances** across all features — without scaling, the feature with the largest numerical scale dominates the distance. A tree looks at every feature **individually** and only asks "value above threshold?" — any monotone transformation (and therefore scaling too) does not change the possible splits.
</details>

<details><summary><b>5. Explain in one sentence each why random forest lowers variance and gradient boosting lowers bias.</b></summary>

Random forest averages many trees that are **decorrelated** by bootstrapping and feature randomness and that individually overfit — the average of independent estimates scatters less than each single one. Gradient boosting adds shallow (strongly biased) trees one after another, each of which corrects the remaining systematic errors of the sum — the sum can represent structures that every individual model misses.
</details>

<details><summary><b>6. You scale the data with StandardScaler on the WHOLE data set and do CV afterwards. What is wrong, how bad is it, and what is the clean solution?</b></summary>

The scaler has learned mean/spread from the later validation folds too — leakage. With pure scaling the effect is usually small, but the principle scales: with imputation, feature selection or target encoding on the whole data set, the CV estimate is flattered massively. Cleanly: put the preprocessing into a `Pipeline`, which within CV is fitted per fold on the training part only.
</details>

<details><summary><b>7. The learning curve shows: training and validation performance lie close together, both bad, and they stop moving beyond 10,000 examples. Does collecting data help?</b></summary>

No — this is a **bias problem**: the model is too simple (or the features do not carry the information). More data of the same kind changes nothing. Remedy: a more expressive model, better features (feature engineering!), less regularisation. Collecting data helps in the opposite case: a large gap between train and validation (variance).
</details>

<details><summary><b>8. Random forest: the feature "customer ID" has the highest impurity importance. What has happened?</b></summary>

A (quasi-)ID has extremely many unique values — with it the tree can practically address training rows by heart; impurity importance rewards exactly such split-friendly features. This is overfitting plus a metric artefact, not genuine importance: remove the ID (in general: no identifiers as features) and use permutation importance on validation data — there the ID would drop to about 0.
</details>

<details><summary><b>9. Why is the 0.5 threshold for classification scores often the wrong one — and how do you choose better?</b></summary>

0.5 only makes sense with symmetric error costs and calibrated scores. If FP and FN cost differently (almost always) or the classes are imbalanced, the threshold belongs where the expected costs are minimal or the required precision/recall is reached — determined on the validation set (e.g. via the PR curve), not on the test set.
</details>

<details><summary><b>10. A colleague says: "The random forest has shown that smoking is the most important feature for disease X — so smoking causes X." Two objections.</b></summary>

(1) Importance measures only what **the model uses for prediction** — a feature can be important because it correlates with the true cause (confounding, module 02). (2) With correlated features, importance is distributed arbitrarily; a causally irrelevant but well-measured proxy can "crowd out" the real cause. Causal statements need study design (randomisation, adjustment), not feature rankings.
</details>

---

## 6. Literature and sources

**Textbooks**

- **James, Witten, Hastie, Tibshirani — "An Introduction to Statistical Learning" (ISLR), 2nd ed.** — covers almost the whole module (ch. 2 bias/variance, 4 classification, 5 CV/bootstrap, 6 regularisation, 8 trees/ensembles). **Free**: https://www.statlearning.com. There is a Python edition (ISLP). *(the main recommendation)*
- **Géron — "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow", 3rd ed.** — the practical classic; part 1 is exactly this module. *(beginner-friendly, paid)*
- **Hastie, Tibshirani, Friedman — "The Elements of Statistical Learning"** — the deeper theoretical reference, **free** as a PDF. *(advanced, only after ISLR)*

**Online courses (free)**

- **Andrew Ng — Machine Learning Specialization** (Coursera, free to audit) — the most didactically polished introduction *(beginner-friendly)*
- **scikit-learn MOOC** (inria.github.io/scikit-learn-mooc) — by the sklearn developers, with exactly the pipeline/CV practices of this module *(beginner-friendly, highly recommended as a companion)*
- **StatQuest** (YouTube): playlists on bias/variance, ROC/AUC, random forest, gradient boosting *(beginner-friendly)*

**Interactive / blog posts (free)**

- *MLU-Explain* (mlu-explain.github.io): interactive visualisations of bias-variance, ROC, random forest, cross-validation — a must-visit
- *A visual introduction to machine learning* (r2d3.us): decision trees explained visually
- scikit-learn user guide, the chapter "Common pitfalls and recommended practices" — leakage and friends, officially

**Advanced**

- Bergstra & Bengio (2012): *Random Search for Hyper-Parameter Optimization* (JMLR)
- Grinsztajn et al. (2022): *Why do tree-based models still outperform deep learning on tabular data?* (NeurIPS)

---

**Next step:** `projects/01-basic/` (kNN by hand + making bias-variance visible) → `projects/02-medium/` (the honest model race: pipelines, CV, tuning) → `projects/03-final/` (income prediction on real census data: imbalanced, mixed types, choosing a threshold).

---
---

# Modul 04 — Machine Learning 1 (deutsche Fassung)

**Worum geht es?** Jetzt wird das Lernen aus Daten systematisch: Wie baut, bewertet und verbessert man Vorhersagemodelle — ehrlich, reproduzierbar und ohne sich selbst zu betrügen? Du lernst die wichtigsten Modellfamilien des klassischen Supervised Learning (kNN, lineare Modelle mit Regularisierung, Entscheidungsbäume, Ensembles), vor allem aber das **Handwerk drumherum**: Validierung, Metriken, Pipelines, Hyperparameter-Tuning. Das ist das Modul, dessen Inhalte in der Praxis am häufigsten gebraucht werden.

**Vorkenntnisse:** Module 02/03 (pandas, Train/Test-Gedanke, Regression, Leakage). Modul 01 (Naive Bayes) hilft.

**Vorher zu machen:** Modul 01–03.

---

## Lernziele

Nach diesem Modul kannst du:

- den Kern des Supervised Learning präzise formulieren (Hypothese, Verlustfunktion, Generalisierung) und **Overfitting/Underfitting** über den **Bias-Variance-Tradeoff** erklären,
- Modelle korrekt evaluieren: Train/Validation/Test, **k-fache Kreuzvalidierung**, und begründen, warum man den Testsatz nur einmal anfasst,
- die richtigen **Metriken** wählen und interpretieren (Confusion Matrix, Precision/Recall/F1, ROC-AUC; MAE/RMSE/$R^2$) — insbesondere bei **unausgewogenen Klassen**,
- **kNN, logistische Regression (mit Regularisierung), Entscheidungsbäume, Random Forests und Gradient Boosting** erklären (wie lernen sie? wann nutzt man was?) und mit scikit-learn anwenden,
- saubere **Pipelines** bauen (Skalierung, One-Hot-Encoding), die Leakage konstruktiv verhindern,
- **Hyperparameter** systematisch tunen (Grid/Random Search mit CV) und **Lernkurven** lesen,
- Modelle inspizieren: Feature Importances, Permutation Importance — und deren Fallstricke.

---

## 1. Grundlagen (Basics)

### 1.1 Was heißt „Lernen" formal?

Gegeben: Trainingsdaten $(x_1, y_1), \dots, (x_n, y_n)$ aus einer unbekannten Verteilung. Gesucht: eine Funktion $h$ (Hypothese) aus einer Modellfamilie $\mathcal{H}$, die den **erwarteten Fehler auf neuen Daten** minimiert — nicht den Trainingsfehler! Formal minimiert man ersatzweise den Trainingsverlust

$$\hat{h} = \arg\min_{h \in \mathcal{H}} \frac{1}{n} \sum_i L(h(x_i), y_i) \; (+ \text{Regularisierung})$$

mit einer **Verlustfunktion** $L$ (z. B. quadratischer Fehler für Regression, Log-Loss für Klassifikation). Die ganze Kunst des Machine Learning liegt in der Lücke zwischen Trainingsverlust und echtem Fehler — der **Generalisierungslücke**.

**Die drei Zutaten jedes ML-Verfahrens:** (1) Modellfamilie $\mathcal{H}$ (welche Funktionen sind überhaupt möglich?), (2) Verlustfunktion (was heißt „gut"?), (3) Optimierungsverfahren (wie findet man das beste $h$?). Wenn du ein neues Verfahren kennenlernst, stelle immer diese drei Fragen.

### 1.2 Das erste Modell: k-Nearest Neighbors (kNN)

**Idee:** Um einen neuen Punkt zu klassifizieren, schau dir die $k$ nächstgelegenen Trainingspunkte an und nimm deren Mehrheitsklasse (bzw. für Regression: deren Mittelwert). Kein „Training" im eigentlichen Sinn — das Modell *ist* der Datensatz.

kNN ist didaktisch Gold, weil man an ihm alles Wichtige sieht:

- **Hyperparameter $k$ steuert die Komplexität**: $k = 1$ → extrem flexible, zackige Entscheidungsgrenze (jeder Trainingspunkt bekommt recht — Overfitting-Gefahr). $k = n$ → immer die häufigste Klasse (maximal starr — Underfitting). Dazwischen liegt das Optimum.
- **Skalierung ist Pflicht**: Distanzen mischen Einheiten! Ohne Standardisierung dominiert das Merkmal mit den größten Zahlen (Gramm schlägt Millimeter — dieselbe Lektion wie bei PCA in Modul 03).
- **Fluch der Dimensionalität**: In hohen Dimensionen sind alle Punkte ungefähr gleich weit voneinander entfernt — „Nähe" verliert ihre Bedeutung. Deshalb skaliert kNN schlecht mit vielen Features.

### 1.3 Overfitting, Underfitting und der Bias-Variance-Tradeoff

Der erwartete Vorhersagefehler eines Modells lässt sich (für quadratischen Verlust) zerlegen:

$$\text{Fehler} = \underbrace{\text{Bias}^2}_{\text{systematisch daneben}} + \underbrace{\text{Varianz}}_{\text{instabil je nach Trainingsdaten}} + \underbrace{\sigma^2}_{\text{irreduzibles Rauschen}}$$

- **Hoher Bias (Underfitting):** Das Modell ist zu einfach für die Wahrheit (Gerade für eine Parabel). Symptom: Trainings- *und* Testfehler hoch, nah beieinander.
- **Hohe Varianz (Overfitting):** Das Modell ist so flexibel, dass es das Rauschen der konkreten Stichprobe mitlernt. Symptom: Trainingsfehler klein, Testfehler deutlich größer.

> **Intuition:** Stell dir vor, du trainierst dasselbe Modell auf 100 verschiedenen Stichproben. Bias = wie weit liegt der *Durchschnitt* der 100 Modelle von der Wahrheit weg. Varianz = wie stark *streuen* die 100 Modelle untereinander. Einfache Modelle: alle ähnlich, alle ähnlich falsch. Komplexe Modelle: im Mittel richtig, einzeln wild verschieden.

**Stellschrauben für die Komplexität**: $k$ bei kNN, Baumtiefe, Polynomgrad, Regularisierungsstärke, … Die richtige Einstellung findet man nie auf den Trainingsdaten (dort gewinnt immer das komplexeste Modell), sondern per **Validierung**.

### 1.4 Ehrliche Evaluation: Train / Validation / Test und Kreuzvalidierung

Drei Rollen, die man nicht vermischen darf:

1. **Training**: Parameter lernen.
2. **Validierung**: Hyperparameter wählen, Modelle vergleichen.
3. **Test**: EINMAL ganz am Ende die finale Leistung messen.

Warum reicht Train/Test nicht? Wer 50 Modellvarianten am Testsatz vergleicht, hat den Testsatz zum Validierungssatz gemacht — die beste der 50 Zahlen ist nach oben verzerrt (Multiple-Testing-Logik aus Modul 02!).

**k-fache Kreuzvalidierung (CV)** nutzt die Daten effizienter: Teile die Trainingsdaten in $k$ (üblich 5 oder 10) Blöcke; trainiere $k$-mal auf $k{-}1$ Blöcken und validiere auf dem übrigen; mittle die Ergebnisse. Vorteile: jede Beobachtung wird einmal validiert, und man bekommt zusätzlich die **Streuung** der Schätzung (eine Modellwahl, die auf ±0,5 % Unterschied bei ±2 % Streuung beruht, ist Kaffeesatzleserei). Bei Klassifikation **stratifiziert** (Klassenanteile in jedem Fold gleich); bei Zeitreihen zeitlich geordnet (Modul 03!); bei gruppierten Daten (mehrere Zeilen pro Patient!) gruppenweise splitten — sonst Leakage.

### 1.5 Metriken: Woran misst man „gut"?

**Klassifikation — die Confusion Matrix** ist die Basis von allem:

|  | vorhergesagt + | vorhergesagt − |
|--|--|--|
| **echt +** | True Positive (TP) | False Negative (FN) |
| **echt −** | False Positive (FP) | True Negative (TN) |

- **Accuracy** $= (TP{+}TN)/n$ — irreführend bei unausgewogenen Klassen (Modul 02: 87 % „ham" raten!)
- **Precision** $= TP/(TP{+}FP)$ — wenn Alarm, wie oft zurecht? (Kosten von Fehlalarmen)
- **Recall** $= TP/(TP{+}FN)$ — wie viel vom Echten gefunden? (Kosten des Übersehens)
- **F1** = harmonisches Mittel aus Precision und Recall — Kompromisszahl
- **ROC-Kurve & AUC**: Fast alle Klassifikatoren geben Scores/Wahrscheinlichkeiten aus; die Entscheidungsschwelle ist frei wählbar. Die ROC-Kurve zeigt True-Positive-Rate gegen False-Positive-Rate über alle Schwellen; die Fläche darunter (**AUC**) misst schwellenunabhängig, wie gut das Modell positiv von negativ trennt (0,5 = Münzwurf, 1,0 = perfekt). Bei starker Unausgewogenheit ist die **Precision-Recall-Kurve** aussagekräftiger.

**Welche Metrik?** Ist keine Statistikfrage, sondern eine **Kostenfrage der Anwendung**: Beim Spamfilter sind False Positives teuer (echte Mail weg → Precision hoch halten), beim Krebs-Screening False Negatives (Fall übersehen → Recall hoch halten). Diese Abwägung *vor* dem Modellieren festlegen.

**Regression:** MAE (robust, direkt interpretierbar), RMSE (bestraft große Fehler stärker), $R^2$ (erklärte Varianz, Vergleich gegen „immer Mittelwert raten").

---

## 2. Aufbau (Intermediate)

### 2.1 Lineare Modelle mit Regularisierung

Lineare/logistische Regression kennst du aus Modul 03. Neu ist die **Regularisierung**: ein Strafterm gegen zu große Koeffizienten,

$$\min_\beta \; \text{Verlust}(\beta) + \alpha \cdot \text{Strafe}(\beta)$$

- **Ridge (L2)**: Strafe $\sum \beta_j^2$ — schrumpft alle Koeffizienten gleichmäßig Richtung 0. Erste Wahl bei vielen korrelierten Features (stabilisiert die Multikollinearität aus Modul 03).
- **Lasso (L1)**: Strafe $\sum |\beta_j|$ — drückt unwichtige Koeffizienten auf *exakt* 0, macht also automatisch Feature-Selektion.
- $\alpha$ steuert den Bias-Variance-Tradeoff direkt: $\alpha = 0$ → OLS (volle Varianz), $\alpha \to \infty$ → Nullmodell (voller Bias). Gewählt wird $\alpha$ per CV.
- **Skalierung Pflicht** (die Strafe vergleicht Koeffizientengrößen — unfair bei verschiedenen Einheiten). In sklearn heißt der Regularisierungsparameter bei `LogisticRegression` übrigens `C` = $1/\alpha$ (groß = wenig Regularisierung) — beliebte Verwechslung.

### 2.2 Entscheidungsbäume

**Idee:** Stelle nacheinander Ja/Nein-Fragen an die Features („temp > 12 °C?"), bis eine Entscheidung fällt. Gelernt wird gierig: An jedem Knoten wird der Split gewählt, der die Klassen am besten trennt — gemessen an der Verunreinigung (**Gini-Index** $1 - \sum_c p_c^2$ oder Entropie).

- **Stärken:** interpretierbar (der Baum *ist* die Erklärung), keine Skalierung nötig, kategoriale und numerische Features gemischt, nichtlineare Zusammenhänge und Interaktionen automatisch.
- **Schwäche:** Einzelbäume overfitten brutal (voll ausgewachsen lernen sie jede Trainingszeile auswendig) und sind **instabil** — kleine Datenänderung, ganz anderer Baum (hohe Varianz!). Gegenmittel: Tiefe begrenzen (`max_depth`), Mindestblattgröße — oder besser: Ensembles.

### 2.3 Ensembles: Weisheit der vielen Modelle

**Random Forest (Bagging):** Trainiere viele (z. B. 300) tiefe Bäume, jeder auf einer Bootstrap-Stichprobe (Modul 03!) der Daten und mit zufälliger Feature-Auswahl pro Split; mittle die Vorhersagen. Die Zufälligkeit **dekorreliert** die Bäume, und der Durchschnitt vieler dekorrelierter Hochvarianz-Modelle hat **niedrige Varianz** bei fast unverändertem Bias. Ergebnis: robust, wenig tuningbedürftig, hervorragende Standardwahl für Tabellendaten.

**Gradient Boosting:** Baue flache Bäume **nacheinander**, jeder korrigiert die Fehler (genauer: den Gradienten des Verlusts) der bisherigen Summe:

$$F_{m}(x) = F_{m-1}(x) + \eta \cdot h_m(x), \qquad h_m \approx -\nabla L(F_{m-1})$$

Mit kleiner Lernrate $\eta$ und genügend Bäumen entsteht ein sehr genaues Modell — Gradient-Boosting-Varianten (XGBoost, LightGBM, sklearns `HistGradientBoosting`) **gewinnen auf Tabellendaten fast alles** (auch gegen Deep Learning!). Preis: mehr Hyperparameter, overfittet ohne Sorgfalt (Lernrate, Baumzahl, Tiefe per CV/Early Stopping).

**Bagging vs. Boosting in einem Satz:** Bagging mittelt unabhängige komplexe Modelle (senkt Varianz), Boosting addiert abhängige einfache Modelle (senkt Bias).

### 2.4 Pipelines: Preprocessing ohne Selbstbetrug

Reale Daten brauchen Vorverarbeitung: numerische Features skalieren (`StandardScaler`), kategoriale one-hot-encoden (`OneHotEncoder`), fehlende Werte imputieren. Die Leakage-Regel aus Modul 03 („alles Lernbare nur auf Train lernen") gilt für **jeden** dieser Schritte — auch der Mittelwert fürs Imputieren ist gelernt!

Die sklearn-**`Pipeline`** (+ `ColumnTransformer` für „Spalte A skalieren, Spalte B encoden") erzwingt das automatisch: `fit` lernt alle Schritte nur auf den Trainingsdaten des jeweiligen (CV-)Splits, `transform` wendet sie auf Validierung/Test an. **Pipelines sind keine Kür, sondern der einzige Weg, CV + Preprocessing korrekt zu kombinieren.**

### 2.5 Hyperparameter-Tuning

- **Grid Search**: alle Kombinationen eines Rasters per CV durchprobieren (`GridSearchCV`). Vollständig, aber exponentiell teuer.
- **Random Search**: zufällige Kombinationen — bei gleichem Budget meist besser, weil wichtige Parameter feiner abgetastet werden (klassisches Ergebnis von Bergstra & Bengio).
- Danach: finale Leistung **auf dem unberührten Testsatz** messen — die beste CV-Zahl selbst ist leicht optimistisch (sie wurde ja ausgewählt).
- **Lernkurven** (Leistung vs. Trainingsmenge) beantworten die Frage „brauche ich mehr Daten oder ein anderes Modell?": Konvergieren Train- und Validierungskurve auf niedrigem Niveau → Bias-Problem (mehr Daten helfen nicht). Große Lücke → Varianz-Problem (mehr Daten oder mehr Regularisierung helfen).

### 2.6 Unausgewogene Klassen

Bei 1:99-Verhältnissen (Betrug, Krankheit, Kündigung) braucht es Sonderbehandlung:

1. **Richtige Metrik** (PR-AUC, Recall bei fixierter Precision — nie Accuracy),
2. **Schwelle verschieben**: Die 0,5-Schwelle ist Konvention, nicht Gesetz — wähle sie nach Kostenabwägung auf der Validierungsmenge,
3. **Klassengewichte** (`class_weight="balanced"`): Fehler auf der seltenen Klasse zählen mehr,
4. Resampling (Over-/Undersampling, SMOTE) — mit Vorsicht und nur auf den Trainingsdaten.

---

## 3. Advanced-Themen

### 3.1 Modelle verstehen: Importances und ihre Fallen

- **Impurity-based Feature Importance** (Random Forest Standard): misst, wie viel ein Feature die Verunreinigung reduziert. Fallen: bevorzugt Features mit vielen möglichen Splitwerten; auf Trainingsdaten berechnet (Overfitting färbt ab).
- **Permutation Importance**: Verwürfle ein Feature in den *Validierungsdaten* und miss den Leistungseinbruch — modellagnostisch und ehrlicher. Falle: Bei korrelierten Features teilt sich die Wichtigkeit unvorhersehbar auf („mein Feature ist unwichtig" kann heißen „sein Zwilling trägt schon dieselbe Information").
- **Partial Dependence Plots**: mittlerer Vorhersageverlauf, wenn man ein Feature durch seinen Wertebereich schiebt — zeigt die *Form* des Zusammenhangs (z. B. die Wohlfühltemperatur-Parabel aus Modul 03).
- Ausblick: SHAP-Werte (spieltheoretisch fundierte Beitragszerlegung pro Vorhersage) — Standard in der Praxis, Details sprengen dieses Modul.

**Warnung:** Alle diese Werkzeuge beschreiben, **was das Modell nutzt** — nicht, was in der Welt kausal wirkt.

### 3.2 Kalibrierung: Sind 80 % wirklich 80 %?

Viele Modelle geben „Wahrscheinlichkeiten" aus, die keine sind: Von allen Vorhersagen mit Score 0,8 sollten ~80 % tatsächlich positiv sein (**Kalibrierung**). Random Forests sind oft zu vorsichtig (Scores gestaucht Richtung 0,5), Boosting oft zu selbstsicher; logistische Regression ist von Haus aus meist gut kalibriert. Prüfen mit dem **Reliability Diagram** (vorhergesagte vs. beobachtete Häufigkeit in Score-Bins), reparieren mit Platt Scaling / Isotonic Regression (`CalibratedClassifierCV`). Wichtig überall dort, wo Scores in Entscheidungen mit Schwellen oder Erwartungswerten eingehen (Risiko-Scores, Preise).

### 3.3 Was dieses Modul bewusst auslässt

- **Neuronale Netze & Deep Learning** → Modul 05 (Machine Learning 2). Merksatz bis dahin: Auf strukturierten Tabellendaten sind Gradient-Boosting-Modelle meist mindestens ebenbürtig; Deep Learning dominiert bei Rohdaten (Bilder, Audio, Text).
- **Unsupervised Learning** (Clustering, Dichteschätzung) → Modul 05.
- **SVMs & Kernel-Methoden**: heute seltener erste Wahl, aber konzeptionell wichtig (Maximum-Margin, Kernel-Trick) — Kurzkontakt im Medium-Projekt, Theorie in Modul 06/07.
- **AutoML**: automatisiertes Tuning/Modellwahl — nützlich, ersetzt aber keine der ehrlichen Evaluationsregeln dieses Moduls.

---

## 4. Zusammenfassung / Cheat-Sheet

**Grundgerüst**
- Lernen = Verlust auf Train minimieren, Ziel = Fehler auf Neuem; die Lücke heißt Generalisierung
- Fehler = Bias² + Varianz + Rauschen; einfach = Bias, flexibel = Varianz
- Symptome: Train schlecht ≈ Test schlecht → Underfitting · Train gut, Test schlecht → Overfitting

**Evaluation**
- Train (Parameter) / Validation (Wahl) / Test (einmal!); k-fach-CV stratifiziert; Zeit → zeitlich, Gruppen → gruppenweise
- Klassifikation: Confusion Matrix; Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1; ROC-AUC (schwellenfrei); unausgewogen → PR-Kurve, nie Accuracy
- Metrik = Kostenfrage der Anwendung, vorab festlegen

**Modelle**
- kNN: k klein → Varianz, k groß → Bias; skalieren!; leidet in hohen Dimensionen
- Ridge (L2) schrumpft, Lasso (L1) selektiert; α per CV; skalieren; sklearn-`C` = 1/α
- Baum: gierige Splits nach Gini/Entropie; interpretierbar, aber instabil → Tiefe begrenzen
- Random Forest: Bagging + Feature-Zufall = Varianz runter; robuste Standardwahl
- Gradient Boosting: sequentielle Fehlerkorrektur, Lernrate klein; Tabellendaten-Champion
- Naive Bayes (Modul 01) und logistische Regression: schnelle, starke Baselines

**Workflow**
- Pipeline + ColumnTransformer: Preprocessing wird im CV mitgelernt → keine Leakage
- GridSearchCV/RandomizedSearchCV; danach EINMAL Test
- Lernkurve: Lücke = Varianz (Daten/Regularisierung), kein Fortschritt = Bias (Modell wechseln)
- Unausgewogen: Metrik, Schwelle, class_weight — in dieser Reihenfolge

**Interpretation**
- Permutation Importance > Impurity Importance; Korrelation verwässert beide
- PDP für Formen; Kalibrierung prüfen, wenn Scores Entscheidungen steuern
- Modell-Nutzung ≠ Kausalität

---

## 5. Selbsttest

<details><summary><b>1. Dein Modell: Trainingsfehler 2 %, Testfehler 19 %. Diagnose und drei mögliche Gegenmittel?</b></summary>

Klassisches **Overfitting** (hohe Varianz): Das Modell lernt Rauschen der Trainingsdaten. Gegenmittel: Komplexität senken (kleinerer Baum, größeres $k$, stärkere Regularisierung $\alpha$↑/`C`↓), mehr Trainingsdaten, Ensembles mit Bagging (Random Forest), Feature-Menge reduzieren. Und prüfen, ob nicht Leakage die Trainingsleistung schönt.
</details>

<details><summary><b>2. Warum darf man Hyperparameter nicht auf dem Testsatz wählen — und was ist die Lösung, wenn Daten knapp sind?</b></summary>

Wer viele Varianten am Testsatz vergleicht und die beste nimmt, wählt teilweise das Zufallsrauschen des Testsatzes aus — die berichtete Leistung ist nach oben verzerrt und der Testsatz „verbraucht". Lösung: Kreuzvalidierung auf den Trainingsdaten für alle Entscheidungen; der Testsatz wird genau einmal am Ende benutzt.
</details>

<details><summary><b>3. Betrugserkennung: 0,5 % Positive. Modell A: Accuracy 99,5 %. Modell B: Accuracy 97 %, Recall 80 %, Precision 12 %. Welches ist vermutlich nützlicher und warum?</b></summary>

Modell A erreicht 99,5 % vermutlich durch „nie Betrug sagen" (Recall ≈ 0) — nutzlos. Modell B findet 80 % der Betrugsfälle; die niedrige Precision (12 %) heißt viele Fehlalarme, was je nach Prüfkosten akzeptabel sein kann. Bewerten muss man mit PR-Metriken und den echten Kosten von FP (Prüfung) vs. FN (Schaden) — nicht mit Accuracy.
</details>

<details><summary><b>4. Warum braucht kNN Skalierung, ein Entscheidungsbaum aber nicht?</b></summary>

kNN vergleicht **Distanzen** über alle Features hinweg — ohne Skalierung dominiert das Feature mit der größten Zahlenskala die Distanz. Ein Baum betrachtet jedes Feature **einzeln** und fragt nur „Wert > Schwelle?" — jede monotone Transformation (also auch Skalierung) verändert die möglichen Splits nicht.
</details>

<details><summary><b>5. Erkläre in je einem Satz, warum Random Forest die Varianz senkt und Gradient Boosting den Bias.</b></summary>

Random Forest mittelt viele durch Bootstrap und Feature-Zufall **dekorrelierte**, einzeln overfittende Bäume — der Durchschnitt unabhängiger Schätzungen streut weniger als jede einzelne. Gradient Boosting addiert nacheinander flache (stark verzerrte) Bäume, von denen jeder die verbleibenden systematischen Fehler der Summe korrigiert — die Summe kann Strukturen abbilden, die jedes Einzelmodell verfehlt.
</details>

<details><summary><b>6. Du skalierst die Daten mit StandardScaler auf dem GANZEN Datensatz und machst danach CV. Was ist falsch, wie schlimm ist es, und was ist die saubere Lösung?</b></summary>

Der Scaler hat Mittel/Streuung auch aus den späteren Validierungs-Folds gelernt — Leakage. Bei purem Skalieren ist der Effekt meist klein, aber das Prinzip skaliert: Bei Imputation, Feature-Selektion oder Target-Encoding auf dem Gesamtdatensatz wird die CV-Schätzung massiv geschönt. Sauber: Preprocessing in eine `Pipeline`, die im CV pro Fold nur auf dem Trainingsteil gefittet wird.
</details>

<details><summary><b>7. Die Lernkurve zeigt: Trainings- und Validierungsleistung liegen eng beieinander, beide schlecht, und bewegen sich ab 10.000 Beispielen nicht mehr. Hilft Datensammeln?</b></summary>

Nein — das ist ein **Bias-Problem**: Das Modell ist zu einfach (oder die Features tragen die Information nicht). Mehr gleichartige Daten verschieben nichts. Abhilfe: ausdrucksstärkeres Modell, bessere Features (Feature Engineering!), weniger Regularisierung. Datensammeln hilft im umgekehrten Fall: große Lücke zwischen Train und Validierung (Varianz).
</details>

<details><summary><b>8. Random Forest: Feature „Kunden-ID" hat die höchste Impurity-Importance. Was ist passiert?</b></summary>

Eine (Quasi-)ID hat extrem viele eindeutige Werte — der Baum kann damit Trainingszeilen praktisch auswendig adressieren; Impurity-Importance belohnt genau solche Split-freudigen Features. Das ist Overfitting + Metrik-Artefakt, keine echte Wichtigkeit: ID entfernen (generell: keine Identifikatoren als Features) und Permutation Importance auf Validierungsdaten verwenden — dort fiele die ID auf ~0.
</details>

<details><summary><b>9. Warum ist die 0,5-Schwelle bei Klassifikationsscores oft die falsche — und wie wählt man besser?</b></summary>

0,5 ist nur bei symmetrischen Fehlkosten und kalibrierten Scores sinnvoll. Sind FP und FN unterschiedlich teuer (fast immer) oder die Klassen unausgewogen, gehört die Schwelle dorthin, wo die erwarteten Kosten minimal sind bzw. die geforderte Precision/Recall erreicht wird — bestimmt auf der Validierungsmenge (z. B. über die PR-Kurve), nicht auf dem Test.
</details>

<details><summary><b>10. Ein Kollege sagt: „Der Random Forest hat gezeigt, dass Rauchen das wichtigste Feature für Krankheit X ist — also verursacht Rauchen X." Zwei Einwände.</b></summary>

(1) Importance misst nur, was **das Modell zur Vorhersage nutzt** — ein Feature kann wichtig sein, weil es mit der wahren Ursache korreliert (Confounding, Modul 02). (2) Bei korrelierten Features verteilt sich Wichtigkeit willkürlich; ein kausal irrelevanter, aber gut gemessener Proxy kann die echte Ursache „verdrängen". Kausalaussagen brauchen Studiendesign (Randomisierung, Adjustierung), nicht Feature-Rankings.
</details>

---

## 6. Literatur & Quellen

**Lehrbücher**

- **James, Witten, Hastie, Tibshirani — „An Introduction to Statistical Learning" (ISLR), 2. Aufl.** — deckt fast das ganze Modul ab (Kap. 2 Bias/Variance, 4 Klassifikation, 5 CV/Bootstrap, 6 Regularisierung, 8 Bäume/Ensembles). **Kostenlos**: https://www.statlearning.com. Es gibt eine Python-Ausgabe (ISLP). *(die Hauptempfehlung)*
- **Géron — „Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow", 3. Aufl.** — der Praxis-Klassiker; Teil 1 ist exakt dieses Modul. *(einsteigerfreundlich, kostenpflichtig)*
- **Hastie, Tibshirani, Friedman — „The Elements of Statistical Learning"** — die vertiefte Theorie-Referenz, **kostenlos** als PDF. *(vertiefend, erst nach ISLR)*

**Onlinekurse (kostenlos)**

- **Andrew Ng — Machine Learning Specialization** (Coursera, Audit kostenlos) — die didaktisch geschliffenste Einführung *(einsteigerfreundlich)*
- **scikit-learn MOOC** (inria.github.io/scikit-learn-mooc) — von den sklearn-Entwicklern, mit exakt den Pipeline-/CV-Praktiken dieses Moduls *(einsteigerfreundlich, sehr empfohlen als Begleitung)*
- **StatQuest** (YouTube): Playlists zu Bias/Variance, ROC/AUC, Random Forest, Gradient Boosting *(einsteigerfreundlich)*

**Interaktiv / Blogposts (kostenlos)**

- *MLU-Explain* (mlu-explain.github.io): interaktive Visualisierungen zu Bias-Variance, ROC, Random Forest, Cross-Validation — Pflichtbesuch
- *A visual introduction to machine learning* (r2d3.us): Entscheidungsbäume visuell erklärt
- scikit-learn User Guide, Kapitel „Common pitfalls and recommended practices" — Leakage & Co. offiziell

**Vertiefend**

- Bergstra & Bengio (2012): *Random Search for Hyper-Parameter Optimization* (JMLR)
- Grinsztajn et al. (2022): *Why do tree-based models still outperform deep learning on tabular data?* (NeurIPS)

---

**Nächster Schritt:** `projects/01-basic/` (kNN von Hand + Bias-Variance sichtbar machen) → `projects/02-medium/` (das ehrliche Modellrennen: Pipelines, CV, Tuning) → `projects/03-final/` (Einkommensvorhersage auf echten Zensusdaten: unausgewogen, gemischte Typen, Schwellenwahl).
