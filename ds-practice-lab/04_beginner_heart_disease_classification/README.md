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

---

# Deutsche Übersetzung

# 04 — Deine erste Klassifikation: Herzerkrankungen ❤️

Schwierigkeit: 🟢 Einsteiger | Thema: Klassifikation mit tabellarischen Daten (scikit-learn)

## 🎯 Projektziel
Sage anhand von 13 klinischen Merkmalen vorher, ob eine Person an einer Herzerkrankung leidet, und lerne, warum **Accuracy allein bei Klassifikationsproblemen irreführend sein kann**.

## 📚 Das lernst du
- Klassifikation im Vergleich zur Regression: gleicher Ablauf, aber andere Zielvariable und Metriken
- `LogisticRegression` und `KNeighborsClassifier`
- Warum kNN skalierte Merkmale benötigt und wie `StandardScaler` sicher in einer Pipeline eingesetzt wird
- Die Konfusionsmatrix mit TP, FP, TN und FN sowie die Bedeutung dieser Fehler im medizinischen Kontext
- Precision, Recall und F1 sowie die Wahl einer geeigneten Zielmetrik
- Vorhergesagte Wahrscheinlichkeiten mit `predict_proba` im Vergleich zu festen Klassenvorhersagen

## 🗂️ Beschreibung des Datensatzes
**UCI Heart Disease (Cleveland)** enthält 303 Personen und 13 Merkmale wie Alter, Geschlecht, Art der Brustschmerzen, Ruheblutdruck, Cholesterin und maximale Herzfrequenz. Die Zielvariable beschreibt das Vorliegen einer Herzerkrankung: 0 bedeutet nein, 1 bis 4 bedeuten ja. Du wandelst sie in eine binäre Variable um.

Der folgende Code im ersten Notebook-Abschnitt lädt die Daten herunter:

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"]
df = pd.read_csv(url, names=cols, na_values="?")
```

Falls der UCI-Server nicht erreichbar ist, suche nach „heart disease cleveland processed“. Es gibt mehrere Spiegelserver; speichere die Datei unter `data/raw/`.

## 🚀 Erste Schritte
```bash
cd 04_beginner_heart_disease_classification
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/solution_template.ipynb
```

## 📋 Schritt-für-Schritt-Anleitung
1. **Laden und bereinigen** — Lade die Daten, prüfe die Anzahl fehlender Werte, entferne die wenigen Zeilen mit fehlenden Werten in `ca` oder `thal` und binarisiere das Ziel mit `(df["target"] > 0).astype(int)`. *Warum:* Die ursprüngliche Skala 0–4 vermischt Schweregrad und Vorhandensein der Erkrankung; zunächst soll eine klare Ja-Nein-Frage beantwortet werden.
2. **Klassenverteilung untersuchen** — Verwende `value_counts(normalize=True)`. *Warum:* Sind 54 % der Personen gesund, erreicht ein Modell, das immer „gesund“ vorhersagt, bereits 54 % Accuracy. Das ist deine Baseline.
3. **Aufteilen** — Erzeuge Trainings- und Testdaten mit `stratify=y`. *Warum:* Dadurch bleibt das Verhältnis erkrankter und gesunder Personen in beiden Datensätzen gleich, was bei nur etwa 300 Zeilen wichtig ist.
4. **Logistische Regression** — Trainiere das Modell, erzeuge Vorhersagen und berechne die Accuracy. Berechne anschließend sofort die **Konfusionsmatrix**. *Warum:* Accuracy fasst TP, FP, TN und FN zu einer Zahl zusammen und verbirgt den Unterschied zwischen einer übersehenen Erkrankung und einem falschen Alarm.
5. **Precision, Recall und F1** — Verwende `classification_report`. Beantworte schriftlich, ob im medizinischen Screening ein falsch positives oder falsch negatives Ergebnis schwerer wiegt und welche Metrik daher optimiert werden sollte.
6. **kNN mit Pipeline** — Verwende `make_pipeline(StandardScaler(), KNeighborsClassifier())` und teste k = 3, 5 und 11. *Warum skalieren:* kNN berechnet Abstände; ohne Skalierung dominiert Cholesterin mit Werten um 250 das binäre Geschlecht. *Warum eine Pipeline:* Sie stellt sicher, dass der Scaler nur mit Trainingsdaten angepasst wird und verhindert Informationslecks aus den Testdaten.
7. **Wahrscheinlichkeiten** — `predict_proba` liefert P(Erkrankung). Gib die fünf Personen aus, bei denen das Modell am unsichersten ist, also deren Wahrscheinlichkeit am nächsten bei 0,5 liegt. In der Praxis wären dies Kandidaten für weitere Untersuchungen.
8. **Vergleichen und abschließen** — Welches Modell erzielt den besseren Recall? Formuliere eine Empfehlung in drei Sätzen.

## ✅ Checkliste zum Abschluss
- [ ] Ich habe die Zielvariable binarisiert und kann erklären, warum.
- [ ] Ich kenne die Accuracy eines Modells, das nichts lernt.
- [ ] Ich kann alle vier Felder einer Konfusionsmatrix benennen.
- [ ] Ich kann Precision und Recall im Kontext dieses Datensatzes erklären.
- [ ] Ich habe eine Pipeline verwendet und kann erklären, welches Datenleck sie verhindert.
- [ ] Ich habe die Personen mit Vorhersagen am nächsten bei 0,5 gefunden.
- [ ] Ich habe mein abschließendes Modell anhand einer begründeten Metrik und nicht nur anhand der Accuracy gewählt.

## 💡 Hinweise und Tipps
- `ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)` zeichnet die Matrix in einer Zeile.
- `classification_report(y_test, y_pred)` gibt Precision, Recall und F1 für jede Klasse aus.
- Finde die unsichersten Fälle mit `proba = model.predict_proba(X_test)[:, 1]` und sortiere anschließend nach `np.abs(proba - 0.5)`.
- Falls `LogisticRegression` vor fehlender Konvergenz warnt, setze `max_iter=1000`.
- Erhöhe k für kNN und beobachte, wie die Entscheidung glatter wird: Ein kleines k ist flexibel und neigt zu Overfitting, ein großes k ist starr und neigt zu Underfitting. Dies entspricht der Abwägung bei der Baumtiefe in Projekt 03.

## 🔗 Weiterführende Informationen
- [scikit-learn: Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Google ML Crash Course: Klassifikationsmetriken](https://developers.google.com/machine-learning/crash-course/classification)
- [UCI Heart Disease-Datensatz](https://archive.ics.uci.edu/dataset/45/heart+disease)
