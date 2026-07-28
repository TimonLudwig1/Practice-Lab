# 11 — NLP: Sentiment Classification on Movie Reviews

Difficulty: 🟠 Medium-Hard | Topic: NLP / Text Classification

## 🎯 Project Goal
Build a sentiment classifier for movie reviews using classical NLP (no deep learning in this project), including a proper text preprocessing pipeline and an honest error analysis of where bag-of-words approaches break down.

## 📊 Dataset + Evaluation Metric
- **Dataset:** Large Movie Review Dataset (IMDB), 50,000 labeled reviews (25k train / 25k test, balanced). Download: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz — extracting and assembling the folder-of-text-files structure into a DataFrame is part of the task.
- **Evaluation metric:** Accuracy on the official 25k test split (it's balanced, so accuracy is fine here). Report F1 alongside.

## 🏁 Success Criteria
- ≥ 88% test accuracy with a non-deep-learning approach
- The official train/test split respected; no test data touched before final evaluation
- An error analysis section: ≥5 misclassified reviews examined, with a hypothesis per failure mode (negation, sarcasm, mixed sentiment, …)
- Reusable preprocessing/training code factored into `src/`, not just notebook cells

Relevant techniques (look them up yourself): TF-IDF, n-grams, `sklearn` text vectorizers, linear classifiers for sparse data, regularization strength tuning, learning curves.

---

# Deutsche Übersetzung

# 11 — NLP: Sentimentklassifikation von Filmrezensionen

Schwierigkeit: 🟠 Mittel bis anspruchsvoll | Thema: NLP und Textklassifikation

## 🎯 Projektziel
Erstelle mit klassischen NLP-Verfahren einen Sentimentklassifikator für Filmrezensionen, in diesem Projekt ohne Deep Learning. Dazu gehören eine saubere Textvorverarbeitungspipeline und eine ehrliche Fehleranalyse der Grenzen von Bag-of-Words-Ansätzen.

## 📊 Datensatz und Bewertungsmetrik
- **Datensatz:** Large Movie Review Dataset von IMDB mit 50.000 beschrifteten Rezensionen, ausgeglichen auf 25.000 Trainings- und 25.000 Testbeispiele. Download: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. Das Entpacken und Zusammenführen der Textdateien in einen DataFrame ist Teil der Aufgabe.
- **Bewertungsmetrik:** Accuracy auf dem offiziellen Testsatz mit 25.000 Beispielen. Da die Klassen ausgeglichen sind, ist Accuracy hier geeignet; berichte zusätzlich den F1-Wert.

## 🏁 Erfolgskriterien
- Mindestens 88 % Test-Accuracy mit einem Ansatz ohne Deep Learning
- Einhaltung der offiziellen Trainings-/Testaufteilung; keine Nutzung der Testdaten vor der abschließenden Bewertung
- Fehleranalyse von mindestens fünf falsch klassifizierten Rezensionen mit einer Hypothese je Fehlertyp, etwa Negation, Sarkasmus oder gemischte Stimmung
- Wiederverwendbarer Code für Vorverarbeitung und Training unter `src/` statt ausschließlich in Notebook-Zellen

Relevante Verfahren zum selbstständigen Nachschlagen: TF-IDF, n-Gramme, Textvektorizierer aus `sklearn`, lineare Klassifikatoren für dünn besetzte Daten, Abstimmung der Regularisierungsstärke und Lernkurven.
