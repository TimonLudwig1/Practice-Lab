# Project 03 (final) — Spam filter with naive Bayes on real SMS data

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format:** Jupyter notebook (`spam_filter.ipynb`) + download script.
**Why this format?** This is a data project: load, explore, model, evaluate — exactly the workflow notebooks were made for, and at the same time a preview of the data science and ML modules.

**Data: real.** We use the **SMS Spam Collection** (UCI Machine Learning Repository, Almeida & Hidalgo 2011): 5,574 real text messages, about 13 % of them spam. Real data are chosen deliberately here — the practical connection (spam filtering was *the* success story of naive Bayes) is precisely the point of this final project. The file (about 470 KB) is fetched by script and not committed to the repository.

## Goal

You translate Bayes' rule from the script into a working spam classifier with your own hands (including Laplace smoothing and log probabilities), evaluate it with the right metrics and compare it with scikit-learn. Reference result: **about 98.6 % accuracy** — identical to the scikit-learn implementation.

## Prior knowledge

- Sections 2.4 (Bayes, naive Bayes) and 2.5 (supervised learning, generalisation) of the module script
- Projects 01 and 02
- As a final project you write **all the code yourself** — the notebook only contains the step-by-step instructions, formulas and self-checks (no pre-written lines of code). You learn pandas systematically only in module 02; the few necessary pandas calls (e.g. `read_csv`) are therefore described as **prose hints** in the tasks so that you can reconstruct them.

## Tasks

1. **Fetch the data** (once, in the folder `03-final`, venv active):
   ```bash
   python datasets/download_data.py
   ```
2. Open the notebook `spam_filter.ipynb` and work through it from the top. Every code cell is **empty** — you fill it in yourself; the markdown cell above it says what to do (with formulas, hints and self-checks). The route: load and explore the data → train/test split → `tokenize(text)` → naive Bayes by hand (`train`, `log_word_prob`, `log_posterior`, `classify`) → evaluation → learned words → comparison with scikit-learn.
3. Run the evaluation and **interpret** it: look closely at the confusion matrix — which kind of error is worse for a spam filter, and how does your model fare on it?
4. Run the section "What did the model learn?" and the scikit-learn comparison.

## What should work in the end

- The tokenizer mini test prints `True`.
- The example messages are correctly recognised as `spam` / `ham`.
- Accuracy on the test set **above 98 %** (reference: 0.9857; naive "always ham" baseline: 0.8664).
- Your model and `MultinomialNB` are practically neck and neck.

## Solution

Fully executed reference solution: [`solution/solution.ipynb`](solution/solution.ipynb) — try it yourself first. There you also see all reference outputs (confusion matrix, top spam words, etc.).

---
---

# Projekt 03 (final) — Spamfilter mit Naive Bayes auf echten SMS-Daten (deutsche Fassung)

**Format:** Jupyter Notebook (`spam_filter.ipynb`) + Download-Skript.
**Warum dieses Format?** Das ist ein Daten-Projekt: laden, explorieren, modellieren, auswerten — genau der Workflow, für den Notebooks gemacht sind, und zugleich eine Vorschau auf die Data-Science- und ML-Module.

**Daten: echt.** Wir nutzen die **SMS Spam Collection** (UCI Machine Learning Repository, Almeida & Hidalgo 2011): 5.574 echte SMS, davon ~13 % Spam. Echte Daten sind hier bewusst gewählt — der Praxisbezug (Spamfilterung war *die* Erfolgsgeschichte von Naive Bayes) ist genau der Punkt dieses Abschlussprojekts. Die Datei (~470 KB) wird per Skript geladen und nicht ins Repo committet.

## Ziel

Du übersetzt die Bayes-Regel aus dem Skript eigenhändig in einen funktionierenden Spam-Klassifikator (inkl. Laplace-Glättung und Log-Wahrscheinlichkeiten), bewertest ihn mit den richtigen Metriken und vergleichst ihn mit scikit-learn. Referenzergebnis: **~98,6 % Accuracy** — identisch mit der scikit-learn-Implementierung.

## Vorwissen

- Modul-Skript Abschnitte 2.4 (Bayes, Naive Bayes) und 2.5 (Supervised Learning, Generalisierung)
- Projekte 01 und 02
- Als Final-Projekt schreibst du **allen Code selbst** — das Notebook enthält nur noch die Schritt-für-Schritt-Anleitung, Formeln und Selbstchecks (keine vorgeschriebenen Code-Zeilen). pandas lernst du systematisch erst in Modul 02; die wenigen nötigen pandas-Aufrufe (z. B. `read_csv`) sind darum als **Prosa-Hinweis** in den Aufgaben beschrieben, sodass du sie nachbauen kannst.

## Aufgaben

1. **Daten holen** (einmalig, im Ordner `03-final`, venv aktiv):
   ```bash
   python datasets/download_data.py
   ```
2. Notebook `spam_filter.ipynb` öffnen und von oben durcharbeiten. Jede Code-Zelle ist **leer** — du füllst sie selbst; die Markdown-Zelle darüber sagt jeweils, was zu tun ist (mit Formeln, Hinweisen und Selbstchecks). Der Weg: Daten laden & explorieren → Train/Test-Split → `tokenisiere(text)` → Naive Bayes von Hand (`trainiere`, `log_wort_wkt`, `log_posterior`, `klassifiziere`) → Auswertung → gelernte Wörter → scikit-learn-Vergleich.
3. Auswertung ausführen und **interpretieren**: Schau dir die Confusion Matrix genau an — welche Fehlerart ist bei einem Spamfilter schlimmer, und wie schlägt sich dein Modell da?
4. Abschnitt „Was hat das Modell gelernt?" und den scikit-learn-Vergleich ausführen.

## Was am Ende funktionieren soll

- Der Tokenizer-Mini-Test gibt `True` aus.
- Die Beispiel-SMS werden korrekt als `spam` / `ham` erkannt.
- Accuracy auf dem Test-Set **> 98 %** (Referenz: 0,9857; naive „immer ham"-Baseline: 0,8664).
- Dein Modell und `MultinomialNB` liegen praktisch gleichauf.

## Lösung

Vollständig ausgeführte Musterlösung: [`solution/solution.ipynb`](solution/solution.ipynb) — erst selbst probieren. Dort siehst du auch alle Referenz-Outputs (Confusion Matrix, Top-Spam-Wörter etc.).
