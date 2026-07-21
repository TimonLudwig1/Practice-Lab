# Project 02 (medium) — Text classification: naive Bayes vs. TF-IDF + logistic regression

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 08 — NLP 1** · Format: **Python project** (several modules + tests)

## Why this format?

Text classification is a small pipeline with clear building blocks — load the
data, tokenize, train a model, evaluate, compare with a second method. As a
**code base** (rather than a notebook) you see the separation of data, model and
evaluation, and you implement **naive Bayes from scratch** — exactly what stays
hidden inside libraries.

## Goal

You implement a **multinomial naive Bayes classifier by hand** and compare it
with **TF-IDF + logistic regression** (scikit-learn) on real text data (**20
newsgroups**, 4 categories). Along the way you learn:

- the generative NB model in **log space** with **Laplace smoothing** (part 2.1);
- the discriminative alternative (TF-IDF + logistic regression, parts 2.2/2.3);
- **clean evaluation** with precision/recall/F1 (macro/weighted) on a test set;
- the (surprisingly good) competitiveness of naive Bayes despite its "naive"
  independence assumption.

## Prior knowledge

- Part 2 of the script (naive Bayes, logistic regression, TF-IDF, precision/recall/F1).
- Python: `collections.Counter`, `math.log`, classes.
- The basics of scikit-learn (the pipeline is given).

## Project structure

```
02-medium/
  data.py          # load 20 newsgroups + the tokenizer  (given)
  naive_bayes.py   # MultinomialNaiveBayes               <- YOU: fit + predict_one
  classify.py      # training + comparison + report      (given, uses your NB)
  test_nb.py       # the test suite                      (given)
  solution/        # the complete reference solution
```

What is given: loading the data and tokenizing, the comparison and reporting
logic (including the sklearn pipeline) and the tests. **You implement the core**
— `MultinomialNaiveBayes.fit` and `.predict_one` (the log prior, the smoothed log
likelihoods, the argmax), marked with `TODO`.

## Tasks (step by step)

1. **`fit`**: determine the classes and the vocabulary; per class the word counts
   $C(w,c)$ and the number of documents; $\log P(c)$; the smoothed
   $\log P(w\mid c)=\log\frac{C(w,c)+\alpha}{\sum_{w'}C(w',c)+\alpha|V|}$.
2. **`predict_one`**: for every class $\log P(c)+\sum_{w\in\text{doc}}\log P(w\mid c)$
   (skipping OOV words); return the argmax class.

Test incrementally with `python test_nb.py` (a mini data set → a normalized
likelihood → the real data). Then run `python classify.py` for the full
comparison with the classification report.

## What should work in the end

```bash
source ../../../../.venv/bin/activate    # sklearn is needed; the 1st run downloads 20ng (about 14 MB)
python test_nb.py     # -> "All tests passed."
python classify.py    # -> reports for NB and TF-IDF+LogReg + a conclusion
```

Reference (4 categories): both reach about **0.88–0.89 accuracy / macro-F1**; the
hand-written naive Bayes is **competitive** (here even slightly ahead). The
classification report shows precision/recall/F1 per category.

> **Note:** the first call downloads 20 newsgroups and caches it in
> `~/scikit_learn_data` (not in the repository). After that everything works
> offline.

## Reflection (in writing, briefly)

1. Naive Bayes is **generative**, logistic regression **discriminative**. Explain
   the difference in terms of $P(d\mid c)$ vs. $P(c\mid d)$ and when each model
   has the advantage.
2. Why does one compute naive Bayes in **log space**? What would otherwise go
   wrong?
3. Why is **accuracy** alone fine here but misleading with strongly unbalanced
   classes — and what does macro-F1 measure differently from weighted-F1?
4. How would the result change if you did *not* remove the headers/footers
   (`remove=()`)? (Keyword: the model learns metadata instead of content — a
   classical data leakage mistake with 20 newsgroups.)

## Reference solution

Complete in **`solution/`** — all tests pass, `classify.py` runs through. Look
only after your own attempt.

---
---

# Projekt 02 (medium) — Textklassifikation: Naive Bayes vs. TF-IDF + LogReg (deutsche Fassung)

**Modul 08 — NLP 1** · Format: **Python-Projekt** (mehrere Module + Tests)

## Warum dieses Format?

Textklassifikation ist eine kleine Pipeline mit klaren Bausteinen — Daten laden,
tokenisieren, ein Modell trainieren, evaluieren, mit einer zweiten Methode
vergleichen. Als **Codebasis** (statt Notebook) siehst du die Trennung von Daten,
Modell und Auswertung und implementierst den **Naive Bayes von Grund auf** — genau
das, was in Bibliotheken verborgen bleibt.

## Ziel

Du implementierst einen **multinomialen Naive-Bayes-Klassifikator von Hand** und
vergleichst ihn mit einer **TF-IDF + logistischen Regression** (scikit-learn) auf
echten Textdaten (**20 Newsgroups**, 4 Kategorien). Dabei lernst du:

- das generative NB-Modell im **Log-Raum** mit **Laplace-Glättung** (Teil 2.1);
- die diskriminative Alternative (TF-IDF + LogReg, Teil 2.2/2.3);
- **saubere Evaluation** mit Precision/Recall/F1 (macro/weighted) auf einem Testset;
- die (überraschend gute) Konkurrenzfähigkeit von Naive Bayes trotz seiner
  „naiven" Unabhängigkeitsannahme.

## Vorwissen

- Skript Teil 2 (Naive Bayes, logistische Regression, TF-IDF, Precision/Recall/F1).
- Python: `collections.Counter`, `math.log`, Klassen.
- scikit-learn-Grundlagen (die Pipeline ist vorgegeben).

## Projektstruktur

```
02-medium/
  data.py          # 20-Newsgroups laden + Tokenizer   (vorgegeben)
  naive_bayes.py   # MultinomialNaiveBayes             <- DU: fit + predict_one
  classify.py      # Training + Vergleich + Report      (vorgegeben, nutzt dein NB)
  test_nb.py       # Testsuite                          (vorgegeben)
  solution/         # vollständige Musterlösung
```

Vorgegeben sind Datenladen/Tokenisierung, die Vergleichs- und Reporting-Logik
(inkl. der sklearn-Pipeline) und die Tests. **Du implementierst den Kern** —
`MultinomialNaiveBayes.fit` und `.predict_one` (Log-Prior, geglättete
Log-Likelihoods, argmax), markiert mit `TODO`.

## Aufgabenstellung (Schritt für Schritt)

1. **`fit`**: Klassen und Vokabular bestimmen; pro Klasse Wortzählungen $C(w,c)$
   und Dokumentzahl; $\log P(c)$; geglättete $\log P(w\mid c)=\log\frac{C(w,c)+\alpha}{\sum_{w'}C(w',c)+\alpha|V|}$.
2. **`predict_one`**: für jede Klasse $\log P(c)+\sum_{w\in\text{doc}}\log P(w\mid c)$
   (OOV-Wörter überspringen); die argmax-Klasse zurückgeben.

Teste inkrementell mit `python test_nb.py` (Mini-Datensatz → normierte
Likelihood → echte Daten). Führe dann `python classify.py` für den vollen
Vergleich mit dem Klassifikationsreport aus.

## Was am Ende funktionieren soll

```bash
source ../../../../.venv/bin/activate    # sklearn nötig; lädt beim 1. Mal 20ng (~14 MB)
python test_nb.py     # -> "Alle Tests bestanden."
python classify.py    # -> Reports für NB und TF-IDF+LogReg + Fazit
```

Referenz (4 Kategorien): beide erreichen **~0.88–0.89 Accuracy / macro-F1**; der
handgeschriebene Naive Bayes ist **konkurrenzfähig** (hier sogar leicht vorn). Der
Klassifikationsreport zeigt Precision/Recall/F1 pro Kategorie.

> **Hinweis:** Der erste Aufruf lädt 20 Newsgroups herunter und cached ihn in
> `~/scikit_learn_data` (nicht im Repo). Danach funktioniert alles offline.

## Reflexion (schriftlich, kurz)

1. Naive Bayes ist **generativ**, LogReg **diskriminativ**. Erkläre den Unterschied
   an $P(d\mid c)$ vs. $P(c\mid d)$ und wann welches Modell im Vorteil ist.
2. Warum rechnet man Naive Bayes im **Log-Raum**? Was ginge sonst schief?
3. Warum ist **Accuracy** allein hier okay, bei stark unbalancierten Klassen aber
   irreführend — und was misst macro-F1 anders als weighted-F1?
4. Wie würde sich das Ergebnis ändern, wenn du die Header/Footer *nicht* entfernst
   (`remove=()`)? (Stichwort: das Modell lernt Metadaten statt Inhalt — ein
   klassischer Data-Leakage-Fehler bei 20 Newsgroups.)

## Musterlösung

Vollständig in **`solution/`** — alle Tests bestehen, `classify.py` läuft durch.
Erst nach eigenem Versuch ansehen.
