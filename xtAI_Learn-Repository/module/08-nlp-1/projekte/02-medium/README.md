# Projekt 02 (medium) — Textklassifikation: Naive Bayes vs. TF-IDF + LogReg

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
  loesung/         # vollständige Musterlösung
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

Vollständig in **`loesung/`** — alle Tests bestehen, `classify.py` läuft durch.
Erst nach eigenem Versuch ansehen.
