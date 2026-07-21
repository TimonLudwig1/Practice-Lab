# Project 01 (basic) — Sentiment classification with an LSTM

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Module 09 — NLP 2** · Format: **Jupyter notebook** (`sentiment_lstm.ipynb`)

## Why this format?

You understand a neural sequence model once you *see* the whole path: raw sentences
→ vocabulary → integer tensors → embeddings → LSTM → loss → prediction. A notebook
connects code, explanation and intermediate outputs (vocabulary size, batch shapes,
epoch-by-epoch accuracy) at exactly the places where it matters didactically —
ideal for this guided start with plenty of instruction.

## Goal

You build an **LSTM classifier** in PyTorch that sorts short reviews into *positive*
or *negative*, and you measure it honestly against a **bag-of-words baseline**. The
transition from module 08 (counting models) to module 09 (neural models) becomes
concrete:

- **Embeddings** instead of one-hot: learned, dense word vectors (`nn.Embedding`).
- **Recurrence**: the LSTM reads the sentence word by word and summarizes it in a
  hidden state $\mathbf{h}_T$.
- **Practical craft**: variable sequence lengths via **padding** +
  `pack_padded_sequence`, `BCEWithLogitsLoss`, a clean training loop.
- **The aha moment**: on this small dataset of very *short* sentences the **BoW
  baseline wins** (~0.86) against the LSTM (~0.79). Neural models are data-hungry
  and overfit here — more capacity does not automatically mean more accuracy.
  Exactly this insight motivates attention/transformers (projects 02/03) *and*
  larger amounts of data.

## Prior knowledge

- **Script** parts 1–3 (embeddings, RNN/LSTM, vanishing gradient / long-term
  dependencies).
- **PyTorch from module 05** (`nn.Module`, optimizer, `loss.backward()`,
  `optimizer.step()`).
- Python: `re`, `collections.Counter`.

## Setup

Requires `torch`, `scikit-learn`, `numpy` (all in the repo `requirements.txt`). The
first code cell downloads the dataset automatically into `datasets/` (see below) and
caches it; `.gitignore` keeps it out of the repository.

```bash
source ../../../../.venv/bin/activate
jupyter lab      # or open sentiment_lstm.ipynb in VS Code, kernel = repo .venv
```

Training runs in **a few minutes on the CPU** (seconds on Apple MPS/CUDA — the
notebook picks the device automatically).

## Dataset

**UCI *Sentiment Labelled Sentences*** — 3000 short, binary-labelled sentences from
real reviews on **IMDb, Amazon and Yelp** (1000 each, exactly balanced). Real data,
but tiny enough for CPU training. Source: UCI ML Repository (dataset 331),
downloaded automatically in the notebook.

> Why real and not synthetic? Sentiment lives off real phrasing (irony, emphasis,
> word choice). A real, small, cleanly balanced dataset illustrates the BoW-vs-LSTM
> comparison more honestly than generated sentences.

## Assignment (step by step)

**Part A** (download, split, tokenization, vocabulary, `Dataset`/`DataLoader` with
padding) is **given**. After that three tasks — each marked with `# TODO`:

1. **Assemble the LSTM model** (`forward`): embedding lookup →
   `pack_padded_sequence` → `nn.LSTM` → last hidden state $\mathbf{h}_T$ → dropout →
   linear onto a single logit. Understand *why* we use `h_n` (not `output`) and
   packing.
2. **Training step** (`train_one_epoch`): `zero_grad` → forward →
   `BCEWithLogitsLoss` → `backward` → `step`.
3. **Evaluate & compare**: put the BoW baseline (given) up against the LSTM and
   write a `predict_sentiment` function for your own sentences.

At the end a short written **reflection part** (4 questions).

## What should work in the end

- A trained LSTM with epoch-by-epoch output (train accuracy rising towards ~0.99,
  test accuracy ~0.78–0.80 → visible overfitting).
- BoW+LogReg baseline ~0.85–0.86 → **beats** the LSTM (the learning point).
- `predict_sentiment("...")` delivers plausible probabilities for your own sentences.

## Reference solution

A fully filled-in, **executed** notebook is in
[`solution/sentiment_lstm_solution.ipynb`](solution/sentiment_lstm_solution.ipynb).
Try it yourself first — the stub cells raise `NotImplementedError` until you fill
them. The reference answers to the reflection questions are at the end of that
solution notebook.

---

# Projekt 01 (basic) — Sentiment-Klassifikation mit einem LSTM (deutsche Fassung)

**Modul 09 — NLP 2** · Format: **Jupyter Notebook** (`sentiment_lstm.ipynb`)

## Warum dieses Format?

Ein neuronales Sequenzmodell versteht man, wenn man den ganzen Weg *sieht*: rohe Sätze
→ Vokabular → Integer-Tensoren → Embeddings → LSTM → Loss → Vorhersage. Ein Notebook
verbindet Code, Erklärung und Zwischenausgaben (Vokabulargröße, Batch-Formen,
Epoch-für-Epoch-Genauigkeit) an genau den Stellen, an denen es didaktisch zählt —
ideal für diesen geführten Einstieg mit viel Anleitung.

## Ziel

Du baust einen **LSTM-Klassifikator** in PyTorch, der kurze Reviews als *positiv* oder
*negativ* einordnet, und misst ihn ehrlich gegen eine **Bag-of-Words-Baseline**. Der
Übergang von Modul 08 (Zähl-Modelle) zu Modul 09 (neuronale Modelle) wird konkret:

- **Embeddings** statt One-Hot: gelernte, dichte Wortvektoren (`nn.Embedding`).
- **Rekurrenz**: das LSTM liest den Satz Wort für Wort und fasst ihn in einem
  versteckten Zustand $\mathbf{h}_T$ zusammen.
- **Praxis-Handwerk**: variable Sequenzlängen per **Padding** + `pack_padded_sequence`,
  `BCEWithLogitsLoss`, eine saubere Trainingsschleife.
- **Der Aha-Moment**: Auf diesem kleinen Datensatz aus sehr *kurzen* Sätzen **gewinnt
  die BoW-Baseline** (~0.86) gegen das LSTM (~0.79). Neuronale Modelle sind
  datenhungrig und overfitten hier — mehr Kapazität heißt nicht automatisch mehr
  Genauigkeit. Genau diese Erkenntnis motiviert Attention/Transformer (Projekt 02/03)
  *und* größere Datenmengen.

## Vorwissen

- **Skript** Teil 1–3 (Embeddings, RNN/LSTM, Vanishing Gradient / Long-Term-Dependencies).
- **PyTorch aus Modul 05** (`nn.Module`, Optimizer, `loss.backward()`, `optimizer.step()`).
- Python: `re`, `collections.Counter`.

## Setup

Benötigt `torch`, `scikit-learn`, `numpy` (alle in der Repo-`requirements.txt`). Die
erste Code-Zelle lädt den Datensatz automatisch nach `datasets/` (siehe unten) und cached
ihn; per `.gitignore` wird er nicht eingecheckt.

```bash
source ../../../../.venv/bin/activate
jupyter lab      # oder sentiment_lstm.ipynb in VS Code öffnen, Kernel = Repo-.venv
```

Training läuft in **wenigen Minuten auf der CPU** (auf Apple-MPS/CUDA in Sekunden — das
Notebook wählt das Device automatisch).

## Datensatz

**UCI *Sentiment Labelled Sentences*** — 3000 kurze, binär gelabelte Sätze aus echten
Reviews von **IMDb, Amazon und Yelp** (je 1000, exakt balanciert). Echte Daten, aber
winzig genug fürs CPU-Training. Quelle: UCI ML Repository (Dataset 331), automatischer
Download im Notebook.

> Warum echt und nicht synthetisch? Sentiment lebt von echter Formulierung (Ironie,
> Betonung, Wortwahl). Ein realer, kleiner, sauber balancierter Datensatz illustriert
> den BoW-vs-LSTM-Vergleich ehrlicher als generierte Sätze.

## Aufgabenstellung (Schritt für Schritt)

**Teil A** (Download, Split, Tokenisierung, Vokabular, `Dataset`/`DataLoader` mit
Padding) ist **vorgegeben**. Danach drei Aufgaben — jeweils an `# TODO` markiert:

1. **LSTM-Modell zusammensetzen** (`forward`): Embedding-Lookup → `pack_padded_sequence`
   → `nn.LSTM` → letzten versteckten Zustand $\mathbf{h}_T$ → Dropout → Linear auf einen
   Logit. Verstehe, *warum* wir `h_n` (nicht `output`) und Packing nutzen.
2. **Trainingsschritt** (`train_one_epoch`): `zero_grad` → Forward → `BCEWithLogitsLoss`
   → `backward` → `step`.
3. **Auswerten & Vergleichen**: BoW-Baseline (vorgegeben) gegen das LSTM stellen und
   eine `predict_sentiment`-Funktion für eigene Sätze schreiben.

Zum Schluss ein kurzer schriftlicher **Reflexionsteil** (4 Fragen).

## Was am Ende funktionieren soll

- Ein trainiertes LSTM mit Epoch-für-Epoch-Ausgabe (Train-Acc steigt gegen ~0.99,
  Test-Acc ~0.78–0.80 → sichtbares Overfitting).
- BoW+LogReg-Baseline ~0.85–0.86 → **schlägt** das LSTM (der Lernpunkt).
- `predict_sentiment("...")` liefert plausible Wahrscheinlichkeiten für eigene Sätze.

## Musterlösung

Voll ausgefülltes, **ausgeführtes** Notebook unter
[`solution/sentiment_lstm_solution.ipynb`](solution/sentiment_lstm_solution.ipynb). Erst
selbst probieren — die Stub-Zellen werfen `NotImplementedError`, bis du sie füllst.
