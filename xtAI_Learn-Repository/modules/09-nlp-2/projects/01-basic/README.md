# Projekt 01 (basic) — Sentiment-Klassifikation mit einem LSTM

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
