# Project 02 (medium) — Self-attention & a mini transformer by hand

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 09 — NLP 2** · Format: **Python project** (several modules + test suite)

## Why this format?

You only understand attention once you multiply the matrices yourself. As a
**codebase** (instead of a notebook) you cleanly separate the building blocks —
attention, encoder block, model, data, training, tests — and implement exactly the
parts that libraries (`nn.MultiheadAttention`, `nn.TransformerEncoderLayer`)
otherwise hide. The **test suite** checks your implementation against PyTorch's
reference and against the mathematical properties (softmax normalization, masking).

## Goal

You build the **core computation of the transformer from scratch** and use it to
train a small encoder-only classifier. Concretely:

- **Scaled dot-product attention** $\operatorname{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}\big)V$
  including masking — checked against `F.scaled_dot_product_attention`.
- **Multi-head attention**: projection into $h$ heads, parallel attention, merging.
- **Sinusoidal positional encoding** and a **transformer encoder block**
  (MHA → residual → LayerNorm → FFN → residual → LayerNorm).
- **The aha moment (connecting to project 01):** a synthetic *negation* dataset in
  which the label depends on the interaction "not + polarity word". A **unigram BoW
  model fails** (test accuracy ≈ 0.50, pure chance), because no *single* word
  reveals the label — the **transformer solves it** (≈ 1.00), because attention can
  link the negator with the following word. Exactly the "not … good" limit from
  project 01, now cracked.

## Prior knowledge

- **Script** part 4 (self-attention, multi-head, positional encoding, encoder block).
- **PyTorch** (`nn.Module`, `nn.Linear`, broadcasting, `view`/`transpose`).
- Project 01 (embeddings, padding masks, training loop).

## Project structure

```
02-medium/
  attention.py       # SDPA + MultiHeadAttention            <- YOU (task 1 + 2)
  transformer.py     # positional encoding + encoder block   <- YOU (task 3)
                     #   (TransformerClassifier is given)
  data.py            # synthetic negation dataset             (given)
  train.py           # training + BoW comparison + attention  (given)
  test_attention.py  # test suite (8 tests)                   (given)
  solution/          # complete, tested reference solution
```

## Assignment

Little is given, much is up to you — the hard parts are yours. Every function has a
`# TODO` block listing the necessary steps as bullet points (only *inspiration*, no
finished code).

1. **`scaled_dot_product_attention`** in `attention.py`: the formula including
   masking (`-inf` before the softmax) and the softmax over the key axis.
2. **`MultiHeadAttention.forward`** in `attention.py`: project, split into heads
   (helper given), attention per head, merge, output projection.
3. **`sinusoidal_positional_encoding`** and **`TransformerEncoderBlock.forward`** in
   `transformer.py`: the sine/cosine encoding and the two sublayers with residual +
   LayerNorm.

**How to proceed:**

```bash
source ../../../../.venv/bin/activate
python test_attention.py     # red -> fill in the TODOs -> all 8 tests green
python train.py              # trains the transformer, compares against BoW
```

## What should work in the end

- `python test_attention.py` → **all 8 tests green** (SDPA == PyTorch, softmax
  normalization, causal & padding masking, MHA shapes, positional encoding, encoder
  block shape, padding invariance of the classifier).
- `python train.py` → transformer **test accuracy ≈ 1.00**, unigram BoW **≈ 0.50**;
  at the end a small attention matrix in which the tokens look strongly at the
  **negator**.

## Reference solution

Complete in [`solution/`](solution/) (identical structure, all tests green). Try it
yourself first — the root files raise `NotImplementedError` until you fill in the
TODOs.

> **Reference** (fixed seed): 8/8 tests green; transformer 1.000, BoW 0.503;
> training ~1 min on CPU, seconds on MPS/CUDA.

---

# Projekt 02 (medium) — Self-Attention & ein Mini-Transformer von Hand (deutsche Fassung)

**Modul 09 — NLP 2** · Format: **Python-Projekt** (mehrere Module + Testsuite)

## Warum dieses Format?

Attention versteht man nur, wenn man die Matrizen selbst multipliziert. Als **Codebasis**
(statt Notebook) trennst du sauber die Bausteine — Attention, Encoder-Block, Modell,
Daten, Training, Tests — und implementierst genau die Teile, die Bibliotheken (`nn.
MultiheadAttention`, `nn.TransformerEncoderLayer`) sonst verstecken. Die **Testsuite**
prüft deine Implementierung gegen PyTorchs Referenz und gegen die mathematischen
Eigenschaften (Softmax-Normierung, Maskierung).

## Ziel

Du baust die **Kernrechnung des Transformers von Grund auf** und trainierst damit einen
kleinen Encoder-only-Klassifikator. Konkret:

- **Scaled-Dot-Product-Attention** $\operatorname{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}\big)V$
  inkl. Maskierung — geprüft gegen `F.scaled_dot_product_attention`.
- **Multi-Head-Attention**: Projektion in $h$ Köpfe, parallele Attention, Zusammenführen.
- **Sinusoidale Positional Encoding** und ein **Transformer-Encoder-Block**
  (MHA → Residual → LayerNorm → FFN → Residual → LayerNorm).
- **Der Aha-Moment (Anschluss an Projekt 01):** ein synthetischer *Negations*-Datensatz,
  bei dem das Label von der Interaktion „not + Polaritätswort" abhängt. Ein
  **Unigramm-BoW-Modell scheitert** (Test-Acc ≈ 0.50, reiner Zufall), weil kein
  *einzelnes* Wort das Label verrät — der **Transformer löst es** (≈ 1.00), weil
  Attention den Negator mit dem folgenden Wort verknüpfen kann. Genau die „not … good"-
  Grenze aus Projekt 01, jetzt geknackt.

## Vorwissen

- **Skript** Teil 4 (Self-Attention, Multi-Head, Positional Encoding, Encoder-Block).
- **PyTorch** (`nn.Module`, `nn.Linear`, Broadcasting, `view`/`transpose`).
- Projekt 01 (Embeddings, Padding-Masken, Trainingsschleife).

## Projektstruktur

```
02-medium/
  attention.py       # SDPA + MultiHeadAttention           <- DU (Aufgabe 1 + 2)
  transformer.py     # Positional Encoding + Encoder-Block  <- DU (Aufgabe 3)
                     #   (TransformerClassifier vorgegeben)
  data.py            # synthetischer Negations-Datensatz     (vorgegeben)
  train.py           # Training + BoW-Vergleich + Attention  (vorgegeben)
  test_attention.py  # Testsuite (8 Tests)                   (vorgegeben)
  solution/           # vollständige, getestete Musterlösung
```

## Aufgabenstellung

Wenig Vorgabe, viel Eigenleistung — die schweren Teile sind deine. Jede Funktion hat
einen `# TODO`-Block mit den nötigen Schritten als Stichpunkte (nur *Inspiration*, kein
fertiger Code).

1. **`scaled_dot_product_attention`** in `attention.py`: die Formel inkl. Maskierung
   (`-inf` vor dem Softmax) und Softmax über die Key-Achse.
2. **`MultiHeadAttention.forward`** in `attention.py`: projizieren, in Köpfe splitten
   (Helfer vorgegeben), Attention pro Kopf, zusammenführen, Ausgangsprojektion.
3. **`sinusoidal_positional_encoding`** und **`TransformerEncoderBlock.forward`** in
   `transformer.py`: die Sinus/Cosinus-Codierung und die beiden Sub-Schichten mit
   Residual + LayerNorm.

**Vorgehen:**

```bash
source ../../../../.venv/bin/activate
python test_attention.py     # rot -> fülle die TODOs -> alle 8 Tests grün
python train.py              # trainiert den Transformer, vergleicht mit BoW
```

## Was am Ende funktionieren soll

- `python test_attention.py` → **alle 8 Tests grün** (SDPA == PyTorch, Softmax-Normierung,
  Causal- & Padding-Maskierung, MHA-Formen, Positional Encoding, Encoder-Block-Form,
  Padding-Invarianz des Classifiers).
- `python train.py` → Transformer **Test-Acc ≈ 1.00**, Unigramm-BoW **≈ 0.50**; am Ende
  eine kleine Attention-Matrix, in der die Tokens stark auf den **Negator** schauen.

## Musterlösung

Vollständig in [`solution/`](solution/) (identische Struktur, alle Tests grün). Erst selbst
versuchen — die Root-Dateien werfen `NotImplementedError`, bis du die TODOs füllst.

> **Referenz** (fester Seed): 8/8 Tests grün; Transformer 1.000, BoW 0.503; Training
> ~1 min auf CPU, Sekunden auf MPS/CUDA.
