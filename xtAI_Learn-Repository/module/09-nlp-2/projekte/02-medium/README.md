# Projekt 02 (medium) — Self-Attention & ein Mini-Transformer von Hand

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
  loesung/           # vollständige, getestete Musterlösung
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

Vollständig in [`loesung/`](loesung/) (identische Struktur, alle Tests grün). Erst selbst
versuchen — die Root-Dateien werfen `NotImplementedError`, bis du die TODOs füllst.

> **Referenz** (fester Seed): 8/8 Tests grün; Transformer 1.000, BoW 0.503; Training
> ~1 min auf CPU, Sekunden auf MPS/CUDA.
