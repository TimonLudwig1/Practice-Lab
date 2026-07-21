# Project 03 (final) — A character-level GPT from scratch

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 09 — NLP 2** · Format: **Python project** (free implementation, *no* code given)

## Why this format & this practical relevance?

The final project consolidates the whole module: you build a **decoder-only
transformer (GPT)** and train it **autoregressively** — the same model class that
stands behind GPT-2/3/4 and practically all modern LLMs, only small. On the
**character level** (instead of subword BPE), so that the whole pipeline runs in **a
few minutes** without external tokenizers and without a GPU cluster, and so that you
write *every* building block yourself. This is the direct, honest "build GPT from
scratch" practical relevance: in the end *your* model generates new text in the style
of the training corpus.

**No code given.** This README is the specification. You design the files, classes
and training loop yourself and apply what you learned in project 01 (embeddings,
training) and 02 (self-attention, multi-head, encoder block, masks). A complete,
tested reference solution is in [`solution/`](solution/) — **build it yourself first.**

## Goal

Train a language model that predicts the **next character**, and use it for **text
generation**. In the end `generate(...)` should produce new, stylistically fitting
text from a starting context.

Dataset: **Tiny Shakespeare** (~1.1 MB, a single stream of text from Shakespeare
plays). Small enough for CPU/MPS training, large enough for recognizable structure
(speaker names in capitals, line breaks, dialogue). Automatic download, see
`datasets/`.

## Prior knowledge

- **Script** parts 4–5 (self-attention, multi-head, positional encoding,
  decoder/causal mask, autoregressive language model, GPT vs. BERT).
- **Project 02** — your attention building blocks are almost directly reusable; the
  only new thing is the **causal mask**.
- PyTorch (`nn.Module`, `nn.Embedding`, `AdamW`, `F.cross_entropy`,
  `torch.multinomial`).

## Technical specification

Build a **GPT** with this architecture (Radford et al. 2018; pre-LayerNorm as in GPT-2):

1. **Tokenization (character level):** a bijective mapping `stoi`/`itos` over all
   characters occurring in the corpus (vocabulary ≈ 65). `encode: str → [int]`,
   `decode` inverse.
2. **Batches:** random windows of length `block_size` from the text; input
   `x = data[i:i+T]`, target `y = data[i+1:i+1+T]` (shifted by one = "next
   character").
3. **Embedding:** token embedding $\mathbf{E}\in\mathbb{R}^{V\times d}$ **plus** a
   **learned** positional embedding $\mathbf{P}\in\mathbb{R}^{T_{\max}\times d}$
   (the sinusoidal alternative from project 02 is also allowed — justify your choice).
4. **Causal multi-head self-attention:** as in project 02, but with a **lower
   triangular mask**, so that position $t$ looks only at $\le t$:
   $$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d_k}}+M\Big)V,\qquad
   M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j> i\end{cases}$$
   This makes the model **autoregressive**: no token sees its future.
5. **Transformer block (pre-LN):**
   $x \leftarrow x + \text{Attn}(\text{LN}(x))$, then
   $x \leftarrow x + \text{FFN}(\text{LN}(x))$, with FFN = linear→GELU→linear (inner
   dimension $4d$). Stack $N$ blocks.
6. **Head:** final LayerNorm, then a linear projection onto $V$ logits. **Weight
   tying** (head weight = token embedding) is recommended.
7. **Loss:** cross entropy between logits $(B,T,V)$ and targets $(B,T)$ over *all*
   positions simultaneously. Optimizer: **AdamW**.
8. **Generation:** autoregressive — crop the context to `block_size`, take the logits
   of the last time step, divide by the **temperature**, optionally filter by
   **top-k**, `softmax` → `torch.multinomial` → append, repeat.

## Milestones (recommended order)

1. **Data & tokenizer**: download/cache, `encode`/`decode`, train/validation split
   (e.g. 90/10), `get_batch`.
2. **Model**: causal attention → block → GPT. Check the **shapes** (one forward pass
   with a dummy batch, logits $(B,T,V)$) and that the **initial loss** is
   ≈ $\ln V \approx 4.17$ (uniform prediction — a good sanity check!).
3. **Training**: a loop with periodic train/validation loss estimation.
4. **Generation**: sampling with temperature & top-k; observe how the text becomes
   more readable as the loss falls.
5. **Analysis** (written, see below).

## What should work in the end

- Training runs stably, the **validation loss falls** from ≈ 4.2 to **≈ 1.8**
  (reference setup: 4 blocks, 4 heads, $d=128$, `block_size=128`, ~3000 iterations;
  **~2 min on MPS**, a few minutes on the CPU).
- `generate` produces **recognizably Shakespeare-like** text: speaker names in
  capitals, line/dialogue structure, mostly real English words (at this loss still
  with invented words and without global meaning — exactly the expected state).
- A short **analysis part** (`ANALYSIS.md` or a docstring/notebook) that shows:

  1. **Sanity check:** why is the initial loss ≈ $\ln V$? (Show the calculation.)
  2. **Causality:** demonstrate that your mask works — e.g. that a change at
     position $t{+}k$ does **not** alter the logits at position $t$.
  3. **Temperature:** compare samples at $T=0.5$, $0.8$, $1.2$. What happens at the
     extremes (repetitive vs. incoherent) and why?
  4. **BPE outlook:** why do real LLMs work on **subword tokens** (BPE/WordPiece,
     script part 6) instead of characters? Name at least two concrete advantages.

## Assessment criteria (master's level)

- **Correctness of the causal mask** (no information leak from the future) — the
  central point of a decoder.
- A complete, clean architecture (pre-LN blocks, residuals, positional embedding,
  weight tying) and stable optimization.
- Working generation with temperature & top-k.
- A comprehensible, quantitatively supported analysis.

## Setup

```bash
source ../../../../.venv/bin/activate
# build your own implementation, then e.g.:  python gpt.py
```

Requires only `torch` (in the repo `requirements.txt`). The corpus is downloaded
into `datasets/` on the first run (see `datasets/README.md`) and is not checked in.

## Reference solution

[`solution/gpt.py`](solution/gpt.py) — a complete, tested GPT of ~0.8 million
parameters (causal attention, pre-LN blocks, weight tying, temperature/top-k
sampling). Reference run: validation loss 1.77 after 3000 iterations, ~1:50 min on
Apple MPS. **Build it yourself first.**

---

# Projekt 03 (final) — Ein Zeichen-Level-GPT von Grund auf (deutsche Fassung)

**Modul 09 — NLP 2** · Format: **Python-Projekt** (freie Umsetzung, *keine* Code-Vorgabe)

## Warum dieses Format & dieser Praxisbezug?

Das Abschlussprojekt konsolidiert das ganze Modul: Du baust einen **Decoder-only-
Transformer (GPT)** und trainierst ihn **autoregressiv** — dieselbe Modellklasse, die
hinter GPT-2/3/4 und praktisch allen modernen LLMs steht, nur klein. Auf **Zeichen-Ebene**
(statt Subword-BPE), damit die ganze Pipeline ohne externe Tokenizer und ohne GPU-Cluster
in **wenigen Minuten** läuft und du *jeden* Baustein selbst schreibst. Das ist der
direkte, ehrliche „build GPT from scratch"-Praxisbezug: am Ende generiert *dein* Modell
neuen Text im Stil des Trainingskorpus.

**Keine Code-Vorgabe.** Diese README ist die Spezifikation. Du entwirfst Dateien,
Klassen und Trainingsschleife selbst und wendest an, was du in Projekt 01 (Embeddings,
Training) und 02 (Self-Attention, Multi-Head, Encoder-Block, Masken) gelernt hast. Eine
vollständige, getestete Musterlösung liegt in [`solution/`](solution/) — **erst selbst
bauen.**

## Ziel

Trainiere ein Sprachmodell, das den **nächsten Buchstaben** vorhersagt, und nutze es zur
**Textgenerierung**. Am Ende soll `generate(...)` aus einem Startkontext neuen,
stilistisch passenden Text produzieren.

Datensatz: **Tiny Shakespeare** (~1,1 MB, ein einziger Textstrom aus Shakespeare-Stücken).
Klein genug fürs CPU/MPS-Training, groß genug für erkennbare Struktur (Sprechernamen in
Großbuchstaben, Zeilenumbrüche, Dialog). Automatischer Download, siehe `datasets/`.

## Vorwissen

- **Skript** Teil 4–5 (Self-Attention, Multi-Head, Positional Encoding, Decoder/kausale
  Maske, autoregressives Sprachmodell, GPT vs. BERT).
- **Projekt 02** — deine Attention-Bausteine sind fast direkt wiederverwendbar; neu ist
  nur die **kausale Maske**.
- PyTorch (`nn.Module`, `nn.Embedding`, `AdamW`, `F.cross_entropy`, `torch.multinomial`).

## Fachliche Spezifikation

Baue einen **GPT** mit dieser Architektur (Radford et al. 2018; Pre-LayerNorm wie GPT-2):

1. **Tokenisierung (Zeichen-Level):** bijektive Abbildung `stoi`/`itos` über alle im
   Korpus vorkommenden Zeichen (Vokabular ≈ 65). `encode: str → [int]`, `decode` invers.
2. **Batches:** Zufällige Fenster der Länge `block_size` aus dem Text; Eingabe `x = data[i:i+T]`,
   Ziel `y = data[i+1:i+1+T]` (um eins verschoben = „nächstes Zeichen").
3. **Einbettung:** Token-Embedding $\mathbf{E}\in\mathbb{R}^{V\times d}$ **plus** eine
   **gelernte** Positional-Embedding $\mathbf{P}\in\mathbb{R}^{T_{\max}\times d}$
   (Alternative sinusoidal aus Projekt 02 ist ebenfalls erlaubt — begründe die Wahl).
4. **Kausale Multi-Head-Self-Attention:** wie in Projekt 02, aber mit einer **unteren
   Dreiecks-Maske**, sodass Position $t$ nur auf $\le t$ schaut:
   $$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d_k}}+M\Big)V,\qquad
   M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j> i\end{cases}$$
   Das macht das Modell **autoregressiv**: kein Token sieht seine Zukunft.
5. **Transformer-Block (Pre-LN):**
   $x \leftarrow x + \text{Attn}(\text{LN}(x))$, danach
   $x \leftarrow x + \text{FFN}(\text{LN}(x))$, mit FFN = Linear→GELU→Linear (innere
   Dimension $4d$). Stapele $N$ Blöcke.
6. **Kopf:** finale LayerNorm, dann lineare Projektion auf $V$ Logits. **Weight tying**
   (Kopf-Gewicht = Token-Embedding) ist empfohlen.
7. **Verlust:** Cross-Entropy zwischen Logits $(B,T,V)$ und Zielen $(B,T)$ über *alle*
   Positionen gleichzeitig. Optimizer: **AdamW**.
8. **Generierung:** autoregressiv — Kontext auf `block_size` kürzen, Logits des letzten
   Zeitschritts nehmen, durch **Temperatur** teilen, optional **top-k** filtern,
   `softmax` → `torch.multinomial` → anhängen, wiederholen.

## Milestones (empfohlene Reihenfolge)

1. **Daten & Tokenizer**: Download/Cache, `encode`/`decode`, Train/Val-Split (z. B. 90/10),
   `get_batch`.
2. **Modell**: kausale Attention → Block → GPT. Prüfe die **Formen** (ein Forward mit
   Dummy-Batch, Logits $(B,T,V)$) und dass der **Anfangs-Loss** ≈ $\ln V \approx 4.17$
   ist (uniforme Vorhersage — ein guter Sanity-Check!).
3. **Training**: Schleife mit periodischer Train/Val-Loss-Schätzung.
4. **Generierung**: Sampling mit Temperatur & top-k; beobachte, wie der Text mit
   sinkendem Loss lesbarer wird.
5. **Analyse** (schriftlich, s. u.).

## Was am Ende funktionieren soll

- Training läuft stabil, **Val-Loss fällt** von ≈ 4.2 auf **≈ 1.8** (Referenz-Setup:
  4 Blöcke, 4 Köpfe, $d=128$, `block_size=128`, ~3000 Iterationen; **~2 min auf MPS**,
  einige Minuten auf CPU).
- `generate` produziert **erkennbar Shakespeare-artigen** Text: Sprechernamen in
  Großbuchstaben, Zeilen-/Dialogstruktur, überwiegend echte englische Wörter (bei diesem
  Loss noch mit Fantasiewörtern und ohne globalen Sinn — genau der erwartete Stand).
- Ein kurzer **Analyseteil** (`ANALYSIS.md` oder Docstring/Notebook), der belegt:

  1. **Sanity-Check:** Warum ist der Start-Loss ≈ $\ln V$? (Zeige die Rechnung.)
  2. **Kausalität:** Weise nach, dass deine Maske wirkt — z. B. dass eine Änderung an
     Position $t{+}k$ die Logits an Position $t$ **nicht** verändert.
  3. **Temperatur:** Vergleiche Proben bei $T=0.5$, $0.8$, $1.2$. Was passiert an den
     Extremen (repetitiv vs. inkohärent) und warum?
  4. **BPE-Ausblick:** Warum arbeiten echte LLMs auf **Subword-Tokens** (BPE/WordPiece,
     Skript Teil 6) statt auf Zeichen? Nenne mindestens zwei konkrete Vorteile.

## Bewertungsmaßstab (Master-Niveau)

- **Korrektheit der Kausalmaske** (kein Informationsleck aus der Zukunft) — der zentrale
  Punkt eines Decoders.
- Vollständige, saubere Architektur (Pre-LN-Blöcke, Residuals, Positional-Embedding,
  Weight-Tying) und stabile Optimierung.
- Funktionierende Generierung mit Temperatur & top-k.
- Nachvollziehbare, quantitativ belegte Analyse.

## Setup

```bash
source ../../../../.venv/bin/activate
# eigene Umsetzung bauen, dann z. B.:  python gpt.py
```

Benötigt nur `torch` (in der Repo-`requirements.txt`). Der Korpus wird beim ersten Lauf
nach `datasets/` geladen (siehe `datasets/README.md`) und nicht eingecheckt.

## Musterlösung

[`solution/gpt.py`](solution/gpt.py) — ein vollständiges, getestetes ~0,8-Mio.-Parameter-GPT
(kausale Attention, Pre-LN-Blöcke, Weight-Tying, Temperatur/top-k-Sampling). Referenzlauf:
Val-Loss 1.77 nach 3000 Iterationen, ~1:50 min auf Apple-MPS. **Erst selbst bauen.**
