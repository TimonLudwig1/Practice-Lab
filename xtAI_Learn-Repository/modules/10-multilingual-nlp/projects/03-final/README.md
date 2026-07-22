# Project 03 (final) — Machine translation: statistical **and** neural

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 10 — Multilingual NLP** · Format: **Python project** (free implementation, *no* code given)

## Why this scope?

The final project consolidates the whole module on **real German→English translation** —
and does so **laptop-friendly**. A full NMT transformer on real data would need a GPU and a
lot of compute (a laptop gets hot doing it). Therefore you combine two parts that both run
on the **CPU in seconds** and together span the arc *classical → neural*:

- **Part A — IBM Model 1 (statistical MT):** real DE→EN translation + word **alignment**,
  learned via **EM** from purely parallel sentences. This is the historical foundation of MT
  and the **conceptual predecessor of attention** ("soft alignment", script 4.2).
- **Part B — encoder-decoder transformer:** the full neural architecture with
  **cross-attention** (script 4.3) *from scratch* — verified via a test suite and trained on
  a **tiny synthetic task** (sequence reversal) that can only be solved with working
  cross-attention. This way you learn the architecture completely, without expensive
  training on real translation data.

**No code given.** This README is the specification. A complete, tested reference solution is
in [`solution/`](solution/) — **build it yourself first.**

## Goal

In the end you should (1) learn a translation table + alignments from parallel sentences and
translate roughly (with **BLEU**), and (2) own an encoder-decoder transformer whose
cross-attention demonstrably works.

## Prior knowledge

- **Script** section 4 (seq2seq, attention as alignment, encoder-decoder transformer,
  cross-attention, teacher forcing, decoding) and 6 (BLEU).
- **Module 05** — the **EM algorithm** (here for IBM Model 1).
- **Module 09 / 10-P02** — self-attention & multi-head (the building blocks for part B).

---

## Part A — IBM Model 1 (statistical MT via EM)

Given parallel sentences (source $f$ = German, target $e$ = English). Model 1 assumes **all**
alignments are equally probable and learns only the translation table $t(e\mid f)$. For a
sentence pair:
$$P(e\mid f)=\frac{\epsilon}{(m+1)^{l}}\prod_{j=1}^{l}\sum_{i=0}^{m} t(e_j\mid f_i),\qquad f_0=\text{NULL}.$$

**To build:**

1. **EM training** of $t(e\mid f)$:
   - *Initialization* uniform (over co-occurring pairs); extend the source by a **NULL**
     token (captures target words without a source counterpart).
   - *E-step:* for every target word $e_j$ accumulate the expected counts
     $c(e_j,f_i)=\dfrac{t(e_j\mid f_i)}{\sum_{i'} t(e_j\mid f_{i'})}$.
   - *M-step:* $t(e\mid f)=\dfrac{\text{count}(e,f)}{\sum_{e'}\text{count}(e',f)}$.
   - 5 iterations suffice; the **log likelihood must increase monotonically** (Model 1 is
     convex → a global optimum — a good sanity check!).
2. **Viterbi alignment:** each target word to $\arg\max_i t(e_j\mid f_i)$.
3. **Word-for-word translation** DE→EN (each source word to its most probable target
   counterpart) and **corpus BLEU** for it. Expect a **modest** BLEU (Model 1 has *no*
   reordering and *no* language model) — the **alignments** are the real gain.

**Data:** Tatoeba DE-EN (filter short sentences for cleaner alignments), ~20 000 pairs for
training.

---

## Part B — encoder-decoder transformer with cross-attention

Build the full NMT architecture (script 4.3):

- **Multi-head attention** with *separate* query and key/value inputs (for cross-attn).
- **Encoder layer:** self-attention (source padding mask) + FFN, each with residual + LayerNorm.
- **Decoder layer:** **causal** self-attention → **cross-attention** (query=decoder,
  key/value=encoder output, source padding mask) → FFN, each with residual + LayerNorm.
- Token embedding + positional encoding; output projection onto the vocabulary.
- **Masks:** source padding $(B,1,1,S)$; decoder = causal $\wedge$ target padding $(B,1,T,T)$;
  cross = source padding.

**Instead of expensive real training** you train on a **synthetic toy task**: *sequence
reversal* (target = source reversed). It is only solvable if cross-attention works — decoder
step $j$ must look at source position $L{-}1{-}j$. It trains in **~20 s on the CPU**. Expected
result:

- **exact match ≈ 0.85+** on unseen sequences (the whole sequence correctly reversed);
- the head-averaged **cross-attention matrix** of the last decoder block is clearly
  **anti-diagonal** — the visual confirmation that cross-attention has learned the correct
  alignment.

**BLEU** (script 6.1) you implement once by hand (clipped n-gram precision × brevity penalty)
and use it in part A.

---

## Milestones (recommended)

1. **BLEU** by hand + test (identical→100, brevity penalty, clipping).
2. **IBM Model 1**: EM (check the likelihood increase), alignment, translation + BLEU.
3. **Transformer**: building blocks + masks; check shapes/causality/cross-attention via tests.
4. **Toy training**: sequence reversal up to exact match ≈ 0.85+, show the anti-diagonal.
5. **Analysis** (written, see below).

## What should work in the end

- IBM Model 1: EM log likelihood increases monotonically; sensible alignments
  (`i←ich`, `know←weiß`, `tom←tom`); word-for-word BLEU in the range **~15–20**.
- Transformer: all architecture tests green; toy **exact match ≈ 0.85+**; anti-diagonal
  cross-attention.
- A short **analysis** (`ANALYSIS.md`) that shows:
  1. **Why does the EM likelihood increase monotonically**, and why does Model 1 find a
     *global* optimum (convexity)?
  2. **Why is the word-for-word BLEU low** — which two components does Model 1 lack (name
     them) that a real MT system would have?
  3. **Alignment ↔ attention:** in what sense is $t(e\mid f)$ / the Viterbi alignment the
     predecessor of (cross-)attention $\alpha_{ti}$? Where is the difference (hard vs. soft,
     context-dependent)?
  4. **Cross-attention evidence:** why does the anti-diagonal attention matrix on the
     reversal task show that cross-attention works "correctly"?

## Assessment criteria (master's level)

- **Correct EM** (E/M step, NULL token, likelihood increase) and Viterbi alignment.
- **Complete, correct transformer architecture** — in particular the three attention types
  and the **causal** decoder mask (no look into the future).
- Clean **BLEU** (clipping + brevity penalty) by hand.
- A comprehensible, quantitatively supported analysis connecting classical and neural MT.

## Setup

```bash
source ../../../../.venv/bin/activate
# build your own implementation, then e.g.:
#   python run_ibm.py           # IBM Model 1: alignments + BLEU  (~5 s)
#   python run_transformer.py   # toy reversal: exact match + cross-attn  (~20 s)
#   python test_mt.py           # test suite (fast)
```

Requires `torch` and `sentencepiece`/`numpy` (in the repo `requirements.txt`). The corpus is
downloaded into `datasets/` on the first run. **Everything CPU, nothing compute-intensive.**

## Reference solution

Complete in [`solution/`](solution/): `ibm_model1.py`, `transformer.py`, `toy_task.py`,
`bleu.py`, `data.py`, `run_ibm.py`, `run_transformer.py`, `test_mt.py` (10 tests green).
Reference: IBM BLEU ≈ 18, toy exact match ≈ 0.87, anti-diagonal cross-attention. **Build it
yourself first.**

---

# Projekt 03 (final) — Maschinelle Übersetzung: statistisch **und** neuronal (deutsche Fassung)

**Modul 10 — Multilingual NLP** · Format: **Python-Projekt** (freie Umsetzung, *keine* Code-Vorgabe)

## Warum dieser Zuschnitt?

Das Abschlussprojekt konsolidiert das ganze Modul an **echter Deutsch→Englisch-Übersetzung**
— und zwar **laptop-freundlich**. Ein vollwertiger NMT-Transformer auf echten Daten würde
eine GPU und viel Rechenzeit brauchen (ein Laptop wird dabei heiß). Deshalb kombinierst du
zwei Teile, die beide auf der **CPU in Sekunden** laufen und zusammen den Bogen
*klassisch → neuronal* schlagen:

- **Teil A — IBM Model 1 (statistische MÜ):** echte DE→EN-Übersetzung + Wort-**Alignment**,
  gelernt per **EM** aus rein parallelen Sätzen. Das ist das historische Fundament der MÜ
  und der **konzeptuelle Vorläufer der Attention** („weiches Alignment", Skript 4.2).
- **Teil B — Encoder-Decoder-Transformer:** die vollständige neuronale Architektur mit
  **Cross-Attention** (Skript 4.3) *von Grund auf* — verifiziert per Testsuite und trainiert
  auf einer **winzigen synthetischen Aufgabe** (Sequenz-Umkehr), die sich nur mit
  funktionierender Cross-Attention lösen lässt. So lernst du die Architektur vollständig,
  ohne teures Training auf echten Übersetzungsdaten.

**Keine Code-Vorgabe.** Diese README ist die Spezifikation. Eine vollständige, getestete
Musterlösung liegt in [`solution/`](solution/) — **erst selbst bauen.**

## Ziel

Am Ende sollst du (1) aus parallelen Sätzen eine Übersetzungstabelle + Alignments lernen
und grob übersetzen (mit **BLEU**), und (2) einen Encoder-Decoder-Transformer besitzen,
dessen Cross-Attention nachweislich funktioniert.

## Vorwissen

- **Skript** Abschnitt 4 (seq2seq, Attention als Alignment, Encoder-Decoder-Transformer,
  Cross-Attention, Teacher Forcing, Decoding) und 6 (BLEU).
- **Modul 05** — der **EM-Algorithmus** (hier für IBM Model 1).
- **Modul 09 / 10-P02** — Self-Attention & Multi-Head (die Bausteine für Teil B).

---

## Teil A — IBM Model 1 (statistische MÜ via EM)

Gegeben parallele Sätze (Quelle $f$ = Deutsch, Ziel $e$ = Englisch). Model 1 nimmt **alle**
Alignments als gleich wahrscheinlich an und lernt nur die Übersetzungstabelle
$t(e\mid f)$. Für ein Satzpaar:
$$P(e\mid f)=\frac{\epsilon}{(m+1)^{l}}\prod_{j=1}^{l}\sum_{i=0}^{m} t(e_j\mid f_i),\qquad f_0=\text{NULL}.$$

**Zu bauen:**

1. **EM-Training** von $t(e\mid f)$:
   - *Initialisierung* uniform (über kookkurrierende Paare); Quelle um ein **NULL**-Token
     erweitern (fängt Zielwörter ohne Quell-Entsprechung ab).
   - *E-Schritt:* für jedes Zielwort $e_j$ die erwarteten Zählungen
     $c(e_j,f_i)=\dfrac{t(e_j\mid f_i)}{\sum_{i'} t(e_j\mid f_{i'})}$ akkumulieren.
   - *M-Schritt:* $t(e\mid f)=\dfrac{\text{count}(e,f)}{\sum_{e'}\text{count}(e',f)}$.
   - 5 Iterationen genügen; die **Log-Likelihood muss monoton steigen** (Model 1 ist
     konvex → globales Optimum — ein guter Sanity-Check!).
2. **Viterbi-Alignment:** jedes Zielwort auf $\arg\max_i t(e_j\mid f_i)$.
3. **Wort-für-Wort-Übersetzung** DE→EN (jedes Quellwort auf sein wahrscheinlichstes
   Ziel-Pendant) und **Korpus-BLEU** dazu. Erwarte eine **bescheidene** BLEU (Model 1 hat
   *kein* Reordering und *kein* Sprachmodell) — die **Alignments** sind der eigentliche
   Gewinn.

**Daten:** Tatoeba DE-EN (kurze Sätze filtern für sauberere Alignments), ~20 000 Paare
zum Training.

---

## Teil B — Encoder-Decoder-Transformer mit Cross-Attention

Baue die vollständige NMT-Architektur (Skript 4.3):

- **Multi-Head-Attention** mit *getrennten* Query- und Key/Value-Eingaben (für Cross-Attn).
- **Encoder-Layer:** Self-Attention (Quell-Padding-Maske) + FFN, je mit Residual + LayerNorm.
- **Decoder-Layer:** **kausale** Self-Attention → **Cross-Attention** (Query=Decoder,
  Key/Value=Encoder-Output, Quell-Padding-Maske) → FFN, je mit Residual + LayerNorm.
- Token-Embedding + Positional Encoding; Output-Projektion aufs Vokabular.
- **Masken:** Quell-Padding $(B,1,1,S)$; Decoder = kausal $\wedge$ Ziel-Padding $(B,1,T,T)$;
  Cross = Quell-Padding.

**Statt teurem Real-Training** trainierst du auf einer **synthetischen Toy-Aufgabe**:
*Sequenz-Umkehr* (Ziel = Quelle rückwärts). Diese ist nur lösbar, wenn die Cross-Attention
funktioniert — Decoder-Schritt $j$ muss auf Quellposition $L{-}1{-}j$ schauen. Sie
trainiert in **~20 s auf der CPU**. Erwartetes Ergebnis:

- **Exact-Match ≈ 0.85+** auf ungesehenen Sequenzen (ganze Folge korrekt umgekehrt);
- die kopf-gemittelte **Cross-Attention-Matrix** des letzten Decoder-Blocks ist deutlich
  **anti-diagonal** — die visuelle Bestätigung, dass Cross-Attention das richtige
  Alignment gelernt hat.

**BLEU** (Skript 6.1) implementierst du einmal von Hand (geclippte n-Gramm-Präzision ×
Brevity Penalty) und nutzt es in Teil A.

---

## Milestones (empfohlen)

1. **BLEU** von Hand + Test (identisch→100, Brevity Penalty, Clipping).
2. **IBM Model 1**: EM (Likelihood-Anstieg prüfen), Alignment, Übersetzung + BLEU.
3. **Transformer**: Bausteine + Masken; per Test Formen/Kausalität/Cross-Attention prüfen.
4. **Toy-Training**: Sequenz-Umkehr bis Exact-Match ≈ 0.85+, Anti-Diagonale zeigen.
5. **Analyse** (schriftlich, s. u.).

## Was am Ende funktionieren soll

- IBM Model 1: EM-Log-Likelihood steigt monoton; sinnvolle Alignments
  (`i←ich`, `know←weiß`, `tom←tom`); Wort-für-Wort-BLEU im Bereich **~15–20**.
- Transformer: alle Architektur-Tests grün; Toy-**Exact-Match ≈ 0.85+**; anti-diagonale
  Cross-Attention.
- Eine kurze **Analyse** (`ANALYSIS.md`), die belegt:
  1. **Warum steigt die EM-Likelihood monoton**, und warum findet Model 1 ein *globales*
     Optimum (Konvexität)?
  2. **Warum ist die Wort-für-Wort-BLEU niedrig** — welche zwei Komponenten fehlen Model 1
     (nenne sie), die ein echtes MÜ-System hätte?
  3. **Alignment ↔ Attention:** Inwiefern ist $t(e\mid f)$ / das Viterbi-Alignment der
     Vorläufer der (Cross-)Attention $\alpha_{ti}$? Wo ist der Unterschied (hart vs. weich,
     kontextabhängig)?
  4. **Cross-Attention-Beleg:** Warum belegt die anti-diagonale Attention-Matrix bei der
     Umkehr-Aufgabe, dass die Cross-Attention „richtig" funktioniert?

## Bewertungsmaßstab (Master-Niveau)

- **Korrektes EM** (E-/M-Schritt, NULL-Token, Likelihood-Anstieg) und Viterbi-Alignment.
- **Vollständige, korrekte Transformer-Architektur** — insbesondere die drei Attention-Typen
  und die **kausale** Decoder-Maske (kein Blick in die Zukunft).
- Sauberes **BLEU** (Clipping + Brevity Penalty) von Hand.
- Nachvollziehbare, quantitativ belegte Analyse, die klassische und neuronale MÜ verbindet.

## Setup

```bash
source ../../../../.venv/bin/activate
# eigene Umsetzung bauen, dann z. B.:
#   python run_ibm.py           # IBM Model 1: Alignments + BLEU  (~5 s)
#   python run_transformer.py   # Toy-Umkehr: Exact-Match + Cross-Attn  (~20 s)
#   python test_mt.py           # Testsuite (schnell)
```

Benötigt `torch` und `sentencepiece`/`numpy` (in der Repo-`requirements.txt`). Der Korpus
lädt beim ersten Lauf nach `datasets/`. **Alles CPU, nichts Rechenintensives.**

## Musterlösung

Vollständig in [`solution/`](solution/): `ibm_model1.py`, `transformer.py`, `toy_task.py`,
`bleu.py`, `data.py`, `run_ibm.py`, `run_transformer.py`, `test_mt.py` (10 Tests grün).
Referenz: IBM-BLEU ≈ 18, Toy-Exact-Match ≈ 0.87, anti-diagonale Cross-Attention. **Erst
selbst bauen.**
