# Project 01 (basic) — An n-gram language model

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Module 08 — NLP 1** · Format: **Jupyter notebook** (`language_model.ipynb`)

## Why this format?

You understand a language model once you *feed it, measure it and let it
generate*. A notebook connects loading the corpus, counting the n-grams,
measuring the perplexity and generating text step by step, with visible
intermediate results — ideal for this guided start with plenty of instruction.

## Goal

You build a statistical **n-gram language model** on real text and check the core
statements of part 1 of the script empirically:

- n-gram counts (unigram/bigram/trigram) and **add-$k$ smoothing**;
- **perplexity** as the evaluation measure on a held-out test set;
- **one central aha moment:** with blunt add-1 the *trigram is worse* than the
  unigram (over-smoothing with a large vocabulary) — only **interpolation**
  (mixing the orders with a small $k$) lowers the perplexity substantially;
- **text generation** by sampling — locally plausible, globally nonsense: the
  limit of n-grams (an outlook on the neural models in module 09).

## Prior knowledge

- Part 1 of the script (tokenization, n-gram models, add-$k$/interpolation, perplexity).
- Python: `collections.Counter`/`defaultdict`, `math.log`, `random.choices`.

## Setup

Only the standard library. The first cell **downloads the corpus** (*The
Adventures of Sherlock Holmes*, about 600 KB) from Project Gutenberg into
`datasets/` and caches it (not checked in, via `.gitignore`). In Jupyter/VS Code
select the kernel of the repository `.venv` and run the cells from top to bottom.

```bash
source ../../../../.venv/bin/activate
jupyter lab    # or open language_model.ipynb in VS Code
```

## Tasks (step by step)

Part A (the data, tokenization, the vocabulary with `<unk>`, the n-gram counts) is
given. Then three tasks:

1. **Add-$k$ and perplexity:** implement `p_unigram/p_bigram/p_trigram` (the
   add-$k$ formula) and `perplexity`. Observe that add-1 makes the trigram
   *worse*.
2. **Interpolation:** mix the three orders ($\lambda_1,\lambda_2,\lambda_3$) with
   a small $k$ — the perplexity should fall substantially.
3. **Text generation:** sample sentences from the bigram model.

At the end there is a short written part for reflection.

## What should work in the end

- Perplexities for the uni-/bi-/trigram (add-1) — with the observation that the
  higher order gets *worse* here (reference order of magnitude: unigram about
  350, trigram about 2000).
- **Interpolation** clearly better (reference: about 190).
- Generated sentences that "sound like" the corpus but are grammatically crude.

The exact numbers depend slightly on the random split (the seed is set), the
tendencies do not.

## Reference solution

The folder **`solution/`** holds `language_model_solution.ipynb` — fully
implemented and **executed**. The reflection questions are deliberately left
unanswered — they are yours to work through. Look only after your own attempt.

---
---

# Projekt 01 (basic) — Ein N-Gramm-Sprachmodell (deutsche Fassung)

**Modul 08 — NLP 1** · Format: **Jupyter Notebook** (`language_model.ipynb`)

## Warum dieses Format?

Ein Sprachmodell versteht man, wenn man es *füttert, misst und generieren lässt*.
Ein Notebook verbindet das Laden des Korpus, das Zählen der N-Gramme, die
Perplexitäts-Messung und die Textgenerierung Schritt für Schritt mit sichtbaren
Zwischenergebnissen — ideal für diesen geführten Einstieg mit viel Anleitung.

## Ziel

Du baust ein statistisches **N-Gramm-Sprachmodell** auf echtem Text und prüfst die
Kernaussagen aus Teil 1 des Skripts empirisch:

- N-Gramm-Zählungen (Unigramm/Bigramm/Trigramm) und **Add-$k$-Glättung**;
- **Perplexität** als Bewertungsmaß auf einem Held-out-Testset;
- **ein zentraler Aha-Moment:** Mit plumpem Add-1 ist das *Trigramm schlechter* als
  das Unigramm (Über-Glättung bei großem Vokabular) — erst **Interpolation** (Mischen
  der Ordnungen mit kleinem $k$) senkt die Perplexität deutlich;
- **Textgenerierung** durch Sampling — lokal plausibel, global Unsinn: die Grenze
  von N-Grammen (Ausblick auf neuronale Modelle in Modul 09).

## Vorwissen

- Skript Teil 1 (Tokenisierung, N-Gramm-Modelle, Add-$k$/Interpolation, Perplexität).
- Python: `collections.Counter`/`defaultdict`, `math.log`, `random.choices`.

## Setup

Nur Standardbibliothek. Die erste Zelle **lädt den Korpus** (*The Adventures of
Sherlock Holmes*, ~600 KB) von Project Gutenberg nach `datasets/` und cached ihn (per
`.gitignore` nicht eingecheckt). In Jupyter/VS Code den Kernel des Repo-`.venv`
wählen und die Zellen von oben nach unten ausführen.

```bash
source ../../../../.venv/bin/activate
jupyter lab    # oder language_model.ipynb in VS Code öffnen
```

## Aufgabenstellung (Schritt für Schritt)

Teil A (Daten, Tokenisierung, Vokabular mit `<unk>`, N-Gramm-Zählungen) ist
vorgegeben. Dann drei Aufgaben:

1. **Add-$k$ & Perplexität:** implementiere `p_unigram/p_bigram/p_trigram`
   (Add-$k$-Formel) und `perplexity`. Beobachte, dass Add-1 das Trigramm
   *verschlechtert*.
2. **Interpolation:** mische die drei Ordnungen ($\lambda_1,\lambda_2,\lambda_3$)
   mit kleinem $k$ — die Perplexität soll deutlich fallen.
3. **Textgenerierung:** sample Sätze aus dem Bigramm-Modell.

Zum Schluss ein kurzer schriftlicher Reflexionsteil.

## Was am Ende funktionieren soll

- Perplexitäten für Uni-/Bi-/Trigramm (Add-1) — mit der Beobachtung, dass höhere
  Ordnung hier *schlechter* wird (Referenz-Größenordnung: Unigramm ~350, Trigramm ~2000).
- **Interpolation** deutlich besser (Referenz: ~190).
- Generierte Sätze, die nach dem Korpus „klingen", aber grammatisch grob sind.

Die exakten Zahlen hängen leicht vom Zufalls-Split ab (Seed gesetzt), die Tendenzen
nicht.

## Musterlösung

In **`solution/`** liegt `language_model_solution.ipynb` — vollständig implementiert und
**ausgeführt**. Die Reflexionsfragen bleiben bewusst unbeantwortet — die sind zum
Selbstdurcharbeiten da. Erst nach eigenem Versuch ansehen.
