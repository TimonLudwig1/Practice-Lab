# Project 01 (basic) — Multilingual subword tokenization with SentencePiece

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Module 10 — Multilingual NLP** · Format: **Jupyter notebook** (`tokenization.ipynb`)

## Why this format?

You understand tokenization by *trying and measuring*: seeing a sentence split, computing
fertility, comparing vocabularies. A notebook connects training, splitting and metrics with
visible intermediate results — ideal for this guided start.

## Goal

Using **SentencePiece** you train a **shared** subword vocabulary on real German–English
data (Tatoeba) and check the core claims from script section 2 empirically:

- **BPE splitting** and the `▁` space symbol; why there is **never OOV**.
- **fertility** (avg. subword tokens per word) for German vs. English.
- **Vocabulary sharing**: what share of the tokens do the two languages share?
- **Vocabulary bias**: an English-dominated vocabulary **doubles** the German fertility —
  the cost/fairness disadvantage of low-resource languages in large LLMs.
- **Subword vs. word**: word vocabulary size and OOV rate as an argument for subwords.

## Prior knowledge

- **Script** section 2 (BPE, WordPiece, unigram LM/SentencePiece, fertility, shared
  vocabulary).
- Python basics (`Counter`, sets, `str.split`).

## Setup

Requires `sentencepiece` (in the repo `requirements.txt`). The first cell downloads the
Tatoeba dataset (~12 MB) automatically into `datasets/` (a browser header is needed, it is
in the code) and caches it; not checked in.

```bash
source ../../../../.venv/bin/activate
jupyter lab      # or open tokenization.ipynb in VS Code, kernel = repo .venv
```

Runs in **under a minute** (SentencePiece is in C++).

## Dataset

**Tatoeba** German–English sentence pairs (~331k, open, CC-BY) via manythings.org/anki.
Real, short everyday sentences — small, clean, ideal for making tokenization visible and for
reusing later (projects 02/03) for embeddings and translation.

## Assignment (step by step)

**Part A** (download, parsing, writing the corpus) is given. Then three tasks (marked with
`# TODO`):

1. **Train BPE & encode**: the shared model is given via a call — split two example
   sentences and read the subword structure.
2. Implement **fertility**: subword tokens per word, separately for EN and DE.
3. Measure the **vocabulary sharing** (given) and show the **vocabulary bias**: train an
   EN-only vocabulary and compare the German fertility with it.

At the end a short written **reflection part** (4 questions).

## What should work in the end

- Splits for EN & DE with visible subword structure (`▁`, word pieces).
- fertility EN ≈ 1.5, DE ≈ 1.4 (shared vocabulary).
- **Bias result**: EN-only vocabulary → German fertility ≈ 2.9 (factor ~2.0).
- Word vocabulary ~20k types with ~11 % OOV, while BPE never has OOV.

## Reference solution

A fully filled-in, **executed** notebook is in
[`solution/tokenization_solution.ipynb`](solution/tokenization_solution.ipynb). Try it
yourself first — the stub cells raise `NotImplementedError` until you fill them. The
reference answers to the reflection questions are at the end of that solution notebook.

---

# Projekt 01 (basic) — Mehrsprachige Subword-Tokenisierung mit SentencePiece (deutsche Fassung)

**Modul 10 — Multilingual NLP** · Format: **Jupyter Notebook** (`tokenization.ipynb`)

## Warum dieses Format?

Tokenisierung versteht man durch *Ausprobieren und Messen*: einen Satz zerlegen sehen,
fertility ausrechnen, Vokabulare vergleichen. Ein Notebook verbindet Training, Zerlegung
und Kennzahlen mit sichtbaren Zwischenergebnissen — ideal für diesen geführten Einstieg.

## Ziel

Du trainierst mit **SentencePiece** ein **gemeinsames** Subword-Vokabular auf echten
Deutsch–Englisch-Daten (Tatoeba) und prüfst die Kernaussagen aus Skript-Abschnitt 2
empirisch:

- **BPE-Zerlegung** und das `▁`-Leerzeichen-Symbol; warum es **nie OOV** gibt.
- **fertility** (Ø Subword-Tokens pro Wort) für Deutsch vs. Englisch.
- **Vokabular-Teilung**: welchen Anteil der Tokens teilen sich beide Sprachen?
- **Vokabular-Bias**: ein englisch-dominiertes Vokabular **verdoppelt** die deutsche
  fertility — der Kosten-/Fairness-Nachteil ressourcenarmer Sprachen in großen LLMs.
- **Subword vs. Wort**: Wort-Vokabulargröße und OOV-Rate als Argument für Subwords.

## Vorwissen

- **Skript** Abschnitt 2 (BPE, WordPiece, Unigram-LM/SentencePiece, fertility, gemeinsames
  Vokabular).
- Python-Basics (`Counter`, Mengen, `str.split`).

## Setup

Benötigt `sentencepiece` (in der Repo-`requirements.txt`). Die erste Zelle lädt den
Tatoeba-Datensatz (~12 MB) automatisch nach `datasets/` (Browser-Header nötig, ist im Code)
und cached ihn; nicht eingecheckt.

```bash
source ../../../../.venv/bin/activate
jupyter lab      # oder tokenization.ipynb in VS Code öffnen, Kernel = Repo-.venv
```

Läuft in **unter einer Minute** (SentencePiece ist in C++).

## Datensatz

**Tatoeba** Deutsch–Englisch-Satzpaare (~331k, offen, CC-BY) über manythings.org/anki.
Echte, kurze Alltagssätze — klein, sauber, ideal, um Tokenisierung sichtbar zu machen und
später (Projekt 02/03) für Embeddings und Übersetzung weiterzuverwenden.

## Aufgabenstellung (Schritt für Schritt)

**Teil A** (Download, Parsing, Korpus schreiben) ist vorgegeben. Dann drei Aufgaben (an
`# TODO` markiert):

1. **BPE trainieren & encoden**: das gemeinsame Modell ist per Aufruf gegeben — zerlege
   zwei Beispielsätze und lies die Subword-Struktur.
2. **fertility** implementieren: Subword-Tokens pro Wort, für EN und DE getrennt.
3. **Vokabular-Teilung** (vorgegeben) messen und den **Vokabular-Bias** zeigen: ein
   EN-only-Vokabular trainieren und die deutsche fertility damit vergleichen.

Zum Schluss ein kurzer schriftlicher **Reflexionsteil** (4 Fragen).

## Was am Ende funktionieren soll

- Zerlegungen für EN & DE mit sichtbarer Subword-Struktur (`▁`, Wortstücke).
- fertility EN ≈ 1.5, DE ≈ 1.4 (gemeinsames Vokabular).
- **Bias-Ergebnis**: EN-only-Vokabular → deutsche fertility ≈ 2.9 (Faktor ~2.0).
- Wort-Vokabular ~20k Typen mit ~11 % OOV, während BPE nie OOV hat.

## Musterlösung

Voll ausgefülltes, **ausgeführtes** Notebook unter
[`solution/tokenization_solution.ipynb`](solution/tokenization_solution.ipynb). Erst selbst
probieren — die Stub-Zellen werfen `NotImplementedError`, bis du sie füllst.
