# Project 02 (medium) — Aligning cross-lingual embeddings (Procrustes + CSLS)

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 10 — Multilingual NLP** · Format: **Python project** (several modules + test suite)

## Why this format?

The core is linear algebra — the **closed-form orthogonal Procrustes solution** via the SVD
— plus a retrieval measure against **hubness**. As a **codebase** you separate the
infrastructure (building embeddings, mining anchors) from the actual learning goal
(alignment + retrieval), and the **test suite** verifies your maths on controlled data where
the correct answer is known.

## Goal

You align two **separately** trained word-embedding spaces (German & English) so that one
can "translate" a dictionary by mapping the source word into the target space and taking the
nearest neighbour (script section 3). Concretely you learn:

- the **orthogonal Procrustes solution** $W=UV^\top$ from $\mathrm{SVD}(X^\top Y)$ —
  including the **correct direction** (source$^\top$·target);
- why the **orthogonality constraint** preserves the geometry (and thus cosine
  neighbourhoods);
- **precision@1** as a retrieval metric and the **hubness** problem;
- **CSLS** as a correction against hubs — and the honest observation of *when* it helps.

## Prior knowledge

- **Script** section 3 (alignment, Procrustes, SVD solution, hubness, CSLS).
- Linear algebra: SVD, orthogonality, Frobenius norm.
- NumPy (`np.linalg.svd`, broadcasting, `argmax`, `argpartition`).

## Project structure

```
02-medium/
  align.py         # Procrustes + NN + CSLS + P@1          <- YOU (task 1 + 2)
  embeddings.py    # PPMI+SVD embeddings, mine anchors       (given)
  data.py          # load Tatoeba + test lexicon             (given)
  run.py           # full pipeline + NN-vs-CSLS comparison   (given)
  test_align.py    # test suite (8 tests, synthetic)         (given)
  solution/        # complete, tested reference solution
```

## Assignment

Little is given — the two mathematical cores are yours (`# TODO` in `align.py`):

1. **`orthogonal_procrustes(X, Y)`**: the closed-form solution $W=UV^\top$ with
   $U\Sigma V^\top=\mathrm{SVD}(X^\top Y)$. Mind the direction: $X^\top Y$
   (source$^\top$ target), so that $XW\approx Y$ holds. `nearest_neighbor` and
   `precision_at_1` are given.
2. **`csls(query_vecs, tgt_emb, src_pool, k)`**: the CSLS retrieval $2\cos - r_T - r_S$
   with the two neighbourhood densities (the helper `_topk_mean` is given).

**How to proceed:**

```bash
source ../../../../.venv/bin/activate
python test_align.py     # red -> fill in the tasks -> all 8 tests green
python run.py            # real DE-EN data: embeddings, anchors, alignment, P@1
```

The infrastructure (`embeddings.py`) builds PPMI+SVD embeddings per language (on Tatoeba
usable vectors emerge: *king~queen*, *water~boils*) and **automatically mines ~2800 anchor
pairs** from the parallel sentences (mutual nearest neighbour over co-occurrence). This way
anchors (training) and test lexicon are cleanly separated.

## What should work in the end

- `python test_align.py` → **all 8 tests green**: Procrustes reconstructs a known rotation
  exactly, is orthogonal, has the correct direction, denoises; NN & CSLS are perfect on
  clean spaces; **CSLS corrects a constructed hub** where NN fails.
- `python run.py` → **precision@1 ≈ 0.95–0.98** on the 40-word test lexicon, with example
  translations (`water→wasser`, `king→könig`, …).

> **Honest observation:** on well-separated content words NN is already almost perfect; CSLS
> is on par or minimally below here. Its benefit shows above all with **hubness** in harder,
> noisier retrieval — exactly what the synthetic hub test isolates. "More method" is not
> always "more accuracy" when the problem is already easy.

## Reference solution

Complete in [`solution/`](solution/) (all tests green, P@1 ≈ 0.95–0.98). Try it yourself
first — the root `align.py` raises `NotImplementedError` until you fill in the TODOs.

---

# Projekt 02 (medium) — Cross-linguale Embeddings alignen (Procrustes + CSLS) (deutsche Fassung)

**Modul 10 — Multilingual NLP** · Format: **Python-Projekt** (mehrere Module + Testsuite)

## Warum dieses Format?

Das Kernstück ist lineare Algebra — die **geschlossene orthogonale Procrustes-Lösung** über
die SVD — plus ein Retrieval-Maß gegen **Hubness**. Als **Codebasis** trennst du die
Infrastruktur (Embeddings bauen, Anker schürfen) vom eigentlichen Lernziel (Alignment +
Retrieval), und die **Testsuite** verifiziert deine Mathematik auf kontrollierten Daten,
bei denen die richtige Antwort bekannt ist.

## Ziel

Du richtest zwei **getrennt** trainierte Wort-Embedding-Räume (Deutsch & Englisch) so
aus, dass man ein Wörterbuch „übersetzen" kann, indem man das Quellwort in den Zielraum
abbildet und den nächsten Nachbarn nimmt (Skript-Abschnitt 3). Konkret lernst du:

- die **orthogonale Procrustes-Lösung** $W=UV^\top$ aus $\mathrm{SVD}(X^\top Y)$ — inklusive
  der **richtigen Richtung** (Quelle$^\top$·Ziel);
- warum die **Orthogonalitäts-Beschränkung** die Geometrie (und damit Kosinus-Nachbarschaften)
  erhält;
- **Precision@1** als Retrieval-Metrik und das **Hubness**-Problem;
- **CSLS** als Korrektur gegen Hubs — und die ehrliche Beobachtung, *wann* es hilft.

## Vorwissen

- **Skript** Abschnitt 3 (Alignment, Procrustes, SVD-Lösung, Hubness, CSLS).
- Lineare Algebra: SVD, Orthogonalität, Frobenius-Norm.
- NumPy (`np.linalg.svd`, Broadcasting, `argmax`, `argpartition`).

## Projektstruktur

```
02-medium/
  align.py         # Procrustes + NN + CSLS + P@1         <- DU (Aufgabe 1 + 2)
  embeddings.py    # PPMI+SVD-Embeddings, Anker schürfen    (vorgegeben)
  data.py          # Tatoeba laden + Test-Lexikon           (vorgegeben)
  run.py           # volle Pipeline + NN-vs-CSLS-Vergleich  (vorgegeben)
  test_align.py    # Testsuite (8 Tests, synthetisch)       (vorgegeben)
  solution/         # vollständige, getestete Musterlösung
```

## Aufgabenstellung

Wenig Vorgabe — die beiden mathematischen Kerne sind deine (`# TODO` in `align.py`):

1. **`orthogonal_procrustes(X, Y)`**: die geschlossene Lösung $W=UV^\top$ mit
   $U\Sigma V^\top=\mathrm{SVD}(X^\top Y)$. Achte auf die Richtung: $X^\top Y$ (Quelle$^\top$
   Ziel), damit $XW\approx Y$ gilt. `nearest_neighbor` und `precision_at_1` sind gegeben.
2. **`csls(query_vecs, tgt_emb, src_pool, k)`**: das CSLS-Retrieval
   $2\cos - r_T - r_S$ mit den beiden Nachbarschafts-Dichten (Helfer `_topk_mean` gegeben).

**Vorgehen:**

```bash
source ../../../../.venv/bin/activate
python test_align.py     # rot -> Aufgaben füllen -> alle 8 Tests grün
python run.py            # echte DE-EN-Daten: Embeddings, Anker, Alignment, P@1
```

Die Infrastruktur (`embeddings.py`) baut PPMI+SVD-Embeddings pro Sprache (auf Tatoeba
ergeben sich brauchbare Vektoren: *king~queen*, *water~boils*) und **schürft automatisch
~2800 Anker-Paare** aus den parallelen Sätzen (Mutual-Nearest-Neighbour über Kookkurrenz).
So sind Anker (Training) und Test-Lexikon sauber getrennt.

## Was am Ende funktionieren soll

- `python test_align.py` → **alle 8 Tests grün**: Procrustes rekonstruiert eine bekannte
  Rotation exakt, ist orthogonal, hat die richtige Richtung, denoised; NN & CSLS sind auf
  sauberen Räumen perfekt; **CSLS korrigiert einen konstruierten Hub**, wo NN scheitert.
- `python run.py` → **Precision@1 ≈ 0.95–0.98** auf dem 40-Wörter-Test-Lexikon, mit
  Beispielübersetzungen (`water→wasser`, `king→könig`, …).

> **Ehrliche Beobachtung:** Auf gut getrennten Inhaltswörtern ist NN bereits nahezu
> perfekt; CSLS liegt hier gleichauf oder minimal darunter. Sein Nutzen zeigt sich vor
> allem bei **Hubness** in schwierigerem, verrauschterem Retrieval — genau das isoliert der
> synthetische Hub-Test. „Mehr Methode" ist nicht immer „mehr Genauigkeit", wenn das
> Problem schon leicht ist.

## Musterlösung

Vollständig in [`solution/`](solution/) (alle Tests grün, P@1 ≈ 0.95–0.98). Erst selbst
versuchen — die Root-`align.py` wirft `NotImplementedError`, bis du die TODOs füllst.
