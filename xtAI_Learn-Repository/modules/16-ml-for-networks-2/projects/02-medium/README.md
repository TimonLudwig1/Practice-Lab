# Project 02 (medium) — a GCN from scratch (and what the evaluation makes of it)

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Format: Python project** (`.py` + tests). **Why?** The normalized adjacency has to be **exactly**
right — and unit tests are the right tool for that (several test cases recompute it by hand). It
also separates things cleanly: data ↔ model ↔ experiment.

---

## Goal

You build a **graph convolutional network** (Kipf & Welling 2017) **by hand** in plain PyTorch —
without `torch_geometric`. In the end it is ~15 lines, and afterwards the library holds no more
mystery.

Then you let it compete against the heuristics from project 01 on the real AS topology — **under
two evaluation protocols**. The actual question of this project is not "which model is better?",
but: **does the answer depend on how I measure?**

## Prior knowledge

Module 16 script, **section 2.3** (message passing, the GCN derivation), **2.4** (over-smoothing),
**2.5** (link prediction + the three traps). Project 01 (heuristics, degree confound). PyTorch
(module 05).

## Files

| File | Role |
|---|---|
| `graph_data.py` | Load the graph, link split, **both** negative samplers. **Given.** |
| `gcn.py` | Normalized adjacency + GCN + control group. **This is your work** (4 TODOs). |
| `run.py` | The experiment: all methods × both protocols + plot. Given. |
| `test_gcn.py` | Test suite (**14 tests**). |

## Task

Four TODOs in `gcn.py`. **The difficulty is not the code, it is understanding the formula:**

1. **`normalized_adjacency(edges, n)`** — $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$ as a
   **sparse** COO tensor. The blueprint is in the docstring. Think of: both edge directions,
   self-loops, and catching `Inf` (isolated nodes!).
2. **`GCN.forward`** — two layers: `relu(W1(A_hat @ emb))`, then `W2(A_hat @ H)`.
3. **`MLPWithoutStructure.forward`** — the same thing **without** any `A_hat` multiplication (the
   control group).
4. **`edge_score`** — the row-wise dot product of the endpoint embeddings.

The tests recompute $\hat A$ **by hand** (path 0–1–2: $\hat A_{01}=1/\sqrt{2\cdot3}$), check
symmetry, self-loops, eigenvalues in $[-1,1]$, hub damping, and that after 1 hop message passing
sees the neighbour but **not yet** the next-but-one node.

## What comes out at the end

`python test_gcn.py` → **14/14 green** (~1 s). `python run.py` → the experiment (~15 s):

**ROC-AUC — the same models, two evaluation protocols:**

| Method | uniform (naive) | **degree-matched (honest)** |
|---|---|---|
| Common neighbors | 0.715 | 0.552 |
| Adamic-Adar | 0.720 | 0.560 |
| **Pref. attachment** | **0.765** ← the winner | **0.407** ← chance level |
| MLP (without graph) | 0.660 | 0.473 |
| **GCN (2 layers)** | 0.754 | **0.642** ← the winner |

### The three insights

1. **The measurement protocol reverses the ranking.** Evaluated naively, **preferential
   attachment** "wins" (0.765) and the GCN (0.754) looks like expensive nonsense — you would
   **discard** the GNN. Evaluated honestly, PA is at **0.407** (worse than guessing) and the GCN is
   clearly ahead with **0.642**. **The same models, the same data, the opposite conclusion.**
2. **The control group proves that the structure makes the difference.** The MLP has exactly the
   same parameters and the same embedding table as the GCN — only **no message passing**. It ends
   up at **0.473** degree-matched (≈ chance), the GCN at **0.642**. So the gain demonstrably comes
   from $\hat A$, not from the model size.
3. **A GNN is no free lunch.** 0.642 is better than everything else — but far from "solved". Link
   prediction on a real, incompletely measured AS topology is **hard**. The naive evaluation only
   covered that up.

> **The core of it:** it is not only the model that can deceive — the **construction of the test
> set** already decides what you measure. Uniformly drawn negative examples are the standard in
> countless link prediction papers.

## Running / setup

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_gcn.py   # 14 tests, ~1 s
/.../xtAI_Learn-Repository/.venv/bin/python run.py        # experiment + plot, ~15 s
```
`torch`, `networkx`, `numpy`, `scikit-learn` (+ `matplotlib`). **CPU** — thanks to sparse matmul
the GCN trains on 10,670 nodes in ~5 s. The graph is downloaded to `datasets/` on the first run
(69 KB, gitignored).

> **Why sparse is mandatory:** dense, $\hat A$ would be **455 MB** — and 99.96 % of it zeros.
> `torch.sparse.mm` computes only on the 50,274 non-zero entries.

## Solution

Complete in [`solution/`](solution/). Try it yourself first — the tests tell you very precisely
which part of $\hat A$ is not right yet (symmetry? self-loops? normalization?).

## Going further

- **Measure over-smoothing** (script 2.4): build 1/2/4/8 layers. From when on does it get worse?
  Measure the mean pairwise cosine similarity of the embeddings — does it converge to 1?
- **The GraphSAGE idea**: replace the embedding table with **structural features** (degree,
  clustering) as $H^{(0)}$. This makes the model **inductive** — test it on nodes that never
  occurred in training.
- **GAT**: replace the fixed $\hat A$ weights with learned attention (script 2.3). Is it worth it
  here?
- **Base rate** (module 15): compute the **PPV** of the GCN at $\pi\approx3.9\cdot10^{-4}$. How
  many of the top 100 suggestions would be real?

---

# Projekt 02 (medium) — Ein GCN from scratch (und was die Evaluation daraus macht) (deutsche Fassung)

**Format: Python-Projekt** (`.py` + Tests). **Warum?** Die normalisierte Adjazenz muss
**exakt** stimmen — dafür sind Unit-Tests das richtige Werkzeug (mehrere Testfälle rechnen sie
von Hand nach). Außerdem trennt es sauber: Daten ↔ Modell ↔ Experiment.

---

## Ziel

Du baust ein **Graph Convolutional Network** (Kipf & Welling 2017) **von Hand** in reinem
PyTorch — ohne `torch_geometric`. Es sind am Ende ~15 Zeilen, und danach ist die Bibliothek
entzaubert.

Dann lässt du es auf der echten AS-Topologie gegen die Heuristiken aus Projekt 01 antreten —
**unter zwei Evaluationsprotokollen**. Die eigentliche Frage dieses Projekts ist nicht „welches
Modell ist besser?", sondern: **hängt die Antwort davon ab, wie ich messe?**

## Vorwissen

Skript **Modul 16, Abschnitt 2.3** (Message Passing, GCN-Herleitung), **2.4** (Over-Smoothing),
**2.5** (Link Prediction + die drei Fallen). Projekt 01 (Heuristiken, Grad-Confound). PyTorch
(Modul 05).

## Dateien

| Datei | Rolle |
|---|---|
| `graph_data.py` | Graph laden, Link-Split, **beide** Negativ-Sampler. **Vorgegeben.** |
| `gcn.py` | Normalisierte Adjazenz + GCN + Kontrollgruppe. **Hier ist deine Arbeit** (4 TODOs). |
| `run.py` | Das Experiment: alle Verfahren × beide Protokolle + Plot. Vorgegeben. |
| `test_gcn.py` | Test-Suite (**14 Tests**). |

## Aufgabe

In `gcn.py` vier TODOs. **Die Schwierigkeit ist nicht der Code, sondern die Formel zu
verstehen:**

1. **`normalized_adjacency(edges, n)`** — $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$ als
   **sparse** COO-Tensor. Der Bauplan steht im Docstring. Denk an: beide Kantenrichtungen,
   Self-Loops, und `Inf` abfangen (isolierte Knoten!).
2. **`GCN.forward`** — zwei Schichten: `relu(W1(A_hat @ emb))`, dann `W2(A_hat @ H)`.
3. **`MLPWithoutStructure.forward`** — dasselbe **ohne** jede `A_hat`-Multiplikation (Kontrollgruppe).
4. **`edge_score`** — zeilenweises Skalarprodukt der Endpunkt-Embeddings.

Die Tests rechnen $\hat A$ **von Hand nach** (Pfad 0–1–2: $\hat A_{01}=1/\sqrt{2\cdot3}$),
prüfen Symmetrie, Self-Loops, Eigenwerte in $[-1,1]$, Hub-Dämpfung, und dass Message Passing
nach 1 Hop den Nachbarn, aber noch **nicht** den Übernächsten sieht.

## Was am Ende herauskommt

`python test_gcn.py` → **14/14 grün** (~1 s). `python run.py` → das Experiment (~15 s):

**ROC-AUC — dieselben Modelle, zwei Evaluationsprotokolle:**

| Verfahren | uniform (naiv) | **grad-gematcht (ehrlich)** |
|---|---|---|
| Common Neighbors | 0,715 | 0,552 |
| Adamic-Adar | 0,720 | 0,560 |
| **Pref. Attachment** | **0,765** ← Sieger | **0,407** ← Zufallsniveau |
| MLP (ohne Graph) | 0,660 | 0,473 |
| **GCN (2 Schichten)** | 0,754 | **0,642** ← Sieger |

### Die drei Erkenntnisse

1. **Das Messprotokoll dreht die Rangfolge um.** Naiv evaluiert „gewinnt" **Preferential
   Attachment** (0,765) und das GCN (0,754) sieht aus wie teurer Unsinn — man würde das GNN
   **verwerfen**. Ehrlich evaluiert ist PA bei **0,407** (schlechter als Raten) und das GCN mit
   **0,642** klar vorn. **Dieselben Modelle, dieselben Daten, gegenteilige Schlussfolgerung.**
2. **Die Kontrollgruppe beweist, dass die Struktur den Unterschied macht.** Das MLP hat exakt
   dieselben Parameter und dieselbe Embedding-Tabelle wie das GCN — nur **kein Message
   Passing**. Es landet grad-gematcht bei **0,473** (≈ Zufall), das GCN bei **0,642**. Der
   Gewinn kommt also nachweislich aus $\hat A$, nicht aus der Modellgröße.
3. **Ein GNN ist kein Selbstläufer.** 0,642 ist besser als alles andere — aber weit von „gelöst"
   entfernt. Link Prediction auf einer echten, unvollständig gemessenen AS-Topologie ist
   **schwer**. Die naive Evaluation hat das nur verdeckt.

> **Der Kern:** Nicht nur das Modell kann täuschen — schon die **Konstruktion der Testmenge**
> entscheidet, was du misst. Uniform gezogene Negativbeispiele sind der Standard in unzähligen
> Link-Prediction-Papers.

## Ausführen / Setup

```bash
/.../xtAI_Learn-Repository/.venv/bin/python test_gcn.py   # 14 Tests, ~1 s
/.../xtAI_Learn-Repository/.venv/bin/python run.py        # Experiment + Plot, ~15 s
```
`torch`, `networkx`, `numpy`, `scikit-learn` (+ `matplotlib`). **CPU** — dank sparse Matmul
trainiert das GCN auf 10 670 Knoten in ~5 s. Der Graph wird beim ersten Lauf nach `datasets/`
geladen (69 KB, gitignored).

> **Warum sparse zwingend ist:** dicht wäre $\hat A$ **455 MB** — und 99,96 % davon Nullen.
> `torch.sparse.mm` rechnet nur auf den 50 274 Nicht-Null-Einträgen.

## Lösung

Vollständig in [`solution/`](solution/). Erst selbst versuchen — die Tests sagen dir sehr genau,
welcher Teil von $\hat A$ noch nicht stimmt (Symmetrie? Self-Loops? Normierung?).

## Weiterdenken

- **Over-Smoothing messen** (Skript 2.4): Baue 1/2/4/8 Schichten. Ab wann wird es schlechter?
  Miss die mittlere paarweise Kosinus-Ähnlichkeit der Embeddings — konvergiert sie gegen 1?
- **GraphSAGE-Idee**: Ersetze die Embedding-Tabelle durch **strukturelle Features** (Grad,
  Clustering) als $H^{(0)}$. Das Modell wird dadurch **induktiv** — teste es auf Knoten, die
  im Training nie vorkamen.
- **GAT**: Ersetze die festen $\hat A$-Gewichte durch gelernte Attention (Skript 2.3). Lohnt es
  sich hier?
- **Basisrate** (Modul 15): Berechne den **PPV** des GCN bei $\pi\approx3{,}9\cdot10^{-4}$.
  Wie viele der Top-100-Vorschläge wären echt?
