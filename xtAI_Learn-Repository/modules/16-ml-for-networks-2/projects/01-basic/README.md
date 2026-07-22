# Project 01 (basic) — the internet as a graph

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Format: Jupyter notebook** (`internet_graph.ipynb`). **Why?** You understand graphs by
**looking at them**: plotting degree distributions, finding hubs, breaking the network apart.
Exploratory analysis with many plots — exactly the notebook format.

---

## Goal

You work with the **real internet topology** and reproduce two of the most famous results of
network research — in order to then learn how easily you deceive yourself in exactly this kind of
analysis.

**Dataset:** `oregon1_010331` (SNAP/Stanford) — the AS peering topology from **BGP** data of the
Oregon route views, 31 March 2001. **10,670 autonomous systems, 22,002 peerings.** Real
measurement data, 69 KB, downloaded in ~1 s, cached locally afterwards (`datasets/`, gitignored).

*Why real data and not a synthetic graph?* Because the interesting properties (scale-free, hub
fragility) are exactly the ones you **put into** synthetic graphs instead of **finding** them.
Here you measure them.

## Prior knowledge

Module 16 script, **section 1** (graph, scale-free, robustness) and **2.1/2.5** (heuristics,
traps). `networkx` is introduced in the notebook.

## Task

Most of it is given (this is the basic project!). You fill in **four** `# TODO` blocks:

1. **Degree statistics** (a `degrees` array from `G.degree()`),
2. **Power-law exponent** — `alpha_naive` (log-log fit) **and** `alpha_mle` (Clauset MLE),
3. **The robustness loop** — remove nodes at random vs. by degree,
4. **The four link prediction heuristics** (common neighbors, Jaccard, Adamic-Adar, pref.
   attachment).

## What comes out at the end

**1. Scale-free.** Degree: min 1, **median 2**, mean 4.12, **max 2312**. Density 0.039 %.
Clustering 0.297. There is **no typical degree** — the mean describes nobody (the same pattern as
the flow sizes in module 15).

**2. The exponent — and the first trap** ⚠️

| Method | α |
|---|---|
| naive log-log fit (PMF) | **1.14** ❌ |
| MLE, $d_{\min}=5$ | 2.12 |
| **MLE, $d_{\min}=10$** | **2.08** ✅ |

The literature value is **≈ 2.1** (Faloutsos³ 1999). The MLE hits it, the naive fit is off by a
**factor of 2** — and appears in a frightening number of papers nevertheless.

**3. Robustness** — the Albert/Jeong/Barabási result (2000), on real data:

| removed | random failure | **targeted (hubs)** |
|---|---|---|
| 1 % | 0.989 | **0.413** |
| 5 % | 0.939 | **0.003** |
| 10 % | 0.800 | 0.001 |

**Remove 5 % of the hubs → the internet crumbles to dust (0.3 %)**, while 10 % of *random*
failures barely bother it (0.80). The same structure that immunizes it against mishaps makes it
fragile against attackers. (The top hubs *are* also the betweenness bottlenecks: 16/20 overlap.)

**4. Link prediction — and the decisive trap**

The first run (uniformly drawn negative examples, as is customary everywhere):

| Heuristic | ROC-AUC | PR-AUC |
|---|---|---|
| Common neighbors | 0.719 | 0.712 |
| Adamic-Adar | 0.724 | 0.732 |
| **Preferential attachment** | **0.763** | **0.857** |

**Preferential attachment wins** — of all things the *dumbest* heuristic, which only looks at
degrees and ignores the neighbourhood completely. That is suspicious. And indeed:

Median degree product $d_u\cdot d_v$: **real edges 500**, **random pairs 4**. In a scale-free
graph a random pair is almost always **leaf × leaf** — so PA does not have to recognize any
structure at all, it only distinguishes *hub involved* from *leaf×leaf*.

With **degree-matched** negative examples (the same degree distribution as the real edges):

| Heuristic | uniform | **degree-matched** |
|---|---|---|
| Common neighbors | 0.719 | 0.559 |
| Jaccard | 0.698 | 0.565 |
| **Adamic-Adar** | 0.724 | **0.566** ← now the best |
| **Preferential attachment** | **0.763** | **0.406** ← chance level |

**PA crashes from 0.763 to 0.406.** Its entire lead was **not a signal but an artifact of the
negative sampling**. And all methods drop to ~0.56: the task is **far harder** than the first
table suggests.

> ### The three traps of this project — all demonstrated, not claimed
> 1. **Estimation trap:** a naive log-log fit instead of the MLE → wrong by a factor of 2.
> 2. **Leakage:** the test edges have to be removed **before** the features are computed —
>    otherwise common neighbors already "sees" the edge you are looking for.
> 3. **Degree confound:** uniformly drawn negatives make every degree-based quantity artificially
>    strong.
>
> On top of that the base rate trap from module 15: in reality $\pi\approx3.9\cdot10^{-4}$
> (22,002 edges among 57 million pairs), but the evaluation is 1:1 balanced. **It is not only the
> model that can deceive — the construction of the test set already decides what you measure.**

Runtime: **~6 s** in total.

## Running / setup

Repo `venv`. Open the notebook:
`/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (or the `.venv` kernel in VS Code).
Only `networkx`, `numpy`, `matplotlib`, `scikit-learn` — all available. Internet is needed only
for the **first** load (the download needs a `User-Agent` header, otherwise 403).

## Solution

Fully solved and **executed** in
[`solution/internet_graph_solution.ipynb`](solution/internet_graph_solution.ipynb). Try it
yourself first! It contains four extension tasks (estimate the true PPV via Bayes from module 15,
deliberately provoke leakage, 1 % instead of 10 % test edges, compare the robustness against a
**random graph** of the same size — the actual proof that the asymmetry comes from the scale-free
structure).

> **Caveat** (script 4.4): **nobody** knows the "real" AS topology. Route collectors only see what
> BGP announces; **peerings between small AS are systematically missing**. We work on an
> *incomplete, biasedly measured* graph — that applies to **all** the numbers above.

## Transfer

The graph fundamentals and especially the **sound link prediction evaluation** are the basis for
**project 02** (a GCN from scratch that has to compete against exactly these heuristics).

---

# Projekt 01 (basic) — Das Internet als Graph (deutsche Fassung)

**Format: Jupyter Notebook** (`internet_graph.ipynb`). **Warum?** Graphen versteht man durch
**Hinschauen**: Gradverteilungen plotten, Hubs finden, das Netz zerbrechen lassen. Explorative
Analyse mit vielen Plots — genau das Notebook-Format.

---

## Ziel

Du arbeitest mit der **echten Internet-Topologie** und reproduzierst zwei der berühmtesten
Ergebnisse der Netzwerkforschung — um dann zu lernen, wie leicht man sich bei genau solchen
Analysen selbst betrügt.

**Datensatz:** `oregon1_010331` (SNAP/Stanford) — die AS-Peering-Topologie aus **BGP**-Daten der
Oregon Route-Views, 31. März 2001. **10 670 autonome Systeme, 22 002 Peerings.** Echte
Messdaten, 69 KB, Download in ~1 s, danach lokal gecached (`datasets/`, gitignored).

*Warum echte Daten und kein synthetischer Graph?* Weil die spannenden Eigenschaften
(scale-free, Hub-Fragilität) genau die sind, die man bei synthetischen Graphen **hineinsteckt**
statt zu **finden**. Hier misst man sie.

## Vorwissen

Skript **Modul 16, Abschnitt 1** (Graph, Scale-Free, Robustheit) und **2.1/2.5** (Heuristiken,
Fallen). `networkx` wird im Notebook eingeführt.

## Aufgabe

Das meiste ist vorgegeben (basic!). Du füllst **vier** `# TODO`-Blöcke:

1. **Grad-Statistiken** (`degrees`-Array aus `G.degree()`),
2. **Power-Law-Exponent** — `alpha_naive` (log-log-Fit) **und** `alpha_mle` (Clauset-MLE),
3. **Robustheits-Schleife** — Knoten zufällig vs. nach Grad entfernen,
4. **Die vier Link-Prediction-Heuristiken** (Common Neighbors, Jaccard, Adamic-Adar, Pref.
   Attachment).

## Was am Ende herauskommt

**1. Scale-free.** Grad: min 1, **Median 2**, Mittel 4,12, **Max 2312**. Dichte 0,039 %.
Clustering 0,297. Es gibt **keinen typischen Grad** — der Mittelwert beschreibt niemanden
(dasselbe Muster wie die Flow-Größen in Modul 15).

**2. Der Exponent — und die erste Falle** ⚠️

| Methode | α |
|---|---|
| naiver log-log-Fit (PMF) | **1,14** ❌ |
| MLE, $d_{\min}=5$ | 2,12 |
| **MLE, $d_{\min}=10$** | **2,08** ✅ |

Der Literaturwert ist **≈ 2,1** (Faloutsos³ 1999). Der MLE trifft ihn, der naive Fit liegt um
**Faktor 2** daneben — und steht trotzdem in erschreckend vielen Papers.

**3. Robustheit** — der Albert/Jeong/Barabási-Befund (2000), auf echten Daten:

| entfernt | zufälliger Ausfall | **gezielt (Hubs)** |
|---|---|---|
| 1 % | 0,989 | **0,413** |
| 5 % | 0,939 | **0,003** |
| 10 % | 0,800 | 0,001 |

**5 % der Hubs entfernt → das Internet zerfällt in Staub (0,3 %)**, während 10 % *zufällige*
Ausfälle es kaum jucken (0,80). Dieselbe Struktur, die es gegen Pannen immunisiert, macht es
gegen Angreifer fragil. (Die Top-Hubs *sind* auch die Betweenness-Flaschenhälse: 16/20
Überschneidung.)

**4. Link Prediction — und die entscheidende Falle** 🕵️

Erster Durchlauf (uniform gezogene Negativbeispiele, wie überall üblich):

| Heuristik | ROC-AUC | PR-AUC |
|---|---|---|
| Common Neighbors | 0,719 | 0,712 |
| Adamic-Adar | 0,724 | 0,732 |
| **Preferential Attachment** | **0,763** | **0,857** |

**Preferential Attachment gewinnt** — ausgerechnet die *dümmste* Heuristik, die nur Grade
anschaut und die Nachbarschaft komplett ignoriert. Das ist verdächtig. Und tatsächlich:

Median-Gradprodukt $d_u\cdot d_v$: **echte Kanten 500**, **Zufallspaare 4**. In einem
scale-free-Graphen ist ein zufälliges Paar fast immer **Blatt × Blatt** — PA muss also gar keine
Struktur erkennen, es unterscheidet nur *Hub-beteiligt* von *Blatt×Blatt*.

Mit **grad-gematchten** Negativbeispielen (gleiche Gradverteilung wie die echten Kanten):

| Heuristik | uniform | **grad-gematcht** |
|---|---|---|
| Common Neighbors | 0,719 | 0,559 |
| Jaccard | 0,698 | 0,565 |
| **Adamic-Adar** | 0,724 | **0,566** ← jetzt bester |
| **Preferential Attachment** | **0,763** | **0,406** ← Zufallsniveau |

**PA stürzt von 0,763 auf 0,406.** Sein ganzer Vorsprung war **kein Signal, sondern ein Artefakt
der Negativ-Auswahl**. Und alle Verfahren fallen auf ~0,56: die Aufgabe ist **viel schwerer** als
die erste Tabelle suggeriert.

> ### Die drei Fallen dieses Projekts — alle vorgeführt, nicht behauptet
> 1. **Schätz-Falle:** naiver log-log-Fit statt MLE → Faktor 2 falsch.
> 2. **Leakage:** Testkanten müssen **vor** der Feature-Berechnung entfernt sein — sonst „sieht"
>    Common Neighbors die gesuchte Kante schon.
> 3. **Grad-Confound:** uniform gezogene Negative machen jede gradbasierte Größe künstlich stark.
>
> Dazu die Basisraten-Falle aus Modul 15: real ist $\pi\approx3{,}9\cdot10^{-4}$ (22 002 Kanten
> unter 57 Mio. Paaren), evaluiert wird 1:1-balanciert. **Nicht nur das Modell kann täuschen —
> schon die Konstruktion der Testmenge entscheidet, was man misst.**

Laufzeit: **~6 s** komplett.

## Ausführen / Setup

Repo-`venv`. Notebook öffnen:
`/.../xtAI_Learn-Repository/.venv/bin/python -m jupyter lab` (oder `.venv`-Kernel in VS Code).
Nur `networkx`, `numpy`, `matplotlib`, `scikit-learn` — alles vorhanden. Internet nur beim
**ersten** Laden nötig (der Download braucht einen `User-Agent`-Header, sonst 403).

## Lösung

Vollständig gelöst und **ausgeführt** in
[`solution/internet_graph_solution.ipynb`](solution/internet_graph_solution.ipynb). Erst selbst
probieren! Enthält vier Erweiterungs-Aufgaben (echten PPV via Bayes aus Modul 15 abschätzen,
Leakage bewusst herbeiführen, 1 % statt 10 % Testkanten, Robustheit gegen einen **Zufallsgraphen**
gleicher Größe vergleichen — der eigentliche Beweis, dass die Asymmetrie an der
scale-free-Struktur liegt).

> **Vorbehalt** (Skript 4.4): Die „echte" AS-Topologie kennt **niemand**. Route-Collectors sehen
> nur, was BGP announciert; **Peerings zwischen kleinen AS fehlen systematisch**. Wir arbeiten
> auf einem *unvollständigen, verzerrt gemessenen* Graphen — das gilt für **alle** Zahlen oben.

## Transfer

Die Graph-Grundbegriffe und besonders die **saubere Link-Prediction-Evaluation** sind die Basis
für **Projekt 02** (GCN from scratch, das gegen genau diese Heuristiken antreten muss).
