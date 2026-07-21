# Projekt 02 (medium) — Ein GCN from scratch (und was die Evaluation daraus macht)

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

1. **`normalisierte_adjazenz(kanten, n)`** — $\hat A=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}$ als
   **sparse** COO-Tensor. Der Bauplan steht im Docstring. Denk an: beide Kantenrichtungen,
   Self-Loops, und `Inf` abfangen (isolierte Knoten!).
2. **`GCN.forward`** — zwei Schichten: `relu(W1(A_hat @ emb))`, dann `W2(A_hat @ H)`.
3. **`MLPOhneStruktur.forward`** — dasselbe **ohne** jede `A_hat`-Multiplikation (Kontrollgruppe).
4. **`kanten_score`** — zeilenweises Skalarprodukt der Endpunkt-Embeddings.

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
