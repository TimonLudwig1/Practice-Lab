# Project 01 (basic): MLP + backpropagation by hand

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

## Goal

Implement a complete multilayer perceptron in **pure NumPy**: the forward pass, the four backprop equations (script, section 1.5), gradient checking and training with minibatch SGD + momentum — on the `make_moons` data set, whose 2D structure makes the learned decision boundary visible.

**Why this format (Jupyter notebook):** the learning effect lives on the alternation of derivation → code → immediate visualization (learning curves, decision boundaries). A notebook holds both together.

**Why synthetic data (`make_moons`):** a deliberate choice — not linearly separable (it shows *what* you need hidden layers for), two-dimensional (the learned function is visible as a contour plot), reproducible (`random_state=42`) and generated in a single notebook cell. Real data would only distract here; they are the subject from project 02 onwards.

## Prior knowledge

- Script sections 1.1–1.6 (MLP, loss from maximum likelihood, the backprop derivation, gradient checking) and 2.1–2.2 (SGD/momentum, He initialization)
- NumPy basics (broadcasting, the matrix product `@`)

## Setup

The shared repository environment is enough (see `SETUP.md` in the root directory):

```bash
source .venv/bin/activate
jupyter lab modules/05-machine-learning-2/projects/01-basic/mlp_from_scratch.ipynb
```

## Tasks (step by step)

Work through `mlp_from_scratch.ipynb`. The structure:

1. **Data:** generate `make_moons`, standardize, train/test split (given).
2. **Forward pass:** a $2 \to 32 \to 32 \to 1$ network, ReLU, logit output, a stable BCE-with-logits loss (given — read the code closely anyway, in particular the numerically stable loss).
3. **TODO 1–3 — backpropagation:** in `MLP.backward`, implement the error signal of the output layer ($\boldsymbol{\delta}^{(L)} = \mathbf{p} - \mathbf{y}$), the parameter gradients (an outer product, averaged over the batch) and the $\delta$ recursion.
4. **Gradient check:** the checking cell compares your analytical gradient with central differences. **Passing = a relative deviation $< 10^{-7}$.** Only continue once that is green.
5. **TODO 4 — the momentum update:** $\mathbf{v} \leftarrow \mu\mathbf{v} + \mathbf{g}$, $\theta \leftarrow \theta - \eta\mathbf{v}$.
6. **Training and visualization:** learning curves, the decision boundary, a width experiment ($2{\to}2{\to}1$ vs. $2{\to}16{\to}1$ vs. $2{\to}128{\to}128{\to}1$).

## What should work in the end

- Gradient check: a relative deviation $< 10^{-7}$ (the reference solution reaches about $10^{-10}$).
- Test accuracy $\ge 0.94$ with the standard architecture (reference: $0.95$ at `noise=0.25`).
- You can explain why the $2{\to}2{\to}1$ network fails structurally.

## Reference solution

In [`solution/solution.ipynb`](solution/solution.ipynb) — fully executed with all outputs. Try it yourself first; the gradient check tells you reliably whether your backprop is correct.

---
---

# Projekt 01 (basic): MLP + Backpropagation von Hand (deutsche Fassung)

## Ziel

Ein vollständiges Multilayer-Perzeptron in **purem NumPy** implementieren: Forward Pass, die vier Backprop-Gleichungen (Skript, Abschnitt 1.5), Gradient Checking und Training mit Minibatch-SGD + Momentum — auf dem `make_moons`-Datensatz, dessen 2D-Struktur die gelernte Entscheidungsgrenze sichtbar macht.

**Warum dieses Format (Jupyter Notebook):** Der Lerneffekt lebt vom Wechsel aus Herleitung → Code → sofortiger Visualisierung (Lernkurven, Entscheidungsgrenzen). Ein Notebook hält beides zusammen.

**Warum synthetische Daten (`make_moons`):** bewusst gewählt — nicht linear separierbar (zeigt, *wofür* man verborgene Schichten braucht), zweidimensional (die gelernte Funktion ist als Konturplot sichtbar), reproduzierbar (`random_state=42`) und in einer Notebook-Zelle erzeugt. Echte Daten würden hier nur ablenken; um sie geht es ab Projekt 02.

## Vorwissen

- Skript-Abschnitte 1.1–1.6 (MLP, Loss aus Maximum Likelihood, Backprop-Herleitung, Gradient Checking) und 2.1–2.2 (SGD/Momentum, He-Initialisierung)
- NumPy-Grundlagen (Broadcasting, Matrixprodukt `@`)

## Setup

Gemeinsame Repo-Umgebung reicht (siehe `SETUP.md` im Wurzelverzeichnis):

```bash
source .venv/bin/activate
jupyter lab modules/05-machine-learning-2/projects/01-basic/mlp_from_scratch.ipynb
```

## Aufgabenstellung (Schritt für Schritt)

Arbeite `mlp_from_scratch.ipynb` durch. Die Struktur:

1. **Daten:** `make_moons` erzeugen, standardisieren, Train/Test-Split (fertig vorgegeben).
2. **Forward Pass:** Netz $2 \to 32 \to 32 \to 1$, ReLU, Logit-Ausgabe, stabile BCE-with-Logits-Loss (fertig vorgegeben — lies den Code trotzdem genau, insbesondere die numerisch stabile Loss).
3. **TODO 1–3 — Backpropagation:** Implementiere in `MLP.backward` das Fehlersignal der Ausgabeschicht ($\boldsymbol{\delta}^{(L)} = \mathbf{p} - \mathbf{y}$), die Parametergradienten (äußeres Produkt, batch-gemittelt) und die $\delta$-Rekursion.
4. **Gradient Check:** Die Prüfzelle vergleicht deinen analytischen Gradienten mit zentralen Differenzen. **Bestanden = relative Abweichung $< 10^{-7}$.** Erst weitermachen, wenn das grün ist.
5. **TODO 4 — Momentum-Update:** $\mathbf{v} \leftarrow \mu\mathbf{v} + \mathbf{g}$, $\theta \leftarrow \theta - \eta\mathbf{v}$.
6. **Training & Visualisierung:** Lernkurven, Entscheidungsgrenze, Breiten-Experiment ($2{\to}2{\to}1$ vs. $2{\to}16{\to}1$ vs. $2{\to}128{\to}128{\to}1$).

## Was am Ende funktionieren soll

- Gradient Check: relative Abweichung $< 10^{-7}$ (Referenzlösung erreicht $\sim 10^{-10}$).
- Test-Accuracy $\ge 0{,}94$ mit der Standardarchitektur (Referenz: $0{,}95$ bei `noise=0.25`).
- Du kannst erklären, warum das $2{\to}2{\to}1$-Netz strukturell scheitert.

## Musterlösung

In [`solution/solution.ipynb`](solution/solution.ipynb) — vollständig ausgeführt mit allen Outputs. Erst selbst probieren; der Gradient Check sagt dir zuverlässig, ob dein Backprop stimmt.
