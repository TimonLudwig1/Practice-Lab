# Projekt 01 (basic): MLP + Backpropagation von Hand

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
