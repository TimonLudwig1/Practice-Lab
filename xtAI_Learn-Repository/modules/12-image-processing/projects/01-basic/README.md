# Project 01 (basic) — Point operations & histogram equalization

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Module 12 — Image Processing** · Format: **Jupyter notebook** (`point_ops_histogram.ipynb`)

## Why this format?

You understand image operations by seeing the result *and* the histogram. A notebook connects
the implementation with the image + distribution — ideal for this guided start.

## Goal

You implement **point operations** and **histogram equalization** by hand and make their
effect visible (script section 1):

- a linear point operation ($s=a\,r+b$) and **gamma** ($s=255(r/255)^\gamma$) with clipping;
- the **histogram** as a brightness distribution;
- **histogram equalization** via the **CDF** ($s=(L-1)\cdot\text{cdf}(r)$) — the contrast is
  stretched, and the CDF afterwards becomes almost a straight line.

## Prior knowledge

- **Script** section 1 (point operations, gamma, histogram, equalization).
- NumPy (indexing, `bincount`, `cumsum`), `matplotlib`.

## Setup

Requires `numpy`, `matplotlib`, `Pillow` (repo `requirements.txt`). The example image
(*Grace Hopper*) is **included** in matplotlib — no download.

```bash
source ../../../../.venv/bin/activate
jupyter lab      # or open the notebook in VS Code, kernel = repo .venv
```

Runs in **seconds** (pure NumPy, no training).

## Assignment (step by step)

**Part A** (image + histogram) is given. Then three tasks (marked with `# TODO`):

1. **`linear(img, a, b)`** and **`gamma(img, g)`** — with clipping to $[0,255]$.
2. **`equalize(img)`** — histogram → CDF → mapping table `T = round(255·cdf)` → `T[img]`.
3. **Plot the CDF** before/after equalization (after equalization ≈ the diagonal).

At the end a short written **reflection part** (4 questions).

## What should work in the end

- Linear/gamma change brightness/contrast visibly; gamma 0.5 lifts shadow detail.
- Equalization stretches a low-contrast image to (almost) the full range 0–255; the histogram
  becomes wider/flatter, the CDF approaches the diagonal.

## Reference solution

A fully filled-in, **executed** notebook is in
[`solution/point_ops_histogram_solution.ipynb`](solution/point_ops_histogram_solution.ipynb).
Try it yourself first — the stub cells raise `NotImplementedError`. The reference answers to
the reflection questions are at the end of that solution notebook.

---

# Projekt 01 (basic) — Punktoperationen & Histogrammausgleich (deutsche Fassung)

**Modul 12 — Image Processing** · Format: **Jupyter Notebook** (`point_ops_histogram.ipynb`)

## Warum dieses Format?

Bildoperationen versteht man, indem man das Ergebnis *und* das Histogramm sieht. Ein
Notebook verbindet die Implementierung mit Bild + Verteilung — ideal für diesen geführten
Einstieg.

## Ziel

Du implementierst **Punktoperationen** und den **Histogrammausgleich** von Hand und machst
ihre Wirkung sichtbar (Skript-Abschnitt 1):

- lineare Punktoperation ($s=a\,r+b$) und **Gamma** ($s=255(r/255)^\gamma$) mit Clipping;
- das **Histogramm** als Helligkeitsverteilung;
- **Histogrammausgleich** über die **CDF** ($s=(L-1)\cdot\text{cdf}(r)$) — der Kontrast
  wird gestreckt, die CDF wird danach fast eine Gerade.

## Vorwissen

- **Skript** Abschnitt 1 (Punktoperationen, Gamma, Histogramm, Ausgleich).
- NumPy (Indexierung, `bincount`, `cumsum`), `matplotlib`.

## Setup

Benötigt `numpy`, `matplotlib`, `Pillow` (Repo-`requirements.txt`). Das Beispielbild
(*Grace Hopper*) ist in matplotlib **enthalten** — kein Download.

```bash
source ../../../../.venv/bin/activate
jupyter lab      # oder das Notebook in VS Code öffnen, Kernel = Repo-.venv
```

Läuft in **Sekunden** (reines NumPy, kein Training).

## Aufgabenstellung (Schritt für Schritt)

**Teil A** (Bild + Histogramm) ist vorgegeben. Dann drei Aufgaben (an `# TODO`):

1. **`linear(img, a, b)`** und **`gamma(img, g)`** — mit Clipping auf $[0,255]$.
2. **`equalize(img)`** — Histogramm → CDF → Abbildungstabelle `T = round(255·cdf)` → `T[img]`.
3. **CDF plotten** vor/nach Ausgleich (nach dem Ausgleich ≈ Diagonale).

Zum Schluss ein kurzer schriftlicher **Reflexionsteil** (4 Fragen).

## Was am Ende funktionieren soll

- Linear/Gamma verändern Helligkeit/Kontrast sichtbar; Gamma 0.5 hebt Schattendetails.
- Der Ausgleich streckt ein kontrastarmes Bild auf (fast) den vollen Bereich 0–255; das
  Histogramm wird breiter/flacher, die CDF nähert sich der Diagonale.

## Musterlösung

Voll ausgefülltes, **ausgeführtes** Notebook unter
[`solution/point_ops_histogram_solution.ipynb`](solution/point_ops_histogram_solution.ipynb).
Erst selbst probieren — die Stub-Zellen werfen `NotImplementedError`.
