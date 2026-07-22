# Project 01 (basic) — Convolution & filters by hand, and learned filters

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The notebook itself is English only.

**Module 11 — Computer Vision** · Format: **Jupyter notebook** (`convolution_filters.ipynb`)

## Why this format?

You understand the convolution by *computing and seeing* it: sliding a kernel over the
image, displaying the result. A notebook connects the implementation, the filters and the
visualization with immediately visible images — ideal for this guided start.

## Goal

You implement the **2D convolution** yourself, apply classical **hand-designed** filters to
a real photo and then visualize the first filters *learned* by a **pretrained ResNet**. Core
points (script sections 1–2):

- 2D convolution with zero padding from scratch;
- **Sobel edge detection** via the gradient magnitude $G=\sqrt{G_x^2+G_y^2}$; Gaussian &
  sharpening;
- **the central CNN insight:** a network does not *design* filters, it *learns* them — the
  64 first filters of ResNet18 look like learned edge/color detectors, strikingly similar to
  your Sobel/Gaussian kernels;
- feature maps after the first convolution layer.

## Prior knowledge

- **Script** sections 1–2 (image tensor, convolution, padding/stride, classical filters,
  CNN principles).
- Python/NumPy (slicing, broadcasting), the basic idea of `matplotlib.imshow`.

## Setup

Requires `numpy`, `matplotlib`, `torch`, `torchvision` (repo `requirements.txt`). The
example image (*Grace Hopper*) is **included** in matplotlib — no download. The pretrained
**ResNet18** (~45 MB) loads once into the torch cache on the first run.

```bash
source ../../../../.venv/bin/activate
jupyter lab      # or open convolution_filters.ipynb in VS Code, kernel = repo .venv
```

Runs in **seconds on the CPU**, no training.

## Assignment (step by step)

**Part A** (load the image, grayscale) is given. Then three tasks (marked with `# TODO`):

1. **`convolve2d`** by hand (zero padding, vectorized over the kernel positions); verified
   with an identity and a box-blur test.
2. **Sobel edges**: $G_x, G_y$ and the gradient magnitude $G$ (Gaussian & sharpening are
   given).
3. **Learned filters**: visualize the `conv1` filters of ResNet18 (given) and send the image
   through `conv1` to show **feature maps**.

At the end a short written **reflection part** (4 questions).

## What should work in the end

- The self-implemented convolution passes the identity test; box blur/Gaussian smooth.
- The gradient magnitude clearly traces the **contours** of the image.
- The 64 learned ResNet `conv1` filters show oriented edges and color blobs; the feature
  maps $(1,64,112,112)$ highlight different image structures.

## Reference solution

A fully filled-in, **executed** notebook is in
[`solution/convolution_filters_solution.ipynb`](solution/convolution_filters_solution.ipynb).
Try it yourself first — the stub cells raise `NotImplementedError` until you fill them. The
reference answers to the reflection questions are at the end of that solution notebook.

---

# Projekt 01 (basic) — Faltung & Filter von Hand, und gelernte Filter (deutsche Fassung)

**Modul 11 — Computer Vision** · Format: **Jupyter Notebook** (`convolution_filters.ipynb`)

## Warum dieses Format?

Die Faltung versteht man, indem man sie *rechnet und sieht*: einen Kernel übers Bild
schieben, das Ergebnis anzeigen. Ein Notebook verbindet die Implementierung, die Filter und
die Visualisierung mit sofort sichtbaren Bildern — ideal für diesen geführten Einstieg.

## Ziel

Du implementierst die **2D-Faltung** selbst, wendest klassische **hand-entworfene** Filter
auf ein echtes Foto an und visualisierst dann die von einem **vortrainierten ResNet**
*gelernten* ersten Filter. Kernpunkte (Skript-Abschnitte 1–2):

- 2D-Faltung mit Zero-Padding von Grund auf;
- **Sobel-Kantendetektion** über den Gradientenbetrag $G=\sqrt{G_x^2+G_y^2}$; Gauß & Schärfen;
- **die zentrale CNN-Erkenntnis:** ein Netz *entwirft* Filter nicht, es *lernt* sie — die
  64 ersten Filter von ResNet18 sehen aus wie gelernte Kanten-/Farbdetektoren, verblüffend
  ähnlich zu deinen Sobel/Gauß-Kerneln;
- Feature-Maps nach der ersten Faltungsschicht.

## Vorwissen

- **Skript** Abschnitt 1–2 (Bild-Tensor, Faltung, Padding/Stride, klassische Filter,
  CNN-Prinzipien).
- Python/NumPy (Slicing, Broadcasting), Grundidee von `matplotlib.imshow`.

## Setup

Benötigt `numpy`, `matplotlib`, `torch`, `torchvision` (Repo-`requirements.txt`). Das
Beispielbild (*Grace Hopper*) ist in matplotlib **enthalten** — kein Download. Das
vortrainierte **ResNet18** (~45 MB) lädt beim ersten Lauf einmalig in den torch-Cache.

```bash
source ../../../../.venv/bin/activate
jupyter lab      # oder convolution_filters.ipynb in VS Code öffnen, Kernel = Repo-.venv
```

Läuft in **Sekunden auf der CPU**, kein Training.

## Aufgabenstellung (Schritt für Schritt)

**Teil A** (Bild laden, Graustufen) ist vorgegeben. Dann drei Aufgaben (an `# TODO`):

1. **`convolve2d`** von Hand (Zero-Padding, vektorisiert über die Kernel-Positionen);
   verifiziert per Identitäts- und Box-Blur-Test.
2. **Sobel-Kanten**: $G_x, G_y$ und der Gradientenbetrag $G$ (Gauß & Schärfen sind gegeben).
3. **Gelernte Filter**: die `conv1`-Filter von ResNet18 visualisieren (gegeben) und das Bild
   durch `conv1` schicken, um **Feature-Maps** zu zeigen.

Zum Schluss ein kurzer schriftlicher **Reflexionsteil** (4 Fragen).

## Was am Ende funktionieren soll

- Die selbst implementierte Faltung besteht den Identitäts-Test; Box-Blur/Gauß glätten.
- Der Gradientenbetrag zeichnet die **Konturen** des Bildes klar nach.
- Die 64 gelernten ResNet-`conv1`-Filter zeigen gerichtete Kanten und Farb-Blobs; die
  Feature-Maps $(1,64,112,112)$ heben verschiedene Bildstrukturen hervor.

## Musterlösung

Voll ausgefülltes, **ausgeführtes** Notebook unter
[`solution/convolution_filters_solution.ipynb`](solution/convolution_filters_solution.ipynb). Erst selbst
probieren — die Stub-Zellen werfen `NotImplementedError`, bis du sie füllst.
