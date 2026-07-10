# Projekt 01 (basic) — Faltung & Filter von Hand, und gelernte Filter

**Modul 11 — Computer Vision** · Format: **Jupyter Notebook** (`faltung_filter.ipynb`)

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
jupyter lab      # oder faltung_filter.ipynb in VS Code öffnen, Kernel = Repo-.venv
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
[`loesung/faltung_filter_loesung.ipynb`](loesung/faltung_filter_loesung.ipynb). Erst selbst
probieren — die Stub-Zellen werfen `NotImplementedError`, bis du sie füllst.
