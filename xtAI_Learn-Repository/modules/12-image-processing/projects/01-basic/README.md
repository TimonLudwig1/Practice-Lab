# Projekt 01 (basic) — Punktoperationen & Histogrammausgleich

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
