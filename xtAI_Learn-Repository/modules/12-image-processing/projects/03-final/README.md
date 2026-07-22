# Project 03 (final) — Seam carving (content-aware image resizing)

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 12 — Image Processing** · Format: **Python project** (free implementation, *no* code given)

## Why this project?

Seam carving (Avidan & Shamir 2007) is a real **computational photography** method: it changes
the image size **without distorting important content** — instead of squeezing uniformly, it
specifically removes **unimportant** (low-energy) paths. The project connects two threads of
the studies: **image processing** (gradient energy from modules 11/12) and **dynamic
programming** (modules 06/07). It is elegant, pure `numpy`, CPU-fast (seconds) — and the
result is visually striking.

**No code given.** This README is the specification. A complete, tested reference solution is
in [`solution/`](solution/) — **build it yourself first.**

## Goal

Shrink an image in **width** by iteratively finding and removing the **lowest-energy vertical
seam**. The main subject should stay undistorted while the low-energy background gets
narrower.

## Prior knowledge

- **Script** section 6 (seam carving) and sections 1/2 (gradient/energy).
- **Dynamic programming** (modules 06/07) — the seam is a shortest-path problem.
- NumPy (slicing, `np.gradient`, `np.delete`, `np.minimum`).

## Technical specification

Build the four building blocks:

1. **Energy map** $e(i,j)$: a measure of "importance" per pixel, e.g. the summed **gradient
   magnitude** over the color channels ($|\partial_x I| + |\partial_y I|$). Smooth areas → low
   energy; edges/subjects → high energy.
2. **Cumulative minimum energy** via **DP** (top to bottom):
   $$M(i,j)=e(i,j)+\min\big(M(i{-}1,j{-}1),\,M(i{-}1,j),\,M(i{-}1,j{+}1)\big),$$
   guarding the borders with $+\infty$. $M(i,j)$ = the cost of the cheapest seam from the top
   down to $(i,j)$.
3. **Seam via backtracking**: start at the **minimum of the last row**, then go row by row
   upwards to the smallest of the (up to three) upper neighbours. The seam is **connected**
   (consecutive rows differ by at most 1 column).
4. **Remove the seam** (one pixel per row → width $-1$) and repeat the whole thing **$N$
   times** (recomputing the energy each time).

## Milestones

1. Implement + visualize the **energy** (the energy map alone is already instructive).
2. **DP** `cumulative_energy` — verify on a small hand example.
3. **`find_seam`** (backtracking) — check that the seam is connected.
4. **`remove_seam`** + **`carve`** (the loop); save before/after + the drawn-in seam.
5. **Analysis** (written, see below).

## What should work in the end

- An image gets narrower by e.g. **150 px**; the **main subject stays undistorted**, the
  low-energy background shrinks (for *Grace Hopper*: the person stays, the flag/area gets
  narrower).
- A figure with **original · energy map · first seam · result**.
- (Optional, for extra points) **seam insertion** for *enlarging*, or object
  protection/removal via a manipulated energy map.

## Analysis (written, `ANALYSIS.md`)

1. **Why DP and not greedy?** What would go wrong if you chose the seam greedily (in each row
   the locally smallest $e$) instead of by DP? (Optimality of the path.)
2. **Complexity:** how does one seam scale in $H, W$? And $N$ seams? Why is this practical
   despite recomputing the energy?
3. **Energy choice:** how does the result change with a different energy (e.g. Sobel magnitude
   or entropy)? Where does seam carving visibly remove "wrong" seams, and why?
4. **Limits:** for which images (e.g. dense, uniform texture; clear geometric structures) does
   content-aware resizing fail and produce artifacts?

## Assessment criteria (master's level)

- **Correct DP** (recurrence + border handling) and a **connected** seam.
- A clean, reproducible carve loop; a convincing before/after visualization.
- An analysis that penetrates the DP core (optimality), the complexity and the energy choice.
- Stayed **CPU-friendly** (pure NumPy, seconds — no training).

## Setup

```bash
source ../../../../.venv/bin/activate
# build your own implementation, then e.g.:
#   python run.py --seams 150     # before/after in results/
#   python test_seam.py           # test suite (fast)
```

Requires `numpy`, `matplotlib`, `Pillow`. The example image (*Grace Hopper*) is included in
matplotlib — no download.

## Reference solution

Complete in [`solution/`](solution/): `seam_carving.py`, `run.py`, `test_seam.py` (6 tests
green). Reference: 512→362 px in ~1 s, the subject undistorted. **Build it yourself first.**

---

# Projekt 03 (final) — Seam Carving (Content-Aware Image Resizing) (deutsche Fassung)

**Modul 12 — Image Processing** · Format: **Python-Projekt** (freie Umsetzung, *keine* Code-Vorgabe)

## Warum dieses Projekt?

Seam Carving (Avidan & Shamir 2007) ist ein echtes **Computational-Photography**-Verfahren:
Es ändert die Bildgröße, **ohne wichtige Inhalte zu verzerren** — statt gleichmäßig zu
stauchen, entfernt es gezielt **unwichtige** (energiearme) Pfade. Das Projekt verbindet
zwei Fäden des Studiums: **Bildverarbeitung** (Gradienten-Energie aus Modul 11/12) und
**dynamische Programmierung** (Modul 06/07). Es ist elegant, rein `numpy`, CPU-schnell
(Sekunden) — und das Ergebnis ist visuell verblüffend.

**Keine Code-Vorgabe.** Diese README ist die Spezifikation. Eine vollständige, getestete
Musterlösung liegt in [`solution/`](solution/) — **erst selbst bauen.**

## Ziel

Ein Bild in der **Breite** verkleinern, indem du iterativ die **energieärmste senkrechte
Naht** findest und entfernst. Das Hauptmotiv soll unverzerrt bleiben, während der
energiearme Hintergrund schmaler wird.

## Vorwissen

- **Skript** Abschnitt 6 (Seam Carving) und Abschnitt 1/2 (Gradient/Energie).
- **Dynamische Programmierung** (Modul 06/07) — die Naht ist ein kürzester-Pfad-Problem.
- NumPy (Slicing, `np.gradient`, `np.delete`, `np.minimum`).

## Fachliche Spezifikation

Baue die vier Bausteine:

1. **Energie-Karte** $e(i,j)$: ein Maß für „Wichtigkeit" pro Pixel, z. B. der summierte
   **Gradientenbetrag** über die Farbkanäle ($|\partial_x I| + |\partial_y I|$). Glatte
   Flächen → niedrige Energie; Kanten/Motive → hohe Energie.
2. **Kumulierte Minimalenergie** per **DP** (von oben nach unten):
   $$M(i,j)=e(i,j)+\min\big(M(i{-}1,j{-}1),\,M(i{-}1,j),\,M(i{-}1,j{+}1)\big),$$
   Ränder mit $+\infty$ absichern. $M(i,j)$ = Kosten der billigsten Naht von oben bis
   $(i,j)$.
3. **Naht per Backtracking**: Start am **Minimum der letzten Zeile**, dann Zeile für Zeile
   nach oben je zum kleinsten der (bis zu drei) oberen Nachbarn. Die Naht ist
   **zusammenhängend** (aufeinanderfolgende Zeilen unterscheiden sich um höchstens 1 Spalte).
4. **Naht entfernen** (ein Pixel pro Zeile → Breite $-1$) und das Ganze **$N$-mal**
   wiederholen (Energie jeweils neu berechnen).

## Milestones

1. **Energie** implementieren + visualisieren (die Energie-Karte allein ist schon lehrreich).
2. **DP** `cumulative_energy` — an einem kleinen Handbeispiel verifizieren.
3. **`find_seam`** (Backtracking) — prüfen, dass die Naht zusammenhängend ist.
4. **`remove_seam`** + **`carve`** (Schleife); Vorher/Nachher + eingezeichnete Naht speichern.
5. **Analyse** (schriftlich, s. u.).

## Was am Ende funktionieren soll

- Ein Bild wird um z. B. **150 px** schmaler; das **Hauptmotiv bleibt unverzerrt**, der
  energiearme Hintergrund schrumpft (bei *Grace Hopper*: die Person bleibt, die Flagge/
  Fläche wird schmaler).
- Eine Abbildung mit **Original · Energie-Karte · erster Naht · Ergebnis**.
- (Optional, für Extrapunkte) **Nahteinfügen** zum *Vergrößern*, oder Objektschutz/-entfernung
  über eine manipulierte Energie-Karte.

## Analyse (schriftlich, `ANALYSIS.md`)

1. **Warum DP und nicht Greedy?** Was ginge schief, wenn man die Naht greedy (in jeder Zeile
   lokal das kleinste $e$) statt per DP wählte? (Optimalität des Pfades.)
2. **Komplexität:** Wie skaliert eine Naht in $H, W$? Und $N$ Nähte? Warum ist das trotz
   Neuberechnung der Energie praktikabel?
3. **Energiewahl:** Wie ändert sich das Ergebnis mit einer anderen Energie (z. B. Sobel-Betrag
   oder Entropie)? Wo entfernt Seam Carving sichtbar „falsche" Nähte, und warum?
4. **Grenzen:** Bei welchen Bildern (z. B. dichte, gleichmäßige Textur; klare geometrische
   Strukturen) versagt content-aware Resizing und erzeugt Artefakte?

## Bewertungsmaßstab (Master-Niveau)

- **Korrekte DP** (Rekurrenz + Randbehandlung) und **zusammenhängende** Naht.
- Sauberer, reproduzierbarer Carve-Loop; überzeugende Vorher/Nachher-Visualisierung.
- Analyse, die den DP-Kern (Optimalität), die Komplexität und die Energiewahl durchdringt.
- **CPU-freundlich** (reines NumPy, Sekunden — kein Training).

## Setup

```bash
source ../../../../.venv/bin/activate
# eigene Umsetzung bauen, dann z. B.:
#   python run.py --seams 150     # Vorher/Nachher in results/
#   python test_seam.py           # Testsuite (schnell)
```

Benötigt `numpy`, `matplotlib`, `Pillow`. Beispielbild (*Grace Hopper*) ist in matplotlib
enthalten — kein Download.

## Musterlösung

Vollständig in [`solution/`](solution/): `seam_carving.py`, `run.py`, `test_seam.py`
(6 Tests grün). Referenz: 512→362 px in ~1 s, Motiv unverzerrt. **Erst selbst bauen.**
