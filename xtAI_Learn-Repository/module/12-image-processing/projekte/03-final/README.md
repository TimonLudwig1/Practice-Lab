# Projekt 03 (final) — Seam Carving (Content-Aware Image Resizing)

**Modul 12 — Image Processing** · Format: **Python-Projekt** (freie Umsetzung, *keine* Code-Vorgabe)

## Warum dieses Projekt?

Seam Carving (Avidan & Shamir 2007) ist ein echtes **Computational-Photography**-Verfahren:
Es ändert die Bildgröße, **ohne wichtige Inhalte zu verzerren** — statt gleichmäßig zu
stauchen, entfernt es gezielt **unwichtige** (energiearme) Pfade. Das Projekt verbindet
zwei Fäden des Studiums: **Bildverarbeitung** (Gradienten-Energie aus Modul 11/12) und
**dynamische Programmierung** (Modul 06/07). Es ist elegant, rein `numpy`, CPU-schnell
(Sekunden) — und das Ergebnis ist visuell verblüffend.

**Keine Code-Vorgabe.** Diese README ist die Spezifikation. Eine vollständige, getestete
Musterlösung liegt in [`loesung/`](loesung/) — **erst selbst bauen.**

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

## Analyse (schriftlich, `ANALYSE.md`)

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
#   python run.py --seams 150     # Vorher/Nachher in ergebnisse/
#   python test_seam.py           # Testsuite (schnell)
```

Benötigt `numpy`, `matplotlib`, `Pillow`. Beispielbild (*Grace Hopper*) ist in matplotlib
enthalten — kein Download.

## Musterlösung

Vollständig in [`loesung/`](loesung/): `seam_carving.py`, `run.py`, `test_seam.py`
(6 Tests grün). Referenz: 512→362 px in ~1 s, Motiv unverzerrt. **Erst selbst bauen.**
