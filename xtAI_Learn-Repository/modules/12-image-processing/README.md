# Modul 12 — Image Processing and Computational Photography

> **Worum geht es?** Während Modul 11 Bilder *verstehen* wollte (Klassifikation via CNN),
> geht es hier um das *Verarbeiten und Erzeugen* von Bildern mit **klassischen,
> deterministischen** Methoden — ohne Training. Von Punktoperationen und
> Histogrammausgleich über Filterung im **Orts- und Frequenzraum** (Fourier), Kanten,
> Morphologie und Interpolation bis zur **Computational Photography** (HDR, Entrauschen,
> Panorama, content-aware Resizing). Das ist das Handwerkszeug, das *unter* jeder
> Bild-Pipeline liegt — und alles läuft mit `numpy`/`scipy` auf der CPU.

**Hilfreiche Vorkenntnisse:** lineare Algebra, etwas Signalverarbeitung/Analysis
(Sinus/Cosinus, Fourier-Idee), NumPy.

**Diese Module solltest du vorher gemacht haben:**
- **Modul 11 (Computer Vision)** — die **Faltung** und klassische Filter (Sobel/Gauß) hast
  du dort schon von Hand gebaut; hier vertiefen wir Filterung und ergänzen den
  **Frequenzraum**. Die **dynamische Programmierung** (Modul 06/07) kehrt beim Seam Carving
  wieder.

---

## Lernziele

Nach diesem Modul kannst du …

- Bilder als abgetastete, quantisierte Signale verstehen und **Punktoperationen**
  (Helligkeit, Kontrast, **Gamma**) sowie **Histogrammausgleich** anwenden;
- lineare (Mittel/Gauß) und **nichtlineare** (Median, bilateral) **Filter** einordnen und
  wissen, *welcher gegen welches Rauschen* hilft;
- die **2D-Fourier-Transformation** erklären, ein Bild im **Frequenzraum** interpretieren
  und **Tief-/Hochpassfilter** anwenden — inklusive des **Faltungssatzes**;
- **Kanten** (Gradient, Laplacian, **Canny**-Pipeline), **Morphologie** (Erosion/Dilatation/
  Opening/Closing) und **Interpolation** (Nearest/Bilinear/Bicubic) beschreiben;
- **Farbräume** (RGB/HSV/Lab/YCbCr) und ihren Nutzen benennen;
- **Computational-Photography**-Verfahren erklären: **HDR/Tone-Mapping**, **Entrauschen**,
  **Panorama-Stitching** (Homographie), **Seam Carving** (content-aware Resizing).

---

## 1 · Grundlagen — Bild, Punktoperationen, Histogramm

### 1.1 Das Bild als Signal

Ein digitales Bild entsteht durch **Abtastung** (sampling — das kontinuierliche Bild wird
auf ein Pixelraster diskretisiert) und **Quantisierung** (jeder Pixel bekommt einen
diskreten Wert, meist 8 Bit = $0$–$255$). Zu grobe Abtastung → **Aliasing** (Treppen,
Moiré); zu grobe Quantisierung → **Banding** (sichtbare Helligkeitsstufen). Das
Abtasttheorem (Nyquist) verlangt eine Abtastrate $>2\times$ der höchsten Bildfrequenz —
deshalb **glättet** man vor dem Verkleinern (Anti-Aliasing).

### 1.2 Punktoperationen

Eine **Punktoperation** bildet jeden Pixelwert $r$ *unabhängig* von seinen Nachbarn auf
$s=T(r)$ ab:
- **Helligkeit/Kontrast (linear):** $s = a\,r + b$ ($a$ = Kontrast, $b$ = Helligkeit).
- **Negativ:** $s = 255 - r$.
- **Gamma-Korrektur:** $s = 255\,(r/255)^{\gamma}$. $\gamma<1$ hellt dunkle Bereiche auf,
  $\gamma>1$ dunkelt ab. Wichtig, weil Displays/Kameras nichtlinear (gamma-kodiert) sind.
- **Schwellwert (Thresholding):** $s = 255$ falls $r>t$, sonst $0$ — erzeugt ein
  Binärbild.

### 1.3 Histogramm & Histogrammausgleich

Das **Histogramm** $h(k)$ zählt, wie oft der Grauwert $k$ vorkommt — es beschreibt die
**Helligkeitsverteilung**. Ein kontrastarmes Bild hat ein schmales Histogramm.

**Histogrammausgleich (equalization)** streckt das Histogramm, um den Kontrast zu
maximieren. Idee: benutze die **kumulative Verteilung** (CDF) als Abbildungsfunktion.
Mit normiertem Histogramm $p(k)=h(k)/N$ und
$$\text{cdf}(k)=\sum_{j=0}^{k} p(j)$$
ist die Ausgleichs-Abbildung
$$s = T(r) = \operatorname{round}\!\big((L-1)\cdot \text{cdf}(r)\big),\qquad L=256.$$
Das Ergebnis hat eine (annähernd) **gleichverteilte** Intensität — dunkle und helle
Details werden sichtbar. (Projekt 01 baut genau das.)

---

## 2 · Filterung im Ortsraum

### 2.1 Lineare Filter (Faltung, Recap aus Modul 11)

Eine **lineare** Filterung ist eine **Faltung** mit einem Kernel (Modul 11, Abschnitt 1):
$$(I*K)(i,j)=\sum_{u,v} I(i+u,j+v)\,K(u,v).$$
- **Mittelwert-/Box-Filter** und **Gauß-Filter**: **glätten**, unterdrücken Rauschen,
  verwischen aber auch Kanten. Der Gauß ist **separierbar** ($2$D $=$ zwei $1$D-Faltungen)
  → effizient.
- **Ableitungsfilter** (Sobel, Prewitt): betonen Kanten (Gradient).
- **Laplacian** $\nabla^2$: zweite Ableitung, betont Kanten/Details isotrop.
- **Unsharp Masking (Schärfen):** $I_{\text{scharf}} = I + \alpha\,(I - I_{\text{blur}})$ —
  die „unscharfe Maske" $I-I_{\text{blur}}$ ist der Hochfrequenzanteil, den man verstärkt.

### 2.2 Nichtlineare Filter

- **Median-Filter:** ersetzt jeden Pixel durch den **Median** seiner Nachbarschaft.
  Ideal gegen **Salz-und-Pfeffer-Rauschen** (Ausreißer), weil der Median robust ist und —
  anders als der Gauß — **Kanten erhält**.
- **Bilateraler Filter:** glättet nur zwischen Pixeln, die räumlich **und** in der
  Intensität ähnlich sind → **kantenerhaltende** Glättung. Gewicht
  $w(i,j;k,l)=\exp\!\big(-\tfrac{\lVert(i,j)-(k,l)\rVert^2}{2\sigma_s^2}\big)\,
  \exp\!\big(-\tfrac{(I_{ij}-I_{kl})^2}{2\sigma_r^2}\big)$ (räumlich × tonal).
- **Non-Local Means (NLM):** mittelt über *ganze ähnliche Patches* im Bild — sehr
  wirksames Entrauschen.

**Merke:** Gauß gegen **gaußsches** Rauschen, Median gegen **Impuls**-(Salz&Pfeffer-)
Rauschen. (Projekt 02 zeigt das empirisch.)

---

## 3 · Der Frequenzraum — Fourier

### 3.1 Idee

Die **Fourier-Transformation** zerlegt ein Signal in **Sinus-/Cosinus-Schwingungen**
verschiedener Frequenzen. Auf ein Bild angewandt (2D): **niedrige** Frequenzen =
langsame Helligkeitsverläufe (grobe Struktur, Flächen), **hohe** Frequenzen = schnelle
Wechsel (Kanten, Texturen, Rauschen). Die **2D-DFT** eines $M\times N$-Bildes:
$$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1} I(x,y)\,
e^{-2\pi i\,(ux/M + vy/N)}.$$
Man berechnet sie effizient mit der **FFT** ($O(MN\log MN)$ statt $O(M^2N^2)$). Das
**Magnitudenspektrum** $|F(u,v)|$ (log-skaliert, mit `fftshift` zentriert) visualisiert,
welche Frequenzen/Orientierungen im Bild stecken.

### 3.2 Der Faltungssatz

Der Grund, warum der Frequenzraum so mächtig ist:
$$I * K \;\;\overset{\mathcal F}{\longleftrightarrow}\;\; F\cdot \hat K.$$
**Faltung im Ortsraum = punktweise Multiplikation im Frequenzraum.** Große Faltungen
werden so via FFT viel billiger, und Filter lassen sich als **Frequenz-Masken** entwerfen.

### 3.3 Frequenzfilterung

- **Tiefpass** (z. B. gaußscher LP): behält niedrige Frequenzen → **glättet/verwischt**
  (entfernt Rauschen, aber auch Kanten). Ein harter „ideal"-Tiefpass erzeugt
  **Ringing** (Gibbs-Phänomen) — deshalb weiche (gaußsche) Masken.
- **Hochpass:** behält hohe Frequenzen → **Kanten/Details**, entfernt Flächen.
- **Bandpass/Notch:** gezielt Frequenzbänder entfernen (z. B. periodisches Rauschen,
  Moiré).
- **Hybride Bilder:** Tiefpass von Bild A + Hochpass von Bild B → aus der Nähe sieht man
  B, aus der Ferne A (Oliva et al.).

---

## 4 · Kanten, Morphologie, Interpolation

### 4.1 Kantendetektion & Canny

Kanten sind Orte starker Intensitätsänderung. Der **Canny-Detektor** ist die klassische
Pipeline:
1. **Glätten** (Gauß), um Rauschen zu dämpfen;
2. **Gradient** (Sobel) → Betrag $G$ und Richtung $\theta$;
3. **Non-Maximum Suppression:** dünne die Kanten auf 1 Pixel aus (nur lokale Maxima quer
   zur Kante behalten);
4. **Hysterese-Schwellung:** zwei Schwellen $t_{\text{low}}<t_{\text{high}}$; starke Kanten
   ($>t_{\text{high}}$) sind sicher, schwache nur, wenn sie mit einer starken verbunden sind.

### 4.2 Morphologie (Binärbilder)

Operationen mit einem **Strukturelement** (kleine Maske) auf $0/1$-Bildern:
- **Erosion:** schrumpft Vordergrund (Pixel bleibt 1, wenn *alle* Nachbarn 1) — entfernt
  kleine weiße Störungen.
- **Dilatation:** wächst Vordergrund (Pixel wird 1, wenn *irgendein* Nachbar 1) — schließt
  kleine Löcher.
- **Opening** = Erosion → Dilatation: entfernt kleine Objekte, erhält Form.
- **Closing** = Dilatation → Erosion: schließt kleine Löcher/Risse.
Nützlich zum Bereinigen von Masken (z. B. nach Thresholding/Segmentierung).

### 4.3 Geometrische Transformationen & Interpolation

Rotation/Skalierung/Warping bilden Zielpixel auf (i. d. R. nicht-ganzzahlige)
Quellkoordinaten ab → man muss **interpolieren**:
- **Nearest-Neighbor:** nächster Pixel — schnell, aber blockig.
- **Bilinear:** gewichtetes Mittel der 4 Nachbarn — glatt, Standard.
- **Bicubic:** 16 Nachbarn — schärfer, für Vergrößerung.
Beim **Verkleinern** vorher **tiefpassfiltern** (Anti-Aliasing, siehe 1.1).

---

## 5 · Farbe

Ein Farbbild hat 3 Kanäle, aber der Farbraum ist wählbar:
- **RGB:** additiv, gerätenah, aber Helligkeit/Farbe vermischt.
- **HSV/HSL:** **H**ue, **S**aturation, **V**alue — trennt Farbe von Helligkeit,
  intuitiv für Farbauswahl/-anpassung.
- **YCbCr:** Luma (Y) + Chroma (Cb, Cr) — Grundlage von JPEG/Video (Chroma-Subsampling,
  weil das Auge Helligkeit feiner auflöst als Farbe).
- **CIE Lab:** **perzeptuell** annähernd gleichabständig; Distanzen entsprechen empfundenen
  Farbunterschieden — gut für Farbvergleiche/-korrektur.
**White Balance / Farbkorrektur:** die Kanäle so skalieren, dass Neutralgrau wirklich grau
ist (z. B. „Grey-World"-Annahme: mittlere Farbe ist grau).

**Demosaicing:** Kamerasensoren messen pro Pixel nur *eine* Farbe (Bayer-Muster
RGGB); die fehlenden zwei Kanäle werden interpoliert — der erste Schritt jeder
RAW-Pipeline.

---

## 6 · Computational Photography

Verfahren, die Bilder **erzeugen/verbessern**, indem sie Optik + Rechnung kombinieren:

- **HDR (High Dynamic Range):** mehrere Aufnahmen mit **verschiedenen Belichtungen**
  fusionieren, um einen größeren Helligkeitsumfang einzufangen, als der Sensor auf einmal
  kann; **Tone-Mapping** komprimiert das HDR-Ergebnis zurück auf ein anzeigbares 8-Bit-Bild.
- **Entrauschen (Denoising):** Rauschmodelle (gaußsch, Poisson/Photonenrauschen,
  Salz&Pfeffer) und Filter aus Abschnitt 2 (Median, bilateral, NLM).
- **Deblurring / Deconvolution:** Unschärfe (Bewegung/Defokus) als Faltung mit einer
  Punktspreizfunktion (PSF) modellieren und (regularisiert) invertieren
  (Wiener-Filter im Frequenzraum).
- **Panorama-Stitching:** Merkmale finden (z. B. SIFT), zwischen Bildern **matchen**, eine
  **Homographie** (projektive $3\times3$-Transformation) robust schätzen (**RANSAC**), die
  Bilder warpen und **blenden**.
- **Seam Carving (content-aware Resizing):** Bildgröße ändern, ohne wichtige Inhalte zu
  verzerren, indem man **energiearme „Nähte"** (Pfade geringster Gradienten-Energie)
  entfernt. Die optimale vertikale Naht findet man per **dynamischer Programmierung**
  (Modul 06/07!): mit der Energie $e(i,j)$ (z. B. Gradientenbetrag) ist die kumulierte
  Minimalenergie
  $$M(i,j)=e(i,j)+\min\big(M(i{-}1,j{-}1),\,M(i{-}1,j),\,M(i{-}1,j{+}1)\big),$$
  und die billigste Naht ergibt sich per Backtracking vom Minimum der letzten Zeile.
  (Projekt 03 baut genau das.)
- **Bokeh / synthetische Tiefenschärfe:** Tiefenkarte schätzen und den Hintergrund
  ortsabhängig weichzeichnen.

---

## 7 · Zusammenfassung / Cheat-Sheet

| Begriff | Kern in einem Satz |
|---|---|
| **Sampling/Quantisierung** | Rasterung + diskrete Werte; zu grob → Aliasing/Banding. |
| **Punktoperation** | $s=T(r)$ pixelweise; Helligkeit/Kontrast, Gamma, Threshold. |
| **Gamma** | $s=(r/255)^\gamma$; korrigiert nichtlineare Wahrnehmung/Displays. |
| **Histogrammausgleich** | Abbildung via CDF → gleichverteilte Intensität, mehr Kontrast. |
| **Gauß vs. Median** | Gauß gegen gaußsches Rauschen; Median gegen Salz&Pfeffer (kantenerhaltend). |
| **Bilateral** | kantenerhaltende Glättung (räumlich × tonal gewichtet). |
| **2D-DFT/FFT** | Bild → Frequenzen; niedrig = Flächen, hoch = Kanten/Rauschen. |
| **Faltungssatz** | $I*K \leftrightarrow F\cdot\hat K$: Faltung = Multiplikation im Frequenzraum. |
| **Tief-/Hochpass** | glätten / Kanten; harte Masken → Ringing. |
| **Canny** | Gauß → Gradient → NMS → Hysterese. |
| **Morphologie** | Erosion/Dilatation/Opening/Closing auf Binärmasken. |
| **Interpolation** | Nearest/Bilinear/Bicubic beim Resampling; vor Verkleinern glätten. |
| **Farbräume** | RGB/HSV/YCbCr/Lab — Helligkeit von Farbe trennen; Lab perzeptuell. |
| **HDR / Tone-Mapping** | Belichtungsreihe fusionieren; zurück auf 8 Bit abbilden. |
| **Homographie** | projektive 3×3-Transformation; Panorama via RANSAC + Warp + Blend. |
| **Seam Carving** | energiearme Nähte per DP entfernen → content-aware Resizing. |

**Formeln zum Merken:** Gamma $s=(r/255)^\gamma$; Ausgleich $s=(L-1)\,\text{cdf}(r)$;
Faltungssatz $I*K\leftrightarrow F\cdot\hat K$; Seam-DP
$M(i,j)=e(i,j)+\min(M(i{-}1,j{-}1),M(i{-}1,j),M(i{-}1,j{+}1))$.

---

## 8 · Selbsttest

<details><summary><b>1.</b> Warum glättet man ein Bild, bevor man es verkleinert?</summary>

Verkleinern = Unterabtastung. Ohne vorheriges Glätten (Tiefpass) verletzt man das
Abtasttheorem: hohe Frequenzen falten sich als **Aliasing** (Moiré, Treppen) in das
kleinere Bild. Der Tiefpass entfernt die zu hohen Frequenzen vorab.
</details>

<details><summary><b>2.</b> Wie funktioniert Histogrammausgleich, und warum erhöht er den Kontrast?</summary>

Man benutzt die **kumulative Verteilungsfunktion** (CDF) des Histogramms als
Abbildung $s=(L-1)\cdot\text{cdf}(r)$. Dadurch werden häufige (dicht besetzte)
Intensitätsbereiche gespreizt und seltene gestaucht → die Intensitäten werden
annähernd **gleichverteilt**, was den wahrgenommenen Kontrast maximiert.
</details>

<details><summary><b>3.</b> Wann Gauß- und wann Median-Filter?</summary>

Gauß (linear) gegen **gaußsches** Rauschen — glättet, verwischt aber Kanten. Median
(nichtlinear) gegen **Salz-und-Pfeffer**-(Impuls-)Rauschen — entfernt Ausreißer und
**erhält Kanten**, weil der Median robust gegen Extremwerte ist.
</details>

<details><summary><b>4.</b> Was besagt der Faltungssatz, und warum ist er praktisch?</summary>

Faltung im Ortsraum entspricht **punktweiser Multiplikation** im Frequenzraum
($I*K\leftrightarrow F\cdot\hat K$). Praktisch: große Faltungen via FFT viel billiger
rechnen, und Filter direkt als Frequenz-Masken (Tief-/Hoch-/Bandpass) entwerfen.
</details>

<details><summary><b>5.</b> Nenne die vier Schritte des Canny-Detektors.</summary>

(1) Gauß-Glättung, (2) Gradient (Sobel) → Betrag + Richtung, (3) Non-Maximum Suppression
(Kanten auf 1 Pixel ausdünnen), (4) Hysterese-Schwellung mit zwei Schwellen (schwache
Kanten nur behalten, wenn mit starken verbunden).
</details>

<details><summary><b>6.</b> Was ist der Unterschied zwischen Opening und Closing?</summary>

**Opening** (Erosion → Dilatation) entfernt kleine Vordergrund-Störungen/Objekte und
glättet Ränder. **Closing** (Dilatation → Erosion) schließt kleine **Löcher** und Risse
im Vordergrund. Beide erhalten grob die Objektgröße.
</details>

<details><summary><b>7.</b> Warum trennt man in JPEG/Video Luma und Chroma (YCbCr)?</summary>

Das menschliche Auge löst **Helligkeit** feiner auf als **Farbe**. In YCbCr kann man die
Chroma-Kanäle (Cb, Cr) stärker unterabtasten/komprimieren (Chroma-Subsampling), ohne dass
es auffällt — das spart Daten bei kaum sichtbarem Qualitätsverlust.
</details>

<details><summary><b>8.</b> Wie findet Seam Carving die zu entfernende Naht, und welches Prinzip steckt dahinter?</summary>

Man berechnet eine **Energie** (z. B. Gradientenbetrag) und sucht den vertikalen Pfad
minimaler kumulierter Energie per **dynamischer Programmierung**
($M(i,j)=e(i,j)+\min$ der drei oberen Nachbarn), dann Backtracking vom Minimum der letzten
Zeile. So werden „unwichtige" (glatte) Regionen entfernt statt wichtige Inhalte zu stauchen.
</details>

<details><summary><b>9.</b> Was ist eine Homographie und wozu dient RANSAC beim Panorama?</summary>

Eine **Homographie** ist eine projektive $3\times3$-Transformation, die zwei Bildebenen
(bei reiner Kamerarotation) ineinander abbildet. **RANSAC** schätzt sie **robust** aus
verrauschten Merkmals-Matches, indem es wiederholt eine Homographie aus zufälligen
Match-Stichproben bildet und die mit den meisten „Inliers" wählt (ignoriert Ausreißer).
</details>

<details><summary><b>10.</b> Warum brauchen Kamera-RAW-Bilder Demosaicing?</summary>

Der Sensor misst pro Pixel nur **eine** Farbe (Bayer-Muster). Für ein volles RGB-Bild
müssen die jeweils zwei fehlenden Farbkanäle pro Pixel aus den Nachbarn **interpoliert**
werden — das ist Demosaicing, der erste Schritt der RAW-Verarbeitung.
</details>

---

## 9 · Literatur & Quellen

**Lehrbücher (kostenlos 💰 wo möglich)**
- Gonzalez & Woods, *Digital Image Processing* — das Standardwerk (Punktops, Histogramm,
  Frequenzraum, Morphologie, Farbe). *vertiefend*
- Szeliski, *Computer Vision: Algorithms and Applications* (2. Aufl., **frei online**) —
  Kap. zu Bildverarbeitung, Feature-Matching, Stitching, Computational Photography. 🟢
- Burger & Burge, *Digital Image Processing* (mit sehr konkreten Algorithmen). *einsteigerfreundlich*

**Kurse / interaktiv (kostenlos 💰)**
- **CS231n / CS131** (Stanford) — klassische CV/Bildverarbeitungs-Grundlagen. 🟢
- **3Blue1Brown**, *But what is the Fourier Transform?* — Intuition. 🟢 *einsteigerfreundlich*
- **Setosa.io** — interaktive Visualisierungen (Image Kernels). 🟢 *einsteigerfreundlich*
- `scipy.ndimage`-, `numpy.fft`- und Pillow-Doku — die praktische Referenz. 🟢

**Klassische Papers (frei 💰)**
- Avidan & Shamir (2007): *Seam Carving for Content-Aware Image Resizing*. 🟢
- Tomasi & Manduchi (1998): *Bilateral Filtering*. 🟢
- Oliva, Torralba & Schyns (2006): *Hybrid Images*. 🟢
- Debevec & Malik (1997): *Recovering High Dynamic Range Radiance Maps*. 🟢
- Brown & Lowe (2007): *Automatic Panoramic Image Stitching* (SIFT + RANSAC). 🟢

---

## Die drei Projekte

Alle Projekte sind **numpy/scipy-basiert, ohne Training, CPU-freundlich** (Sekunden) und
arbeiten auf echten Bildern:

- **01 – basic** (`projects/01-basic/`): **Punktoperationen & Histogrammausgleich von
  Hand.** Geführtes Notebook: Helligkeit/Kontrast/Gamma, Histogramm berechnen und den
  **Histogrammausgleich** über die CDF selbst implementieren; Wirkung sichtbar machen.
- **02 – medium** (`projects/02-medium/`): **Frequenzraum & Entrauschen.** Python-Projekt
  mit Testsuite: 2D-FFT, Spektrum, **Tief-/Hochpass**, **Faltungssatz** empirisch prüfen;
  **Gauß vs. Median** gegen verschiedene Rauscharten (PSNR-Vergleich).
- **03 – final** (`projects/03-final/`): **Seam Carving (content-aware Resizing).** Keine
  Code-Vorgabe: Energie-Karte, optimale Naht per **dynamischer Programmierung**, Nähte
  iterativ entfernen — ein echtes Computational-Photography-Verfahren, das
  Bildverarbeitung und DP verbindet.

Details, Setup und Musterlösungen jeweils in der `README.md` des Projektordners.
