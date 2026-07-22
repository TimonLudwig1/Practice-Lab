# Module 12 — Image Processing and Computational Photography

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** Whereas module 11 wanted to *understand* images (classification
> via a CNN), here it is about *processing and generating* images with **classical,
> deterministic** methods — without training. From point operations and histogram
> equalization via filtering in the **spatial and frequency domain** (Fourier), edges,
> morphology and interpolation, to **computational photography** (HDR, denoising,
> panorama, content-aware resizing). This is the toolkit that lies *beneath* every image
> pipeline — and everything runs with `numpy`/`scipy` on the CPU.

**Helpful prior knowledge:** linear algebra, some signal processing/calculus (sine/cosine,
the Fourier idea), NumPy.

**Modules you should have done first:**
- **Module 11 (Computer Vision)** — you already built the **convolution** and classical
  filters (Sobel/Gaussian) by hand there; here we deepen filtering and add the **frequency
  domain**. **Dynamic programming** (modules 06/07) returns with seam carving.

---

## Learning objectives

After this module you can …

- understand images as sampled, quantized signals and apply **point operations**
  (brightness, contrast, **gamma**) as well as **histogram equalization**;
- classify linear (mean/Gaussian) and **nonlinear** (median, bilateral) **filters** and
  know *which one helps against which noise*;
- explain the **2D Fourier transform**, interpret an image in the **frequency domain** and
  apply **low-/high-pass filters** — including the **convolution theorem**;
- describe **edges** (gradient, Laplacian, the **Canny** pipeline), **morphology**
  (erosion/dilation/opening/closing) and **interpolation** (nearest/bilinear/bicubic);
- name **color spaces** (RGB/HSV/Lab/YCbCr) and their use;
- explain **computational photography** methods: **HDR/tone mapping**, **denoising**,
  **panorama stitching** (homography), **seam carving** (content-aware resizing).

---

## 1 · Basics — image, point operations, histogram

### 1.1 The image as a signal

A digital image arises through **sampling** (the continuous image is discretized onto a
pixel grid) and **quantization** (each pixel gets a discrete value, usually 8 bit =
$0$–$255$). Too coarse sampling → **aliasing** (staircases, moiré); too coarse quantization
→ **banding** (visible brightness steps). The sampling theorem (Nyquist) requires a sampling
rate $>2\times$ the highest image frequency — which is why one **smooths** before
downsizing (anti-aliasing).

### 1.2 Point operations

A **point operation** maps every pixel value $r$ *independently* of its neighbours onto
$s=T(r)$:
- **Brightness/contrast (linear):** $s = a\,r + b$ ($a$ = contrast, $b$ = brightness).
- **Negative:** $s = 255 - r$.
- **Gamma correction:** $s = 255\,(r/255)^{\gamma}$. $\gamma<1$ brightens dark areas,
  $\gamma>1$ darkens. Important because displays/cameras are nonlinear (gamma-encoded).
- **Thresholding:** $s = 255$ if $r>t$, else $0$ — produces a binary image.

### 1.3 Histogram & histogram equalization

The **histogram** $h(k)$ counts how often the gray value $k$ occurs — it describes the
**brightness distribution**. A low-contrast image has a narrow histogram.

**Histogram equalization** stretches the histogram to maximize contrast. Idea: use the
**cumulative distribution** (CDF) as the mapping function. With the normalized histogram
$p(k)=h(k)/N$ and
$$\text{cdf}(k)=\sum_{j=0}^{k} p(j)$$
the equalization mapping is
$$s = T(r) = \operatorname{round}\!\big((L-1)\cdot \text{cdf}(r)\big),\qquad L=256.$$
The result has an (approximately) **uniform** intensity — dark and bright details become
visible. (Project 01 builds exactly this.)

---

## 2 · Filtering in the spatial domain

### 2.1 Linear filters (convolution, recap from module 11)

A **linear** filtering is a **convolution** with a kernel (module 11, section 1):
$$(I*K)(i,j)=\sum_{u,v} I(i+u,j+v)\,K(u,v).$$
- **Mean/box filter** and **Gaussian filter**: **smooth**, suppress noise, but also blur
  edges. The Gaussian is **separable** (2D = two 1D convolutions) → efficient.
- **Derivative filters** (Sobel, Prewitt): emphasize edges (gradient).
- **Laplacian** $\nabla^2$: second derivative, emphasizes edges/details isotropically.
- **Unsharp masking (sharpening):** $I_{\text{sharp}} = I + \alpha\,(I - I_{\text{blur}})$ —
  the "unsharp mask" $I-I_{\text{blur}}$ is the high-frequency part that one amplifies.

### 2.2 Nonlinear filters

- **Median filter:** replaces each pixel by the **median** of its neighbourhood. Ideal
  against **salt-and-pepper noise** (outliers), because the median is robust and — unlike
  the Gaussian — **preserves edges**.
- **Bilateral filter:** smooths only between pixels that are similar both spatially **and**
  in intensity → **edge-preserving** smoothing. Weight
  $w(i,j;k,l)=\exp\!\big(-\tfrac{\lVert(i,j)-(k,l)\rVert^2}{2\sigma_s^2}\big)\,
  \exp\!\big(-\tfrac{(I_{ij}-I_{kl})^2}{2\sigma_r^2}\big)$ (spatial × tonal).
- **Non-local means (NLM):** averages over *whole similar patches* in the image — very
  effective denoising.

**Note:** Gaussian against **Gaussian** noise, median against **impulse** (salt&pepper)
noise. (Project 02 shows this empirically.)

---

## 3 · The frequency domain — Fourier

### 3.1 Idea

The **Fourier transform** decomposes a signal into **sine/cosine oscillations** of
different frequencies. Applied to an image (2D): **low** frequencies = slow brightness
gradients (coarse structure, areas), **high** frequencies = fast changes (edges, textures,
noise). The **2D DFT** of an $M\times N$ image:
$$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1} I(x,y)\,
e^{-2\pi i\,(ux/M + vy/N)}.$$
It is computed efficiently with the **FFT** ($O(MN\log MN)$ instead of $O(M^2N^2)$). The
**magnitude spectrum** $|F(u,v)|$ (log-scaled, centered with `fftshift`) visualizes which
frequencies/orientations are in the image.

### 3.2 The convolution theorem

The reason the frequency domain is so powerful:
$$I * K \;\;\overset{\mathcal F}{\longleftrightarrow}\;\; F\cdot \hat K.$$
**Convolution in the spatial domain = pointwise multiplication in the frequency domain.**
Large convolutions thereby become much cheaper via the FFT, and filters can be designed as
**frequency masks**.

### 3.3 Frequency filtering

- **Low-pass** (e.g. a Gaussian LP): keeps low frequencies → **smooths/blurs** (removes
  noise, but also edges). A hard "ideal" low-pass produces **ringing** (Gibbs phenomenon) —
  which is why one uses soft (Gaussian) masks.
- **High-pass:** keeps high frequencies → **edges/details**, removes areas.
- **Band-pass/notch:** remove specific frequency bands (e.g. periodic noise, moiré).
- **Hybrid images:** low-pass of image A + high-pass of image B → up close you see B, from
  afar A (Oliva et al.).

---

## 4 · Edges, morphology, interpolation

### 4.1 Edge detection & Canny

Edges are places of strong intensity change. The **Canny detector** is the classical
pipeline:
1. **Smooth** (Gaussian) to dampen noise;
2. **Gradient** (Sobel) → magnitude $G$ and direction $\theta$;
3. **Non-maximum suppression:** thin the edges to 1 pixel (keep only local maxima across
   the edge);
4. **Hysteresis thresholding:** two thresholds $t_{\text{low}}<t_{\text{high}}$; strong
   edges ($>t_{\text{high}}$) are certain, weak ones only if connected to a strong one.

### 4.2 Morphology (binary images)

Operations with a **structuring element** (a small mask) on $0/1$ images:
- **Erosion:** shrinks the foreground (a pixel stays 1 if *all* neighbours are 1) — removes
  small white disturbances.
- **Dilation:** grows the foreground (a pixel becomes 1 if *any* neighbour is 1) — closes
  small holes.
- **Opening** = erosion → dilation: removes small objects, preserves the shape.
- **Closing** = dilation → erosion: closes small holes/cracks.
Useful for cleaning up masks (e.g. after thresholding/segmentation).

### 4.3 Geometric transformations & interpolation

Rotation/scaling/warping map target pixels onto (usually non-integer) source coordinates →
one has to **interpolate**:
- **Nearest neighbor:** the nearest pixel — fast, but blocky.
- **Bilinear:** a weighted average of the 4 neighbours — smooth, the standard.
- **Bicubic:** 16 neighbours — sharper, for upscaling.
When **downsizing**, **low-pass filter** beforehand (anti-aliasing, see 1.1).

---

## 5 · Color

A color image has 3 channels, but the color space is a choice:
- **RGB:** additive, device-near, but brightness/color are mixed.
- **HSV/HSL:** **h**ue, **s**aturation, **v**alue — separates color from brightness,
  intuitive for color selection/adjustment.
- **YCbCr:** luma (Y) + chroma (Cb, Cr) — the basis of JPEG/video (chroma subsampling,
  because the eye resolves brightness more finely than color).
- **CIE Lab:** approximately **perceptually** uniform; distances correspond to perceived
  color differences — good for color comparison/correction.
**White balance / color correction:** scale the channels so that neutral gray really is gray
(e.g. the "grey-world" assumption: the mean color is gray).

**Demosaicing:** camera sensors measure only *one* color per pixel (Bayer pattern RGGB);
the missing two channels are interpolated — the first step of every RAW pipeline.

---

## 6 · Computational photography

Methods that **generate/improve** images by combining optics + computation:

- **HDR (high dynamic range):** fuse several shots at **different exposures** to capture a
  larger brightness range than the sensor can at once; **tone mapping** compresses the HDR
  result back to a displayable 8-bit image.
- **Denoising:** noise models (Gaussian, Poisson/photon noise, salt&pepper) and the filters
  from section 2 (median, bilateral, NLM).
- **Deblurring / deconvolution:** model the blur (motion/defocus) as a convolution with a
  point spread function (PSF) and invert it (regularized) (Wiener filter in the frequency
  domain).
- **Panorama stitching:** find features (e.g. SIFT), **match** them between images, robustly
  estimate a **homography** (a projective $3\times3$ transformation) (**RANSAC**), warp the
  images and **blend**.
- **Seam carving (content-aware resizing):** change the image size without distorting
  important content, by removing **low-energy "seams"** (paths of least gradient energy).
  The optimal vertical seam is found by **dynamic programming** (modules 06/07!): with the
  energy $e(i,j)$ (e.g. gradient magnitude) the cumulative minimum energy is
  $$M(i,j)=e(i,j)+\min\big(M(i{-}1,j{-}1),\,M(i{-}1,j),\,M(i{-}1,j{+}1)\big),$$
  and the cheapest seam follows from backtracking from the minimum of the last row.
  (Project 03 builds exactly this.)
- **Bokeh / synthetic depth of field:** estimate a depth map and blur the background
  location-dependently.

---

## 7 · Summary / cheat sheet

| Term | Core in one sentence |
|---|---|
| **Sampling/quantization** | rasterization + discrete values; too coarse → aliasing/banding. |
| **Point operation** | $s=T(r)$ pixel-wise; brightness/contrast, gamma, threshold. |
| **Gamma** | $s=(r/255)^\gamma$; corrects nonlinear perception/displays. |
| **Histogram equalization** | mapping via CDF → uniform intensity, more contrast. |
| **Gaussian vs. median** | Gaussian against Gaussian noise; median against salt&pepper (edge-preserving). |
| **Bilateral** | edge-preserving smoothing (spatial × tonal weighted). |
| **2D DFT/FFT** | image → frequencies; low = areas, high = edges/noise. |
| **Convolution theorem** | $I*K \leftrightarrow F\cdot\hat K$: convolution = multiplication in the frequency domain. |
| **Low-/high-pass** | smooth / edges; hard masks → ringing. |
| **Canny** | Gaussian → gradient → NMS → hysteresis. |
| **Morphology** | erosion/dilation/opening/closing on binary masks. |
| **Interpolation** | nearest/bilinear/bicubic when resampling; smooth before downsizing. |
| **Color spaces** | RGB/HSV/YCbCr/Lab — separate brightness from color; Lab perceptual. |
| **HDR / tone mapping** | fuse an exposure series; map back to 8 bit. |
| **Homography** | projective 3×3 transformation; panorama via RANSAC + warp + blend. |
| **Seam carving** | remove low-energy seams via DP → content-aware resizing. |

**Formulas to remember:** gamma $s=(r/255)^\gamma$; equalization $s=(L-1)\,\text{cdf}(r)$;
convolution theorem $I*K\leftrightarrow F\cdot\hat K$; seam DP
$M(i,j)=e(i,j)+\min(M(i{-}1,j{-}1),M(i{-}1,j),M(i{-}1,j{+}1))$.

---

## 8 · Self-test

<details><summary><b>1.</b> Why does one smooth an image before downsizing?</summary>

Downsizing = subsampling. Without prior smoothing (a low-pass) one violates the sampling
theorem: high frequencies fold back as **aliasing** (moiré, staircases) into the smaller
image. The low-pass removes the too-high frequencies beforehand.
</details>

<details><summary><b>2.</b> How does histogram equalization work, and why does it raise contrast?</summary>

One uses the **cumulative distribution function** (CDF) of the histogram as the mapping
$s=(L-1)\cdot\text{cdf}(r)$. This stretches frequent (densely populated) intensity ranges
and compresses rare ones → the intensities become approximately **uniformly distributed**,
which maximizes the perceived contrast.
</details>

<details><summary><b>3.</b> When a Gaussian and when a median filter?</summary>

Gaussian (linear) against **Gaussian** noise — smooths, but blurs edges. Median (nonlinear)
against **salt-and-pepper** (impulse) noise — removes outliers and **preserves edges**,
because the median is robust against extreme values.
</details>

<details><summary><b>4.</b> What does the convolution theorem state, and why is it practical?</summary>

Convolution in the spatial domain corresponds to **pointwise multiplication** in the
frequency domain ($I*K\leftrightarrow F\cdot\hat K$). Practical: compute large convolutions
much more cheaply via the FFT, and design filters directly as frequency masks
(low-/high-/band-pass).
</details>

<details><summary><b>5.</b> Name the four steps of the Canny detector.</summary>

(1) Gaussian smoothing, (2) gradient (Sobel) → magnitude + direction, (3) non-maximum
suppression (thin edges to 1 pixel), (4) hysteresis thresholding with two thresholds (keep
weak edges only if connected to strong ones).
</details>

<details><summary><b>6.</b> What is the difference between opening and closing?</summary>

**Opening** (erosion → dilation) removes small foreground disturbances/objects and smooths
edges. **Closing** (dilation → erosion) closes small **holes** and cracks in the foreground.
Both roughly preserve the object size.
</details>

<details><summary><b>7.</b> Why does one separate luma and chroma (YCbCr) in JPEG/video?</summary>

The human eye resolves **brightness** more finely than **color**. In YCbCr one can subsample
/ compress the chroma channels (Cb, Cr) more strongly (chroma subsampling) without it being
noticeable — this saves data at barely visible quality loss.
</details>

<details><summary><b>8.</b> How does seam carving find the seam to remove, and what principle is behind it?</summary>

One computes an **energy** (e.g. gradient magnitude) and searches for the vertical path of
minimum cumulative energy by **dynamic programming** ($M(i,j)=e(i,j)+\min$ of the three upper
neighbours), then backtracks from the minimum of the last row. This removes "unimportant"
(smooth) regions instead of squeezing important content.
</details>

<details><summary><b>9.</b> What is a homography and what is RANSAC for in panorama stitching?</summary>

A **homography** is a projective $3\times3$ transformation that maps two image planes (under
pure camera rotation) onto each other. **RANSAC** estimates it **robustly** from noisy
feature matches by repeatedly forming a homography from random match samples and choosing
the one with the most "inliers" (ignoring outliers).
</details>

<details><summary><b>10.</b> Why do camera RAW images need demosaicing?</summary>

The sensor measures only **one** color per pixel (Bayer pattern). For a full RGB image the
respective two missing color channels per pixel have to be **interpolated** from the
neighbours — that is demosaicing, the first step of RAW processing.
</details>

---

## 9 · Literature & sources

*Legend: (free) = freely available, (beginner) = beginner-friendly, (in-depth) = advanced.*

**Textbooks (free where possible)**
- Gonzalez & Woods, *Digital Image Processing* — the standard work (point ops, histogram,
  frequency domain, morphology, color). (in-depth)
- Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed., **free online**) —
  chapters on image processing, feature matching, stitching, computational photography. (free)
- Burger & Burge, *Digital Image Processing* (with very concrete algorithms). (beginner)

**Courses / interactive (free)**
- **CS231n / CS131** (Stanford) — classical CV/image-processing foundations. (free)
- **3Blue1Brown**, *But what is the Fourier Transform?* — intuition. (beginner)
- **Setosa.io** — interactive visualizations (image kernels). (beginner)
- The `scipy.ndimage`, `numpy.fft` and Pillow docs — the practical reference. (free)

**Classical papers (free)**
- Avidan & Shamir (2007): *Seam Carving for Content-Aware Image Resizing*.
- Tomasi & Manduchi (1998): *Bilateral Filtering*.
- Oliva, Torralba & Schyns (2006): *Hybrid Images*.
- Debevec & Malik (1997): *Recovering High Dynamic Range Radiance Maps*.
- Brown & Lowe (2007): *Automatic Panoramic Image Stitching* (SIFT + RANSAC).

---

## The three projects

All projects are **numpy/scipy-based, without training, CPU-friendly** (seconds) and work on
real images:

- **01 – basic** (`projects/01-basic/`): **Point operations & histogram equalization by
  hand.** Guided notebook: brightness/contrast/gamma, compute the histogram and implement
  the **histogram equalization** via the CDF yourself; make the effect visible.
- **02 – medium** (`projects/02-medium/`): **Frequency domain & denoising.** Python project
  with a test suite: 2D FFT, spectrum, **low-/high-pass**, check the **convolution theorem**
  empirically; **Gaussian vs. median** against different noise types (PSNR comparison).
- **03 – final** (`projects/03-final/`): **Seam carving (content-aware resizing).** No code
  given: energy map, the optimal seam via **dynamic programming**, remove seams iteratively —
  a real computational-photography method that connects image processing and DP.

Details, setup and reference solutions are in the `README.md` of each project folder.

---
# Modul 12 — Image Processing and Computational Photography (deutsche Fassung)

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
