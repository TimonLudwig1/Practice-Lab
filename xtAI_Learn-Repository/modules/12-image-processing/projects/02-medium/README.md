# Project 02 (medium) — Frequency domain & denoising

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The project code itself is English only.

**Module 12 — Image Processing** · Format: **Python project** (several modules + test suite)

## Why this format?

The frequency domain is maths to *try out*: compute the FFT, build masks, confirm the
**convolution theorem** empirically. As a **codebase** you cleanly separate frequency
filtering and denoising, and the **test suite** checks the core claims (convolution theorem,
DC preservation, median vs. Gaussian).

## Goal

You work in the **frequency domain** and compare **denoising** filters (script sections 2 & 3):

- the **2D FFT** and the (log) **magnitude spectrum**;
- a **Gaussian low-pass/high-pass** as a frequency mask → smooth resp. edges;
- **the convolution theorem** empirically: circular convolution in the spatial domain **=**
  multiplication in the frequency domain ($I*K \leftrightarrow F\cdot\hat K$, difference
  ≈ $10^{-13}$);
- **Gaussian vs. median**: against Gaussian noise similar; against **salt&pepper the median
  wins clearly** (~30 vs. ~23 dB PSNR) — robust against outliers, edge-preserving.

## Prior knowledge

- **Script** section 2 (linear/nonlinear filters, noise types) and 3 (Fourier, convolution
  theorem, low-/high-pass).
- NumPy (`np.fft`), SciPy (`ndimage`), the basic idea of the Fourier transform.

## Project structure

```
02-medium/
  frequency.py    # low-pass mask + filter application   <- YOU (task 1 + 2)
  denoise.py      # PSNR (+ Gaussian/median given)         <- YOU (task 3)
  imaging.py      # image/noise/reference convolution       (given)
  run.py          # demo: spectrum, filters, denoising      (given)
  test_freq.py    # test suite (7 tests)                    (given)
  solution/       # complete, tested reference solution
```

## Assignment

Three short, mathematically clear `# TODO`s:

1. **`gaussian_lowpass_mask(shape, cutoff)`** in `frequency.py`: a Gaussian frequency mask
   (DC at the origin via `np.fft.fftfreq`).
2. **`apply_frequency_filter(img, mask)`** in `frequency.py`: `fft2` → `* mask` → `ifft2` → real.
3. **`psnr(clean, test)`** in `denoise.py`: $10\log_{10}(255^2/\text{MSE})$.

**How to proceed:**

```bash
source ../../../../.venv/bin/activate
python test_freq.py     # red -> fill in the TODOs -> all 7 tests green
python run.py           # spectrum, low-/high-pass, convolution theorem, denoising PSNR
python run.py --save    # additionally save the figures into results/
```

## What should work in the end

- `python test_freq.py` → **all 7 tests green** (convolution theorem circular, low-pass
  smooths & preserves the mean, high-pass removes DC, mask properties, PSNR, median > Gaussian
  on salt&pepper, spectrum shape).
- `python run.py` → convolution-theorem difference ≈ $10^{-13}$; low-pass smooth / high-pass
  ≈ 0-mean; a denoising table with **median ~30 dB vs. Gaussian ~23 dB** on salt&pepper.

## Reference solution

Complete in [`solution/`](solution/) (all tests green). Try it yourself first — the root
`frequency.py`/`denoise.py` raise `NotImplementedError`.

---

# Projekt 02 (medium) — Frequenzraum & Entrauschen (deutsche Fassung)

**Modul 12 — Image Processing** · Format: **Python-Projekt** (mehrere Module + Testsuite)

## Warum dieses Format?

Der Frequenzraum ist Mathematik zum *Ausprobieren*: FFT rechnen, Masken bauen, den
**Faltungssatz** empirisch bestätigen. Als **Codebasis** trennst du Frequenzfilterung und
Entrauschen sauber, und die **Testsuite** prüft die Kernaussagen (Faltungssatz, DC-Erhalt,
Median vs. Gauß).

## Ziel

Du arbeitest im **Frequenzraum** und vergleichst **Entrausch**-Filter (Skript-Abschnitte 2 & 3):

- **2D-FFT** und das (log-)**Magnitudenspektrum**;
- **gaußscher Tiefpass/Hochpass** als Frequenz-Maske → glätten bzw. Kanten;
- **der Faltungssatz** empirisch: zirkuläre Faltung im Ortsraum **=** Multiplikation im
  Frequenzraum ($I*K \leftrightarrow F\cdot\hat K$, Differenz ≈ $10^{-13}$);
- **Gauß vs. Median**: gegen gaußsches Rauschen ähnlich; gegen **Salz&Pfeffer gewinnt der
  Median deutlich** (~30 vs. ~23 dB PSNR) — robust gegen Ausreißer, kantenerhaltend.

## Vorwissen

- **Skript** Abschnitt 2 (lineare/nichtlineare Filter, Rauscharten) und 3 (Fourier,
  Faltungssatz, Tief-/Hochpass).
- NumPy (`np.fft`), SciPy (`ndimage`), Grundidee der Fourier-Transformation.

## Projektstruktur

```
02-medium/
  frequency.py    # Tiefpass-Maske + Filteranwendung   <- DU (Aufgabe 1 + 2)
  denoise.py      # PSNR (+ Gauß/Median vorgegeben)      <- DU (Aufgabe 3)
  imaging.py      # Bild/Rauschen/Referenz-Faltung        (vorgegeben)
  run.py          # Demo: Spektrum, Filter, Entrauschen   (vorgegeben)
  test_freq.py    # Testsuite (7 Tests)                   (vorgegeben)
  solution/        # vollständige, getestete Musterlösung
```

## Aufgabenstellung

Drei kurze, mathematisch klare `# TODO`:

1. **`gaussian_lowpass_mask(shape, cutoff)`** in `frequency.py`: eine gaußsche Frequenz-Maske
   (DC im Ursprung via `np.fft.fftfreq`).
2. **`apply_frequency_filter(img, mask)`** in `frequency.py`: `fft2` → `* mask` → `ifft2` → real.
3. **`psnr(clean, test)`** in `denoise.py`: $10\log_{10}(255^2/\text{MSE})$.

**Vorgehen:**

```bash
source ../../../../.venv/bin/activate
python test_freq.py     # rot -> TODOs füllen -> alle 7 Tests grün
python run.py           # Spektrum, Tief-/Hochpass, Faltungssatz, Entrausch-PSNR
python run.py --save    # zusätzlich Abbildungen in results/ speichern
```

## Was am Ende funktionieren soll

- `python test_freq.py` → **alle 7 Tests grün** (Faltungssatz zirkulär, Tiefpass glättet &
  erhält Mittel, Hochpass entfernt DC, Masken-Eigenschaften, PSNR, Median > Gauß bei
  Salz&Pfeffer, Spektrum-Form).
- `python run.py` → Faltungssatz-Differenz ≈ $10^{-13}$; Tiefpass glatt / Hochpass ≈ 0-Mittel;
  Entrausch-Tabelle mit **Median ~30 dB vs. Gauß ~23 dB** bei Salz&Pfeffer.

## Musterlösung

Vollständig in [`solution/`](solution/) (alle Tests grün). Erst selbst versuchen — die
Root-`frequency.py`/`denoise.py` werfen `NotImplementedError`.
