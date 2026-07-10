# Projekt 02 (medium) — Frequenzraum & Entrauschen

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
  loesung/        # vollständige, getestete Musterlösung
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
python run.py --save    # zusätzlich Abbildungen in ergebnisse/ speichern
```

## Was am Ende funktionieren soll

- `python test_freq.py` → **alle 7 Tests grün** (Faltungssatz zirkulär, Tiefpass glättet &
  erhält Mittel, Hochpass entfernt DC, Masken-Eigenschaften, PSNR, Median > Gauß bei
  Salz&Pfeffer, Spektrum-Form).
- `python run.py` → Faltungssatz-Differenz ≈ $10^{-13}$; Tiefpass glatt / Hochpass ≈ 0-Mittel;
  Entrausch-Tabelle mit **Median ~30 dB vs. Gauß ~23 dB** bei Salz&Pfeffer.

## Musterlösung

Vollständig in [`loesung/`](loesung/) (alle Tests grün). Erst selbst versuchen — die
Root-`frequency.py`/`denoise.py` werfen `NotImplementedError`.
