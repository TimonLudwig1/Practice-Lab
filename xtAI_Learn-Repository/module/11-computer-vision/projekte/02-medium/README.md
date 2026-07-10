# Projekt 02 (medium) — Transfer Learning als Feature-Extraktor (EuroSAT)

**Modul 11 — Computer Vision** · Format: **Python-Projekt** (mehrere Module + Testsuite)

## Warum dieses Format?

Der Kern ist eine **Praxis-Fähigkeit**: ein vortrainiertes Modell korrekt als
Merkmalsextraktor verwenden (laden, einfrieren, Kopf entfernen, richtig vorverarbeiten,
Features ziehen). Als **Codebasis** trennst du diese Bausteine sauber und die **Testsuite**
prüft, dass du das Backbone wirklich eingefroren und richtig angezapft hast.

## Ziel

Du nutzt ein **eingefrorenes** ImageNet-Backbone (MobileNetV3-Small) als
**Feature-Extraktor** auf **EuroSAT** (Satellitenbildern — einer *ganz anderen* Domäne als
ImageNet) und trainierst darauf einen kleinen Klassifikator. Kernpunkte (Skript 3.2, Modus B):

- ein pretrained Modell **laden**, **einfrieren** (`requires_grad = False`) und den
  Klassifikationskopf durch `Identity` **ersetzen**;
- Bilder mit **`weights.transforms()`** exakt so vorverarbeiten wie im Vortraining
  (inkl. ImageNet-Normalisierung) und **Feature-Vektoren** (576-dim) extrahieren;
- **der Aha-Moment:** ein linearer Klassifikator auf diesen Features erreicht **~0.94**,
  während dieselbe Methode auf **rohen Pixeln** nur **~0.41** schafft — *ohne* das Backbone
  je zu trainieren. Übertragbare Merkmale schlagen rohe Pixel deutlich.

## Vorwissen

- **Skript** Abschnitt 3 (was ein pretrained Modell ist, die drei Nutzungsarten,
  Normalisierung, from-scratch vs. Transfer).
- PyTorch/torchvision-Grundlagen; scikit-learn (`LogisticRegression`).

## Projektstruktur

```
02-medium/
  transfer.py        # Feature-Extraktor bauen + Features ziehen   <- DU (Aufgabe 1 + 2)
  data.py            # EuroSAT laden + Teilmengen                    (vorgegeben)
  run.py             # Pipeline: Features vs. Rohpixel + Fehleranalyse (vorgegeben)
  test_transfer.py   # Testsuite (5 Tests, schnell)                  (vorgegeben)
  loesung/           # vollständige, getestete Musterlösung
```

## Aufgabenstellung

Wenig Vorgabe — die beiden Praxis-Kerne sind deine (`# TODO` in `transfer.py`):

1. **`build_feature_extractor`**: pretrained MobileNetV3-Small laden, **einfrieren**,
   `classifier = Identity`, `eval()`, und `WEIGHTS.transforms()` zurückgeben.
2. **`extract_features`**: die Bilder batchweise vorverarbeiten und unter `torch.no_grad()`
   durchs Netz schicken; Feature-Matrix `(N, 576)` zurückgeben.

`raw_pixel_features` (Baseline), das Laden und die Klassifikation sind vorgegeben.

**Vorgehen:**

```bash
source ../../../../.venv/bin/activate
python test_transfer.py     # rot -> Aufgaben füllen -> alle 5 Tests grün
python run.py               # echtes EuroSAT: Feature- vs. Rohpixel-Genauigkeit
```

`run.py` lädt EuroSAT beim ersten Lauf (~90 MB, schnell) und cached die extrahierten
Features in `daten/` (Neuaufbau mit `--rebuild`).

## Was am Ende funktionieren soll

- `python test_transfer.py` → **alle 5 Tests grün**: Backbone eingefroren & ohne Kopf,
  Feature-Form `(N, 576)`, korrekte Vorverarbeitung (Normalisierung erzeugt negative Werte),
  Rohpixel-Baseline, und ein Integrationstest, dass pretrained Features Toy-Klassen trennen.
- `python run.py` → **Rohpixel ~0.41 vs. Pretrained-Features ~0.94** auf EuroSAT, plus eine
  kurze Fehleranalyse der schwächsten Klassen.

## Musterlösung

Vollständig in [`loesung/`](loesung/) (alle Tests grün, ~0.94). Erst selbst versuchen — die
Root-`transfer.py` wirft `NotImplementedError`, bis du die TODOs füllst.
