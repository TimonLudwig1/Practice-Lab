# Projekt 03 (final) — Bildklassifikator für eine neue Domäne: drei Wege im Vergleich

**Modul 11 — Computer Vision** · Format: **Python-Projekt** (freie Umsetzung, *keine* Code-Vorgabe)

## Warum dieser Zuschnitt?

Das Abschlussprojekt konsolidiert die **zentrale praktische Entscheidung** des Moduls: Wie
baue ich einen Klassifikator für **meine eigene** Bilddomäne? Du vergleichst — auf echten
**EuroSAT-Satellitenbildern** (10 Landnutzungsklassen, eine *ganz andere* Domäne als
ImageNet) — die drei Wege aus Skript-Abschnitt 2.3 & 3.2 **direkt gegeneinander**:

- **(a) „ohne pretrained"** — ein CNN **von Grund auf** trainieren;
- **(b) Feature-Extraktion** — ein **eingefrorenes** pretrained Backbone + eigener Klassifikator;
- **(c) Fine-Tuning** — pretrained Backbone (teilweise) **weitertrainieren**.

So erlebst du quantitativ, *warum* Transfer Learning der Standard ist — und wo seine
Grenzen/Kosten liegen. **Alles ist bewusst CPU-freundlich** (kleine Modelle, kleine
Subsets, wenige Epochen) und läuft in **~2–3 Minuten** ohne den Laptop zu überhitzen.

**Keine Code-Vorgabe.** Diese README ist die Spezifikation. Eine vollständige, getestete
Musterlösung liegt in [`solution/`](solution/) — **erst selbst bauen.**

## Ziel

Einen EuroSAT-Klassifikator auf **drei** Arten bauen, sauber **evaluieren** und den
Unterschied **erklären**.

## Vorwissen

- **Skript** Abschnitt 2 (CNN-Bausteine, Training from scratch) und **3** (pretrained
  Modelle, die drei Nutzungsarten, Normalisierung).
- **Projekt 02** — Feature-Extraktion (Modus B) hast du dort schon gebaut; hier kommt es
  im Vergleich mit den anderen Wegen wieder vor.
- PyTorch (`nn.Conv2d`, Trainingsschleife), torchvision (`models`, `datasets.EuroSAT`),
  scikit-learn (`LogisticRegression`).

## Datensatz

**EuroSAT** — 27 000 Sentinel-2-Satellitenbilder (64×64 RGB), 10 Klassen (AnnualCrop,
Forest, River, Residential, …). Automatischer Download via torchvision (~90 MB, schnell).
Eine **andere Domäne** als ImageNet — deshalb ein ehrlicher Transfer-Test.

## Spezifikation der drei Wege

**(a) From-Scratch-CNN** — der „ohne pretrained"-Weg:
- ein kleines CNN nach Skript 2.2: `[Conv → BN → ReLU → Pool] × 3 → GlobalAvgPool → FC`;
- auf den **nativen 64×64**-Bildern; einfache **Flip-Augmentation** (horizontal *und*
  vertikal — bei Satellitenbildern gültig); Adam, ~10 Epochen. **Klein halten!**

**(b) Feature-Extraktion** (Modus B) — Backbone **einfrieren**:
- pretrained **MobileNetV3-Small** laden, Parameter einfrieren, Kopf durch `Identity`
  ersetzen; Bilder mit `weights.transforms()` vorverarbeiten (volle 224 px, korrekte
  Normalisierung) und **Feature-Vektoren** ziehen;
- darauf einen **linearen Klassifikator** (LogReg) trainieren.

**(c) Fine-Tuning** (Modus C) — pretrained Gewichte **teilweise** weitertrainieren:
- neuen Kopf für 10 Klassen setzen; **nur** den letzten Feature-Block **+** den Kopf
  auftauen (Rest eingefroren → billig), mit **kleiner Lernrate** trainieren;
- **bewusst klein**: reduzierte Auflösung (z. B. 96 px), kleines Subset, wenige Epochen.

## Milestones

1. **Daten**: EuroSAT laden (Tensor-Sicht für (a); PIL-Sicht für (b)/(c)); Train/Test-Split.
2. **(a)** From-Scratch-CNN definieren + trainieren; Test-Accuracy messen.
3. **(b)** Feature-Extraktor bauen, Features ziehen, Klassifikator trainieren.
4. **(c)** Fine-Tune-Modell (Kopf + letzter Block) kurz trainieren.
5. **Vergleich + Fehleranalyse** (Konfusionsmatrix, schwächste Klassen) und **Analyse** (s. u.).

## Was am Ende funktionieren soll

Referenz-Größenordnungen (fester Seed, CPU, ~2–3 min):

| Weg | Test-Accuracy |
|---|---|
| (a) From scratch (kein pretrained) | **~0.76** |
| (b) Feature-Extraktion (@224, eingefroren) | **~0.94** |
| (c) Fine-Tuning (klein, @96) | **~0.75** |

Beide Transfer-Wege sind **konkurrenzfähig oder besser** als From-Scratch, und die
**Feature-Extraktion bei voller Auflösung dominiert klar** — bei minimalem Rechenaufwand.

## Analyse (schriftlich, `ANALYSE.md`)

Belege/erkläre:

1. **Warum schlägt Feature-Extraktion das From-Scratch-CNN**, obwohl das Backbone **nie**
   auf Satellitenbildern trainiert wurde? (Bezug: übertragbare Merkmale, Skript 3.1.)
2. **Warum liegt das *billige* Fine-Tuning hier *unter* der Feature-Extraktion**, obwohl
   Fine-Tuning „mächtiger" ist? Welche zwei bewussten Sparmaßnahmen bremsen es (Auflösung,
   Epochen/Datenmenge), und was würdest du ändern, um es über (b) zu heben — und zu welchem
   Preis?
3. **Datenbudget:** From-Scratch erreicht ~0.76 — würde es mit **10×** mehr Bildern und
   Rechenzeit die Transfer-Wege einholen? Warum (nicht)? (Bezug: Skript 3.3.)
4. **Normalisierung:** Was passiert, wenn du die pretrained Modelle *ohne* die
   ImageNet-Normalisierung (`weights.transforms()`) fütterst? Warum?

## Bewertungsmaßstab (Master-Niveau)

- **Alle drei Wege korrekt** umgesetzt — insbesondere: Backbone in (b) wirklich eingefroren
  und mit korrekter Vorverarbeitung; in (c) nur die vorgesehenen Teile auftauen.
- Saubere, reproduzierbare **Evaluation** + Fehleranalyse.
- **Analyse**, die From-Scratch vs. Transfer quantitativ *und* konzeptuell einordnet und
  den Rechenaufwand-/Genauigkeits-Kompromiss ehrlich diskutiert.
- **CPU-freundlich** geblieben (keine überzogenen Trainings).

## Setup

```bash
source ../../../../.venv/bin/activate
# eigene Umsetzung bauen, dann z. B.:
#   python run.py         # trainiert & vergleicht alle drei Wege (~2-3 min)
#   python test_cv.py     # Testsuite (schnell)
```

Benötigt `torch`, `torchvision`, `scikit-learn`, `numpy`. EuroSAT lädt beim ersten Lauf
nach `datasets/`.

## Musterlösung

Vollständig in [`solution/`](solution/): `data.py`, `model.py`, `train_scratch.py`,
`transfer.py`, `run.py`, `test_cv.py` (5 Tests grün). Referenz: (a) ~0.76, (b) ~0.94,
(c) ~0.75. **Erst selbst bauen.**
