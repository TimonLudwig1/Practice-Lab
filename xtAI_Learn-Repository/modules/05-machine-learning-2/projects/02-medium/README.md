# Projekt 02 (medium): CNN-Bildklassifikation mit PyTorch — Ablationsstudie

## Ziel

Ein Convolutional Neural Network in PyTorch bauen und drei Skript-Behauptungen **experimentell prüfen**, sauber als Ablation (immer nur eine Stellgröße ändern):

1. **Architektur-Bias:** CNN (≈66k Parameter) vs. MLP (≈235k Parameter) — schlägt Struktur rohe Kapazität?
2. **Regularisierung:** Dropout + Weight Decay (AdamW) + Datenaugmentierung — was passiert kurzfristig (3 Epochen) vs. langfristig (10 Epochen) mit der Train/Test-Schere?
3. **Optimierer:** Adam vs. SGD+Momentum bei identischem Budget.

Dazu Fehlerdiagnose: Confusion Matrix und die konfidentesten Fehlklassifikationen.

**Warum dieses Format (Jupyter Notebook):** Die Ablation lebt von Lernkurven-Plots direkt neben dem Trainingscode; ein Notebook dokumentiert Experiment und Befund in einem.

**Warum echte Daten (Fashion-MNIST):** 70 000 echte Zalando-Produktbilder, 10 Klassen — anspruchsvoller als MNIST (Shirt/T-Shirt/Pullover sind auch für Menschen schwer), aber klein genug für CPU/Apple-GPU. `torchvision` lädt automatisch (~30 MB nach `datasets/`, per `.gitignore` vom Repo ausgeschlossen). Für die Ablation nutzen wir 20 000 Trainingsbilder (Konstante `SUBSET` — auf 60 000 stellen für Bestleistung).

## Vorwissen

- Projekt 01 (du weißt, was `loss.backward()` intern tut)
- Skript 2.1 (Optimierer), 2.3 (Regularisierung), 2.4 (BatchNorm), 2.5 (CNNs)

## Setup

```bash
source .venv/bin/activate
jupyter lab modules/05-machine-learning-2/projects/02-medium/image_classification_cnn.ipynb
```

Läuft auf Apple-GPU (`mps`), CUDA oder CPU — wird automatisch erkannt. Referenz-Laufzeit auf einem M-Series-Mac: ca. 5–10 Minuten für das ganze Notebook.

## Aufgabenstellung (Schritt für Schritt)

1. **Daten ansehen** (fertig vorgegeben): Lade-Pipeline mit Normalisierung und einem zweiten, augmentierten Loader (RandomCrop + Flip).
2. **TODO 1 — CNN bauen:** zwei `[Conv3×3→BN→ReLU]×2→MaxPool`-Blöcke (32, 64 Kanäle) → Global Average Pooling → Dropout → Linear. Eine Assert-Zelle prüft die Parameterzahl (**erwartet: 66 570** — rechne sie vorher von Hand nach, Formel im Skript 2.5).
3. **TODO 2 — `evaluate` reparieren:** Eine entscheidende Zeile fehlt. Ohne sie sind alle Testmetriken falsch — finde sie und erkläre, welche zwei Mechanismen betroffen sind.
4. **TODO 3 — Ablations-Konfigurationen** vervollständigen (Adam / AdamW+wd / SGD+Momentum, Augmentierungs-Loader für Konfiguration C).
5. **Ablation ausführen und interpretieren** — vergleiche mit den Erwartungen im Notebook-Text.
6. **Fehlerdiagnose:** Confusion Matrix; welche Klassenpaare verwechselt das Netz und warum?
7. **Langzeit-Experiment:** 10 Epochen pur vs. reguarisiert — sichtbare Overfitting-Schere?

## Was am Ende funktionieren soll

- Parameterzahl-Assert besteht (66 570).
- CNN schlägt MLP deutlich (Referenz nach 3 Epochen auf 20k-Subset: CNN ≈ 0,90–0,91, MLP ≈ 0,86–0,88 Test-Accuracy).
- Du kannst anhand deiner Kurven erklären: warum die regularisierte Variante nach 3 Epochen noch nicht vorn liegt, nach 10 aber die kleinere Train/Test-Schere hat.

## Musterlösung

[`solution/solution.ipynb`](solution/solution.ipynb) — vollständig ausgeführt, mit allen Kurven, Tabellen und Interpretationstexten.
