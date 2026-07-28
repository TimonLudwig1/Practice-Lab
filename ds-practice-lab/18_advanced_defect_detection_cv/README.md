# 18 — Project Brief: Automated Casting Defect Detection

Difficulty: ⚫ Advanced / Portfolio | Topic: Computer Vision (TensorFlow)

---

## Project Brief

**From:** Head of Quality Assurance, MetalCast GmbH
**To:** Data Science Team
**Re:** Pilot — automated visual inspection of pump impeller castings

We manufacture submersible pump impellers. Today, every casting is inspected manually; inspectors miss defects when fatigued (audit estimate: 4–7% escape rate) and we lose ~40 person-hours weekly. We want a pilot study answering: can a camera + model replace the first inspection pass, with humans only reviewing flagged or uncertain parts?

## Business Context
- A defective impeller shipped to a customer triggers warranty + reputation cost ≈ 300€/unit
- A false rejection costs a needless manual re-inspection ≈ 2€/unit
- Production volume: ~6,000 units/week
- The model's recommended operating point must be justified against these costs

## Data
Kaggle: "casting product image data for quality inspection" — ~7,300 grayscale images (300×300) of impeller castings, labeled `ok_front` / `def_front`, with a predefined train/test split.
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

## Technical Constraints
- TensorFlow/Keras
- Inference must run on CPU at ≥10 images/second (measure and report)
- The provided test split is the acceptance set: it may be evaluated **once**, by your final chosen model
- Reproducible training: fixed seeds documented, one-command training script

## Deliverables
1. Training pipeline as Python modules (data loading/augmentation, model, training, evaluation) + a short notebook presenting results
2. Two model families compared: a compact CNN trained from scratch vs transfer learning on a pretrained backbone
3. Cost-optimal decision threshold derived from the economics above, with the cost curve plotted
4. Error gallery: every false negative on the acceptance set, displayed and discussed
5. A 1-page management summary: expected weekly savings, escape rate vs human baseline, and your go/no-go recommendation
6. Saved model artifact + measured CPU throughput

## Evaluation Rubric
| Criterion | Bar |
|---|---|
| Recall on defects (acceptance set) | ≥ 0.99 |
| Precision at that recall | ≥ 0.95 |
| CPU throughput | ≥ 10 img/s |
| Code quality | Modules, not notebook spaghetti; runs end-to-end with one command |
| Analysis quality | Threshold tied to € costs; failure cases examined honestly |

---

# Deutsche Übersetzung

# 18 — Projektauftrag: Automatisierte Erkennung von Gussfehlern

Schwierigkeit: ⚫ Fortgeschritten / Portfolio | Thema: Computer Vision mit TensorFlow

---

## Projektauftrag

**Von:** Leitung Qualitätssicherung, MetalCast GmbH
**An:** Data-Science-Team
**Betreff:** Pilotprojekt zur automatisierten Sichtprüfung gegossener Pumpenlaufräder

Wir fertigen Laufräder für Tauchpumpen. Derzeit wird jedes Gussteil manuell geprüft. Bei Ermüdung übersehen Prüfer Fehler; laut Audit gelangen 4–7 % der Fehler durch die Kontrolle. Zugleich entstehen wöchentlich etwa 40 Arbeitsstunden Aufwand. Die Pilotstudie soll klären, ob eine Kamera mit Modell die erste Prüfung übernehmen kann und Menschen nur markierte oder unsichere Teile kontrollieren müssen.

## Geschäftlicher Kontext
- Ein fehlerhaft ausgeliefertes Laufrad verursacht Garantie- und Reputationskosten von ungefähr 300 € je Einheit.
- Eine falsche Zurückweisung verursacht eine unnötige manuelle Nachprüfung von ungefähr 2 € je Einheit.
- Produktionsvolumen: etwa 6.000 Einheiten pro Woche.
- Der empfohlene Betriebspunkt des Modells muss anhand dieser Kosten begründet werden.

## Daten
Kaggle-Datensatz „casting product image data for quality inspection“ mit etwa 7.300 Graustufenbildern von 300×300 Pixeln. Die Laufräder sind als `ok_front` oder `def_front` beschriftet; eine Trainings-/Testaufteilung ist vorgegeben.
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

## Technische Vorgaben
- TensorFlow/Keras
- Inferenz auf der CPU mit mindestens zehn Bildern pro Sekunde; messen und dokumentieren
- Die vorgegebene Testaufteilung ist der Abnahmesatz und darf **einmal** mit dem endgültig gewählten Modell ausgewertet werden.
- Reproduzierbares Training mit dokumentierten festen Seeds und einem Trainingsskript, das mit einem Befehl ausgeführt wird

## Liefergegenstände
1. Trainingspipeline als Python-Module für Laden und Augmentieren der Daten, Modell, Training und Bewertung sowie ein kurzes Notebook zur Ergebnisdarstellung
2. Vergleich zweier Modellfamilien: kompaktes selbst trainiertes CNN und Transfer Learning mit vortrainiertem Backbone
3. Aus den Kosten abgeleiteter optimaler Entscheidungsschwellenwert mit dargestellter Kostenkurve
4. Fehlergalerie aller falsch negativen Fälle des Abnahmesatzes mit Diskussion
5. Einseitige Management-Zusammenfassung mit erwarteter wöchentlicher Einsparung, Durchschlupfrate im Vergleich zur manuellen Prüfung und Go-/No-Go-Empfehlung
6. Gespeichertes Modellartefakt und gemessener CPU-Durchsatz

## Bewertungskriterien
| Kriterium | Anforderung |
|---|---|
| Recall für Defekte im Abnahmesatz | ≥ 0,99 |
| Precision bei diesem Recall | ≥ 0,95 |
| CPU-Durchsatz | ≥ 10 Bilder/s |
| Codequalität | Module statt unstrukturierter Notebooks; vollständige Ausführung mit einem Befehl |
| Analysequalität | Schwellenwert an Eurokosten gebunden; Fehlerfälle ehrlich untersucht |
