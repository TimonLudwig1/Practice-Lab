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
