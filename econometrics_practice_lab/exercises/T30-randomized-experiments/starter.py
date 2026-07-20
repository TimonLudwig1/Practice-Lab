"""Starter für T30: Randomisierte Experimente und RCTs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
BASELINES = ["baseline_score", "motivation_score", "digital_access"]


def make_data(size: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    baseline = rng.normal(60, 10, size=size)
    motivation = rng.normal(size=size)
    digital_access = rng.binomial(1, 0.72, size=size)
    outcome_zero = 20 + 0.72 * baseline + 3 * motivation + 2 * digital_access + rng.normal(0, 7, size=size)
    effect = 5 + 0.8 * motivation
    outcome_one = outcome_zero + effect
    treatment = np.zeros(size, dtype=int)
    treatment[rng.choice(size, size=size // 2, replace=False)] = 1
    observed = np.where(treatment == 1, outcome_one, outcome_zero)
    return pd.DataFrame(
        {"student_id": np.arange(1, size + 1), "baseline_score": baseline,
         "motivation_score": motivation, "digital_access": digital_access,
         "treatment": treatment, "outcome_y0": outcome_zero, "outcome_y1": outcome_one,
         "observed_exam_score": observed}
    )


def main() -> None:
    data = make_data()

    # TODO: Berechne Sample ATE, beobachtete Differenz und Baseline-Balance.
    # TODO: Wiederhole die vollständige Randomisierung 1.500-mal.
    # TODO: Analysiere Verteilung und laufenden Mittelwert der Schätzwerte.
    # TODO: Speichere Daten, Tabellen und die RCT-Grafik.


if __name__ == "__main__":
    main()
