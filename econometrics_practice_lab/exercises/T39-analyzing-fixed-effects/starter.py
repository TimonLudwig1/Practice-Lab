"""Starter für T39: Fixed Effects direkt schätzen und analysieren."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
TRUE_BETA = 1.2


def make_data(hospitals: int = 60, max_periods: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    quality = rng.normal(0, 5, hospitals)
    urban = (quality + rng.normal(0, 5, hospitals) > 0).astype(int)
    rows = []
    for hospital in range(hospitals):
        observed_periods = np.sort(rng.choice(np.arange(1, max_periods + 1),
                                              size=rng.integers(4, max_periods + 1), replace=False))
        for period in observed_periods:
            staffing = 18 + 0.45 * quality[hospital] + rng.normal(0, 2)
            true_intercept = 70 + quality[hospital]
            patient_score = true_intercept + TRUE_BETA * staffing + rng.normal(0, 4)
            rows.append((hospital + 1, period, staffing, patient_score, true_intercept, urban[hospital]))
    return pd.DataFrame(rows, columns=["hospital_id", "period", "nurses_per_100_beds",
                                       "patient_score", "oracle_hospital_intercept", "urban"])


def main() -> None:
    data = make_data()

    # TODO: Schätze Personal-Koeffizient und Krankenhausdummies.
    # TODO: Rekonstruiere Intercepts samt Standardfehlern und Zentrierung.
    # TODO: Analysiere Paneltiefe, Rangstabilität und urbane Unterschiede.
    # TODO: Speichere Daten, Tabellen und die Fixed-Effect-Grafik.


if __name__ == "__main__":
    main()
